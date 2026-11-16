"""Interactive ML03 attack sessions: the ONLY attacker-facing way to query the victim over HTTP.

Roles
-----
* **Evaluator** creates a session (``POST /api/inversion/sessions``): it fixes the victim model, seed, output policy,
  the in-scope targets and the query budgets. The attacker cannot choose or change any of them.
* **Attacker** holds the session id — a 128-bit random capability, never chosen by the client — and can only
  read the session's attacker view (public features + observed outputs of its targets, budget state), send
  candidate queries for its targets, and submit guesses once.
* **Evaluator** scores a session only after the attacker has submitted (``GET …/score``).

Query accounting is server-side and persisted (one JSON file per session under the evidence root), updated under
a lock BEFORE a response is released, so the budget survives restarts and concurrent requests. Every call counts —
repeating a query costs budget like a new one. A query names an in-scope target and a candidate value only: the
server builds the crafted record from that target's public features, so the session cannot be used to probe the
model on arbitrary inputs (which would be a model-extraction surface, ML05). There is no reset operation; the
only way to get more queries is a NEW session, which only the evaluator can create, and the number of sessions
per evidence root is capped.

Session files contain no ground truth: the private attribute is read from the regenerated dataset only when an
evaluator scores a submitted session. Anyone with write access to the evidence directory can edit a session file
(integrity is not authenticity; see docs/reproducibility.md).
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from ..config import LIMITS
from ..hashing import sha256_json
from ..metrics import ratio
from ..schemas import Strict
from ..stats import mcnemar_one_sided_p
from .attacks import auxiliary_guess
from .data import PRIVATE_ATTRIBUTE, PUBLIC_FEATURES, load_inversion_dataset
from .defenses import PredictionService, describe_policy
from .metrics import LEAKAGE_ALPHA
from .models import auxiliary_model, victim_model
from .schemas import FullProbability, ModelId, OutputPolicyConfig, TargetSelection

SESSION_ID_PATTERN = r"^MLB-SES-[0-9a-f]{32}$"
_ID_RE = re.compile(SESSION_ID_PATTERN)
SessionId = Annotated[str, Field(pattern=SESSION_ID_PATTERN)]


# --------------------------------------------------------------------------------------------- requests
class SessionCreateRequest(Strict):
    """EVALUATOR-facing. No field accepts a session id: ids are generated server-side."""

    model_id: ModelId = "logreg"
    seed: int = Field(7, ge=0, le=LIMITS.max_inversion_seed)
    output_policy: OutputPolicyConfig = Field(default_factory=FullProbability)
    targets: TargetSelection = Field(default_factory=lambda: TargetSelection(max_targets=20))
    query_budget_per_target: int = Field(2, ge=1, le=LIMITS.max_inversion_query_budget)
    query_budget_total: int | None = Field(None, ge=1, le=LIMITS.max_session_queries,
                                           description="defaults to per-target budget × targets (capped)")


class SessionQueryRequest(Strict):
    """ATTACKER-facing: an in-scope target and a candidate value. No features, policy or model can be supplied."""

    target_index: int = Field(ge=0, lt=LIMITS.max_inversion_targets)
    candidate: Literal[0, 1]


class SessionGuess(Strict):
    target_index: int = Field(ge=0, lt=LIMITS.max_inversion_targets)
    inferred: Literal[0, 1] | None  # None = abstain


class SessionSubmitRequest(Strict):
    guesses: list[SessionGuess] = Field(max_length=LIMITS.max_inversion_targets)


# ------------------------------------------------------------------------------ attacker-facing responses
class BudgetState(Strict):
    per_target: int
    total: int
    used_total: int
    remaining_total: int
    used_per_target: dict[str, int]


class HiddenAttribute(Strict):
    name: str
    domain: list[int]
    value: Literal["hidden"] = "hidden"


class AttackerTargetView(Strict):
    target_index: int
    public_features: list[float]
    observed_output: float


class AttackerSessionView(Strict):
    """Everything the attacker may know about a session. Response model of the attacker-facing endpoints: FastAPI
    serialises ONLY these fields."""

    session_id: str
    status: Literal["open", "submitted"]
    model_id: str
    seed: int
    output_policy: dict[str, Any]
    public_feature_names: list[str]
    hidden_attribute: HiddenAttribute
    targets: list[AttackerTargetView]
    budget: BudgetState


class AttackerQueryResponse(Strict):
    session_id: str
    target_index: int
    candidate: int
    released: float
    query_number: int
    budget: BudgetState


class AttackerSubmitReceipt(Strict):
    session_id: str
    status: Literal["submitted"]
    n_guesses: int
    guesses_sha256: str
    queries_used: int
    submitted_at: str


# -------------------------------------------------------------------------------- evaluator-facing responses
class TargetsPreview(Strict):
    """EVALUATOR-facing preview of targets (public features + observed output under a chosen policy)."""

    model_id: str
    seed: int
    output_policy: dict[str, Any]
    public_features: list[str]
    hidden_attribute: HiddenAttribute
    targets: list[AttackerTargetView]


class SessionTargetScore(BaseModel):
    target_index: int
    inferred: int | None
    private_attribute: int
    correct: bool
    baseline_guess: int
    baseline_correct: bool
    queries: int


class SessionScore(BaseModel):
    """EVALUATOR-facing: only available after the attacker has submitted."""

    session_id: str
    model_id: str
    seed: int
    output_policy: dict[str, Any]
    n_targets: int
    attribute_inference_accuracy: dict[str, Any]
    baseline_inference_accuracy: dict[str, Any]
    inference_advantage: dict[str, Any]
    abstention_rate: dict[str, Any]
    leakage_p_value: dict[str, Any]
    queries_used: int
    query_budget_total: int
    query_log_sha256: str
    targets: list[SessionTargetScore]


# ---------------------------------------------------------------------------------------------- errors
class SessionNotFound(KeyError):
    """Unknown or malformed session id (404)."""


class SessionRefused(PermissionError):
    """A session operation that is not permitted (403). ``code`` is machine-readable."""

    def __init__(self, code: str, message: str, budget: dict[str, Any] | None = None):
        super().__init__(message)
        self.code, self.message, self.budget = code, message, budget


class SessionNotSubmitted(RuntimeError):
    """Scoring requested before the attacker submitted (409)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------------------------- store
class SessionStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "inversion_sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        if not _ID_RE.fullmatch(session_id or ""):  # also rules out path traversal
            raise SessionNotFound(session_id)
        return self.dir / f"{session_id}.json"

    def _load(self, session_id: str) -> dict[str, Any]:
        p = self._path(session_id)
        if not p.is_file():
            raise SessionNotFound(session_id)
        return json.loads(p.read_text())

    def _save(self, state: dict[str, Any]) -> None:
        p = self._path(state["session_id"])
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(p)

    # -------------------------------------------------------------------- evaluator
    def create(self, req: SessionCreateRequest) -> AttackerSessionView:
        from .engine import select_targets
        ds = load_inversion_dataset(req.seed)
        idx = select_targets(req.targets, req.seed, len(ds.s_eval))
        cap = min(req.query_budget_per_target * len(idx), LIMITS.max_session_queries)
        total = cap if req.query_budget_total is None else min(req.query_budget_total, cap)
        with self._lock:
            if len(list(self.dir.glob("MLB-SES-*.json"))) >= LIMITS.max_inversion_sessions:
                raise SessionRefused("session_limit_reached",
                                     f"at most {LIMITS.max_inversion_sessions} sessions per evidence root")
            state = {"schema_version": "mlbreak.inversion-session/v1", "session_id": f"MLB-SES-{secrets.token_hex(16)}",
                     "created_at": _now(), "status": "open", "request": req.model_dump(mode="json"),
                     "model_id": req.model_id, "seed": req.seed, "output_policy": req.output_policy.model_dump(),
                     "target_indices": idx, "budget_per_target": req.query_budget_per_target, "budget_total": total,
                     "used_total": 0, "used_per_target": {str(i): 0 for i in idx}, "query_log": [],
                     "guesses": None, "submitted_at": None}
            self._save(state)
        return self._view(state)

    def score(self, session_id: str) -> SessionScore:
        state = self._load(session_id)
        if state["status"] != "submitted":
            raise SessionNotSubmitted(f"session {session_id} has not been submitted; scoring before submission "
                                      "would reveal evaluation results to an attacker who can still query")
        ds = load_inversion_dataset(state["seed"])
        idx = state["target_indices"]
        guesses = {int(k): v for k, v in state["guesses"].items()}
        base = auxiliary_guess(auxiliary_model(state["seed"]).proba(ds.X_eval[idx]))
        rows = []
        for i, b in zip(idx, base):
            truth, g = int(ds.s_eval[i]), guesses.get(i)
            rows.append(SessionTargetScore(target_index=i, inferred=g, private_attribute=truth,
                                           correct=g is not None and g == truth, baseline_guess=int(b),
                                           baseline_correct=int(b) == truth,
                                           queries=state["used_per_target"][str(i)]))
        n = len(rows)
        inv, bas = [r.correct for r in rows], [r.baseline_correct for r in rows]
        pb = sum(a and not c for a, c in zip(inv, bas))
        pc = sum(c and not a for a, c in zip(inv, bas))
        p = mcnemar_one_sided_p(pb, pc) if n else None
        pop = "targets in scope of this session"
        return SessionScore(
            session_id=session_id, model_id=state["model_id"], seed=state["seed"],
            output_policy=describe_policy(state["output_policy"]), n_targets=n,
            attribute_inference_accuracy=ratio(sum(inv), n, "submitted guesses equal to the private attribute "
                                                           "(abstention = not correct) / targets", pop),
            baseline_inference_accuracy=ratio(sum(bas), n, "auxiliary-only guesses equal to the private attribute "
                                                           "/ targets (same targets)", pop),
            inference_advantage=ratio(sum(inv) - sum(bas), n, "(correct submitted − correct auxiliary-only) / "
                                                               "targets — signed", pop),
            abstention_rate=ratio(sum(r.inferred is None for r in rows), n, "abstentions / targets", pop),
            leakage_p_value={"value": p, "n": n, "inversion_only_correct": pb, "baseline_only_correct": pc,
                             "alpha": LEAKAGE_ALPHA,
                             "leakage_claimed": bool(p is not None and p < LEAKAGE_ALPHA and pb > pc),
                             "definition": "exact one-sided McNemar, submitted guesses vs auxiliary-only attacker"},
            queries_used=state["used_total"], query_budget_total=state["budget_total"],
            query_log_sha256=sha256_json(state["query_log"]), targets=rows)

    # --------------------------------------------------------------------- attacker
    def view(self, session_id: str) -> AttackerSessionView:
        return self._view(self._load(session_id))

    def query(self, session_id: str, req: SessionQueryRequest) -> AttackerQueryResponse:
        with self._lock:  # check, count and persist BEFORE releasing the response
            state = self._load(session_id)
            if state["status"] != "open":
                raise SessionRefused("session_closed", "the session was submitted; no further queries are accepted")
            key = str(req.target_index)
            if key not in state["used_per_target"]:
                raise SessionRefused("target_out_of_scope",
                                     f"target {req.target_index} is not in scope of this session")
            if state["used_total"] >= state["budget_total"]:
                raise SessionRefused("query_budget_exhausted", "the session's total query budget is exhausted",
                                     self._budget(state).model_dump())
            if state["used_per_target"][key] >= state["budget_per_target"]:
                raise SessionRefused("query_budget_exhausted",
                                     f"the per-target query budget for target {req.target_index} is exhausted",
                                     self._budget(state).model_dump())
            ds = load_inversion_dataset(state["seed"])
            service = PredictionService(victim_model(state["model_id"], state["seed"]), state["output_policy"])
            released = float(service.respond(ds.X_eval[req.target_index:req.target_index + 1],
                                             np.array([req.candidate]))[0])
            state["used_total"] += 1
            state["used_per_target"][key] += 1
            state["query_log"].append({"n": state["used_total"], "target_index": req.target_index,
                                       "candidate": req.candidate, "released": released})
            self._save(state)
        return AttackerQueryResponse(session_id=session_id, target_index=req.target_index, candidate=req.candidate,
                                     released=released, query_number=state["used_total"], budget=self._budget(state))

    def submit(self, session_id: str, req: SessionSubmitRequest) -> AttackerSubmitReceipt:
        with self._lock:
            state = self._load(session_id)
            if state["status"] != "open":
                raise SessionRefused("already_submitted", "guesses were already submitted for this session")
            idx = state["target_indices"]
            got = [g.target_index for g in req.guesses]
            if sorted(got) != sorted(idx) or len(set(got)) != len(got):
                raise SessionRefused("invalid_submission", "submit exactly one guess (0, 1 or null) for every "
                                                           "target in scope, and no other target")
            state["guesses"] = {str(g.target_index): g.inferred for g in req.guesses}
            state["status"], state["submitted_at"] = "submitted", _now()
            self._save(state)
        return AttackerSubmitReceipt(session_id=session_id, status="submitted", n_guesses=len(req.guesses),
                                     guesses_sha256=sha256_json(state["guesses"]), queries_used=state["used_total"],
                                     submitted_at=state["submitted_at"])

    # --------------------------------------------------------------------- helpers
    @staticmethod
    def _budget(state: dict[str, Any]) -> BudgetState:
        return BudgetState(per_target=state["budget_per_target"], total=state["budget_total"],
                           used_total=state["used_total"], remaining_total=state["budget_total"] - state["used_total"],
                           used_per_target=dict(state["used_per_target"]))

    def _view(self, state: dict[str, Any]) -> AttackerSessionView:
        ds = load_inversion_dataset(state["seed"])
        idx = state["target_indices"]
        service = PredictionService(victim_model(state["model_id"], state["seed"]), state["output_policy"])
        observed = service.respond(ds.X_eval[idx], ds.s_eval[idx])  # the real records' released outputs
        return AttackerSessionView(
            session_id=state["session_id"], status=state["status"], model_id=state["model_id"], seed=state["seed"],
            output_policy=describe_policy(state["output_policy"]), public_feature_names=list(PUBLIC_FEATURES),
            hidden_attribute=HiddenAttribute(name=PRIVATE_ATTRIBUTE, domain=[0, 1]),
            targets=[AttackerTargetView(target_index=i, public_features=ds.X_eval[i].tolist(), observed_output=float(o))
                     for i, o in zip(idx, observed)],
            budget=self._budget(state))
