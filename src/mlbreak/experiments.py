"""Experiment engine: Baseline → Attack → Defend, replay and evidence.

Three distinct evaluations are kept deliberately separate:

* ``attack`` — a new attack generated against the **undefended** model.
* ``replay`` (mode ``defend``) — the *exact* adversarial inputs of an earlier
  attack experiment are re-submitted to a **defended** system. This measures a
  non-adaptive attacker (perturbations crafted without knowledge of the defense).
* ``adaptive_attack`` (mode ``attack`` with ``target_defense``) — a **new**
  attack generated against the defended system by an attacker who knows the
  defense. Stored as its own experiment, linked to its parent.

:meth:`ExperimentEngine.replay` is something else again: it re-executes a
stored experiment from its recorded request and checks the results are
bit-for-bit identical (reproducibility, not robustness).

Lineage: a parent experiment (for ``defend`` and for adaptive attacks) must be a
completed, non-adaptive attack against the undefended model, must pass the
evidence integrity check, and must share scenario, model, seed, dataset split and
base-model artifact with the child. Adaptive attacks must also use the same
perturbation budget (norm, ε) and the same test samples, so the comparison is
like for like. The validated provenance is stored in ``lineage``.

Execution limits: every experiment runs inside a concurrency gate and against a
wall-clock deadline that covers model training, adversarial training, the attack
loops and evaluation. Deadlines are cooperative (checked between fits, rounds,
attack steps and samples); see ``docs/reproducibility.md`` for the bound.
"""

from __future__ import annotations

import platform
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import sklearn

from . import __version__
from .attacks import (ABSTAIN, BudgetViolation, ExperimentTimeout, QueryBudgetExceeded, _check_deadline,
                      enforce_constraints, gradient_attack, lp_norm, random_search_attack)
from .config import LIMITS
from .data import load_dataset
from .defenses import DeployedSystem, base_model, build_system, logit_of
from .evidence import EvidenceStore
from .hashing import sha256_json
from .integrity import (metrics_sha256, record_sha256, results_fingerprint, samples_sha256, verify_record)
from .metrics import classify, clean_metrics, outcome_metrics
from .owasp import get_category
from .schemas import (SCHEMA_V1, AttackRequest, BaselineRequest, BoundaryRequest, DefendRequest, ExperimentRecord,
                      SampleResult)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    owasp_id: str
    dataset_id: str
    intentionally_vulnerable: bool
    description: str


SCENARIOS: dict[str, Scenario] = {
    "ml01-moons-evasion": Scenario(
        scenario_id="ml01-moons-evasion",
        title="Evasion of a 2-D classifier with bounded test-time perturbations",
        owasp_id="ML01:2023",
        dataset_id="moons-v1",
        intentionally_vulnerable=True,
        description=(
            "A classifier trained on a synthetic two-moons dataset is attacked at "
            "inference time. The attacker perturbs a correctly classified input "
            "within an Lp budget ε so the deployed system returns the wrong class."
        ),
    )
}


class EngineBusy(RuntimeError):
    """Raised when the concurrent-experiment limit is reached."""


class InvalidExperiment(ValueError):
    """The request is well-formed but cannot be executed (e.g. bad parent)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return f"MLB-EXP-{uuid.uuid4().hex[:12]}"


def _stats(linf: np.ndarray, l2: np.ndarray) -> dict[str, Any]:
    if len(linf) == 0:
        return {"n": 0}
    return {"n": int(len(linf)), "linf_mean": float(np.mean(linf)), "linf_max": float(np.max(linf)),
            "linf_median": float(np.median(linf)), "l2_mean": float(np.mean(l2)),
            "l2_max": float(np.max(l2)), "l2_median": float(np.median(l2))}


def _blas_threads() -> Any:
    try:
        from threadpoolctl import threadpool_info  # shipped with scikit-learn
        return [{"api": i.get("user_api"), "lib": i.get("internal_api"), "threads": i.get("num_threads")}
                for i in threadpool_info()]
    except Exception:  # pragma: no cover - informational only
        return None


# Which record fields are expected to be bit-for-bit identical on replay, and which are not.
REPRODUCIBLE_FIELDS = {
    "bitwise": [
        "request / config_sha256", "dataset.train_sha256", "dataset.test_sha256", "model.artifact_sha256",
        "target_system.model_artifact_sha256", "samples (x_adversarial, predictions, probabilities, queries, outcome)",
        "metrics", "samples_sha256", "metrics_sha256", "results_sha256", "lineage (parent hashes)",
    ],
    "runtime": [
        "experiment_id", "created_at", "duration_ms", "replay_of", "record_sha256 (covers runtime fields)",
        "environment versions / platform / blas_threads", "status when a wall-clock timeout is hit",
    ],
    "scope": "bit-for-bit on the same platform, CPU architecture, BLAS build and library versions",
}


def reproducibility_info() -> dict[str, Any]:
    import pydantic
    return {
        "mlbreak_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pydantic": pydantic.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "blas_threads": _blas_threads(),
        "deterministic": True,
        "fields": REPRODUCIBLE_FIELDS,
        "note": "Bit-identical replay is expected on the same platform and library versions.",
    }


ENVIRONMENT_KEYS = ("mlbreak_version", "python", "numpy", "scikit_learn", "platform", "machine")


class ExperimentEngine:
    def __init__(self, store: EvidenceStore, max_concurrent: int = LIMITS.max_concurrent_experiments,
                 time_limit: float = LIMITS.max_experiment_seconds):
        self.store = store
        self._gate = threading.BoundedSemaphore(max_concurrent)
        self.time_limit = time_limit
        # Only one regression suite at a time (see regression.run_suite).
        self.suite_lock = threading.Lock()

    # ------------------------------------------------------------------ API
    def run_baseline(self, req: BaselineRequest, replay_of: str | None = None) -> ExperimentRecord:
        return self._run("baseline", req, self._baseline, replay_of)

    def run_attack(self, req: AttackRequest, replay_of: str | None = None) -> ExperimentRecord:
        if req.parent_experiment_id and req.target_defense is None:
            raise InvalidExperiment("parent_experiment_id is only valid for adaptive attacks (set target_defense); "
                                    "a non-adaptive attack against the undefended model has no parent")
        parent = self._load_parent(req.parent_experiment_id) if req.parent_experiment_id else None
        return self._run("attack", req, lambda r, deadline: self._attack(r, deadline, parent), replay_of)

    def run_defend(self, req: DefendRequest, replay_of: str | None = None) -> ExperimentRecord:
        parent = self._load_parent(req.parent_experiment_id)  # raises ExperimentNotFound / InvalidExperiment
        return self._run("defend", req, lambda r, deadline: self._defend(r, parent, deadline), replay_of)

    def verify(self, experiment_id: str) -> dict[str, Any]:
        """Evidence integrity check of a stored record (no re-execution)."""
        return verify_record(self.store.load(experiment_id))

    def replay(self, experiment_id: str) -> dict[str, Any]:
        """Re-execute a stored experiment from its request and compare results exactly."""
        original = self.store.load(experiment_id)
        if original.status != "completed":
            raise InvalidExperiment("only completed experiments can be replayed")
        if original.mode == "baseline":
            new = self.run_baseline(BaselineRequest.model_validate(original.request), replay_of=experiment_id)
        elif original.mode == "attack":
            new = self.run_attack(AttackRequest.model_validate(original.request), replay_of=experiment_id)
        else:
            new = self.run_defend(DefendRequest.model_validate(original.request), replay_of=experiment_id)

        o_rep, n_rep = original.reproducibility, new.reproducibility
        legacy = original.schema_version == SCHEMA_V1
        pairs = [
            ("status", original.status, new.status),
            ("config_sha256", o_rep.get("config_sha256"), n_rep.get("config_sha256")),
            ("model_artifact_sha256", original.model["artifact_sha256"], new.model.get("artifact_sha256")),
            ("target_system_artifact_sha256", original.target_system["model_artifact_sha256"],
             new.target_system.get("model_artifact_sha256")),
            ("train_split_sha256", original.dataset["train_sha256"], new.dataset.get("train_sha256")),
            ("test_split_sha256", original.dataset["test_sha256"], new.dataset.get("test_sha256")),
            ("samples_sha256", samples_sha256(original.samples), n_rep.get("samples_sha256")),
        ]
        if not legacy:
            pairs += [("metrics_sha256", o_rep.get("metrics_sha256"), n_rep.get("metrics_sha256")),
                      ("results_sha256", o_rep.get("results_sha256"), n_rep.get("results_sha256"))]
        checks = [{"check": n, "original": a, "replay": b, "match": a == b} for n, a, b in pairs]
        env_diff = {k: {"original": o_rep.get(k), "replay": n_rep.get(k)} for k in ENVIRONMENT_KEYS
                    if o_rep.get(k) != n_rep.get(k)}
        notes = []
        if legacy:
            notes.append("Original record uses schema v1, whose adversarial_accuracy definition counted abstentions "
                         "as correct; metrics are not comparable, so per-sample results are compared instead.")
        if env_diff:
            notes.append("The runtime environment differs from the original; bit-identical results are only expected "
                         "on the same platform and library versions.")
        return {"original_experiment_id": experiment_id, "replay_experiment_id": new.experiment_id,
                "identical": all(c["match"] for c in checks), "checks": checks,
                "environment_differences": env_diff, "notes": notes}

    def boundary(self, req: BoundaryRequest) -> dict[str, Any]:
        """Decision-boundary grid. Runs inside the concurrency gate and the wall-clock deadline because
        building a defended system may trigger adversarial training."""
        def work(deadline: float) -> dict[str, Any]:
            scen = SCENARIOS[req.scenario_id]
            ds = load_dataset(scen.dataset_id, req.seed)
            system = build_system(req.model_id, scen.dataset_id, req.seed, req.defense, deadline)
            xs = np.linspace(*ds.spec.domain[0], req.resolution)
            ys = np.linspace(*ds.spec.domain[1], req.resolution)
            gx, gy = np.meshgrid(xs, ys)
            d = system.decide(np.c_[gx.ravel(), gy.ravel()])
            shape = gx.shape
            return {
                "x": xs.round(5).tolist(), "y": ys.round(5).tolist(),
                "p1": d.p1.reshape(shape).round(5).tolist(),
                "abstain": (d.labels == ABSTAIN).reshape(shape).astype(int).tolist(),
                "target_system": system.describe(),
                "model": system.model.describe(),
            }
        return self._guarded(work)

    def dataset_view(self, scenario_id: str, model_id: str, seed: int) -> dict[str, Any]:
        """Training/test data and base-model predictions (gated: a new seed trains a model)."""
        def work(deadline: float) -> dict[str, Any]:
            scen = SCENARIOS[scenario_id]
            ds = load_dataset(scen.dataset_id, seed)
            model = base_model(model_id, scen.dataset_id, seed)
            _check_deadline(deadline)
            p1 = model.proba(ds.X_test)
            return {
                "scenario": scen.__dict__, "dataset": ds.describe(), "model": model.describe(),
                "train": {"X": ds.X_train.round(6).tolist(), "y": ds.y_train.tolist()},
                "test": {"X": ds.X_test.round(6).tolist(), "y": ds.y_test.tolist(),
                         "pred": model.predict(ds.X_test).tolist(), "p1": np.round(p1, 6).tolist()},
            }
        return self._guarded(work)

    # ------------------------------------------------------------ internals
    def _guarded(self, fn: Callable[[float], Any]) -> Any:
        """Run non-experiment work under the same concurrency gate and deadline as experiments."""
        if not self._gate.acquire(blocking=False):
            raise EngineBusy("too many concurrent experiments; retry shortly")
        try:
            return fn(time.monotonic() + self.time_limit)
        finally:
            self._gate.release()

    def _run(self, mode: str, req, fn, replay_of: str | None) -> ExperimentRecord:
        if not self._gate.acquire(blocking=False):
            raise EngineBusy("too many concurrent experiments; retry shortly")
        started = time.monotonic()
        exp_id = _new_id()
        try:
            try:
                body = fn(req, started + self.time_limit)
                status, error = "completed", None
            except (BudgetViolation, ExperimentTimeout, QueryBudgetExceeded) as exc:
                body, status, error = self._failure_body(req), "failed", f"{type(exc).__name__}: {exc}"
            body["metrics"] = body.get("metrics", {})
            body["samples"] = body.get("samples", [])
            repro = reproducibility_info()
            repro["config_sha256"] = sha256_json(req.model_dump(mode="json"))
            completed = status == "completed"
            repro["samples_sha256"] = samples_sha256(body["samples"]) if completed else None
            repro["metrics_sha256"] = metrics_sha256(body["metrics"]) if completed else None
            repro["results_sha256"] = results_fingerprint(body["samples"], body["metrics"]) if completed else None
            repro["replay_command"] = f"mlbreak replay {exp_id}"
            repro["verify_command"] = f"mlbreak check {exp_id}"
            record = ExperimentRecord(
                experiment_id=exp_id, created_at=_now(), mode=mode, status=status, error=error,
                replay_of=replay_of, request=req.model_dump(mode="json"), reproducibility=repro,
                duration_ms=round((time.monotonic() - started) * 1000, 2), **body,
            )
            record.reproducibility["record_sha256"] = record_sha256(record)
            self.store.save(record)
            return record
        finally:
            self._gate.release()

    def _context(self, scenario_id: str, model_id: str, seed: int, deadline: float | None = None):
        scen = SCENARIOS[scenario_id]
        cat = get_category(scen.owasp_id)
        ds = load_dataset(scen.dataset_id, seed)
        model = base_model(model_id, scen.dataset_id, seed)
        _check_deadline(deadline)
        common = {
            "scenario_id": scen.scenario_id,
            "owasp": {"id": cat.id, "name": cat.name, "taxonomy": cat.taxonomy,
                      "taxonomy_version": cat.taxonomy_version, "reference_url": cat.reference_url},
            "model": model.describe(),
            "dataset": ds.describe(),
            "seed": seed,
        }
        return scen, ds, common

    def _failure_body(self, req) -> dict[str, Any]:
        r = req
        if isinstance(req, DefendRequest):
            parent = self.store.load(req.parent_experiment_id)
            r = AttackRequest.model_validate(parent.request)
        _, ds, common = self._context(r.scenario_id, r.model_id, r.seed)
        evaluation = "clean" if isinstance(req, BaselineRequest) else (
            "replay" if isinstance(req, DefendRequest) else
            "adaptive_attack" if getattr(req, "target_defense", None) is not None else "attack")
        return {**common, "evaluation": evaluation, "intentionally_vulnerable": evaluation in ("attack", "clean"),
                "attacker_capabilities": None, "target_system": {"model_artifact_sha256": None},
                "attack_config": getattr(r, "attack", None) and r.attack.model_dump(),
                "defense_config": None, "parent_experiment_id": getattr(req, "parent_experiment_id", None)}

    # ------------------------------------------------------------- lineage
    def _load_parent(self, parent_id: str) -> ExperimentRecord:
        """Load a parent and check its own validity (status, evaluation type, evidence integrity)."""
        parent = self.store.load(parent_id)  # raises ExperimentNotFound
        if parent.status != "completed":
            raise InvalidExperiment(f"parent {parent_id} did not complete (status {parent.status})")
        if parent.mode != "attack" or parent.evaluation != "attack":
            raise InvalidExperiment(
                f"parent {parent_id} must be a non-adaptive attack against the undefended model "
                f"(got mode={parent.mode}, evaluation={parent.evaluation})")
        integrity = verify_record(parent)
        if not integrity["ok"]:
            failed = [c["check"] for c in integrity["checks"] if c["status"] == "fail"]
            raise InvalidExperiment(f"parent {parent_id} failed the evidence integrity check: {failed}")
        return parent

    def _lineage(self, parent: ExperimentRecord, relation: str, *, scenario_id: str, model_id: str, seed: int,
                 ds, model_hash: str, extra_checks: list[str]) -> dict[str, Any]:
        """Check that a child experiment is compatible with its parent and return the provenance."""
        problems = []
        if parent.scenario_id != scenario_id:
            problems.append(f"scenario ({parent.scenario_id} ≠ {scenario_id})")
        if parent.model.get("model_id") != model_id:
            problems.append(f"model ({parent.model.get('model_id')} ≠ {model_id})")
        if parent.seed != seed:
            problems.append(f"seed ({parent.seed} ≠ {seed})")
        fp = ds.fingerprint()
        if (parent.dataset.get("dataset_id"), parent.dataset.get("train_sha256"), parent.dataset.get("test_sha256")) != \
                (ds.spec.dataset_id, fp["train_sha256"], fp["test_sha256"]):
            problems.append("dataset split hashes")
        if parent.model.get("artifact_sha256") != model_hash:
            problems.append("base-model artifact hash (the parent's perturbations were crafted against a different "
                            "model)")
        if problems:
            raise InvalidExperiment(f"parent {parent.experiment_id} is incompatible: differs in " + ", ".join(problems))
        return {
            "relation": relation,
            "parent_experiment_id": parent.experiment_id,
            "parent_evaluation": parent.evaluation,
            "parent_schema_version": parent.schema_version,
            "parent_config_sha256": parent.reproducibility.get("config_sha256"),
            "parent_results_sha256": parent.reproducibility.get("results_sha256"),
            "parent_record_sha256": parent.reproducibility.get("record_sha256"),
            "shared": {"scenario_id": scenario_id, "model_id": model_id, "seed": seed,
                       "dataset_train_sha256": fp["train_sha256"], "dataset_test_sha256": fp["test_sha256"],
                       "base_model_artifact_sha256": model_hash},
            "checks_passed": ["parent completed", "parent is a non-adaptive attack on the undefended model",
                              "parent evidence integrity", "scenario", "model", "seed", "dataset split hashes",
                              "base-model artifact hash", *extra_checks],
        }

    # ---------------------------------------------------------- evaluations
    @staticmethod
    def _select(ds, selection) -> np.ndarray:
        n = len(ds.y_test)
        if selection.indices is not None:
            idx = np.array(selection.indices, dtype=int)
            if np.any(idx >= n):
                raise InvalidExperiment(f"sample indices must be < {n} (test split size)")
            return idx
        return np.arange(min(selection.max_samples, n))

    @staticmethod
    def _clean_metrics(system: DeployedSystem, ds) -> dict[str, Any]:
        d = system.decide(ds.X_test)
        return clean_metrics(d.labels, ds.y_test, system.reject_threshold is not None)

    def _baseline(self, req: BaselineRequest, deadline: float) -> dict[str, Any]:
        scen, ds, common = self._context(req.scenario_id, req.model_id, req.seed, deadline)
        system = build_system(req.model_id, scen.dataset_id, req.seed, req.defense, deadline)
        d = system.decide(ds.X_test)
        samples = [
            SampleResult(test_index=i, y_true=int(ds.y_test[i]), x_original=ds.X_test[i].tolist(),
                         pred_original=int(d.labels[i]), p1_original=float(d.p1[i]),
                         confidence_original=float(d.confidence[i]),
                         clean_correct=bool(d.labels[i] == ds.y_test[i]))
            for i in range(len(ds.y_test))
        ]
        metrics = self._clean_metrics(system, ds)
        metrics["train_accuracy"] = {
            "numerator": int(np.sum(system.model.predict(ds.X_train) == ds.y_train)),
            "denominator": int(len(ds.y_train)),
            "value": float(np.mean(system.model.predict(ds.X_train) == ds.y_train)),
            "population": "training-split samples",
            "definition": "model argmax correct / training samples (no abstention policy applied)"}
        return {**common, "evaluation": "clean", "intentionally_vulnerable": req.defense is None,
                "attacker_capabilities": None, "target_system": system.describe(),
                "attack_config": None, "defense_config": system.defense,
                "metrics": metrics, "samples": samples}

    def _attack(self, req: AttackRequest, deadline: float, parent: ExperimentRecord | None) -> dict[str, Any]:
        scen, ds, common = self._context(req.scenario_id, req.model_id, req.seed, deadline)
        cfg = req.attack
        idx = self._select(ds, req.samples)
        lineage = None
        if parent is not None:
            pcfg = parent.attack_config or {}
            budget_problems = [f"{k} ({pcfg.get(k)} ≠ {getattr(cfg, k)})" for k in ("norm", "epsilon")
                               if pcfg.get(k) != getattr(cfg, k)]
            parent_idx = [s.test_index for s in parent.samples]
            if parent_idx != idx.tolist():
                budget_problems.append("selected test samples")
            if budget_problems:
                raise InvalidExperiment(
                    f"adaptive attack is not comparable with parent {parent.experiment_id}: differs in "
                    + ", ".join(budget_problems))
            lineage = self._lineage(parent, "adaptive_attack_of", scenario_id=req.scenario_id, model_id=req.model_id,
                                    seed=req.seed, ds=ds, model_hash=common["model"]["artifact_sha256"],
                                    extra_checks=["same perturbation norm and ε", "same selected test samples"])
            lineage["attack_differs"] = pcfg.get("attack") != cfg.attack
        system = build_system(req.model_id, scen.dataset_id, req.seed, req.target_defense, deadline)
        X0, y = ds.X_test[idx], ds.y_test[idx]
        clean = system.decide(X0)
        correct = clean.labels == y
        att = np.flatnonzero(correct)  # only correctly-decided inputs are attacked
        lo, hi = ds.domain_low, ds.domain_high
        rngs = [np.random.default_rng([req.seed, int(i)]) for i in idx[att]]
        adaptive = req.target_defense is not None

        if cfg.knowledge == "white-box":
            min_conf = 0.0
            if system.reject_threshold is not None:
                # Adaptive: land past the rejection threshold, not just past the boundary.
                min_conf = logit_of(system.reject_threshold) + 1e-3
            out = gradient_attack(system.model, X0[att], y[att], cfg, lo, hi, rngs, deadline, min_conf)
        else:
            out = random_search_attack(lambda X: system.decide(X).labels, X0[att], y[att], cfg,
                                       lo, hi, rngs, deadline)
        budget_ratio = enforce_constraints(X0[att], out.x_adv, cfg.epsilon, cfg.norm, lo, hi)
        adv = system.decide(out.x_adv) if len(att) else None
        _check_deadline(deadline)

        caps = {
            "knowledge": cfg.knowledge,
            "access": ("model parameters and exact input gradients" if cfg.knowledge == "white-box"
                       else f"hard-label queries to the deployed system (budget {cfg.query_budget}/sample)"),
            "knows_defense": adaptive,
            "can_modify": "the 2 input features of a test-time sample",
            "constraint": f"‖δ‖_{cfg.norm} ≤ {cfg.epsilon}, input stays in the feature domain",
            "cannot": "modify training data, model parameters, or the decision policy",
        }
        if cfg.knowledge == "white-box":
            caps["effective_confidence_kappa"] = out.effective_confidence
        samples = self._samples(idx, X0, y, clean, att, out.x_adv, adv, out.queries)
        metrics = self._attack_metrics(system, ds, cfg, samples, att, out.x_adv, X0, budget_ratio)
        metrics["queries"] = {
            "total": int(out.queries.sum()),
            "mean_per_attacked_sample": float(out.queries.mean()) if len(att) else 0.0,
            "kind": "black-box label queries" if cfg.knowledge == "black-box"
                    else "white-box model evaluations (forward + gradient)",
        }
        return {**common, "evaluation": "adaptive_attack" if adaptive else "attack",
                "intentionally_vulnerable": not adaptive, "attacker_capabilities": caps,
                "target_system": system.describe(), "attack_config": cfg.model_dump(),
                "defense_config": system.defense, "parent_experiment_id": req.parent_experiment_id,
                "lineage": lineage, "metrics": metrics, "samples": samples}

    def _defend(self, req: DefendRequest, parent: ExperimentRecord, deadline: float) -> dict[str, Any]:
        preq = AttackRequest.model_validate(parent.request)
        scen, ds, common = self._context(preq.scenario_id, preq.model_id, preq.seed, deadline)
        lineage = self._lineage(parent, "replay_against_defense_of", scenario_id=preq.scenario_id,
                                model_id=preq.model_id, seed=preq.seed, ds=ds,
                                model_hash=common["model"]["artifact_sha256"],
                                extra_checks=["parent inputs match the regenerated test split"])
        system = build_system(preq.model_id, scen.dataset_id, preq.seed, req.defense, deadline)
        cfg = preq.attack
        idx = np.array([s.test_index for s in parent.samples], dtype=int)
        X0, y = ds.X_test[idx], ds.y_test[idx]
        if not np.array_equal(X0, np.array([s.x_original for s in parent.samples])):
            raise InvalidExperiment("parent experiment inputs do not match the regenerated dataset")
        # Only inputs the parent actually perturbed are replayed.
        att = np.array([i for i, s in enumerate(parent.samples) if s.x_adversarial is not None], dtype=int)
        x_adv = (np.array([parent.samples[i].x_adversarial for i in att]) if len(att)
                 else np.empty((0, X0.shape[1])))
        budget_ratio = enforce_constraints(X0[att], x_adv, cfg.epsilon, cfg.norm,
                                           ds.domain_low, ds.domain_high)
        clean = system.decide(X0)
        correct = clean.labels == y
        # Replay is only evaluated where the defended system was right on the clean input.
        keep = att[correct[att]]
        keep_pos = np.flatnonzero(correct[att])
        adv = system.decide(x_adv[keep_pos]) if len(keep) else None
        queries = np.zeros(len(keep), dtype=np.int64)
        samples = self._samples(idx, X0, y, clean, keep, x_adv[keep_pos], adv, queries)
        metrics = self._attack_metrics(system, ds, cfg, samples, keep, x_adv[keep_pos], X0, budget_ratio)
        # Undefended side recomputed from the parent's per-sample evidence with the SAME definitions
        # (a v1 parent's stored adversarial_accuracy used a different formula).
        pm = outcome_metrics(parent.samples, can_abstain=False)
        metrics["comparison"] = {
            "clean_accuracy": {"undefended": parent.metrics["clean_accuracy"], "defended": metrics["clean_accuracy"]},
            "adversarial_accuracy": {"undefended": pm["adversarial_accuracy"],
                                     "defended": metrics["adversarial_accuracy"]},
            "robust_accuracy": {"undefended": pm["robust_accuracy"], "defended": metrics["robust_accuracy"]},
            "attack_success_rate": {"undefended": pm["attack_success_rate"],
                                    "defended": metrics["attack_success_rate"]},
            "abstention_rate_adversarial": {"undefended": pm["abstention_rate_adversarial"],
                                            "defended": metrics["abstention_rate_adversarial"]},
        }
        metrics["notes"] = [
            "Non-adaptive replay: the adversarial inputs were crafted against the undefended model "
            f"in {parent.experiment_id} and re-submitted unchanged.",
            "A low replay success rate does not establish robustness. Run an adaptive attack "
            "(a new attack against the defended system) before drawing conclusions.",
            "Abstentions on replayed inputs are neither attack successes nor correct predictions; "
            "see abstention_rate_adversarial and robust_accuracy.",
        ]
        caps = dict(parent.attacker_capabilities or {})
        caps["knows_defense"] = False
        caps["note"] = "perturbations replayed from the parent experiment; no new queries/gradients"
        _check_deadline(deadline)
        return {**common, "evaluation": "replay", "intentionally_vulnerable": False,
                "attacker_capabilities": caps, "target_system": system.describe(),
                "attack_config": cfg.model_dump(), "defense_config": system.defense,
                "parent_experiment_id": parent.experiment_id, "lineage": lineage,
                "metrics": metrics, "samples": samples}

    @staticmethod
    def _samples(idx, X0, y, clean, att, x_adv, adv, queries) -> list[SampleResult]:
        """Build per-sample results. ``att`` are positions (into idx) whose adversarial input was evaluated."""
        pos = {int(p): k for k, p in enumerate(att)}
        samples = []
        for p, ti in enumerate(idx):
            s = SampleResult(
                test_index=int(ti), y_true=int(y[p]), x_original=X0[p].tolist(),
                pred_original=int(clean.labels[p]), p1_original=float(clean.p1[p]),
                confidence_original=float(clean.confidence[p]),
                clean_correct=bool(clean.labels[p] == y[p]),
            )
            if p in pos:
                k = pos[p]
                delta = x_adv[k] - X0[p]
                label = int(adv.labels[k])
                s.x_adversarial = x_adv[k].tolist()
                s.pred_adversarial = label
                s.p1_adversarial = float(adv.p1[k])
                s.confidence_adversarial = float(adv.confidence[k])
                s.perturbation_linf = float(lp_norm(delta, "linf")[0])
                s.perturbation_l2 = float(lp_norm(delta, "l2")[0])
                s.attack_success = bool(label != int(y[p]) and label != ABSTAIN)
                s.queries = int(queries[k])
            s.outcome = classify(s)
            samples.append(s)
        return samples

    def _attack_metrics(self, system, ds, cfg, samples, att, x_adv, X0, budget_ratio) -> dict[str, Any]:
        m = self._clean_metrics(system, ds)
        m.update(outcome_metrics(samples, system.reject_threshold is not None))
        if len(att):
            deltas = x_adv - X0[att]
            linf, l2 = lp_norm(deltas, "linf"), lp_norm(deltas, "l2")
            succ_mask = np.array([samples[int(p)].outcome == "evaded" for p in att], dtype=bool)
        else:
            linf, l2, succ_mask = np.array([]), np.array([]), np.array([], dtype=bool)
        m["perturbation"] = {"norm": cfg.norm, "epsilon": cfg.epsilon, "max_budget_ratio": budget_ratio,
                             "attacked": _stats(linf, l2),
                             "successful": _stats(linf[succ_mask], l2[succ_mask])}
        return m
