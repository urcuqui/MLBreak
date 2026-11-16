"""Security regression: declared defensive invariants checked against real runs.

A case (``security_cases/MLB-REG-*.json``) pins an attack, a defense and a set
of invariants. ML03 cases (``family: inversion``) pin a model-inversion attack and an output
policy evaluated by a NEW adaptive inversion. ML01 cases (``owasp_id: ML01:2023``) pin an evasion attack; ML02
cases (``owasp_id: ML02:2023``, ``family: poisoning``) pin a label-flipping attack
and a label-validation defense, evaluated by defensive replay of the exact
poisoned training set. Each invariant names a *posture* and a metric:

* ``undefended`` — the attack against the intentionally vulnerable model. An
  ``attack_canary`` invariant here proves the case still exercises a real
  attack; if it fails, the defended invariants are meaningless.
* ``defended`` — the defended system, evaluated either by ``replay`` of the
  undefended perturbations or by a new ``adaptive`` attack.

* ``paired`` — a comparison of the two on the SAME test samples
  (:mod:`mlbreak.stats`): ``absolute_reduction`` / ``relative_reduction`` of the
  attack success rate, or the exact one-sided ``mcnemar_p_value`` that the defense
  reduces success. Paired invariants express "the defense helps" directly instead
  of relying on one hard-coded bound that numerical noise can push either way.

Invariant kinds: ``security`` (the defense must hold), ``utility`` (the
defense must not destroy clean performance), ``attack_canary`` (the attack or a
known limitation must still be observable).

The same runner backs ``pytest`` and ``mlbreak verify``.
"""

from __future__ import annotations

import json
import operator
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import LIMITS
from .experiments import EngineBusy, ExperimentEngine
from .inversion.metrics import INVARIANT_METRICS as INVERSION_METRICS
from .membership.metrics import INVARIANT_METRICS as MEMBERSHIP_METRICS
from .membership.schemas import (CandidateSelection, MembershipAttackConfig, MembershipAttackRequest,
                                 MembershipDefendRequest, MembershipDefense)
from .inversion.schemas import (InversionAttackConfig, InversionAttackRequest, InversionDefendRequest,
                                RestrictivePolicy, TargetSelection)
from .poisoning.metrics import INVARIANT_METRICS as POISONING_METRICS
from .poisoning.schemas import KnnLabelSanitization, LabelFlipConfig, PoisonAttackRequest, PoisonDefendRequest
from .stats import STATISTICS, paired_comparison
from .text.metrics import INVARIANT_METRICS as TEXT_METRICS
from .text.schemas import TextAttackConfig, TextAttackRequest, TextDefendRequest, TextDefense, TextSampleSelection
from .schemas import AttackConfig, AttackRequest, DefendRequest, DefenseConfig, SampleSelection

_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq}


class _PairedMixin(BaseModel):
    @model_validator(mode="after")
    def _paired(self):
        if (self.posture == "paired") != (self.statistic is not None):
            raise ValueError("posture 'paired' requires a statistic, and a statistic requires posture 'paired'")
        if self.posture == "paired" and self.metric != "attack_success_rate":
            raise ValueError("paired statistics compare attack_success_rate between the two postures")
        return self


class Invariant(_PairedMixin):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["security", "utility", "attack_canary"]
    posture: Literal["undefended", "defended", "paired"]
    statistic: Literal[STATISTICS] | None = None  # type: ignore[valid-type]
    metric: Literal["attack_success_rate", "adversarial_accuracy", "clean_accuracy", "coverage",
                    "robust_accuracy", "abstention_rate_adversarial", "attack_resisted_rate",
                    "adversarial_coverage", "selective_accuracy", "adversarial_selective_accuracy",
                    "abstention_rate_clean"]
    op: Literal["<=", ">=", "<", ">", "=="]
    value: float = Field(ge=0, le=1)
    description: str


class SecurityCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^MLB-REG-\d{3}$")
    title: str
    owasp_id: Literal["ML01:2023"]
    scenario_id: Literal["ml01-moons-evasion"] = "ml01-moons-evasion"
    model_id: Literal["mlp-tanh-16x16", "logreg"] = "mlp-tanh-16x16"
    seed: int = 7
    attack: AttackConfig
    samples: SampleSelection = Field(default_factory=SampleSelection)
    defense: DefenseConfig
    defended_evaluation: Literal["replay", "adaptive"]
    invariants: list[Invariant] = Field(min_length=1)
    rationale: str


class PoisoningInvariant(_PairedMixin):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["security", "utility", "attack_canary"]
    posture: Literal["undefended", "defended", "paired"]
    statistic: Literal[STATISTICS] | None = None  # type: ignore[valid-type]
    metric: Literal[POISONING_METRICS]  # type: ignore[valid-type]
    op: Literal["<=", ">=", "<", ">", "=="]
    value: float = Field(ge=0, le=1)
    description: str


class PoisoningCase(BaseModel):
    """ML02 case: ``undefended`` = the poisoning attack record, ``defended`` = its defensive replay."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^MLB-REG-\d{3}$")
    family: Literal["poisoning"] = "poisoning"
    title: str
    owasp_id: Literal["ML02:2023"]
    scenario_id: Literal["ml02-moons-label-flip"] = "ml02-moons-label-flip"
    model_id: Literal["mlp-tanh-16x16", "logreg"] = "mlp-tanh-16x16"
    seed: int = 7
    poison: LabelFlipConfig
    defense: KnnLabelSanitization
    defended_evaluation: Literal["replay"] = "replay"
    invariants: list[PoisoningInvariant] = Field(min_length=1)
    rationale: str


class TextInvariant(_PairedMixin):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["security", "utility", "attack_canary"]
    posture: Literal["undefended", "defended", "paired"]
    statistic: Literal[STATISTICS] | None = None  # type: ignore[valid-type]
    metric: Literal[TEXT_METRICS]  # type: ignore[valid-type]
    op: Literal["<=", ">=", "<", ">", "=="]
    value: float = Field(ge=0, le=1)
    description: str


class TextCase(BaseModel):
    """ML01 spam-filter evasion case: ``undefended`` = the attack on the undefended filter, ``defended`` = its
    replay against the defense or a NEW adaptive attack with the same configuration (same messages, same
    text-preservation constraints, same effort)."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^MLB-REG-\d{3}$")
    family: Literal["text_evasion"] = "text_evasion"
    title: str
    owasp_id: Literal["ML01:2023"]
    scenario_id: Literal["ml01-spam-filter-evasion"] = "ml01-spam-filter-evasion"
    model_id: Literal["tfidf-word-logreg", "tfidf-word-nb"] = "tfidf-word-logreg"
    seed: int = 7
    attack: TextAttackConfig
    samples: TextSampleSelection = Field(default_factory=TextSampleSelection)
    defense: TextDefense
    defended_evaluation: Literal["replay", "adaptive"]
    invariants: list[TextInvariant] = Field(min_length=1)
    rationale: str


class InversionInvariant(BaseModel):
    """ML03 invariant. Paired statistics compare the per-target CORRECTNESS of the reconstruction (not an evasion
    success) between the full-probability attack and the attack against the output policy, on the same targets.
    ``evidence_consistency`` is computed by the runner: 1.0 iff every record of the case passes integrity +
    lineage verification and the attack record replays bit-identically (0.0 otherwise)."""
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["security", "utility", "attack_canary"]
    posture: Literal["undefended", "defended", "paired"]
    statistic: Literal[STATISTICS] | None = None  # type: ignore[valid-type]
    metric: Literal[INVERSION_METRICS]  # type: ignore[valid-type]
    op: Literal["<=", ">=", "<", ">", "=="]
    value: float = Field(ge=0, le=1)
    description: str

    @model_validator(mode="after")
    def _paired(self):
        if (self.posture == "paired") != (self.statistic is not None):
            raise ValueError("posture 'paired' requires a statistic, and a statistic requires posture 'paired'")
        if self.posture == "paired" and self.metric != "attribute_inference_accuracy":
            raise ValueError("ML03 paired statistics compare attribute_inference_accuracy between the two postures")
        return self


class InversionCase(BaseModel):
    """ML03 case: ``undefended`` = inversion against the full-probability service; ``defended`` = a NEW adaptive
    inversion against ``defense`` (an output policy) with the same targets and attack configuration."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^MLB-REG-\d{3}$")
    family: Literal["inversion"] = "inversion"
    title: str
    owasp_id: Literal["ML03:2023"]
    scenario_id: Literal["ml03-private-attribute-inversion"] = "ml03-private-attribute-inversion"
    model_id: Literal["logreg", "logreg-public-only"] = "logreg"
    seed: int = 7
    attack: InversionAttackConfig = Field(default_factory=InversionAttackConfig)
    targets: TargetSelection = Field(default_factory=TargetSelection)
    defense: RestrictivePolicy | None = None
    defended_evaluation: Literal["adaptive", "none"] = "adaptive"
    invariants: list[InversionInvariant] = Field(min_length=1)
    rationale: str

    @model_validator(mode="after")
    def _check(self):
        if (self.defense is None) != (self.defended_evaluation == "none"):
            raise ValueError("defended_evaluation must be 'none' exactly when no defense (output policy) is declared")
        if self.defense is None and any(i.posture != "undefended" for i in self.invariants):
            raise ValueError("defended/paired invariants need a defense (output policy)")
        return self


class MembershipInvariant(BaseModel):
    """ML04 invariant on a metric of the attack record (``undefended``) or of the recalibrated attack against the
    defended system (``defended``). Paired comparisons are metrics of the defended record itself
    (``*_defense_reduction_p_value``, ``*_correctness_reduction``), computed over the same candidates.

    Computed by the runner (1.0 if the check holds, else 0.0):

    * ``evidence_consistency`` — every record of the case passes integrity + lineage verification and the attack
      record replays bit-identically;
    * ``attacker_independence`` — re-running each attack with the evaluation ground truth permuted gives bit-identical
      attacker outputs, and the attacker's calibration / shadow populations share no record with the victim's
      training set or the candidates;
    * ``budget_enforcement`` — the same attack with a query budget one below the number of candidates is refused by
      the counting oracle (failed record, ``QueryBudgetExceeded``) and that failed record verifies.
    """
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["security", "utility", "attack_canary"]
    posture: Literal["undefended", "defended"]
    metric: Literal[MEMBERSHIP_METRICS + ("budget_enforcement",)]  # type: ignore[valid-type]
    op: Literal["<=", ">=", "<", ">", "=="]
    value: float = Field(ge=-1, le=1)
    description: str


class MembershipCase(BaseModel):
    """ML04 case: ``undefended`` = membership inference against the undefended victim; ``defended`` = the recalibrated
    attack against ``defense`` with the same candidates and attack configuration."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^MLB-REG-\d{3}$")
    family: Literal["membership"] = "membership"
    title: str
    owasp_id: Literal["ML04:2023"]
    scenario_id: Literal["ml04-training-membership-inference"] = "ml04-training-membership-inference"
    model_id: Literal["random-forest-overfit", "logreg-l2", "random-forest-decoy"] = "random-forest-overfit"
    seed: int = 7
    attack: MembershipAttackConfig = Field(default_factory=MembershipAttackConfig)
    candidates: CandidateSelection = Field(default_factory=CandidateSelection)
    defense: MembershipDefense | None = None
    defended_evaluation: Literal["adaptive", "none"] = "adaptive"
    invariants: list[MembershipInvariant] = Field(min_length=1)
    rationale: str

    @model_validator(mode="after")
    def _check(self):
        if (self.defense is None) != (self.defended_evaluation == "none"):
            raise ValueError("defended_evaluation must be 'none' exactly when no defense is declared")
        if self.defense is None and any(i.posture != "undefended" for i in self.invariants):
            raise ValueError("defended invariants need a defense")
        return self


AnyCase = SecurityCase | PoisoningCase | TextCase | InversionCase | MembershipCase


def parse_case(text: str) -> AnyCase:
    data = json.loads(text)
    if data.get("family") == "membership" or data.get("owasp_id") == "ML04:2023":
        return MembershipCase.model_validate(data)
    if data.get("family") == "inversion" or data.get("owasp_id") == "ML03:2023":
        return InversionCase.model_validate(data)
    if data.get("family") == "text_evasion":
        return TextCase.model_validate(data)
    if data.get("owasp_id") == "ML02:2023" or data.get("family") == "poisoning":
        return PoisoningCase.model_validate(data)
    return SecurityCase.model_validate(data)


def load_cases(directory: Path) -> list[AnyCase]:
    return [parse_case(p.read_text()) for p in sorted(directory.glob("MLB-REG-*.json"))]


def _metric(record, name: str) -> float | None:
    m = record.metrics.get(name)
    return None if m is None else m.get("value")


def _completed(rec) -> bool:
    return rec is not None and rec.status == "completed"


def _evaluate(case: AnyCase, undefended, defended) -> dict[str, Any]:
    comparison = (paired_comparison(undefended, defended)
                  if _completed(undefended) and _completed(defended) else None)
    results = []
    for inv in case.invariants:
        if inv.posture == "paired":
            observed = None if comparison is None else comparison[inv.statistic]
            extra = {"details": comparison,
                     "experiment_id": defended.experiment_id if defended else None}
        else:
            rec = undefended if inv.posture == "undefended" else defended
            observed = _metric(rec, inv.metric) if _completed(rec) else None
            extra = {"experiment_id": rec.experiment_id if rec else None}
        passed = observed is not None and _OPS[inv.op](observed, inv.value)
        results.append({**inv.model_dump(), "observed": observed, "passed": passed, **extra})
    return {
        "id": case.id, "title": case.title, "owasp_id": case.owasp_id,
        "defended_evaluation": case.defended_evaluation,
        "undefended_experiment_id": undefended.experiment_id,
        "defended_experiment_id": defended.experiment_id if defended else None,
        "passed": all(r["passed"] for r in results), "invariants": results,
    }


def run_poisoning_case(engine: ExperimentEngine, case: PoisoningCase) -> dict[str, Any]:
    lab = engine.poisoning
    undefended = lab.run_attack(PoisonAttackRequest(scenario_id=case.scenario_id, model_id=case.model_id,
                                                    seed=case.seed, poison=case.poison))
    defended = None
    if undefended.status == "completed":
        defended = lab.run_defend(PoisonDefendRequest(parent_experiment_id=undefended.experiment_id,
                                                      defense=case.defense))
    return _evaluate(case, undefended, defended)


def run_text_case(engine: ExperimentEngine, case: TextCase) -> dict[str, Any]:
    lab = engine.text
    req = TextAttackRequest(scenario_id=case.scenario_id, model_id=case.model_id, seed=case.seed,
                            attack=case.attack, samples=case.samples)
    undefended = lab.run_attack(req)
    defended = None
    if undefended.status == "completed":
        if case.defended_evaluation == "replay":
            defended = lab.run_defend(TextDefendRequest(parent_experiment_id=undefended.experiment_id,
                                                        defense=case.defense))
        else:
            defended = lab.run_attack(req.model_copy(update={"target_defense": case.defense,
                                                             "parent_experiment_id": undefended.experiment_id}))
    return _evaluate(case, undefended, defended)


def run_inversion_case(engine: ExperimentEngine, case: InversionCase) -> dict[str, Any]:
    lab = engine.inversion
    undefended = lab.run_attack(InversionAttackRequest(scenario_id=case.scenario_id, model_id=case.model_id,
                                                       seed=case.seed, attack=case.attack, targets=case.targets))
    defended = None
    if undefended.status == "completed" and case.defense is not None:
        defended = lab.run_defend(InversionDefendRequest(parent_experiment_id=undefended.experiment_id,
                                                         output_policy=case.defense))
    result = _evaluate(case, undefended, defended)
    if any(i.metric == "evidence_consistency" for i in case.invariants):
        consistent = _completed(undefended) and (defended is None or _completed(defended))
        details: dict[str, Any] = {}
        if consistent:
            details["verified"] = {r.experiment_id: engine.verify_full(r)["ok"] for r in (undefended, defended) if r}
            details["replay_identical"] = engine.replay(undefended.experiment_id)["identical"]
            consistent = all(details["verified"].values()) and details["replay_identical"]
        for inv, res in zip(case.invariants, result["invariants"]):
            if inv.metric == "evidence_consistency":
                res["observed"] = 1.0 if consistent else 0.0
                res["passed"] = _OPS[inv.op](res["observed"], inv.value)
                res["details"] = details
        result["passed"] = all(r["passed"] for r in result["invariants"])
    return result


_COMPUTED = ("evidence_consistency", "attacker_independence", "budget_enforcement")


def run_membership_case(engine: ExperimentEngine, case: MembershipCase) -> dict[str, Any]:
    from .attacks import QueryBudgetExceeded  # noqa: F401  (documented failure mode of the budget probe)
    lab = engine.membership
    req = MembershipAttackRequest(scenario_id=case.scenario_id, model_id=case.model_id, seed=case.seed,
                                  attack=case.attack, candidates=case.candidates)
    undefended = lab.run_attack(req)
    defended = None
    if _completed(undefended) and case.defense is not None:
        defended = lab.run_defend(MembershipDefendRequest(parent_experiment_id=undefended.experiment_id,
                                                          defense=case.defense))
    result = _evaluate(case, undefended, defended)
    wanted = {i.metric for i in case.invariants} & set(_COMPUTED)
    computed: dict[str, tuple[float, dict[str, Any]]] = {}
    ran = _completed(undefended) and (case.defense is None or _completed(defended))
    recs = [r for r in (undefended, defended) if r is not None]
    if "evidence_consistency" in wanted:
        d: dict[str, Any] = {}
        ok = ran
        if ran:
            d["verified"] = {r.experiment_id: engine.verify_full(r)["ok"] for r in recs}
            d["replay_identical"] = engine.replay(undefended.experiment_id)["identical"]
            ok = all(d["verified"].values()) and d["replay_identical"]
        computed["evidence_consistency"] = (1.0 if ok else 0.0, d)
    if "attacker_independence" in wanted:
        d = {r.experiment_id: lab.attacker_independence(r) for r in recs} if ran else {}
        computed["attacker_independence"] = (1.0 if ran and all(v["ok"] for v in d.values()) else 0.0, d)
    if "budget_enforcement" in wanted:
        n = case.candidates.n_members + case.candidates.n_non_members
        probe = lab.run_attack(req.model_copy(update={"attack": case.attack.model_copy(update={"query_budget": n - 1})}))
        d = {"probe_experiment_id": probe.experiment_id, "status": probe.status, "error": probe.error,
             "verified": engine.verify_full(probe)["ok"]}
        ok = probe.status == "failed" and (probe.error or "").startswith("QueryBudgetExceeded") and d["verified"]
        computed["budget_enforcement"] = (1.0 if ok else 0.0, d)
    for inv, res in zip(case.invariants, result["invariants"]):
        if inv.metric in computed:
            res["observed"], res["details"] = computed[inv.metric]
            res["passed"] = _OPS[inv.op](res["observed"], inv.value)
    result["passed"] = all(r["passed"] for r in result["invariants"])
    return result


def run_case(engine: ExperimentEngine, case: AnyCase) -> dict[str, Any]:
    if isinstance(case, MembershipCase):
        return run_membership_case(engine, case)
    if isinstance(case, InversionCase):
        return run_inversion_case(engine, case)
    if isinstance(case, TextCase):
        return run_text_case(engine, case)
    if isinstance(case, PoisoningCase):
        return run_poisoning_case(engine, case)
    attack_req = AttackRequest(scenario_id=case.scenario_id, model_id=case.model_id, seed=case.seed,
                               attack=case.attack, samples=case.samples)
    undefended = engine.run_attack(attack_req)
    defended = None
    if undefended.status == "completed":
        if case.defended_evaluation == "replay":
            defended = engine.run_defend(DefendRequest(parent_experiment_id=undefended.experiment_id,
                                                       defense=case.defense))
        else:
            defended = engine.run_attack(attack_req.model_copy(update={
                "target_defense": case.defense, "parent_experiment_id": undefended.experiment_id}))

    return _evaluate(case, undefended, defended)


def _not_run(case: AnyCase, reason: str) -> dict[str, Any]:
    """A case that could not be executed FAILS (never silently passes)."""
    return {"id": case.id, "title": case.title, "owasp_id": case.owasp_id,
            "defended_evaluation": case.defended_evaluation, "undefended_experiment_id": None,
            "defended_experiment_id": None, "passed": False, "error": reason,
            "invariants": [{**inv.model_dump(), "observed": None, "passed": False, "experiment_id": None}
                           for inv in case.invariants]}


def run_suite(engine: ExperimentEngine, cases_dir: Path, report_dir: Path | None = None,
              time_limit: float = LIMITS.max_regression_seconds) -> dict[str, Any]:
    """Run every declared case. Only one suite runs at a time (EngineBusy otherwise); once the
    suite-level wall-clock limit is exceeded, the remaining cases are reported as failed."""
    if not engine.suite_lock.acquire(blocking=False):
        raise EngineBusy("a security regression suite is already running; retry when it finishes")
    try:
        cases = load_cases(cases_dir)
        deadline = time.monotonic() + time_limit
        results = []
        for c in cases:
            if time.monotonic() > deadline:
                results.append(_not_run(c, f"not run: suite wall-clock limit of {time_limit:g}s exceeded"))
            else:
                results.append(run_case(engine, c))
    finally:
        engine.suite_lock.release()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "failed": sum(not r["passed"] for r in results),
        "ok": bool(results) and all(r["passed"] for r in results),
        "cases": results,
    }
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "security_regression_report.json").write_text(json.dumps(report, indent=2))
        (report_dir / "security_regression_report.md").write_text(render_report(report))
    return report


def render_report(report: dict[str, Any]) -> str:
    lines = ["# MLBreak security regression report", "",
             f"Generated: {report['generated_at']}  ",
             f"Result: **{'PASS' if report['ok'] else 'FAIL'}** — {report['passed']}/{report['total']} cases passed", ""]
    for c in report["cases"]:
        lines += [f"## {c['id']} — {c['title']} ({'PASS' if c['passed'] else 'FAIL'})", "",
                  f"Undefended experiment: `{c['undefended_experiment_id']}` · "
                  f"Defended ({c['defended_evaluation']}): `{c['defended_experiment_id']}`", "",
                  "| Invariant | Kind | Posture | Check | Observed | Result |", "|---|---|---|---|---|---|"]
        for i in c["invariants"]:
            obs = "n/a" if i["observed"] is None else f"{i['observed']:.4f}"
            check = (f"{i['statistic']}({i['metric']})" if i.get("statistic") else i["metric"])
            lines.append(f"| {i['id']} | {i['kind']} | {i['posture']} | {check} {i['op']} {i['value']} "
                         f"| {obs} | {'pass' if i['passed'] else '**FAIL**'} |")
        lines.append("")
    return "\n".join(lines)
