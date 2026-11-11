"""Security regression: declared defensive invariants checked against real runs.

A case (``security_cases/MLB-REG-*.json``) pins an attack, a defense and a set
of invariants. Each invariant names a *posture* and a metric:

* ``undefended`` — the attack against the intentionally vulnerable model. An
  ``attack_canary`` invariant here proves the case still exercises a real
  attack; if it fails, the defended invariants are meaningless.
* ``defended`` — the defended system, evaluated either by ``replay`` of the
  undefended perturbations or by a new ``adaptive`` attack.

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

from pydantic import BaseModel, ConfigDict, Field

from .config import LIMITS
from .experiments import EngineBusy, ExperimentEngine
from .schemas import AttackConfig, AttackRequest, DefendRequest, DefenseConfig, SampleSelection

_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq}


class Invariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["security", "utility", "attack_canary"]
    posture: Literal["undefended", "defended"]
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


def load_cases(directory: Path) -> list[SecurityCase]:
    return [SecurityCase.model_validate_json(p.read_text()) for p in sorted(directory.glob("MLB-REG-*.json"))]


def _metric(record, name: str) -> float | None:
    m = record.metrics.get(name)
    return None if m is None else m.get("value")


def run_case(engine: ExperimentEngine, case: SecurityCase) -> dict[str, Any]:
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

    results = []
    for inv in case.invariants:
        rec = undefended if inv.posture == "undefended" else defended
        observed = None if rec is None or rec.status != "completed" else _metric(rec, inv.metric)
        passed = observed is not None and _OPS[inv.op](observed, inv.value)
        results.append({**inv.model_dump(), "observed": observed, "passed": passed,
                        "experiment_id": rec.experiment_id if rec else None})
    return {
        "id": case.id, "title": case.title, "owasp_id": case.owasp_id,
        "defended_evaluation": case.defended_evaluation,
        "undefended_experiment_id": undefended.experiment_id,
        "defended_experiment_id": defended.experiment_id if defended else None,
        "passed": all(r["passed"] for r in results), "invariants": results,
    }


def _not_run(case: SecurityCase, reason: str) -> dict[str, Any]:
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
            lines.append(f"| {i['id']} | {i['kind']} | {i['posture']} | {i['metric']} {i['op']} {i['value']} "
                         f"| {obs} | {'pass' if i['passed'] else '**FAIL**'} |")
        lines.append("")
    return "\n".join(lines)
