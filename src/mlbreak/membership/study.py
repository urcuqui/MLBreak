"""ML04 multi-seed study: the same configurations on several seeds, so acceptance criteria and documented defense
failures rest on more than one draw of the data.

For every seed the study runs, with the default candidates (250 members + 250 non-members) and the given attack
configuration, through the SAME code path as the experiments (:meth:`.engine.MembershipLab._run_attack` and the
record metrics):

* ``rf-overfit``            — the overfitting random forest (undefended);
* ``regularization``        — leaf-size regularisation (min_samples_leaf 25), recalibrated attack;
* ``complexity_reduction``  — depth 4, 30 trees, recalibrated attack;
* ``quantized_4``           — 4-bucket output quantisation, recalibrated attack;
* ``label_only``            — label-only output, recalibrated attack;
* ``logreg-l2``             — the lower-capacity comparison victim;
* ``decoy``                 — the negative control (no candidate is a member).

Defenses are paired per candidate with the undefended run of the same seed. Per-seed rows are stored; the summary is
re-derived from them on verification. Evaluator-facing only; no record per run is written.
"""

from __future__ import annotations

import json
import secrets
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..config import LIMITS
from ..hashing import sha256_json
from ..schemas import Strict
from .data import load_membership_dataset
from .metrics import ATTACKS
from .models import apply_defense, victim_model, victim_recipe
from .schemas import (CandidateSelection, ComplexityReduction, LabelOnly, MembershipAttackConfig,
                      MembershipAttackRequest, OutputRestriction, QuantizedProbability, Regularization, policy_of)

STUDY_ID_PATTERN = r"^MLB-MIS-[0-9a-f]{12}$"
CONFIGS: dict[str, tuple[str, Any]] = {
    "rf-overfit": ("random-forest-overfit", None),
    "regularization": ("random-forest-overfit", Regularization()),
    "complexity_reduction": ("random-forest-overfit", ComplexityReduction()),
    "quantized_4": ("random-forest-overfit", OutputRestriction(policy=QuantizedProbability(bins=4))),
    "label_only": ("random-forest-overfit", OutputRestriction(policy=LabelOnly())),
    "logreg-l2": ("logreg-l2", None),
    "decoy": ("random-forest-decoy", None),
}
DEFENDED = ("regularization", "complexity_reduction", "quantized_4", "label_only")
_KEYS = ("roc_auc", "membership_advantage", "tpr_at_low_fpr", "fpr_at_low_fpr")


class StudyRequest(Strict):
    seeds: list[int] = Field(default_factory=lambda: list(range(10)), min_length=1, max_length=LIMITS.max_study_seeds)
    attack: MembershipAttackConfig = Field(default_factory=MembershipAttackConfig)

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v) or any(not 0 <= s <= LIMITS.max_membership_seed for s in v):
            raise ValueError(f"seeds must be unique integers in [0, {LIMITS.max_membership_seed}]")
        return v


class StudyReport(BaseModel):
    schema_version: str = "mlbreak.membership-study/v1"
    report_id: str
    created_at: str
    request: dict[str, Any]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    reproducibility: dict[str, Any]
    duration_ms: float


def _row(seed: int, name: str, m: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"seed": seed, "config": name, "train_accuracy": m["train_accuracy"]["value"],
                           "test_accuracy": m["test_accuracy"]["value"], "leakage_claimed": m["leakage_claimed"]["attacks"]}
    for a in ATTACKS:
        for k in _KEYS:
            row[f"{a}_{k}"] = m[f"{a}_{k}"]["value"]
        row[f"{a}_leakage_p_value"] = m[f"{a}_leakage_p_value"]["value"]
        row[f"{a}_fpr_exceeds_target"] = bool(m[f"{a}_fpr_at_low_fpr"].get("exceeds_target"))
        if f"{a}_defense_reduction_p_value" in m:
            row[f"{a}_defense_reduction_p_value"] = m[f"{a}_defense_reduction_p_value"]["value"]
    return row


def _stats(v: list[float | None]) -> dict[str, Any]:
    x = [a for a in v if a is not None]
    if not x:
        return {"n": 0, "undefined": len(v)}
    return {"n": len(x), "min": min(x), "median": statistics.median(x), "max": max(x), "undefined": len(v) - len(x)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    base = {r["seed"]: r for r in rows if r["config"] == "rf-overfit"}
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        s: dict[str, Any] = {"seeds": len(rs), "test_accuracy": _stats([r["test_accuracy"] for r in rs]),
                             "generalization_gap": _stats([r["train_accuracy"] - r["test_accuracy"] for r in rs]),
                             "seeds_with_leakage_claimed": sum(bool(r["leakage_claimed"]) for r in rs)}
        for a in ATTACKS:
            for k in (*_KEYS, "leakage_p_value"):
                s[f"{a}_{k}"] = _stats([r[f"{a}_{k}"] for r in rs])
            s[f"{a}_seeds_fpr_exceeds_target"] = sum(r[f"{a}_fpr_exceeds_target"] for r in rs)
            if name in DEFENDED:
                s[f"{a}_seeds_reduction_significant"] = sum((r.get(f"{a}_defense_reduction_p_value") or 1) < 0.05
                                                            for r in rs)
                s[f"{a}_auc_drop"] = _stats([base[r["seed"]][f"{a}_roc_auc"] - r[f"{a}_roc_auc"] for r in rs
                                            if r["seed"] in base])
        if name in DEFENDED:
            s["utility_drop"] = _stats([base[r["seed"]]["test_accuracy"] - r["test_accuracy"] for r in rs
                                        if r["seed"] in base])
            s["defense_failures"] = [
                {"seed": r["seed"], "attack": a,
                 "reason": "reduction not significant (paired McNemar p ≥ 0.05)"
                 if (r.get(f"{a}_defense_reduction_p_value") or 1) >= 0.05 else "leakage still claimed"}
                for r in rs for a in ATTACKS
                if (r.get(f"{a}_defense_reduction_p_value") or 1) >= 0.05 or a in r["leakage_claimed"]]
        out[name] = s
    return out


def run_seed(lab, req: StudyRequest, seed: int, deadline: float) -> list[dict[str, Any]]:
    from .engine import FULL
    ds = load_membership_dataset(seed)
    rows, base_cands = [], None
    for name, (model_id, defense) in CONFIGS.items():
        areq = MembershipAttackRequest(model_id=model_id, seed=seed, attack=req.attack,
                                       candidates=CandidateSelection())
        d = defense.model_dump() if defense else None
        rec_recipe, _ = apply_defense(victim_recipe(model_id), d)
        policy = policy_of(d) if d else FULL
        victim = victim_model(model_id, seed, rec_recipe)
        cands, cal, _, _ = lab._run_attack(ds, areq, rec_recipe, policy, victim, deadline)
        evaluation = "membership_inference"
        if name in DEFENDED:
            evaluation = "adaptive_membership_inference"
            for c, p in zip(cands, base_cands):
                member = p.truth.group == "member"
                c.undefended_correct = {a: getattr(p, a).member_balanced == member for a in ATTACKS}
        utility = lab._utility_rows(ds, model_id, victim, policy)
        m = lab._metrics(evaluation, areq, cands, utility, cal, ds)
        if name == "rf-overfit":
            base_cands = cands
        rows.append(_row(seed, name, m))
    return rows


def run_study(lab, req: StudyRequest) -> StudyReport:
    """Each seed runs inside the engine's concurrency gate and wall-clock limit (one seed per gated call)."""
    from ..experiments import reproducibility_info
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for seed in req.seeds:
        rows += lab.engine.gated(lambda deadline, s=seed: run_seed(lab, req, s, deadline))
    repro = reproducibility_info()
    repro["config_sha256"] = sha256_json(req.model_dump(mode="json"))
    repro["results_sha256"] = sha256_json(rows)
    rep = StudyReport(report_id=f"MLB-MIS-{secrets.token_hex(6)}", created_at=datetime.now(timezone.utc).isoformat(),
                      request=req.model_dump(mode="json"), rows=rows, summary=summarize(rows), reproducibility=repro,
                      duration_ms=round((time.monotonic() - started) * 1000, 2))
    rep.reproducibility["record_sha256"] = _record_sha(rep)
    return rep


def _record_sha(rep: StudyReport) -> str:
    d = rep.model_dump(mode="json")
    d["reproducibility"] = {k: v for k, v in d["reproducibility"].items() if k != "record_sha256"}
    return sha256_json(d)


def verify_study(rep: StudyReport) -> dict[str, Any]:
    from ..integrity import INTEGRITY_GUARANTEE, _check
    r = rep.reproducibility
    checks = [_check("config_sha256", r.get("config_sha256") == sha256_json(rep.request), "hash of the request"),
              _check("results_sha256", r.get("results_sha256") == sha256_json(rep.rows), "hash of the per-seed rows"),
              _check("record_sha256", r.get("record_sha256") == _record_sha(rep), "hash of the whole report"),
              _check("summary_derivation", sha256_json(summarize(rep.rows)) == sha256_json(rep.summary),
                     "summary re-derived from the per-seed rows")]
    return {"report_id": rep.report_id, "ok": all(c["status"] != "fail" for c in checks),
            "guarantee": INTEGRITY_GUARANTEE, "checks": checks}


class StudyStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "membership_studies"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, report_id: str) -> Path:
        import re

        from ..evidence import ExperimentNotFound
        if not re.fullmatch(STUDY_ID_PATTERN, report_id or ""):  # strict id: also rules out path traversal
            raise ExperimentNotFound(report_id)
        return self.dir / f"{report_id}.json"

    def save(self, rep: StudyReport) -> None:
        p = self.path(rep.report_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rep.model_dump(mode="json"), indent=2))
        tmp.replace(p)
        p.with_suffix(".md").write_text(render_markdown(rep))

    def load(self, report_id: str) -> StudyReport:
        from ..evidence import ExperimentNotFound
        p = self.path(report_id)
        if not p.is_file():
            raise ExperimentNotFound(report_id)
        return StudyReport.model_validate_json(p.read_text())


def _rng(s: dict[str, Any], digits: int = 3) -> str:
    if not s.get("n"):
        return "undefined"
    f = f"{{:.{digits}f}}"
    und = f" ({s['undefined']} undefined)" if s.get("undefined") else ""
    return f"{f.format(s['min'])} / {f.format(s['median'])} / {f.format(s['max'])}{und}"


def render_markdown(rep: StudyReport) -> str:
    seeds = rep.request["seeds"]
    lines = [f"# ML04 multi-seed study {rep.report_id}", "",
             f"Seeds: {seeds} · candidates 250 members + 250 non-members per seed · attack {json.dumps(rep.request['attack'])}",
             "", "Values are min / median / max over seeds.", "",
             "| Configuration | test acc. | gap | threshold AUC | shadow AUC | threshold adv. | shadow adv. | "
             "threshold TPR@low FPR | seeds w/ leakage claimed |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in rep.summary.items():
        lines.append(f"| {name} | {_rng(s['test_accuracy'])} | {_rng(s['generalization_gap'])} | "
                     f"{_rng(s['threshold_roc_auc'])} | {_rng(s['shadow_roc_auc'])} | "
                     f"{_rng(s['threshold_membership_advantage'])} | {_rng(s['shadow_membership_advantage'])} | "
                     f"{_rng(s['threshold_tpr_at_low_fpr'])} | {s['seeds_with_leakage_claimed']}/{s['seeds']} |")
    lines += ["", "## Defenses (paired with the undefended forest of the same seed, recalibrated attacks)", "",
              "| Defense | utility drop | threshold AUC drop | shadow AUC drop | significant reduction (thr / shadow) | "
              "failures |", "|---|---|---|---|---|---|"]
    for name in DEFENDED:
        s = rep.summary[name]
        lines.append(f"| {name} | {_rng(s['utility_drop'])} | {_rng(s['threshold_auc_drop'])} | "
                     f"{_rng(s['shadow_auc_drop'])} | {s['threshold_seeds_reduction_significant']} / "
                     f"{s['shadow_seeds_reduction_significant']} of {s['seeds']} | {len(s['defense_failures'])} |")
    lines += ["", "A defense 'failure' is a (seed, attack) where the paired reduction is not significant or leakage is "
                  "still claimed against the defended system.", "",
              f"Fingerprints: config `{rep.reproducibility['config_sha256']}` · results "
              f"`{rep.reproducibility['results_sha256']}`.", ""]
    return "\n".join(lines)
