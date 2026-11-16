"""Controlled sensitivity study for ML03 (evaluator-side).

The reference experiment (``mlbreak.inversion/v1``) is unchanged: exact candidate matching, full knowledge of the
public features, full-precision observation, 2 queries per target, ties → auxiliary prior. This module varies ONE
factor at a time around it and compares every setting with the auxiliary-only attacker ON THE SAME TARGETS, so
model-induced advantage is separated from accuracy that public data already gives.

Factors
-------
* ``feature_knowledge`` — the attacker knows only some public features (unknown ones are marginalised over rows
  of its own auxiliary population), or knows them with Gaussian error. Both the reference exact matcher and a
  nearest-response matcher (:func:`~.attacks.nearest_matching`) are run.
* ``observation_precision`` — the target's output is OBSERVED rounded to d decimals (e.g. printed in a letter)
  while queries return full precision; the attacker matches within the rounding half-width.
* ``query_budget`` — per-target budgets for the exact matcher (abstaining on ties, to isolate the budget) and for
  the marginalising matcher.
* ``tie_handling`` — auxiliary-prior fallback vs abstention under restrictive output policies.
* ``auxiliary_strength`` — the auxiliary attacker fitted on 0 / 30 / 100 / 600 rows (the baseline AND the tie
  fallback use the same reduced model).
* ``output_policy`` — the defensive evaluation: every policy applied identically to the observed output and to
  every query, with inference, ambiguity, abstention, queries, victim and consumer utility.

Assumptions are listed per setting. Evidence: per-target correctness vectors are stored, every aggregate is
re-derived on verification, and the report has config / results / record fingerprints; re-running the same request
must reproduce ``results_sha256`` exactly.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..attacks import _check_deadline
from ..hashing import sha256_json
from ..integrity import INTEGRITY_GUARANTEE, _check
from ..metrics import ratio
from ..schemas import Strict
from ..stats import mcnemar_one_sided_p
from .attacks import AttackerView, PartialView, candidate_matching, nearest_matching, prior_choice
from .data import PUBLIC_FEATURES, load_inversion_dataset
from .defenses import PredictionService, apply_policy, counting_oracle, describe_policy
from .metrics import LEAKAGE_ALPHA, _service_metrics
from .models import auxiliary_model, auxiliary_model_sized, victim_model
from .schemas import InversionAttackConfig, ModelId, ServiceRow, TargetSelection

SENSITIVITY_SCHEMA = "mlbreak.inversion-sensitivity/v1"
REPORT_ID_PATTERN = r"^MLB-SEN-[0-9a-f]{12}$"
FACTORS = ("reference", "feature_knowledge", "observation_precision", "query_budget", "tie_handling",
           "auxiliary_strength", "output_policy")
Factor = Literal["reference", "feature_knowledge", "observation_precision", "query_budget", "tie_handling",
                 "auxiliary_strength", "output_policy"]
FULL = {"kind": "full_probability"}
LABEL = {"kind": "label_only"}


class SensitivityRequest(Strict):
    model_id: ModelId = "logreg"
    seed: int = Field(7, ge=0, le=100_000)
    targets: TargetSelection = Field(default_factory=TargetSelection)
    factors: list[Factor] = Field(default_factory=lambda: list(FACTORS), min_length=1)

    @field_validator("factors")
    @classmethod
    def _unique(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("factors must be unique")
        return v


class SettingResult(BaseModel):
    factor: str
    label: str
    setting: dict[str, Any]
    attack: str
    assumptions: list[str]
    inversion_correct: list[bool]
    baseline_correct: list[bool]
    ambiguous: list[bool]
    abstained: list[bool]
    queries: list[int]
    metrics: dict[str, Any]
    utility: dict[str, Any] | None = None


class SensitivityReport(BaseModel):
    schema_version: Literal["mlbreak.inversion-sensitivity/v1"] = SENSITIVITY_SCHEMA
    report_id: str
    created_at: str
    request: dict[str, Any]
    model: dict[str, Any]
    dataset: dict[str, Any]
    target_indices: list[int]
    results: list[SettingResult]
    reproducibility: dict[str, Any]
    duration_ms: float


# ----------------------------------------------------------------------------------------------- metrics
def setting_metrics(inv: list[bool], base: list[bool], amb: list[bool], abst: list[bool], q: list[int]) -> dict[str, Any]:
    n, pop = len(inv), "selected held-out targets (identical for every setting)"
    b = sum(i and not a for i, a in zip(inv, base))
    c = sum(a and not i for i, a in zip(inv, base))
    p = mcnemar_one_sided_p(b, c) if n else None
    return {
        "attribute_inference_accuracy": ratio(sum(inv), n, "correct reconstructions / targets", pop),
        "baseline_inference_accuracy": ratio(sum(base), n, "correct auxiliary-only guesses / targets", pop),
        "inference_advantage": ratio(sum(inv) - sum(base), n, "(correct inversion − correct auxiliary-only) / "
                                                              "targets — signed", pop),
        "leakage_p_value": {"value": p, "inversion_only_correct": b, "baseline_only_correct": c,
                            "leakage_claimed": bool(p is not None and p < LEAKAGE_ALPHA and b > c)},
        "ambiguity_rate": ratio(sum(amb), n, "ambiguous targets / targets", pop),
        "abstention_rate": ratio(sum(abst), n, "abstentions / targets", pop),
        "queries": {"mean": (sum(q) / n) if n else None, "max": max(q, default=0), "total": sum(q)},
    }


# ------------------------------------------------------------------------------------------------ runner
class _Ctx:
    def __init__(self, req: SensitivityRequest, deadline: float | None):
        from .engine import select_targets
        self.req, self.deadline = req, deadline
        self.ds = load_inversion_dataset(req.seed)
        self.victim = victim_model(req.model_id, req.seed)
        self.idx = select_targets(req.targets, req.seed, len(self.ds.s_eval))
        self.truth = [int(self.ds.s_eval[i]) for i in self.idx]
        self.aux_mean = self.ds.aux.X.mean(axis=0)

    def prior(self, believed: tuple[float | None, ...], aux_n: int | None) -> float:
        """The auxiliary-only attacker with EXACTLY the attacker's public knowledge: its model applied to the
        believed features, unknown ones replaced by the auxiliary-population mean."""
        m = auxiliary_model(self.req.seed) if aux_n is None else auxiliary_model_sized(self.req.seed, aux_n)
        if m is None:
            return 0.5
        x = [float(self.aux_mean[j]) if v is None else v for j, v in enumerate(believed)]
        return float(m.proba(np.array([x]))[0])

    def finish(self, factor, label, setting, attack, assumptions, inferred, amb, q, p1, utility=None) -> SettingResult:
        base = [prior_choice(float(p)) == t for p, t in zip(p1, self.truth)]
        inv = [g is not None and g == t for g, t in zip(inferred, self.truth)]
        abst = [g is None for g in inferred]
        return SettingResult(factor=factor, label=label, setting=setting, attack=attack, assumptions=assumptions,
                             inversion_correct=inv, baseline_correct=base, ambiguous=list(amb), abstained=abst,
                             queries=list(q), metrics=setting_metrics(inv, base, list(amb), abst, list(q)),
                             utility=utility)

    def exact(self, factor, label, *, policy=FULL, budget=2, tie="auxiliary_prior", decimals=None, aux_n=None,
              noise=0.0, unknown=(), extra=()) -> SettingResult:
        service, p1 = PredictionService(self.victim, policy), []
        tol = 1e-9 if decimals is None else 0.5 * 10.0 ** -decimals + 1e-12
        # Evaluator-side sensitivity setting: tolerance may exceed the request schema's cap (rounded observations).
        cfg = InversionAttackConfig.model_construct(attack="candidate_matching", query_budget=budget,
                                                    tie_policy=tie, tolerance=tol)
        inferred, amb, q = [], [], []
        for i in self.idx:
            _check_deadline(self.deadline)
            obs = float(service.respond(self.ds.X_eval[i:i + 1], self.ds.s_eval[i:i + 1])[0])
            if decimals is not None:
                obs = round(obs, decimals)
            feats = self._believed(i, noise, unknown)
            p1.append(self.prior(feats, aux_n))
            if any(v is None for v in feats):  # exact matching needs every feature: impute the auxiliary mean
                feats = tuple(float(self.aux_mean[j]) if v is None else v for j, v in enumerate(feats))
            query, _ = counting_oracle(service, budget)
            tr = candidate_matching(AttackerView(feats, obs, p1[-1]), query, cfg)
            inferred.append(tr.inferred)
            amb.append(tr.ambiguous)
            q.append(tr.n_queries)
        setting = {"policy": policy, "query_budget": budget, "tie_policy": tie, "observation_decimals": decimals,
                   "auxiliary_rows": aux_n if aux_n is not None else len(self.ds.aux.s), "feature_noise_std": noise,
                   "unknown_features": [PUBLIC_FEATURES[j] for j in unknown], "tolerance": tol}
        assumptions = ["deterministic service", "candidate domain {0, 1} is exhaustive", *extra]
        if not unknown and noise == 0:
            assumptions.append("attacker knows the target's public features exactly")
        return self.finish(factor, label, setting, "exact candidate matching (reference attack)", assumptions,
                           inferred, amb, q, p1)

    def nearest(self, factor, label, *, policy=FULL, budget=2, tie="auxiliary_prior", noise=0.0, unknown=(),
                margin=1e-9, extra=()) -> SettingResult:
        service, p1 = PredictionService(self.victim, policy), []
        inferred, amb, q = [], [], []
        n_imp = max(1, budget // 2)
        for i in self.idx:
            _check_deadline(self.deadline)
            obs = float(service.respond(self.ds.X_eval[i:i + 1], self.ds.s_eval[i:i + 1])[0])
            rows = np.random.default_rng([self.req.seed, 3302, i]).choice(len(self.ds.aux.s), n_imp, replace=False)
            imps = tuple(tuple(float(v) for v in self.ds.aux.X[r]) for r in rows) if unknown else ()
            feats = self._believed(i, noise, unknown)
            p1.append(self.prior(feats, None))
            view = PartialView(feats, obs, p1[-1], imps)
            query, _ = counting_oracle(service, budget)
            r = nearest_matching(view, query, budget, tie, margin)
            inferred.append(r.inferred)
            amb.append(r.ambiguous)
            q.append(r.n_queries)
        setting = {"policy": policy, "query_budget": budget, "tie_policy": tie, "feature_noise_std": noise,
                   "unknown_features": [PUBLIC_FEATURES[j] for j in unknown],
                   "imputations": n_imp if unknown else 0, "margin": margin}
        assumptions = ["deterministic service", "unknown features are marginalised over the attacker's auxiliary rows"
                       if unknown else "no feature is unknown", *extra]
        return self.finish(factor, label, setting, "nearest-response matching (degraded-knowledge attacker)",
                           assumptions, inferred, amb, q, p1)

    def _believed(self, i: int, noise: float, unknown) -> tuple[float | None, ...]:
        x = self.ds.X_eval[i].astype(float)
        if noise > 0:  # the evaluator simulates the attacker's knowledge error; the attacker sees only the result
            x = x + np.random.default_rng([self.req.seed, 3301, i]).normal(0.0, noise, size=x.shape)
        return tuple(None if j in unknown else float(v) for j, v in enumerate(x))

    def utility(self, policy) -> dict[str, Any]:
        full = PredictionService(self.victim, FULL).full_scores(self.ds.X_eval, self.ds.s_eval)
        rel = apply_policy(policy, full)
        rows = [ServiceRow(eval_index=i, y_true=int(self.ds.y_eval[i]), p_full=float(full[i]), released=float(rel[i]),
                           decision=int(full[i] >= 0.5), consumer_decision=int(rel[i] >= 0.5)) for i in range(len(full))]
        return _service_metrics(rows, "inversion")


def _run(req: SensitivityRequest, deadline: float | None) -> list[SettingResult]:
    c = _Ctx(req, deadline)
    out: list[SettingResult] = []
    f = set(req.factors)
    if "reference" in f:
        out.append(c.exact("reference", "reference: exact matching, full probability, 2 queries, ties → prior"))
    if "feature_knowledge" in f:
        for unknown, name in (((), "all 4 features known"), ((3,), "region_index unknown"),
                              ((1, 3), "monthly_activity + region_index unknown"), ((0, 1, 2, 3), "no feature known")):
            out.append(c.exact("feature_knowledge", f"{name} · exact (unknowns = auxiliary mean)", unknown=unknown,
                               extra=["unknown features replaced by the auxiliary-population mean"] if unknown else []))
            out.append(c.nearest("feature_knowledge", f"{name} · nearest, 8 imputations (16 queries)",
                                 budget=16, unknown=unknown))
        for sd in (0.1, 0.5, 1.0):
            extra = [f"attacker's features carry N(0, {sd}²) error it is unaware of"]
            out.append(c.exact("feature_knowledge", f"feature noise σ={sd} · exact", noise=sd, extra=extra))
            out.append(c.nearest("feature_knowledge", f"feature noise σ={sd} · nearest", noise=sd, extra=extra))
    if "observation_precision" in f:
        for d in (None, 3, 2, 1, 0):
            out.append(c.exact("observation_precision", "observed at full precision" if d is None else
                               f"observed rounded to {d} decimal(s)", decimals=d,
                               extra=[] if d is None else ["queries return full precision; only the observation of the "
                                                           "real record is rounded (attacker knows the precision)"]))
    if "query_budget" in f:
        for b in (1, 2, 4):
            out.append(c.exact("query_budget", f"exact · {b} quer{'y' if b == 1 else 'ies'}/target · ties → abstain",
                               budget=b, tie="abstain"))
        for b in (2, 4, 8, 16):
            out.append(c.nearest("query_budget", f"monthly_activity + region_index unknown · nearest · {b} queries",
                                 budget=b, unknown=(1, 3)))
    if "tie_handling" in f:
        for pol, name in ((LABEL, "label only"), ({"kind": "quantized_probability", "bins": 4}, "4 buckets")):
            for tie in ("auxiliary_prior", "abstain"):
                out.append(c.exact("tie_handling", f"{name} · ties → {tie}", policy=pol, tie=tie))
    if "auxiliary_strength" in f:
        for n in (0, 30, 100, 600):
            for pol, name in ((FULL, "full probability"), (LABEL, "label only")):
                out.append(c.exact("auxiliary_strength", f"auxiliary attacker on {n} rows · {name}", policy=pol,
                                   aux_n=n, extra=["no auxiliary data: prior 0.5 for every target (guess 0)"] if n == 0
                                   else []))
    if "output_policy" in f:
        for pol in (FULL, {"kind": "quantized_probability", "bins": 10}, {"kind": "quantized_probability", "bins": 4},
                    {"kind": "quantized_probability", "bins": 2}, LABEL):
            r = c.exact("output_policy", describe_policy(pol)["releases"], policy=pol,
                        extra=["the policy is applied identically to the observed output and to every query"])
            r.utility = c.utility(pol)
            out.append(r)
    return out


def results_sha256(results: list[SettingResult]) -> str:
    return sha256_json([r.model_dump(mode="json") for r in results])


def record_sha256(rep: SensitivityReport) -> str:
    d = rep.model_dump(mode="json")
    d["reproducibility"] = {k: v for k, v in d["reproducibility"].items() if k != "record_sha256"}
    return sha256_json(d)


def run_sensitivity(req: SensitivityRequest, deadline: float | None) -> SensitivityReport:
    from ..experiments import reproducibility_info
    started = time.monotonic()
    results = _run(req, deadline)
    ds = load_inversion_dataset(req.seed)
    repro = reproducibility_info()
    repro.update(config_sha256=sha256_json(req.model_dump(mode="json")), results_sha256=results_sha256(results))
    rep = SensitivityReport(report_id=f"MLB-SEN-{uuid.uuid4().hex[:12]}",
                            created_at=datetime.now(timezone.utc).isoformat(), request=req.model_dump(mode="json"),
                            model=victim_model(req.model_id, req.seed).describe(), dataset=ds.fingerprint(),
                            target_indices=[int(i) for i in _targets(req)],
                            results=results, reproducibility=repro,
                            duration_ms=round((time.monotonic() - started) * 1000, 2))
    rep.reproducibility["record_sha256"] = record_sha256(rep)
    return rep


def _targets(req: SensitivityRequest) -> list[int]:
    from .engine import select_targets
    return select_targets(req.targets, req.seed, len(load_inversion_dataset(req.seed).s_eval))


def verify_report(rep: SensitivityReport) -> dict[str, Any]:
    r = rep.reproducibility
    checks = [_check("config_sha256", r.get("config_sha256") == sha256_json(rep.request), "hash of the stored request"),
              _check("record_sha256", r.get("record_sha256") == record_sha256(rep), "hash of the whole report"),
              _check("results_sha256", r.get("results_sha256") == results_sha256(rep.results),
                     "hash of every setting's per-target vectors and metrics")]
    bad = [s.label for s in rep.results
           if s.metrics != setting_metrics(s.inversion_correct, s.baseline_correct, s.ambiguous, s.abstained, s.queries)
           or len({len(s.inversion_correct), len(s.baseline_correct), len(s.ambiguous), len(s.abstained),
                   len(s.queries), len(rep.target_indices)}) != 1]
    checks.append(_check("metrics_derivation", not bad, "aggregates re-derived from per-target vectors"
                         + (f"; mismatch: {bad[:5]}" if bad else "")))
    base0 = rep.results[0].baseline_correct if rep.results else []
    full_aux = [s.label for s in rep.results if s.setting.get("auxiliary_rows") in (None, 600)
                and not s.setting.get("unknown_features") and not s.setting.get("feature_noise_std")
                and s.baseline_correct != base0]
    checks.append(_check("identical_targets_and_baseline", not full_aux, "every setting with full public knowledge "
                         "and the full auxiliary attacker has the identical baseline vector (same targets)"
                         + (f"; differs: {full_aux[:5]}" if full_aux else "")))
    return {"report_id": rep.report_id, "schema_version": rep.schema_version,
            "ok": all(c["status"] != "fail" for c in checks), "guarantee": INTEGRITY_GUARANTEE, "checks": checks}


class SensitivityStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "sensitivity"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, report_id: str) -> Path:
        from ..evidence import ExperimentNotFound
        if not re.fullmatch(REPORT_ID_PATTERN, report_id or ""):
            raise ExperimentNotFound(report_id)
        return self.dir / f"{report_id}.json"

    def save(self, rep: SensitivityReport) -> None:
        p = self.path(rep.report_id)
        p.write_text(json.dumps(rep.model_dump(mode="json"), indent=2))
        p.with_suffix(".md").write_text(render_markdown(rep))

    def load(self, report_id: str) -> SensitivityReport:
        from ..evidence import ExperimentNotFound
        p = self.path(report_id)
        if not p.is_file():
            raise ExperimentNotFound(report_id)
        return SensitivityReport.model_validate(json.loads(p.read_text()))


def _pct(r: dict) -> str:
    return "undefined" if r["value"] is None else f"{100 * r['value']:.1f}% ({r['numerator']}/{r['denominator']})"


def render_markdown(rep: SensitivityReport) -> str:
    lines = [f"# ML03 sensitivity study {rep.report_id}", "",
             f"Model `{rep.request['model_id']}`, seed {rep.request['seed']}, {len(rep.target_indices)} held-out targets "
             "(identical for every setting). Every setting is compared with the auxiliary-only attacker on the same "
             "targets; only the advantage, with its paired McNemar test, is attributed to the model's outputs.", "",
             "| Factor | Setting | Inversion | Auxiliary-only | Advantage | Leakage (p) | Ambiguous | Abstained | Queries (mean/max) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in rep.results:
        m, lp = s.metrics, s.metrics["leakage_p_value"]
        lines.append(f"| {s.factor} | {s.label} | {_pct(m['attribute_inference_accuracy'])} | "
                     f"{_pct(m['baseline_inference_accuracy'])} | {_pct(m['inference_advantage'])} | "
                     f"{'claimed' if lp['leakage_claimed'] else 'not claimed'} ({lp['value']:.2g}) | "
                     f"{_pct(m['ambiguity_rate'])} | {_pct(m['abstention_rate'])} | "
                     f"{m['queries']['mean']:.2f}/{m['queries']['max']} |")
    util = [s for s in rep.results if s.utility]
    if util:
        lines += ["", "## Output-policy utility (whole evaluation split)", "",
                  "| Policy | Victim decisions | Consumer decisions | Brier ↓ | Ranking AUC | Score distortion |",
                  "|---|---|---|---|---|---|"]
        for s in util:
            u = s.utility
            lines.append(f"| {s.label} | {_pct(u['victim_prediction_accuracy'])} | {_pct(u['consumer_decision_accuracy'])} | "
                         f"{u['released_brier_score']['value']:.4f} | {u['released_ranking_auc']['value']:.4f} | "
                         f"{u['score_distortion']['value']:.4f} |")
    lines += ["", "Output restriction is not differential privacy: each policy is a deterministic function of the "
                  "record, so its privacy loss between records whose outputs differ is unbounded (see "
                  "docs/ml03-model-inversion.md).", "", "## Assumptions per setting", ""]
    lines += [f"- **{s.label}**: {'; '.join(s.assumptions)}" for s in rep.results]
    lines += ["", f"results_sha256 `{rep.reproducibility['results_sha256']}` · record_sha256 "
                  f"`{rep.reproducibility['record_sha256']}`", ""]
    return "\n".join(lines)
