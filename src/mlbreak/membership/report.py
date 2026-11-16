"""Markdown report for ML04 membership records (shared seven-section structure, see :mod:`mlbreak.reporting`)."""

from __future__ import annotations

from .metrics import ATTACKS
from .schemas import MembershipRecord

NOTES = {
    "clean": [
        "No attacker queries were made. The no-signal baseline is the accuracy of always guessing the larger group; "
        "the generalisation gap (training − test accuracy) is the overfitting that membership inference exploits.",
    ],
    "membership_inference": [
        "Operating thresholds were calibrated on the attacker's reference models (its own calibration population); "
        "the final evaluation candidates were used only for scoring. The ROC curve is descriptive.",
        "The attacker is assumed to know the victim's training recipe and training-set size and to hold auxiliary "
        "data from the same distribution — a strong but standard assumption (Shokri et al. 2017).",
    ],
    "adaptive_membership_inference": [
        "The attack was RECALIBRATED against the defended system: reference and shadow models were retrained with "
        "the defended recipe and observed through the same output policy. Comparisons with the parent are paired "
        "over the same candidates.",
        "No defense here is a formal privacy guarantee; output restriction in particular is not differential "
        "privacy (deterministic release, ε = ∞).",
    ],
}


def _ratio(r: dict | None) -> str:
    if not r:
        return "n/a"
    v = r.get("value")
    pct = "undefined" if v is None else f"{100 * v:.1f}%"
    return f"{pct} ({r.get('numerator')}/{r.get('denominator')})"


def _num(r: dict | None, digits: int = 3) -> str:
    v = (r or {}).get("value")
    return "undefined" if v is None else f"{v:.{digits}f}"


def render_membership_markdown(rec: MembershipRecord, verification: dict | None = None) -> str:
    from ..reporting import compose, json_block
    m = rec.metrics
    summary = [("Victim model", f"`{rec.model['model_id']}` ({rec.model['estimator']})"),
               ("Output policy", f"`{rec.output_policy['kind']}`")]
    ds = rec.dataset
    config = ["| Item | Value |", "|---|---|",
              f"| Victim artifact SHA-256 | `{rec.model['artifact_sha256']}` |",
              f"| Victim training population | `{rec.model['training_population']}` "
              f"(`{rec.model['training_population_sha256']}`) |",
              f"| Dataset / split | `{ds['dataset_id']}` / `{ds['split_id']}` (synthetic) |",
              f"| Populations disjoint | {ds['disjointness']['disjoint']} "
              f"(duplicates {ds['disjointness']['duplicate_records']}) |",
              f"| Candidates | {rec.candidate_selection['n_members']} members + "
              f"{rec.candidate_selection['n_non_members']} non-members |",
              "", "### Victim recipe", "", *json_block(rec.model["hyperparameters"])]
    if rec.attack_config:
        config += ["", "### Attack", "", *json_block(rec.attack_config)]
    if rec.defense_config:
        config += ["", "### Defense", "", *json_block(rec.defense_config)]
    if rec.calibration:
        config += ["", "### Calibration provenance (attacker reference models)", "",
                   f"Population `{rec.calibration['population_sha256']}` · {rec.calibration['n_reference_models']} "
                   f"reference model(s): " + ", ".join(f"`{r['artifact_sha256'][:16]}…`"
                                                       for r in rec.calibration["reference_models"])]
    if rec.shadow:
        mc = rec.shadow["meta_classifier"]
        config += ["", "### Shadow models", "",
                   f"{rec.shadow['n_shadow_models']} shadow models on population `{rec.shadow['population_sha256']}`; "
                   f"meta-classifier `{mc['artifact_sha256'][:16]}…` trained on {mc['n_training_examples']} shadow "
                   f"examples ({mc['n_member_examples']} members), features {mc['features']}."]
    out: list[str] = []
    if m:
        out += ["### Victim utility", "", "| Metric | Value |", "|---|---|",
                f"| train_accuracy | {_ratio(m.get('train_accuracy'))} |",
                f"| test_accuracy | {_ratio(m.get('test_accuracy'))} |",
                f"| generalization_gap | {_num(m.get('generalization_gap'))} |",
                f"| test_roc_auc (released) | {_num(m.get('test_roc_auc'))} |",
                f"| consumer_decision_accuracy | {_ratio(m.get('consumer_decision_accuracy'))} |",
                f"| no_signal_accuracy (evaluation) | {_ratio(m.get('no_signal_accuracy'))} |"]
        if rec.evaluation != "clean":
            out += ["", "### Final evaluation (untouched candidates)", "",
                    "| Metric | Threshold attack | Shadow-model attack |", "|---|---|---|"]
            for key in ("roc_auc", "membership_advantage", "balanced_accuracy"):
                out.append(f"| {key} | {_num(m.get(f'threshold_{key}'))} | {_num(m.get(f'shadow_{key}'))} |")
            for key in ("membership_accuracy", "membership_precision", "membership_recall", "false_positive_rate",
                        "tpr_at_low_fpr", "fpr_at_low_fpr", "undefended_membership_accuracy", "correctness_reduction"):
                if f"threshold_{key}" in m:
                    out.append(f"| {key} | {_ratio(m.get(f'threshold_{key}'))} | {_ratio(m.get(f'shadow_{key}'))} |")
            out.append("| leakage p (Mann–Whitney, one-sided) | " + " | ".join(
                f"{m[f'{a}_leakage_p_value']['value']:.3g} → "
                f"{'claimed' if m[f'{a}_leakage_p_value']['leakage_claimed'] else 'not claimed'}" for a in ATTACKS) + " |")
            out += ["", "### Confusion matrices (balanced operating point)", "",
                    "| Attack | TP | FP | TN | FN |", "|---|---|---|---|---|"]
            out += [f"| {a} | {m[f'{a}_confusion']['tp']} | {m[f'{a}_confusion']['fp']} | {m[f'{a}_confusion']['tn']} "
                    f"| {m[f'{a}_confusion']['fn']} |" for a in ATTACKS]
            out += ["", "### Calibration results (attacker reference models — NOT the evaluation population)", "",
                    "| Attack | calibration AUC | balanced: TPR / FPR | low-FPR: TPR / FPR | low-FPR status |",
                    "|---|---|---|---|---|"]
            for a in ATTACKS:
                c = m.get(f"{a}_calibration") or {}
                b, lo = c.get("balanced", {}), c.get("low_fpr", {})
                out.append(f"| {a} | {c.get('calibration_auc') or 0:.3f} | {b.get('calibration_tpr', 0):.3f} / "
                           f"{b.get('calibration_fpr', 0):.3f} | {lo.get('calibration_tpr', 0):.3f} / "
                           f"{lo.get('calibration_fpr', 0):.3f} | {lo.get('status')} |")
            for a in ATTACKS:
                f = m.get(f"{a}_fpr_at_low_fpr") or {}
                if f.get("exceeds_target"):
                    out.append(f"\n**{a}: the evaluation FPR at the low-FPR threshold exceeds the "
                               f"{f.get('target_fpr')} target (Wilson lower bound {f['wilson_95'][0]:.3f}) — the "
                               "calibrated threshold did not transfer.**")
            out += ["", "### Operational precision (projected from the balanced operating point)", "",
                    "| Prevalence | Threshold precision / FDR | Shadow precision / FDR |", "|---|---|---|"]
            th, sh = m["threshold_operational_precision"]["values"], m["shadow_operational_precision"]["values"]
            for a, b in zip(th, sh):
                fmt = lambda r: "undefined" if r["precision"] is None else (  # noqa: E731
                    f"{r['precision']:.3f} / {r['false_discovery_rate']:.3f}")
                out.append(f"| {a['prevalence']} | {fmt(a)} | {fmt(b)} |")
            q = m.get("queries", {})
            out += ["", f"Victim queries: {q.get('total')} of {q.get('budget')} (one per candidate; shadow and reference "
                        "models never query the victim)."]
            out += ["", "### Per-candidate results (first 25)", "",
                    "| candidate | released | threshold score → member? | shadow score → member? | group (eval only) |",
                    "|---|---|---|---|---|"]
            for c in rec.candidates[:25]:
                out.append(f"| {c.candidate_id} | {c.observation.released_output:.4g} | {c.threshold.score:.4g} → "
                           f"{c.threshold.member_balanced} | {c.shadow.score:.4g} → {c.shadow.member_balanced} | "
                           f"{c.truth.group} |")
        notes = NOTES.get(rec.evaluation, [])
        if notes:
            out += ["", "### Notes", ""] + [f"- {n}" for n in notes]
        out += ["", "### Metric definitions", ""]
        out += [f"- **{k}**: {v['definition']}." + (f" *Population:* {v['population']}." if v.get("population") else "")
                for k, v in m.items() if isinstance(v, dict) and "definition" in v]
    return compose(rec, summary_rows=summary, configuration=config, outcomes=out, verification=verification)
