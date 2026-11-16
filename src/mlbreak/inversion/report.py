"""Markdown report for ML03 inversion records (shared seven-section structure, see :mod:`mlbreak.reporting`)."""

from __future__ import annotations

from .schemas import InversionRecord

NOTES = {
    "clean": [
        "No attacker queries were made. The auxiliary-only guess uses only public features and the attacker's "
        "auxiliary population; its accuracy is the reference every inversion result is compared with.",
        "accuracy_without_private_attribute / log-loss compare the victim with a reference model trained without "
        "the private attribute: the attribute is relevant to the prediction, which is why outputs can leak it.",
    ],
    "inversion": [
        "High attribute_inference_accuracy alone is NOT evidence of model-induced leakage: compare it with "
        "baseline_inference_accuracy on the same targets (inference_advantage) and the paired leakage test.",
        "The undefended service releases full-precision probabilities and is deterministic, and the attacker knows "
        "the target's public features exactly; under these assumptions any output that changes with the attribute "
        "identifies it. Imperfect knowledge of the public features is not modelled.",
    ],
    "adaptive_inversion": [
        "A NEW reconstruction was run against this output policy (observed output and every query restricted the "
        "same way). The comparison with the parent is paired over the same targets.",
        "Output restriction is not a formal privacy guarantee: it only removes information that this attack uses.",
    ],
}

_HEADLINE = ("attribute_inference_accuracy", "baseline_inference_accuracy", "inference_advantage",
             "undefended_attribute_inference_accuracy", "defense_accuracy_reduction", "recall_private_0",
             "recall_private_1", "baseline_recall_private_0", "baseline_recall_private_1", "ambiguity_rate",
             "unambiguous_accuracy", "abstention_rate", "budget_exhaustion_rate", "query_budget_utilisation",
             "victim_prediction_accuracy", "consumer_decision_accuracy", "accuracy_without_private_attribute")
_MEANS = ("released_brier_score", "score_distortion", "released_ranking_auc", "log_loss_with_private_attribute",
          "log_loss_without_private_attribute")


def _ratio(r: dict | None) -> str:
    if not r:
        return "n/a"
    v = r.get("value")
    pct = "undefined (empty denominator)" if v is None else f"{100 * v:.1f}%"
    return f"{pct} ({r['numerator']}/{r['denominator']})"


def render_inversion_markdown(rec: InversionRecord, verification: dict | None = None) -> str:
    from ..reporting import compose, json_block
    m = rec.metrics
    summary = [("Victim model", f"`{rec.model['model_id']}` ({rec.model['estimator']})"),
               ("Output policy", f"`{rec.output_policy['kind']}`")]
    config = ["| Item | Value |", "|---|---|",
              f"| Victim artifact SHA-256 | `{rec.model['artifact_sha256']}` |",
              f"| Auxiliary (attacker) model SHA-256 | `{(rec.auxiliary_model or {}).get('artifact_sha256')}` |",
              f"| Dataset / split | `{rec.dataset['dataset_id']}` / `{rec.dataset['split_id']}` (synthetic) |",
              f"| Train / auxiliary / evaluation SHA-256 | `{rec.dataset['train_sha256']}` / "
              f"`{rec.dataset['auxiliary_sha256']}` / `{rec.dataset['test_sha256']}` |",
              f"| Targets | {len(rec.targets)} evaluation records ({rec.target_selection}) |",
              "", "### Output policy", "", *json_block(rec.target_system.get("output_policy", rec.output_policy))]
    if rec.attack_config:
        config += ["", "### Attack", "", *json_block(rec.attack_config)]
    out: list[str] = []
    if m:
        out += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {_ratio(m[k])} |" for k in _HEADLINE if k in m]
        for k in _MEANS:
            if k in m:
                v = m[k]["value"]
                out.append(f"| {k} | {'undefined' if v is None else format(v, '.4f')} (n={m[k]['n']}) |")
        if "leakage_p_value" in m:
            lp = m["leakage_p_value"]
            out += ["", "### Leakage attributable to the model (paired with the auxiliary-only attacker)", "",
                    f"Inversion-only correct: {lp['inversion_only_correct']} · baseline-only correct: "
                    f"{lp['baseline_only_correct']} · one-sided McNemar p = {lp['value']:.3g} → "
                    f"**leakage {'claimed' if lp['leakage_claimed'] else 'NOT claimed'}** (alpha {lp['alpha']})."]
        if "defense_reduction_p_value" in m:
            d = m["defense_reduction_p_value"]
            out += ["", f"Policy vs full probability (same targets): {d['only_undefended_correct']} targets correct "
                        f"only without the policy, {d['only_defended_correct']} only with it; one-sided McNemar "
                        f"p = {d['value']:.3g}."]
        if "queries" in m:
            q = m["queries"]
            out += ["", f"Queries per target: mean {q['mean']:.2f}, max {q['max']}, total {q['total']} "
                        f"(budget {q['budget_per_target']} per target). Inference rules: {m.get('inference_rules')}."]
        traced = [t for t in rec.targets if t.trace is not None]
        if traced:
            out += ["", "### Per-target reconstruction (first 25)", "",
                    "| target | observed | queries (s=c → response) | rule | inferred | true | correct | aux guess |",
                    "|---|---|---|---|---|---|---|---|"]
            for t in traced[:25]:
                tr = t.trace
                qs = "; ".join(f"s={q.candidate}→{q.response:.4g}" for q in tr.queries) or "–"
                out.append(f"| {t.target_index} | {tr.observed_output:.4g} | {qs} | {tr.rule} | "
                           f"{'–' if tr.inferred is None else tr.inferred} | {t.ground_truth.private_attribute} | "
                           f"{t.inversion_correct} | {t.baseline_inferred} |")
        notes = NOTES.get(rec.evaluation, [])
        if notes:
            out += ["", "### Notes", ""] + [f"- {n}" for n in notes]
        out += ["", "### Metric definitions", ""]
        out += [f"- **{k}**: {v['definition']}." + (f" *Population:* {v['population']}." if v.get("population") else "")
                for k, v in m.items() if isinstance(v, dict) and "definition" in v]
    return compose(rec, summary_rows=summary, configuration=config, outcomes=out, verification=verification)
