"""Human-readable Markdown report for ML02 poisoning records."""

from __future__ import annotations

import json

from .schemas import PoisoningRecord

NOTES = {
    "clean_training": [
        "Clean training: no labels were modified. This is the reference model for every poisoning experiment "
        "with the same model and seed.",
    ],
    "poisoning": [
        "The model was RETRAINED on the poisoned labels; every prediction comes from that retrained model.",
        "Test inputs and test labels are clean: poisoning only acts through training.",
        "attack_success_rate counts test samples the clean model gets right and the poisoned model gets wrong "
        "(restricted to the target class for targeted attacks).",
    ],
    "poisoning_replay": [
        "Defensive replay: the defense was applied to the EXACT poisoned labels of the parent experiment; the "
        "regenerated poisoned model reproduces the parent's artifact hash.",
        "The poisoner did not know the defense. An attacker who anticipates kNN sanitisation can flip clustered "
        "or boundary samples whose neighbours agree with the flipped label (see the boundary-targeted "
        "regression case), so a good replay result does not establish robustness.",
        "clean_data_* metrics apply the same defense to the CLEAN labels: the defense's cost with no attack.",
    ],
}

_HEADLINE = ("test_accuracy", "clean_test_accuracy", "undefended_test_accuracy", "train_accuracy",
             "train_accuracy_observed", "recall_class_0", "recall_class_1", "clean_target_class_recall",
             "undefended_target_class_recall", "target_class_recall", "poison_rate", "attack_success_rate",
             "undefended_attack_success_rate", "collateral_damage_rate", "flagged", "detection_recall",
             "detection_precision", "false_positive_rate", "labels_repaired", "labels_corrupted", "removed",
             "recovery_rate", "clean_data_test_accuracy", "clean_data_flag_rate")


def _ratio(r: dict | None) -> str:
    if not r:
        return "n/a"
    v = r.get("value")
    pct = "undefined (empty denominator)" if v is None else f"{100 * v:.1f}%"
    return f"{pct} ({r['numerator']}/{r['denominator']})"


def render_poisoning_markdown(rec: PoisoningRecord, verification: dict | None = None) -> str:
    """ML02 poisoning report (shared seven-section structure, see mlbreak.reporting)."""
    from ..reporting import compose, json_block
    m = rec.metrics
    summary = [("Model", f"`{rec.model['model_id']}` ({rec.model['estimator']})")]
    config = ["| Item | Value |", "|---|---|"]
    config += [f"| {which} model artifact SHA-256 | `{d['artifact_sha256']}` |" for which, d in rec.models.items() if d]
    config += [f"| Dataset / split | `{rec.dataset['dataset_id']}` / `{rec.dataset['split_id']}` |",
               f"| Train split SHA-256 | `{rec.dataset['train_sha256']}` |"]
    if rec.poison_config:
        config += ["", "### Poisoning (attack)", "", *json_block(rec.poison_config)]
    if rec.poisoning and rec.poisoning.get("selection_rule"):
        p = rec.poisoning
        config += ["", f"Selection: {p['selection_rule']}. Flipped {p['n_flips']} labels (budget {p['budget']}, "
                       "verified).", "", f"Poisoned training indices: `{p['poisoned_indices']}`"]
    config += ["", "### Defense", ""]
    config += json_block(rec.defense_config) if rec.defense_config else \
        ["None — the undefended (intentionally vulnerable) training pipeline was evaluated."]
    out: list[str] = []
    if m:
        out += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {_ratio(m[k])} |" for k in _HEADLINE if k in m]
        if "test_accuracy_delta" in m and m["test_accuracy_delta"]["value"] is not None:
            out.append(f"| test_accuracy_delta | {100 * m['test_accuracy_delta']['value']:+.1f} pp |")
        out += ["", "### Model comparison (test accuracy / recall)", "",
                "| Model | Test accuracy | Recall class 0 | Recall class 1 |", "|---|---|---|---|"]
        out += [f"| {w} | {_ratio(b['test_accuracy'])} | {_ratio(b['recall_class_0'])} | {_ratio(b['recall_class_1'])} |"
                for w, b in m.get("by_model", {}).items()]
        for key in ("test_outcomes", "defended_test_outcomes"):
            if key in m:
                o = m[key]
                out += ["", f"### Test outcomes: {o['reference']} → {o['compared']}", "", "| Outcome | Count |",
                        "|---|---|"] + [f"| {k} | {v} |" for k, v in o["counts"].items()]
        if "detection_counts" in m:
            out += ["", "### Suspicious-label detection", "", "| | Count |", "|---|---|"]
            out += [f"| {k} | {v} |" for k, v in m["detection_counts"].items()]
        notes = NOTES.get(rec.evaluation, [])
        if notes:
            out += ["", "### Notes", ""] + [f"- {n}" for n in notes]
        out += ["", "### Metric definitions", ""]
        out += [f"- **{k}**: {m[k]['definition']}. *Population:* {m[k]['population']}." for k in _HEADLINE
                if isinstance(m.get(k), dict) and "definition" in m[k]]
    return compose(rec, summary_rows=summary, configuration=config, outcomes=out, verification=verification)
