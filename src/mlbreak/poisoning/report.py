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


def render_poisoning_markdown(rec: PoisoningRecord) -> str:
    m = rec.metrics
    lines = [
        f"# MLBreak experiment {rec.experiment_id}",
        "",
        "> **Intentionally vulnerable scenario:** "
        f"{'yes' if rec.intentionally_vulnerable else 'no (defended training pipeline)'}. Educational use only.",
        "",
        "| Field | Value |", "|---|---|",
        f"| OWASP category | {rec.owasp['id']} — {rec.owasp['name']} ({rec.owasp['taxonomy']} "
        f"{rec.owasp['taxonomy_version']}) |",
        f"| Scenario | `{rec.scenario_id}` |",
        f"| Mode / evaluation | {rec.mode} / {rec.evaluation} |",
        f"| Status | **{rec.status}** |",
        f"| Created | {rec.created_at} |",
        f"| Seed | {rec.seed} |",
        f"| Model | `{rec.model['model_id']}` ({rec.model['estimator']}) |",
    ]
    for which, d in rec.models.items():
        if d:
            lines.append(f"| {which} model artifact SHA-256 | `{d['artifact_sha256']}` |")
    lines += [f"| Dataset / split | `{rec.dataset['dataset_id']}` / `{rec.dataset['split_id']}` |",
              f"| Train split SHA-256 | `{rec.dataset['train_sha256']}` |",
              f"| Evidence schema | `{rec.schema_version}` |"]
    if rec.parent_experiment_id:
        lines.append(f"| Parent experiment | `{rec.parent_experiment_id}` |")
    if rec.replay_of:
        lines.append(f"| Replay of | `{rec.replay_of}` |")
    if rec.error:
        lines += ["", f"**Error:** {rec.error}"]
    if rec.attacker_capabilities:
        lines += ["", "## Attacker capabilities", ""] + [f"- **{k}**: {v}" for k, v in rec.attacker_capabilities.items()]
    if rec.poison_config:
        lines += ["", "## Poisoning configuration", "", "```json", json.dumps(rec.poison_config, indent=2), "```"]
    if rec.poisoning and rec.poisoning.get("selection_rule"):
        p = rec.poisoning
        lines += ["", f"Selection: {p['selection_rule']}. Flipped {p['n_flips']} labels "
                      f"(budget {p['budget']}, verified).", "",
                  f"Poisoned training indices: `{p['poisoned_indices']}`"]
    lines += ["", "## Defense configuration", ""]
    lines += (["```json", json.dumps(rec.defense_config, indent=2), "```"] if rec.defense_config
              else ["None — the undefended (intentionally vulnerable) training pipeline was evaluated."])
    if m:
        lines += ["", "## Metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {_ratio(m[k])} |" for k in _HEADLINE if k in m]
        if "test_accuracy_delta" in m and m["test_accuracy_delta"]["value"] is not None:
            lines.append(f"| test_accuracy_delta | {100 * m['test_accuracy_delta']['value']:+.1f} pp |")
        lines += ["", "### Model comparison (test accuracy / recall)", "",
                  "| Model | Test accuracy | Recall class 0 | Recall class 1 |", "|---|---|---|---|"]
        for which, b in m.get("by_model", {}).items():
            lines.append(f"| {which} | {_ratio(b['test_accuracy'])} | {_ratio(b['recall_class_0'])} | "
                         f"{_ratio(b['recall_class_1'])} |")
        for key in ("test_outcomes", "defended_test_outcomes"):
            if key in m:
                o = m[key]
                lines += ["", f"### Test outcomes: {o['reference']} → {o['compared']}", "",
                          "| Outcome | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in o["counts"].items()]
        if "detection_counts" in m:
            lines += ["", "### Suspicious-label detection", "", "| | Count |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in m["detection_counts"].items()]
        lines += ["", "### Metric definitions", ""]
        for key in _HEADLINE:
            v = m.get(key)
            if isinstance(v, dict) and "definition" in v:
                lines.append(f"- **{key}**: {v['definition']}. *Population:* {v['population']}.")
    notes = NOTES.get(rec.evaluation, [])
    if notes:
        lines += ["", "### Notes", ""] + [f"- {n}" for n in notes]
    if rec.lineage:
        lines += ["", "## Lineage", "", "```json", json.dumps(rec.lineage, indent=2), "```"]
    lines += ["", "## Reproducibility", "", "```json", json.dumps(rec.reproducibility, indent=2), "```", "",
              f"Replay with: `mlbreak replay {rec.experiment_id}` · verify with: `mlbreak check {rec.experiment_id}`",
              ""]
    return "\n".join(lines)
