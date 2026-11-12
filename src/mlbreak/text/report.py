"""Markdown report for ML01 spam-filter evasion records."""

from __future__ import annotations

import json
import unicodedata

from .schemas import TextEvasionRecord

NOTES = {
    "clean": ["Clean evaluation on the held-out test split; no message was modified."],
    "attack": ["Black-box attack: only the deployed system's class probabilities were used, through a counting, "
               "budget-enforcing oracle.",
               "An outcome is 'evaded' only if the target class was reached AND every declared constraint holds; the "
               "constraints are re-validated from the stored text and edit script by `mlbreak check`."],
    "adaptive_attack": ["Adaptive attack: a NEW search that queried the defended pipeline (normalisation is inside "
                        "its loop), with the parent's selected messages and text-preservation constraints."],
    "replay": ["Non-adaptive replay: the parent's adversarial messages were re-submitted unchanged.",
               "A low replay success rate does not establish robustness; compare with the adaptive attack."],
}

_METRICS = ("clean_accuracy", "spam_recall", "spam_precision", "ham_false_positive_rate", "clean_accuracy_selected",
            "attack_success_rate", "accuracy_under_attack", "budget_exhaustion_rate", "constraint_violation_rate")


def visible(text: str | None) -> str:
    """Render invisible / format characters explicitly (they are part of the attack)."""
    if text is None:
        return "–"
    return "".join(f"⟨U+{ord(c):04X}⟩" if unicodedata.category(c) in ("Cf", "Cc") else c for c in text)


def _ratio(r: dict | None) -> str:
    if not r:
        return "n/a"
    v = r.get("value")
    return ("undefined (empty denominator)" if v is None else f"{100 * v:.1f}%") + f" ({r['numerator']}/{r['denominator']})"


def render_text_markdown(rec: TextEvasionRecord) -> str:
    m = rec.metrics
    lines = [f"# MLBreak experiment {rec.experiment_id}", "",
             f"> **Intentionally vulnerable scenario:** {'yes' if rec.intentionally_vulnerable else 'no (defended system)'}."
             " Educational use only.", "",
             "| Field | Value |", "|---|---|",
             f"| OWASP category | {rec.owasp['id']} — {rec.owasp['name']} ({rec.owasp['taxonomy']} {rec.owasp['taxonomy_version']}) |",
             f"| Scenario | `{rec.scenario_id}` (spam-filter evasion, character-level) |",
             f"| Mode / evaluation | {rec.mode} / {rec.evaluation} |",
             f"| Status | **{rec.status}** |", f"| Created | {rec.created_at} |", f"| Seed | {rec.seed} |",
             f"| Model | `{rec.model['model_id']}` ({rec.model['estimator']}) |",
             f"| Base model artifact SHA-256 | `{rec.model['artifact_sha256']}` |",
             f"| Deployed system artifact SHA-256 | `{rec.target_system.get('model_artifact_sha256')}` |",
             f"| Dataset / split | `{rec.dataset['dataset_id']}` / `{rec.dataset['split_id']}` |",
             f"| Train / test split SHA-256 | `{rec.dataset['train_sha256']}` / `{rec.dataset['test_sha256']}` |",
             f"| Evidence schema | `{rec.schema_version}` |"]
    if rec.parent_experiment_id:
        lines.append(f"| Parent experiment | `{rec.parent_experiment_id}` |")
    if rec.replay_of:
        lines.append(f"| Replay of | `{rec.replay_of}` |")
    if rec.error:
        lines += ["", f"**Error:** {rec.error}"]
    lines += ["", "## TF-IDF configuration", "", "```json", json.dumps(rec.model.get("tfidf"), indent=2), "```"]
    if rec.attacker_capabilities:
        lines += ["", "## Attacker capabilities", ""] + [f"- **{k}**: {v}" for k, v in rec.attacker_capabilities.items()]
    if rec.attack_config:
        lines += ["", "## Attack configuration", "", "```json", json.dumps(rec.attack_config, indent=2), "```"]
    lines += ["", "## Defense configuration", ""]
    lines += (["```json", json.dumps(rec.defense_config, indent=2), "```"] if rec.defense_config
              else ["None — the undefended (intentionally vulnerable) classifier was evaluated."])
    if m:
        lines += ["", "## Metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {_ratio(m[k])} |" for k in _METRICS if k in m]
        if "outcomes" in m:
            lines += ["", "### Outcome partition", "", "| Outcome | Count |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in m["outcomes"]["counts"].items()]
            lines.append(f"| **total** | **{m['outcomes']['n_selected']}** |")
        if "effort" in m:
            e = m["effort"]
            lines += ["", "### Attack effort", ""] + [f"- **{k}**: {v}" for k, v in e.items() if k != "definition"]
        if "comparison" in m:
            lines += ["", "### Undefended vs defended", "", "| | Undefended | Defended |", "|---|---|---|"]
            lines += [f"| {k} | {_ratio(v['undefended'])} | {_ratio(v['defended'])} |" for k, v in m["comparison"].items()]
        lines += ["", "### Metric definitions", ""]
        lines += [f"- **{k}**: {m[k]['definition']}. *Population:* {m[k]['population']}." for k in _METRICS if k in m]
    attacked = [s for s in rec.samples if s.text_adversarial is not None]
    if attacked:
        lines += ["", "## Messages (first 15 attacked)", ""]
        for s in attacked[:15]:
            lines += [f"**test #{s.test_index} — {s.outcome}** · P(spam) {s.p_spam_original:.3f} → "
                      f"{s.p_spam_adversarial:.3f} · {s.queries} queries · edit distance {s.edit_distance}", "",
                      f"- original: `{visible(s.text_original)}`", f"- adversarial: `{visible(s.text_adversarial)}`"]
            if s.edits:
                lines.append("- edits: " + "; ".join(f"{e.op} `{visible(e.word_before)}`→`{visible(e.word_after)}`"
                                                      for e in s.edits))
            failed = [k for k, v in (s.constraints or {}).items() if not v]
            lines += [f"- constraints: {'all satisfied' if not failed else 'VIOLATED: ' + ', '.join(failed)}", ""]
    notes = NOTES.get(rec.evaluation, [])
    if notes:
        lines += ["", "### Notes", ""] + [f"- {n}" for n in notes]
    if rec.lineage:
        lines += ["", "## Lineage", "", "```json", json.dumps(rec.lineage, indent=2), "```"]
    lines += ["", "## Reproducibility", "", "```json", json.dumps(rec.reproducibility, indent=2), "```", "",
              f"Replay with: `mlbreak replay {rec.experiment_id}` · verify with: `mlbreak check {rec.experiment_id}`", ""]
    return "\n".join(lines)
