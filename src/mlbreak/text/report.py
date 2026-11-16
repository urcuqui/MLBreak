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


def render_text_markdown(rec: TextEvasionRecord, verification: dict | None = None) -> str:
    """ML01 spam-filter evasion report (shared seven-section structure, see mlbreak.reporting)."""
    from ..reporting import SEMANTIC_DISCLAIMER, compose, json_block
    m = rec.metrics
    summary = [("Model", f"`{rec.model['model_id']}` ({rec.model['estimator']})")]
    config = ["| Item | Value |", "|---|---|",
              f"| Base model artifact SHA-256 | `{rec.model['artifact_sha256']}` |",
              f"| Deployed system artifact SHA-256 | `{rec.target_system.get('model_artifact_sha256')}` |",
              f"| Dataset / split | `{rec.dataset['dataset_id']}` / `{rec.dataset['split_id']}` |",
              f"| Dataset license | {rec.dataset.get('license')} (DOI {rec.dataset.get('doi')}) |",
              f"| Train / test split SHA-256 | `{rec.dataset['train_sha256']}` / `{rec.dataset['test_sha256']}` |",
              "", "### TF-IDF configuration", "", *json_block(rec.model.get("tfidf"))]
    if rec.attack_config:
        config += ["", "### Attack configuration", "", *json_block(rec.attack_config)]
    config += ["", "### Defense configuration", ""]
    config += json_block(rec.defense_config) if rec.defense_config else \
        ["None — the undefended (intentionally vulnerable) classifier was evaluated."]
    out: list[str] = []
    if m:
        out += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {_ratio(m[k])} |" for k in _METRICS if k in m]
        if "outcomes" in m:
            out += ["", "### Outcome partition", "", "| Outcome | Count |", "|---|---|"]
            out += [f"| {k} | {v} |" for k, v in m["outcomes"]["counts"].items()]
            out.append(f"| **total** | **{m['outcomes']['n_selected']}** |")
        if "effort" in m:
            out += ["", "### Attack effort", ""] + [f"- **{k}**: {v}" for k, v in m["effort"].items() if k != "definition"]
        if "comparison" in m:
            out += ["", "### Undefended vs defended", "", "| | Undefended | Defended |", "|---|---|---|"]
            out += [f"| {k} | {_ratio(v['undefended'])} | {_ratio(v['defended'])} |" for k, v in m["comparison"].items()]
        out += ["", "### Metric definitions", ""]
        out += [f"- **{k}**: {m[k]['definition']}. *Population:* {m[k]['population']}." for k in _METRICS if k in m]
    attacked = [x for x in rec.samples if x.text_adversarial is not None]
    if attacked:
        out += ["", "### Messages (first 15 attacked)", "", f"> {SEMANTIC_DISCLAIMER}", ""]
        for x in attacked[:15]:
            out += [f"**test #{x.test_index} — {x.outcome}** · P(spam) {x.p_spam_original:.3f} → "
                    f"{x.p_spam_adversarial:.3f} · {x.queries} queries · edit distance {x.edit_distance}", "",
                    f"- original: `{visible(x.text_original)}`", f"- adversarial: `{visible(x.text_adversarial)}`"]
            if x.edits:
                out.append("- edits: " + "; ".join(f"{e.op} `{visible(e.word_before)}`→`{visible(e.word_after)}`"
                                                   for e in x.edits))
            failed = [k for k, v in (x.constraints or {}).items() if not v]
            out += [f"- constraints: {'all satisfied' if not failed else 'VIOLATED: ' + ', '.join(failed)}", ""]
    notes = NOTES.get(rec.evaluation, [])
    if notes:
        out += ["", "### Notes", ""] + [f"- {n}" for n in notes]
    return compose(rec, summary_rows=summary, configuration=config, outcomes=out, verification=verification)
