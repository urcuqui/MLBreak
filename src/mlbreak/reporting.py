"""Shared structure for every experiment report (ML01 2-D, ML01 spam filter, ML02 poisoning).

Every report has the same seven sections, in this order, so evidence reads the same across laboratories:

1. Summary                      — identity, status, OWASP category, scenario
2. Threat model                 — scenario threat model + the attacker capabilities of THIS experiment
3. Experiment configuration     — model, data, attack, defense, lineage (family-specific)
4. Observed outcomes            — metrics with their own definitions and denominators (family-specific)
5. Defense limitations          — what the result does NOT show, backed by the regression cases
6. Reproducibility              — fingerprints, numerical environment, replay command
7. Integrity verification       — result of the integrity check when the report was rendered

Sections 2, 5, 6 and 7 are written here; each family supplies 1-specific rows, 3 and 4, so specialised metric
definitions are preserved. Reports served by the API / CLI are rendered from the stored JSON with a FRESH
integrity check, so a tampered record is reported as such instead of showing a stale clean report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

SECTIONS = ("1. Summary", "2. Threat model", "3. Experiment configuration", "4. Observed outcomes",
            "5. Defense limitations", "6. Reproducibility", "7. Integrity verification")

THREAT_MODELS: dict[str, dict[str, str]] = {
    "ml01-moons-evasion": {
        "asset": "integrity of the classifier's decision on a test-time input",
        "trust boundary": "input → model (the 2 input features are attacker-controlled within an Lp budget)",
        "attacker goal": "untargeted evasion: a correctly decided input is decided as the other class (abstention "
                         "is not a success)",
        "out of scope": "training-data poisoning, model extraction, physical-world or semantic constraints",
        "details": "docs/threat-model.md",
    },
    "ml01-spam-filter-evasion": {
        "asset": "integrity of the spam filter's decision on an incoming message",
        "trust boundary": "message sender → classifier (the message text is attacker-controlled)",
        "attacker goal": "spam delivered as ham (or ham flagged as spam) by character edits within declared "
                         "text-preservation constraints",
        "out of scope": "training-data poisoning, model access beyond class probabilities, proof that the "
                        "meaning of the message is preserved",
        "details": "docs/ml01-spam-filter-evasion.md#2-threat-model",
    },
    "ml02-moons-label-flip": {
        "asset": "integrity of the training labels, hence of the trained model",
        "trust boundary": "label source → training pipeline (a bounded fraction of labels is attacker-controlled)",
        "attacker goal": "degrade accuracy (random flipping) or make the model miss a target class (targeted)",
        "out of scope": "feature or clean-label poisoning, backdoors, test-time manipulation, an attacker who "
                        "optimises against the sanitiser",
        "details": "docs/ml02-data-poisoning.md#2-threat-model",
    },
}

SEMANTIC_DISCLAIMER = ("A constraint-valid adversarial message is NOT proof that its meaning is preserved: edit "
                       "distance, character/word ratios and protected tokens are proxies; no human or semantic "
                       "check was performed.")


def _scope_note(rec) -> str:
    return (f"Results are specific to scenario `{rec.scenario_id}`, model `{rec.model.get('model_id')}`, seed "
            f"{rec.seed} and the bundled dataset; they are measurements, not guarantees, and margins are a few "
            "samples (see the regression cases).")


def limitations(rec) -> list[str]:
    """What this experiment's result does NOT establish. Statements are backed by declared regression cases."""
    out = [_scope_note(rec)]
    ev, scen = rec.evaluation, rec.scenario_id
    defense = rec.defense_config or {}
    kind = defense.get("kind")
    if ev == "replay" or ev == "poisoning_replay":
        out.append("Non-adaptive replay: the adversarial inputs were produced WITHOUT knowledge of the defense. A low "
                   "success rate under replay does not establish robustness; compare with an adaptive evaluation.")
    if ev == "adaptive_attack":
        out.append("The adaptive attacker is one specific strategy with a bounded budget: its success rate is a LOWER "
                   "bound on what a stronger attacker could achieve against this defense.")
    if ev in ("attack", "adaptive_attack") and scen == "ml01-moons-evasion":
        out.append("PGD/FGSM without restarts and random search are not the strongest attacks; success rates are lower "
                   "bounds.")
    if scen == "ml01-spam-filter-evasion" and ev != "clean":
        out.append(SEMANTIC_DISCLAIMER)
        out.append("The DeepWordBug-style search is greedy and samples candidate edits: a lower bound, and not "
                   "comparable number-for-number with the original DeepWordBug or BERT-based results.")
    if kind == "adversarial_training":
        out.append("Adversarial training reduces but does not eliminate adaptive success: across 7 numerical "
                   "environments the residual adaptive success for MLB-REG-002 was 25.0–31.7%, and 25.5–43.4% across "
                   "seeds (docs/regression-analysis-mlb-reg-002.md).")
    if kind == "confidence_rejection":
        out.append("Confidence rejection is bypassed by an adaptive attacker that targets confidence above the "
                   "threshold (MLB-REG-004), and abstention costs coverage on clean inputs.")
    if kind == "text_normalization":
        if defense.get("spell_snap"):
            out.append("Spell-snap undoes single-character typos, but adaptive edits that land on OTHER real words "
                       "(e.g. msg→msn) bypass it (MLB-REG-009); for naive Bayes it does not significantly reduce an "
                       "adaptive attacker (MLB-REG-013).")
        if defense.get("unicode") and not defense.get("spell_snap") and not defense.get("leetspeak"):
            out.append("Unicode-only normalisation covers invisible and homoglyph characters only (MLB-REG-011); it "
                       "does nothing against ASCII edits.")
    if kind == "text_adversarial_training":
        out.append("Augmentation uses adversarial examples from ONE attack configuration on training spam; it reduces "
                   "but does not remove adaptive success (MLB-REG-012) and may not generalise to other attacks.")
    if kind == "knn_label_sanitization":
        out.append("kNN sanitisation is blind to boundary-targeted flips (MLB-REG-007) and backfires when targeted "
                   "flips are dense (MLB-REG-008); its test-time effect on random/sparse flips is too small to be "
                   "statistically significant on 150 test samples (MLB-REG-005/006).")
    if scen == "ml02-moons-label-flip" and ev == "poisoning":
        out.append("Only label flipping is modelled; no adaptive poisoner (one who anticipates the sanitiser) is "
                   "implemented.")
    if not kind and ev in ("clean", "attack", "clean_training", "poisoning"):
        out.append("No defense was evaluated in this experiment.")
    return out


def _threat_model(rec) -> list[str]:
    tm = THREAT_MODELS.get(rec.scenario_id, {})
    lines = [f"- **{k}**: {v}" for k, v in tm.items() if k != "details"]
    if tm.get("details"):
        lines.append(f"- **details**: `{tm['details']}`")
    lines += ["", "### Attacker capabilities in this experiment", ""]
    caps = rec.attacker_capabilities
    lines += [f"- **{k}**: {v}" for k, v in caps.items()] if caps else \
        ["None — this is a clean evaluation; no attacker was simulated."]
    return lines


def _reproducibility(rec) -> list[str]:
    r = rec.reproducibility or {}
    rows = [("config_sha256", "the validated request"), ("results_sha256", "per-sample results + metrics"),
            ("metrics_sha256", "metrics block"), ("record_sha256", "whole record except this field"),
            ("numerical_environment_sha256", "platform, BLAS build/kernel, library versions")]
    lines = ["| Fingerprint | Covers | Value |", "|---|---|---|"]
    lines += [f"| {k} | {d} | `{r.get(k)}` |" for k, d in rows if k in r]
    deps = r.get("dependencies") or {}
    if deps:
        lines += ["", "Dependencies: " + ", ".join(f"{k} {v}" for k, v in deps.items()) + "."]
    backend = r.get("numerical_backend") or {}
    if backend:
        lines.append(f"Numerical backend: {json.dumps(backend)}.")
    lines += ["", f"Replay: `mlbreak replay {rec.experiment_id}` (refused if integrity verification fails). "
                  "Bit-identical results are expected only in the same numerical environment.", "",
              "<details><summary>Full reproducibility block</summary>", "", "```json", json.dumps(r, indent=2), "```",
              "", "</details>"]
    return lines


def _integrity(rec, verification: dict | None) -> list[str]:
    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if verification is None:
        lines = ["Not verified when this report was rendered."]
    else:
        status = "PASS" if verification["ok"] else "**FAIL — do not rely on this evidence**"
        lines = [f"Result at {when}: {status}", "", "| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {c['check']} | {c['status']} | {c['detail']} |" for c in verification["checks"]]
        lines += ["", f"Guarantee: {verification.get('guarantee', '')}."]
    lines += ["", f"Re-verify with `mlbreak check {rec.experiment_id}` or `GET /api/experiments/{rec.experiment_id}/verify`. "
                  "Integrity is not authenticity: see docs/reproducibility.md#integrity-is-not-authenticity."]
    return lines


def compose(rec, *, summary_rows: list[tuple[str, str]], configuration: list[str], outcomes: list[str],
            verification: dict | None) -> str:
    vuln = "yes" if rec.intentionally_vulnerable else "no (defended system)"
    rows = [("OWASP category", f"{rec.owasp['id']} — {rec.owasp['name']} ({rec.owasp['taxonomy']} "
                               f"{rec.owasp['taxonomy_version']})"),
            ("Scenario", f"`{rec.scenario_id}`"), ("Mode / evaluation", f"{rec.mode} / {rec.evaluation}"),
            ("Status", f"**{rec.status}**"), ("Created", rec.created_at), ("Seed", str(rec.seed)), *summary_rows,
            ("Evidence schema", f"`{rec.schema_version}`")]
    if rec.parent_experiment_id:
        rows.append(("Parent experiment", f"`{rec.parent_experiment_id}`"))
    if rec.replay_of:
        rows.append(("Replay of", f"`{rec.replay_of}`"))
    lines = [f"# MLBreak experiment {rec.experiment_id}", "",
             f"> **Intentionally vulnerable scenario:** {vuln}. Educational and defensive research use only.", "",
             f"## {SECTIONS[0]}", "", "| Field | Value |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows]]
    if rec.error:
        lines += ["", f"**Error:** {rec.error}"]
    lines += ["", f"## {SECTIONS[1]}", "", *_threat_model(rec)]
    lines += ["", f"## {SECTIONS[2]}", "", *configuration]
    if rec.lineage:
        lin = rec.lineage
        lines += ["", "### Lineage", "",
                  f"Relation: `{lin['relation']}` → parent `{lin['parent_experiment_id']}` ({lin['parent_evaluation']}, "
                  f"{lin['parent_schema_version']}).", "",
                  "Validated before execution: " + "; ".join(lin.get("checks_passed", [])) + "."]
    lines += ["", f"## {SECTIONS[3]}", "", *(outcomes or ["No results (the experiment did not complete)."])]
    lines += ["", f"## {SECTIONS[4]}", "", *[f"- {x}" for x in limitations(rec)]]
    lines += ["", f"## {SECTIONS[5]}", "", *_reproducibility(rec)]
    lines += ["", f"## {SECTIONS[6]}", "", *_integrity(rec, verification), ""]
    return "\n".join(lines)


def json_block(obj: Any) -> list[str]:
    return ["```json", json.dumps(obj, indent=2), "```"]
