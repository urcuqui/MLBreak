"""Evidence storage: one JSON (machine-readable) and one Markdown report per experiment."""

from __future__ import annotations

import io
import json
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import EXPERIMENT_ID_PATTERN, ExperimentRecord

_ID_RE = re.compile(EXPERIMENT_ID_PATTERN)


class ExperimentNotFound(KeyError):
    pass


def _check_id(experiment_id: str) -> str:
    # Strict pattern match also rules out path traversal through the id.
    if not _ID_RE.fullmatch(experiment_id or ""):
        raise ExperimentNotFound(experiment_id)
    return experiment_id


class EvidenceStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir = self.root / "experiments"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def json_path(self, experiment_id: str) -> Path:
        return self.dir / f"{_check_id(experiment_id)}.json"

    def md_path(self, experiment_id: str) -> Path:
        return self.dir / f"{_check_id(experiment_id)}.md"

    def save(self, record: ExperimentRecord) -> None:
        data = record.model_dump(mode="json")
        with self._lock:
            tmp = self.json_path(record.experiment_id).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.json_path(record.experiment_id))
            self.md_path(record.experiment_id).write_text(render_markdown(record))

    def load(self, experiment_id: str) -> ExperimentRecord:
        path = self.json_path(experiment_id)
        if not path.is_file():
            raise ExperimentNotFound(experiment_id)
        return ExperimentRecord.model_validate_json(path.read_text())

    def markdown(self, experiment_id: str) -> str:
        path = self.md_path(experiment_id)
        if not path.is_file():
            raise ExperimentNotFound(experiment_id)
        return path.read_text()

    def samples_csv(self, experiment_id: str) -> str:
        rec = self.load(experiment_id)
        rows = []
        for s in rec.samples:
            row = s.model_dump()
            for key in ("x_original", "x_adversarial"):
                vals = row.pop(key) or [None, None]
                row[f"{key}_0"], row[f"{key}_1"] = vals
            rows.append(row)
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue()

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in self.dir.glob("MLB-EXP-*.json"):
            try:
                rec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append(summarize(rec))
        return sorted(out, key=lambda r: r["created_at"], reverse=True)


def summarize(rec: dict[str, Any]) -> dict[str, Any]:
    m = rec.get("metrics") or {}
    asr = m.get("attack_success_rate") or {}
    return {
        "experiment_id": rec["experiment_id"],
        "created_at": rec["created_at"],
        "mode": rec["mode"],
        "evaluation": rec["evaluation"],
        "status": rec["status"],
        "scenario_id": rec["scenario_id"],
        "owasp_id": rec["owasp"]["id"],
        "model_id": rec["model"].get("model_id"),
        "attack": (rec.get("attack_config") or {}).get("attack"),
        "epsilon": (rec.get("attack_config") or {}).get("epsilon"),
        "norm": (rec.get("attack_config") or {}).get("norm"),
        "defense": (rec.get("defense_config") or {}).get("kind"),
        "n_samples": len(rec.get("samples") or []),
        "clean_accuracy": (m.get("clean_accuracy") or {}).get("value"),
        "attack_success_rate": asr.get("value"),
        "attack_success": f"{asr['numerator']}/{asr['denominator']}" if asr else None,
        "parent_experiment_id": rec.get("parent_experiment_id"),
        "replay_of": rec.get("replay_of"),
        "error": rec.get("error"),
    }


def _ratio(r: dict | None) -> str:
    if not r:
        return "n/a"
    v = r.get("value")
    pct = "undefined (empty denominator)" if v is None else f"{100 * v:.1f}%"
    return f"{pct} ({r['numerator']}/{r['denominator']})"


def render_markdown(rec: ExperimentRecord) -> str:
    m = rec.metrics
    lines = [
        f"# MLBreak experiment {rec.experiment_id}",
        "",
        f"> **Intentionally vulnerable scenario:** {'yes' if rec.intentionally_vulnerable else 'no (defended system)'}. "
        "Educational use only.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| OWASP category | {rec.owasp['id']} — {rec.owasp['name']} ({rec.owasp['taxonomy']} {rec.owasp['taxonomy_version']}) |",
        f"| Scenario | `{rec.scenario_id}` |",
        f"| Mode / evaluation | {rec.mode} / {rec.evaluation} |",
        f"| Status | **{rec.status}** |",
        f"| Created | {rec.created_at} |",
        f"| Seed | {rec.seed} |",
        f"| Model | `{rec.model['model_id']}` ({rec.model['estimator']}) |",
        f"| Model artifact SHA-256 | `{rec.model['artifact_sha256']}` |",
        f"| Target system artifact SHA-256 | `{rec.target_system['model_artifact_sha256']}` |",
        f"| Dataset / split | `{rec.dataset['dataset_id']}` / `{rec.dataset['split_id']}` |",
        f"| Test split SHA-256 | `{rec.dataset['test_sha256']}` |",
    ]
    if rec.parent_experiment_id:
        lines.append(f"| Parent experiment | `{rec.parent_experiment_id}` |")
    if rec.replay_of:
        lines.append(f"| Replay of | `{rec.replay_of}` |")
    if rec.error:
        lines += ["", f"**Error:** {rec.error}"]

    if rec.attacker_capabilities:
        lines += ["", "## Attacker capabilities", ""]
        lines += [f"- **{k}**: {v}" for k, v in rec.attacker_capabilities.items()]
    if rec.attack_config:
        lines += ["", "## Attack configuration", "", "```json", json.dumps(rec.attack_config, indent=2), "```"]
    lines += ["", "## Defense configuration", ""]
    lines += (["```json", json.dumps(rec.defense_config, indent=2), "```"] if rec.defense_config
              else ["None — the undefended (intentionally vulnerable) model was evaluated."])

    if m:
        lines += ["", "## Metrics", "", "| Metric | Value |", "|---|---|"]
        for key in ("clean_accuracy", "clean_accuracy_selected", "coverage",
                    "adversarial_accuracy", "attack_success_rate"):
            if key in m:
                lines.append(f"| {key} | {_ratio(m[key])} |")
        for key in ("attack_success_rate",):
            if key in m:
                lines += ["", f"*Attack success definition:* {m[key]['definition']}"]
        if "perturbation" in m:
            p = m["perturbation"]
            lines += ["", "### Perturbation magnitude", "",
                      f"Budget: ‖δ‖_{p['norm'].replace('linf', '∞').replace('l2', '2')} ≤ ε = {p['epsilon']}. "
                      f"Largest observed ‖δ‖/ε = {p['max_budget_ratio']:.4f} (verified ≤ 1).", "",
                      "| Subset | n | mean L∞ | max L∞ | mean L2 | max L2 |", "|---|---|---|---|---|---|"]
            for subset in ("attacked", "successful"):
                s = p[subset]
                if s["n"]:
                    lines.append(f"| {subset} | {s['n']} | {s['linf_mean']:.4f} | {s['linf_max']:.4f} | "
                                 f"{s['l2_mean']:.4f} | {s['l2_max']:.4f} |")
                else:
                    lines.append(f"| {subset} | 0 | – | – | – | – |")
        if "comparison" in m:
            lines += ["", "### Undefended vs defended", "", "| | Undefended | Defended |", "|---|---|---|"]
            for k, v in m["comparison"].items():
                lines.append(f"| {k} | {_ratio(v['undefended'])} | {_ratio(v['defended'])} |")
        if "notes" in m:
            lines += ["", "### Notes", ""] + [f"- {n}" for n in m["notes"]]

    if rec.samples and rec.mode != "baseline":
        lines += ["", "## Per-sample results (first 25)", "",
                  "| test idx | y | pred (clean) | pred (adv) | ‖δ‖∞ | success |", "|---|---|---|---|---|---|"]
        for s in rec.samples[:25]:
            pa = "–" if s.pred_adversarial is None else s.pred_adversarial
            d = "–" if s.perturbation_linf is None else f"{s.perturbation_linf:.4f}"
            lines.append(f"| {s.test_index} | {s.y_true} | {s.pred_original} | {pa} | {d} | {s.attack_success} |")
        lines += ["", "Prediction `-1` means the system abstained (confidence rejection)."]

    r = rec.reproducibility
    lines += ["", "## Reproducibility", "", "```json", json.dumps(r, indent=2), "```", "",
              f"Replay with: `mlbreak replay {rec.experiment_id}`", ""]
    return "\n".join(lines)
