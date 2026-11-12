"""Evidence integrity for ML01 spam-filter evasion records (integrity, NOT authenticity — see
``docs/reproducibility.md#integrity-is-not-authenticity``).

Beyond the shared fingerprints, the check RE-VALIDATES every adversarial message from the stored original
text, adversarial text and edit script: edit distance, character and word ratios, permitted operations and
characters, protected tokens, Unicode validity and that the edit script reproduces the stored text. A record
claiming an evasion that does not satisfy its own constraints fails verification.
"""

from __future__ import annotations

from typing import Any

from ..hashing import sha256_json
from ..integrity import INTEGRITY_GUARANTEE, _check, record_sha256
from .attacks import validate_constraints
from .metrics import consistency_errors, derive_text_metrics
from .schemas import TextAttackConfig, TextEvasionRecord


def results_fingerprint(samples, test_predictions, metrics) -> str:
    return sha256_json({"samples": [s.model_dump() for s in samples],
                        "test_predictions": [p.model_dump() for p in test_predictions], "metrics": metrics})


def verify_text_record(rec: TextEvasionRecord) -> dict[str, Any]:
    repro = rec.reproducibility
    checks = [
        _check("config_sha256", repro.get("config_sha256") == sha256_json(rec.request), "hash of the stored request"),
        _check("record_sha256", repro.get("record_sha256") == record_sha256(rec), "hash of the whole record"),
    ]
    if rec.status != "completed":
        checks.append(_check("results_sha256", None, "failed experiments have no results"))
        return _report(rec, checks)
    checks.append(_check("results_sha256", repro.get("results_sha256") ==
                         results_fingerprint(rec.samples, rec.test_predictions, rec.metrics),
                         "hash of samples + clean test predictions + metrics"))
    if rec.attack_config is not None:
        cfg = TextAttackConfig.model_validate(rec.attack_config)
        bad = []
        for s in rec.samples:
            if s.text_adversarial is None or s.edits is None:
                continue
            chk, stats = validate_constraints(s.text_original, s.text_adversarial, cfg, s.edits)
            if chk != s.constraints or stats["edit_distance"] != s.edit_distance:
                bad.append(s.test_index)
        checks.append(_check("constraints_revalidated", not bad,
                             "constraints re-validated from original text, adversarial text and edit script"
                             + (f"; mismatch at test indices {bad[:10]}" if bad else "")))
        errs = consistency_errors(rec.samples, cfg.min_target_confidence, cfg.target_class)
        checks.append(_check("sample_consistency", not errs, "; ".join(errs[:5]) or "outcomes consistent"))
    derived = derive_text_metrics(rec.samples, rec.test_predictions)
    stored = {k: v for k, v in rec.metrics.items() if k != "comparison"}
    diff = sorted(k for k in set(derived) | set(stored) if derived.get(k) != stored.get(k))
    checks.append(_check("metrics_derivation", not diff,
                         "metrics re-derived from per-sample evidence" + (f"; mismatch: {diff[:8]}" if diff else "")))
    return _report(rec, checks)


def _report(rec, checks):
    return {"experiment_id": rec.experiment_id, "schema_version": rec.schema_version,
            "ok": all(c["status"] != "fail" for c in checks), "guarantee": INTEGRITY_GUARANTEE, "checks": checks}
