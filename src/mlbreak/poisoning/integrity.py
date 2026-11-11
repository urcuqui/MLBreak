"""Evidence integrity for ML02 poisoning records.

Same model as ML01 (:mod:`mlbreak.integrity`): unkeyed SHA-256 fingerprints stored in the record, plus
re-derivation of every metric from the per-sample evidence. This is an INTEGRITY check (consistency), not
an authenticity check: a writer who recomputes every digest is not detected (see
``docs/reproducibility.md#integrity-is-not-authenticity``).

Poisoning-specific checks: the label vectors (clean / observed / sanitised) match their recorded
hashes, the recorded poisoned indices are exactly the rows whose observed label differs from the true
label, and the flip count respects the configured budget.
"""

from __future__ import annotations

from typing import Any

from ..hashing import sha256_json
from ..integrity import INTEGRITY_GUARANTEE, _check, record_sha256
from .metrics import consistency_errors, derive_metrics
from .schemas import LabelFlipConfig, PoisoningRecord


def labels_sha256(values: list[int | None]) -> str:
    return sha256_json([None if v is None else int(v) for v in values])


def results_fingerprint(parts) -> str:
    """Fingerprint of ``(training_samples, test_samples, metrics)``."""
    train, test, metrics = parts
    return sha256_json({"training_samples": [s.model_dump() for s in train],
                        "test_samples": [s.model_dump() for s in test], "metrics": metrics})


def label_hashes(train) -> dict[str, str]:
    out = {"clean_labels_sha256": labels_sha256([s.y_true for s in train]),
           "observed_labels_sha256": labels_sha256([s.y_observed for s in train])}
    if any(s.y_sanitized is not None or s.removed for s in train):  # defensive replay records
        out["sanitized_labels_sha256"] = labels_sha256([s.y_sanitized for s in train])
    return out


def verify_poisoning_record(rec: PoisoningRecord) -> dict[str, Any]:
    repro = rec.reproducibility
    checks = [
        _check("config_sha256", repro.get("config_sha256") == sha256_json(rec.request), "hash of the stored request"),
        _check("record_sha256", repro.get("record_sha256") == record_sha256(rec), "hash of the whole record"),
    ]
    if rec.status != "completed":
        checks.append(_check("results_sha256", None, "failed experiments have no results"))
        return _report(rec, checks)
    checks.append(_check("results_sha256", repro.get("results_sha256") ==
                         results_fingerprint((rec.training_samples, rec.test_samples, rec.metrics)),
                         "hash of training rows + test rows + metrics"))
    stored = rec.poisoning or {}
    hashes = label_hashes(rec.training_samples)
    bad = [k for k, v in hashes.items() if stored.get(k) is not None and stored.get(k) != v]
    missing = [k for k in hashes if stored.get(k) is None and rec.evaluation != "clean_training"]
    checks.append(_check("label_vectors", not bad and not missing,
                         "clean/observed/sanitised label vectors match their recorded hashes"
                         + (f"; mismatch: {bad}" if bad else "") + (f"; missing: {missing}" if missing else "")))
    errs = consistency_errors(rec.evaluation, rec.training_samples, rec.test_samples)
    checks.append(_check("sample_consistency", not errs, "; ".join(errs[:5]) or "per-sample invariants hold"))
    if rec.evaluation != "clean_training":
        flipped = [s.train_index for s in rec.training_samples if s.poisoned]
        ok = flipped == stored.get("poisoned_indices")
        cfg = LabelFlipConfig.model_validate(rec.poison_config)
        budget = int(round(cfg.fraction * len(rec.training_samples)))
        ok_budget = len(flipped) <= budget and (cfg.strategy == "random" or all(
            s.y_true == cfg.target_class for s in rec.training_samples if s.poisoned))
        checks.append(_check("poisoned_indices", ok, "recorded poisoned indices = rows whose label was flipped"))
        checks.append(_check("poison_budget", ok_budget,
                             f"{len(flipped)} flips ≤ budget {budget}" +
                             ("" if cfg.strategy == "random" else f", all of target class {cfg.target_class}")))
    try:
        derived = derive_metrics(rec.evaluation, rec.training_samples, rec.test_samples, rec.poison_config,
                                 rec.defense_config)
        same = derived == rec.metrics
        detail = "metrics re-derived from per-sample evidence"
        if not same:
            diff = sorted(k for k in set(derived) | set(rec.metrics) if derived.get(k) != rec.metrics.get(k))
            detail += f"; mismatch: {diff[:8]}"
    except (TypeError, KeyError, ValueError) as exc:  # malformed rows
        same, detail = False, f"could not re-derive metrics: {exc}"
    checks.append(_check("metrics_derivation", same, detail))
    return _report(rec, checks)


def _report(rec: PoisoningRecord, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"experiment_id": rec.experiment_id, "schema_version": rec.schema_version,
            "ok": all(c["status"] != "fail" for c in checks), "guarantee": INTEGRITY_GUARANTEE, "checks": checks}
