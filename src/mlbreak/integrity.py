"""Evidence integrity: canonical fingerprints and verification of stored experiment records.

Fingerprints (all SHA-256 over canonical JSON, see :mod:`mlbreak.hashing`):

* ``config_sha256``   — the validated request.
* ``samples_sha256``  — per-sample results *without* the derived ``outcome`` field, so it is
  comparable across schema versions.
* ``metrics_sha256``  — the metrics block.
* ``results_sha256``  — samples (with ``outcome`` when present) + metrics. Identical formula to
  schema v1 for v1 records.
* ``record_sha256``   — the whole record except this field itself (v2 only). Detects any edit of a
  stored evidence file, including timestamps, lineage and configuration.

:func:`verify_record` recomputes every fingerprint and re-derives the outcome metrics from the
per-sample evidence, so a record whose metrics were edited (or computed by a buggy formula) is
detected without re-running the experiment.
"""

from __future__ import annotations

from typing import Any

from .hashing import sha256_json
from .metrics import (OUTCOME_METRIC_KEYS, InconsistentSample, classify, clean_metrics, consistency_errors,
                      outcome_metrics)
from .schemas import SCHEMA_V1, ExperimentRecord, SampleResult


def sample_dump(s: SampleResult, include_outcome: bool = True) -> dict[str, Any]:
    d = s.model_dump()
    if not include_outcome or d.get("outcome") is None:
        d.pop("outcome", None)
    return d


def samples_sha256(samples: list[SampleResult]) -> str:
    return sha256_json([sample_dump(s, include_outcome=False) for s in samples])


def metrics_sha256(metrics: dict[str, Any]) -> str:
    return sha256_json(metrics)


def results_fingerprint(samples: list[SampleResult], metrics: dict[str, Any]) -> str:
    return sha256_json({"samples": [sample_dump(s) for s in samples], "metrics": metrics})


def record_sha256(record: ExperimentRecord) -> str:
    data = record.model_dump(mode="json")
    data["reproducibility"] = {k: v for k, v in data["reproducibility"].items() if k != "record_sha256"}
    return sha256_json(data)


def can_abstain(record: ExperimentRecord) -> bool:
    return (record.defense_config or {}).get("kind") == "confidence_rejection"


def _check(name: str, ok: bool | None, detail: str = "") -> dict[str, Any]:
    status = "skipped" if ok is None else ("pass" if ok else "fail")
    return {"check": name, "status": status, "detail": detail}


def verify_record(record: ExperimentRecord) -> dict[str, Any]:
    """Verify a stored record without re-running it. ``ok`` is False if any check fails."""
    repro = record.reproducibility
    checks: list[dict[str, Any]] = []
    checks.append(_check("config_sha256", repro.get("config_sha256") == sha256_json(record.request),
                         "hash of the stored request"))
    if record.status == "completed":
        checks.append(_check("results_sha256", repro.get("results_sha256") ==
                             results_fingerprint(record.samples, record.metrics),
                             "hash of per-sample results + metrics"))
    else:
        checks.append(_check("results_sha256", None, "failed experiments have no results"))
    if record.schema_version == SCHEMA_V1:
        checks.append(_check("record_sha256", None, "not available in schema v1 records"))
    else:
        checks.append(_check("record_sha256", repro.get("record_sha256") == record_sha256(record),
                             "hash of the whole record"))

    if record.status == "completed" and record.mode != "baseline":
        try:
            derived = [classify(s) for s in record.samples]
            consistent = True
        except InconsistentSample as exc:
            derived, consistent = [], False
            checks.append(_check("sample_consistency", False, str(exc)))
        if consistent:
            flags_ok = all((s.attack_success is True) == (o == "evaded") for s, o in zip(record.samples, derived)
                           if s.pred_adversarial is not None)
            checks.append(_check("sample_consistency", flags_ok, "attack_success flag ⇔ outcome 'evaded'"))
            if record.schema_version != SCHEMA_V1:
                stored = [s.outcome for s in record.samples]
                checks.append(_check("sample_outcomes", stored == derived, "stored outcome = derived outcome"))
                recomputed = outcome_metrics(record.samples, can_abstain(record))
                bad = [k for k in OUTCOME_METRIC_KEYS if k in recomputed and record.metrics.get(k) != recomputed[k]]
                bad += ["outcomes"] if record.metrics.get("outcomes") != recomputed["outcomes"] else []
                checks.append(_check("metrics_derivation", not bad,
                                     "metrics re-derived from per-sample outcomes" + (f"; mismatch: {bad}" if bad else "")))
                errs = consistency_errors(record.metrics)
                checks.append(_check("metrics_arithmetic", not errs, "; ".join(errs) or "partition invariants hold"))
            else:
                checks.append(_check("metrics_derivation", None,
                                     "v1 metrics used a different adversarial_accuracy definition; not re-derived"))
    if record.status == "completed" and record.mode == "baseline":
        recomputed = clean_metrics([s.pred_original for s in record.samples], [s.y_true for s in record.samples],
                                   can_abstain(record))
        stored = record.metrics.get("clean_accuracy", {})
        ok = (stored.get("numerator"), stored.get("denominator")) == (recomputed["clean_accuracy"]["numerator"],
                                                                    recomputed["clean_accuracy"]["denominator"])
        checks.append(_check("metrics_derivation", ok, "clean accuracy re-derived from per-sample decisions"))
    return {
        "experiment_id": record.experiment_id,
        "schema_version": record.schema_version,
        "ok": all(c["status"] != "fail" for c in checks),
        "checks": checks,
    }
