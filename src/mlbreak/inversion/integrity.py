"""Evidence integrity for ML03 inversion records (integrity, NOT authenticity — see
``docs/reproducibility.md#integrity-is-not-authenticity``).

Beyond the shared fingerprints, verification re-derives every reconstruction from the stored trace: query count
and budget, query order, distances, policy-valid responses, the consistent set, the ambiguity flag, the inference
rule and the inferred value (with the same :func:`~.attacks.decide` the attack used), correctness against the
stored ground truth, the observed output against the service row of the same record, and finally every metric.
"""

from __future__ import annotations

from typing import Any

from ..hashing import sha256_json
from ..integrity import INTEGRITY_GUARANTEE, _check, record_sha256
from .attacks import decide, prior_choice
from .defenses import apply_policy, is_valid_output
from .metrics import derive_inversion_metrics
from .schemas import InversionRecord


def results_fingerprint(targets, service, metrics) -> str:
    return sha256_json({"targets": [t.model_dump() for t in targets],
                        "service": [r.model_dump() for r in service], "metrics": metrics})


def trace_errors(rec: InversionRecord) -> list[str]:
    errs: list[str] = []
    policy = rec.output_policy
    cfg = rec.attack_config or {}
    released = {r.eval_index: r.released for r in rec.service}
    for t in rec.targets:
        k = t.target_index
        if t.baseline_inferred != prior_choice(t.auxiliary_p1):
            errs.append(f"target {k}: baseline guess ≠ auxiliary prior choice")
        if t.baseline_correct != (t.baseline_inferred == t.ground_truth.private_attribute):
            errs.append(f"target {k}: baseline_correct inconsistent with ground truth")
        tr = t.trace
        if rec.evaluation == "clean":
            if tr is not None or t.inversion_correct is not None:
                errs.append(f"target {k}: a clean evaluation has no attacker trace")
            continue
        if tr is None:
            errs.append(f"target {k}: missing attacker trace")
            continue
        if tr.auxiliary_p1 != t.auxiliary_p1:
            errs.append(f"target {k}: trace prior ≠ auxiliary prior")
        if released.get(k) != tr.observed_output:
            errs.append(f"target {k}: observed output ≠ the service's released output for that record")
        if tr.n_queries != len(tr.queries) or tr.n_queries > tr.query_budget or tr.query_budget != cfg.get("query_budget"):
            errs.append(f"target {k}: query count exceeds or misstates the budget")
        if [q.query_index for q in tr.queries] != list(range(1, len(tr.queries) + 1)):
            errs.append(f"target {k}: query order")
        cands = [q.candidate for q in tr.queries]
        if len(set(cands)) != len(cands) or not set(cands) <= set(tr.candidate_domain):
            errs.append(f"target {k}: duplicate or out-of-domain candidate")
        for q in tr.queries:
            if not is_valid_output(policy, q.response) or q.distance != abs(q.response - tr.observed_output):
                errs.append(f"target {k}: query {q.query_index} response/distance invalid under the policy")
        if not is_valid_output(policy, tr.observed_output):
            errs.append(f"target {k}: observed output is not a value the policy can release")
        responses = {q.candidate: q.response for q in tr.queries}
        cons, amb, rule, inferred = decide(tr.observed_output, responses, tr.candidate_domain, tr.auxiliary_p1,
                                           cfg.get("tie_policy", "auxiliary_prior"), cfg.get("tolerance", 0.0))
        if (cons, amb, rule, inferred) != (tr.consistent_candidates, tr.ambiguous, tr.rule, tr.inferred):
            errs.append(f"target {k}: stored decision does not follow from the stored queries")
        if t.inversion_correct != (tr.inferred is not None and tr.inferred == t.ground_truth.private_attribute):
            errs.append(f"target {k}: inversion_correct inconsistent with ground truth")
    for r in rec.service:
        if r.released != float(apply_policy(policy, [r.p_full])[0]) or r.decision != int(r.p_full >= 0.5) \
                or r.consumer_decision != int(r.released >= 0.5):
            errs.append(f"service row {r.eval_index}: released output/decisions inconsistent with the policy")
    return errs


def verify_inversion_record(rec: InversionRecord) -> dict[str, Any]:
    repro = rec.reproducibility
    checks = [
        _check("config_sha256", repro.get("config_sha256") == sha256_json(rec.request), "hash of the stored request"),
        _check("record_sha256", repro.get("record_sha256") == record_sha256(rec), "hash of the whole record"),
    ]
    if rec.status != "completed":
        checks.append(_check("results_sha256", None, "failed experiments have no results"))
        return _report(rec, checks)
    checks.append(_check("results_sha256", repro.get("results_sha256") ==
                         results_fingerprint(rec.targets, rec.service, rec.metrics),
                         "hash of per-target results + service rows + metrics"))
    errs = trace_errors(rec)
    checks.append(_check("reconstruction_revalidated", not errs,
                         "every decision re-derived from its stored queries; budgets, policy outputs and ground-truth "
                         "comparisons re-checked" + (f"; {errs[:5]}" if errs else "")))
    derived = derive_inversion_metrics(rec.evaluation, rec.targets, rec.service)
    diff = sorted(k for k in set(derived) | set(rec.metrics) if derived.get(k) != rec.metrics.get(k))
    checks.append(_check("metrics_derivation", not diff,
                         "metrics re-derived from per-target evidence" + (f"; mismatch: {diff[:8]}" if diff else "")))
    return _report(rec, checks)


def _report(rec, checks):
    return {"experiment_id": rec.experiment_id, "schema_version": rec.schema_version,
            "ok": all(c["status"] != "fail" for c in checks), "guarantee": INTEGRITY_GUARANTEE, "checks": checks}
