"""Evidence integrity for ML04 membership records (integrity, NOT authenticity — see
``docs/reproducibility.md#integrity-is-not-authenticity``).

Beyond the shared fingerprints, verification re-derives, without re-running any model:

* every threshold-attack score from the stored released output (and label) with the declared signal;
* every operating point of both attacks from the stored CALIBRATION scores and the attacker-known calibration
  membership (thresholds, calibration rates, status, class-conditional fallbacks, calibration AUC);
* every member / non-member decision from the stored score and the re-derived thresholds;
* query order and budget, policy-valid released outputs, utility rows under the policy;
* ground-truth consistency (designated group ⇔ source population; real training membership false for the decoy
  control), and finally every metric.

The shadow meta-classifier's scores cannot be re-derived without re-training it; they are covered by the results
fingerprint and by exact replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..hashing import sha256_json
from ..integrity import INTEGRITY_GUARANTEE, _check, record_sha256
from .attacks import calibrate, class_keys, decide, signal
from .defenses import apply_policy, is_valid_output
from .metrics import ATTACKS, derive_membership_metrics


def _dump(x):
    return x.model_dump() if hasattr(x, "model_dump") else x


def results_fingerprint(candidates, utility, metrics, calibration, shadow) -> str:
    return sha256_json({"candidates": [_dump(c) for c in candidates], "utility": [_dump(u) for u in utility],
                        "metrics": metrics, "calibration": calibration, "shadow": shadow})


def attacker_side_sha256(candidates) -> str:
    """Fingerprint of everything the ADVERSARY observed or produced (no ground truth)."""
    return sha256_json([{k: _dump(getattr(c, k)) for k in ("candidate_id", "observation", "threshold", "shadow")}
                        for c in candidates])


def _calibration_errors(rec) -> list[str]:
    errs: list[str] = []
    cfg = rec.attack_config or {}
    attacks = (rec.calibration or {}).get("attacks", {})
    n_ref = sum(int(m["n_in"]) + int(m["n_out"]) for m in (rec.calibration or {}).get("reference_models", []))
    for a in ATTACKS:
        cal = attacks.get(a)
        if cal is None:
            errs.append(f"{a}: missing calibration")
            continue
        if len(cal["scores"]) != n_ref or len(cal["is_member"]) != n_ref or len(cal["class_keys"]) != n_ref:
            errs.append(f"{a}: calibration size ≠ reference-model records ({n_ref})")
            continue
        if sum(cal["is_member"]) != sum(int(m["n_in"]) for m in rec.calibration["reference_models"]):
            errs.append(f"{a}: calibration member count ≠ reference-model training sizes")
        if cal["mode"] != cfg.get("threshold_mode") or cal["target_fpr"] != cfg.get("target_fpr"):
            errs.append(f"{a}: calibration mode/target differs from the attack configuration")
        again = calibrate(cal["scores"], cal["is_member"], cal["class_keys"], cal["mode"],
                          cfg.get("min_class_calibration", 50), cal["target_fpr"])
        stored = {"class_modes": cal["class_modes"], "operating_points": cal["operating_points"],
                  "calibration_auc": cal["calibration_auc"]}
        if sha256_json(again) != sha256_json(stored):
            errs.append(f"{a}: operating points do not follow from the stored calibration data")
    return errs


def candidate_errors(rec) -> list[str]:
    errs: list[str] = []
    cfg = rec.attack_config or {}
    policy = rec.output_policy
    decoy = rec.model.get("training_population") != "victim_train"
    sel = rec.candidate_selection
    if len(rec.candidates) != sel["n_members"] + sel["n_non_members"] or \
            sum(c.truth.group == "member" for c in rec.candidates) != sel["n_members"]:
        errs.append("candidate counts differ from the selection")
    if [c.candidate_id for c in rec.candidates] != [f"C-{k:04d}" for k in range(len(rec.candidates))]:
        errs.append("candidate identifiers are not sequential")
    points = {a: {p["name"]: p for p in rec.calibration["attacks"][a]["operating_points"]}
              for a in ATTACKS} if rec.calibration else {}
    for k, c in enumerate(rec.candidates):
        t = c.truth
        if (t.group == "member") != (t.population == "victim_train"):
            errs.append(f"{c.candidate_id}: designated group ≠ source population")
        if t.in_training != (t.group == "member" and not decoy):
            errs.append(f"{c.candidate_id}: in_training inconsistent with the victim's training population")
        if rec.evaluation == "clean":
            if c.observation or c.threshold or c.shadow:
                errs.append(f"{c.candidate_id}: a clean evaluation has no attacker output")
            continue
        o = c.observation
        if o is None or c.threshold is None or c.shadow is None:
            errs.append(f"{c.candidate_id}: missing attacker output")
            continue
        if o.query_index != k + 1:
            errs.append(f"{c.candidate_id}: query order")
        if not is_valid_output(policy, o.released_output):
            errs.append(f"{c.candidate_id}: released output is not a value the policy can release")
        if (o.label is None) == bool(cfg.get("label_known")):
            errs.append(f"{c.candidate_id}: label presence contradicts label_known")
        lab = None if o.label is None else np.array([o.label])
        rel = np.array([o.released_output])
        s = float(signal(cfg["threshold_signal"], rel, lab)[0])
        key = int(class_keys(rel, lab)[0])
        if s != c.threshold.score:
            errs.append(f"{c.candidate_id}: threshold score ≠ {cfg['threshold_signal']} of the released output")
        cond = cfg.get("threshold_mode") == "class_conditional"
        for a in ATTACKS:
            d = getattr(c, a)
            if d.class_key != (key if cond else None):
                errs.append(f"{c.candidate_id}: {a} class key")
            bal, low = points[a]["balanced"], points[a]["low_fpr"]
            if d.member_balanced != decide(d.score, key, bal["thresholds"]):
                errs.append(f"{c.candidate_id}: {a} balanced decision does not follow from score and threshold")
            exp_low = None if low["status"] == "undefined" else decide(d.score, key, low["thresholds"])
            if d.member_low_fpr != exp_low:
                errs.append(f"{c.candidate_id}: {a} low-FPR decision does not follow from score and threshold")
        if rec.evaluation == "adaptive_membership_inference" and set((c.undefended_correct or {})) != set(ATTACKS):
            errs.append(f"{c.candidate_id}: missing paired undefended outcome")
    for r in rec.utility:
        if r.released != float(apply_policy(policy, [r.p_full])[0]):
            errs.append(f"utility row {r.split}/{r.index}: released output inconsistent with the policy")
    return errs


def verify_membership_record(rec) -> dict[str, Any]:
    repro = rec.reproducibility
    checks = [
        _check("config_sha256", repro.get("config_sha256") == sha256_json(rec.request), "hash of the stored request"),
        _check("record_sha256", repro.get("record_sha256") == record_sha256(rec), "hash of the whole record"),
    ]
    if rec.status != "completed":
        checks.append(_check("results_sha256", None, "failed experiments have no results"))
        return _report(rec, checks)
    checks.append(_check("results_sha256", repro.get("results_sha256") == results_fingerprint(
        rec.candidates, rec.utility, rec.metrics, rec.calibration, rec.shadow),
        "hash of per-candidate results + utility rows + metrics + calibration + shadow provenance"))
    if rec.evaluation != "clean":
        cerrs = _calibration_errors(rec)
        checks.append(_check("calibration_rederived", not cerrs,
                             "both attacks' operating points re-derived from the stored calibration data (attacker "
                             "reference models only)" + (f"; {cerrs[:4]}" if cerrs else "")))
    errs = candidate_errors(rec)
    checks.append(_check("candidates_revalidated", not errs,
                         "scores, decisions, query order, policy outputs and ground-truth consistency re-checked"
                         + (f"; {errs[:5]}" if errs else "")))
    budget = (rec.attack_config or {}).get("query_budget")
    derived = derive_membership_metrics(rec.evaluation, rec.candidates, rec.utility,
                                        rec.candidate_selection.get("operational_prevalences", []), rec.calibration,
                                        budget, bool(rec.dataset.get("disjointness", {}).get("disjoint")))
    diff = sorted(k for k in set(derived) | set(rec.metrics) if derived.get(k) != rec.metrics.get(k))
    checks.append(_check("metrics_derivation", not diff,
                         "metrics re-derived from per-candidate evidence" + (f"; mismatch: {diff[:8]}" if diff else "")))
    return _report(rec, checks)


def _report(rec, checks):
    return {"experiment_id": rec.experiment_id, "schema_version": rec.schema_version,
            "ok": all(c["status"] != "fail" for c in checks), "guarantee": INTEGRITY_GUARANTEE, "checks": checks}
