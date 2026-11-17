"""ML04 membership-inference metrics, derived ONLY from per-candidate and per-record evidence.

:func:`derive_membership_metrics` builds a record's metrics; integrity verification re-derives them. Every ratio
carries numerator, denominator, population and definition; an empty denominator gives ``value: None`` (undefined).

Populations are named explicitly:

* EVALUATION — the final candidates (members from the victim's training set, non-members from the disjoint
  evaluation pool). Every attack metric below is computed on it, against the evaluator-only designated group.
* CALIBRATION — the attacker's reference models (``<attack>_calibration``). Reported separately; thresholds come
  from here, never from the evaluation population.
* UTILITY — the victim's full training split and its held-out test split (generalisation gap, legitimate utility).

Statistics (implemented here, no SciPy dependency):

* ROC AUC = Mann–Whitney U / (P·N), ties count ½. One-sided test of AUC > ½ with the normal approximation, tie
  correction and continuity correction. Leakage is CLAIMED for an attack only if its p < α/2 (Bonferroni over the
  two attacks, α = 0.05) and AUC > ½.
* Operating-point test: one-sided Fisher exact test that members are called members more often than non-members.
* Rates at the low-FPR operating point carry 95% Wilson intervals; an evaluation FPR whose Wilson LOWER bound exceeds
  the target is flagged (the calibrated threshold did not transfer).

Labels (see :mod:`.schemas` for the schema versions):

* ``contract=None`` (schema v1) — metrics use the DESIGNATED group, exactly as v1 records were written;
* ``contract="membership"`` / ``"placebo"`` (schema v2) — ordinary membership metrics use ACTUAL membership
  (``truth.in_training``). With no actual member (placebo) every positive-class metric (AUC, TPR, advantage, precision
  projections, the leakage test) is UNDEFINED (``None``) and the false-positive metrics are measured on real
  non-members. The placebo contract adds ``*_placebo_*`` metrics: population discrimination between placebo group A
  (victim-training pool, NOT this victim's training set) and group B (evaluation pool). They never use the word
  member for group A and never produce a membership-leakage claim.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..metrics import ratio
from ..stats import mcnemar_one_sided_p

ALPHA = 0.05
ATTACKS = ("threshold", "shadow")
EVAL_POP = ("final membership-evaluation candidates (members: victim training records; non-members: disjoint "
            "evaluation pool); designated group is evaluator-only")
UTIL_POP = {"train": "every record of the victim's training population", "test": "every record of the held-out "
            "victim test population (never attacked)"}

_PER_ATTACK = ("roc_auc", "leakage_p_value", "membership_accuracy", "balanced_accuracy", "membership_precision",
               "membership_recall", "false_positive_rate", "membership_advantage", "accuracy_over_no_signal",
               "tpr_at_low_fpr", "fpr_at_low_fpr", "operating_point_p_value", "undefended_membership_accuracy",
               "correctness_reduction", "defense_reduction_p_value")
# Flat metrics (``record.metrics[k]["value"]``) that regression invariants may reference.
INVARIANT_METRICS = tuple(f"{a}_{m}" for a in ATTACKS for m in _PER_ATTACK) + (
    "no_signal_accuracy", "leakage_claimed", "train_accuracy", "test_accuracy", "generalization_gap",
    "test_roc_auc", "consumer_decision_accuracy", "query_budget_utilisation", "population_disjointness",
    "evidence_consistency", "attacker_independence", "actual_member_fraction", "placebo_discrimination_claimed") + tuple(
    f"{a}_placebo_{m}" for a in ATTACKS for m in ("auc", "p_value", "advantage"))
PLACEBO_POP = ("PLACEBO: the same candidates split by source pool — group A drawn from the victim-training pool and group "
               "B from the evaluation pool; NO candidate is a training member of this victim")


def _g(o: Any, k: str):
    return o.get(k) if isinstance(o, dict) else getattr(o, k)


def _ranks(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Average ranks (1-based) and Σ(t³ − t) over tie groups."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    ties = 0.0
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    return r, ties


def auc_mann_whitney(pos: Sequence[float], neg: Sequence[float]) -> dict[str, Any]:
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    P, N = len(pos), len(neg)
    if not P or not N:
        return {"auc": None, "u": None, "z": None, "p_value": None}
    r, ties = _ranks(np.r_[pos, neg])
    u = float(r[:P].sum() - P * (P + 1) / 2.0)
    n = P + N
    var = P * N / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return {"auc": u / (P * N), "u": u, "z": 0.0, "p_value": 1.0}
    z = (u - P * N / 2.0 - 0.5) / math.sqrt(var)
    return {"auc": u / (P * N), "u": u, "z": z, "p_value": 0.5 * math.erfc(z / math.sqrt(2.0))}


def fisher_one_sided(tp: int, fn: int, fp: int, tn: int) -> float:
    """P(TP ≥ observed) under the hypergeometric null (calls independent of group)."""
    P, N, K = tp + fn, fp + tn, tp + fp
    total = math.comb(P + N, K)
    if total == 0:
        return 1.0
    hi = min(P, K)
    return float(sum(math.comb(P, x) * math.comb(N, K - x) for x in range(tp, hi + 1)) / total)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def roc_points(pos: np.ndarray, neg: np.ndarray) -> list[list[float]] | None:
    """(FPR, TPR) for every distinct threshold, from (0, 0) to (1, 1); None when either class is empty."""
    if not len(pos) or not len(neg):
        return None
    s = np.r_[pos, neg]
    m = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    order = np.argsort(-s, kind="stable")
    s, m = s[order], m[order]
    pts, tp, fp, i = [[0.0, 0.0]], 0, 0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            tp += int(m[j])
            fp += int(not m[j])
            j += 1
        pts.append([fp / len(neg), tp / len(pos)])
        i = j
    return pts


def _utility(utility: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    acc = {}
    for split in ("train", "test"):
        rows = [r for r in utility if _g(r, "split") == split]
        correct = sum(int(_g(r, "p_full") >= 0.5) == _g(r, "y_true") for r in rows)
        out[f"{split}_accuracy"] = ratio(correct, len(rows), "records whose service decision 1[P ≥ 0.5] equals the "
                                         "true label / records", UTIL_POP[split])
        acc[split] = out[f"{split}_accuracy"]["value"]
    out["generalization_gap"] = {
        "value": None if None in acc.values() else acc["train"] - acc["test"], "population": "victim train vs test",
        "definition": "training accuracy − held-out test accuracy (overfitting; the signal membership inference uses)"}
    test = [r for r in utility if _g(r, "split") == "test"]
    y = np.array([_g(r, "y_true") for r in test], dtype=np.int64)
    rel = np.array([_g(r, "released") for r in test], dtype=np.float64)
    if len(test):
        auc = auc_mann_whitney(rel[y == 1], rel[y == 0])["auc"]
        out["test_roc_auc"] = {"value": auc, "n": len(test), "population": UTIL_POP["test"],
                               "definition": "ROC AUC of the RELEASED output for the true label (utility of what a "
                                             "consumer receives; label-only can only rank two groups)"}
        out["released_brier_score"] = {"value": float(np.mean((rel - y) ** 2)), "n": len(test),
                                       "population": UTIL_POP["test"],
                                       "definition": "mean (released − y)² on the test split (lower is better)"}
        out["consumer_decision_accuracy"] = ratio(int(((rel >= 0.5).astype(int) == y).sum()), len(test),
                                                  "test records whose decision derived from the RELEASED output, "
                                                  "1[released ≥ 0.5], equals the true label / records",
                                                  UTIL_POP["test"])
    return out


def _attack_metrics(a: str, cands: Sequence[Any], member: list[bool], prevalences: Sequence[float],
                    calibration: dict[str, Any] | None, evaluation: str) -> dict[str, Any]:
    dec = [_g(c, a) for c in cands]
    score = np.array([_g(d, "score") for d in dec], dtype=np.float64)
    is_m = np.array(member, dtype=bool)
    P, N, n = int(is_m.sum()), int((~is_m).sum()), len(member)
    m: dict[str, Any] = {}
    mw = auc_mann_whitney(score[is_m], score[~is_m])
    m[f"{a}_roc_auc"] = {"value": mw["auc"], "n_members": P, "n_non_members": N, "population": EVAL_POP,
                         "definition": "P(score of a random member > score of a random non-member), ties ½ "
                                       "(threshold-free; 0.5 = no membership signal)"}
    claimed = bool(mw["p_value"] is not None and mw["p_value"] < ALPHA / 2 and (mw["auc"] or 0) > 0.5)
    m[f"{a}_leakage_p_value"] = {
        "value": mw["p_value"], "z": mw["z"], "alpha": ALPHA, "alpha_per_attack": ALPHA / 2,
        "leakage_claimed": claimed, "population": EVAL_POP,
        "definition": "one-sided Mann–Whitney test of AUC > 0.5 (normal approximation, tie and continuity "
                      "corrections); leakage is claimed only if p < alpha/2 (Bonferroni over the two attacks) and "
                      "AUC > 0.5"}
    pred = np.array([bool(_g(d, "member_balanced")) for d in dec])
    tp, fp = int((pred & is_m).sum()), int((pred & ~is_m).sum())
    fn, tn = P - tp, N - fp
    tpr = tp / P if P else None
    fpr = fp / N if N else None
    m[f"{a}_confusion"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "operating_point": "balanced",
                           "population": EVAL_POP}
    m[f"{a}_membership_accuracy"] = ratio(tp + tn, n, "candidates whose membership decision (balanced operating "
                                          "point) equals the designated group / candidates", EVAL_POP)
    m[f"{a}_balanced_accuracy"] = {"value": None if tpr is None or fpr is None else 0.5 * (tpr + 1 - fpr),
                                   "population": EVAL_POP, "definition": "(TPR + TNR) / 2 at the balanced operating "
                                                                         "point (0.5 = no signal at any prevalence)"}
    m[f"{a}_membership_precision"] = ratio(tp, tp + fp, "members called members / candidates called members "
                                           "(at the EVALUATION prevalence; see operational_precision)", EVAL_POP)
    m[f"{a}_membership_recall"] = ratio(tp, P, "members called members / members (TPR)", EVAL_POP)
    m[f"{a}_false_positive_rate"] = ratio(fp, N, "non-members called members / non-members (FPR)", EVAL_POP)
    m[f"{a}_membership_advantage"] = {
        "value": None if tpr is None or fpr is None else tpr - fpr, "population": EVAL_POP,
        "definition": "TPR − FPR at the balanced operating point (Yeom et al. membership advantage; 0 = the no-signal "
                      "baseline, which calls members and non-members members equally often)"}
    acc = (tp + tn) / n if n else None
    m[f"{a}_accuracy_over_no_signal"] = {"value": None if acc is None else acc - max(P, N) / n,
                                         "population": EVAL_POP,
                                         "definition": "membership accuracy − accuracy of always guessing the "
                                                       "majority group (signed)"}
    m[f"{a}_operating_point_p_value"] = {"value": fisher_one_sided(tp, fn, fp, tn) if P and N else None,
                                         "population": EVAL_POP,
                                         "definition": "one-sided Fisher exact test that members are called members "
                                                       "more often than non-members at the balanced operating point"}
    low = [_g(d, "member_low_fpr") for d in dec]
    cal_low = next((p for p in (calibration or {}).get("operating_points", []) if p["name"] == "low_fpr"), None)
    target = (calibration or {}).get("target_fpr")
    if all(v is None for v in low):
        reason = cal_low["note"] if cal_low else "no calibration"
        m[f"{a}_tpr_at_low_fpr"] = {"value": None, "numerator": None, "denominator": P, "target_fpr": target,
                                    "status": "undefined", "reason": reason, "population": EVAL_POP,
                                    "definition": "TPR on the evaluation members at the threshold calibrated for "
                                                  "FPR ≤ target on the attacker's reference models"}
        m[f"{a}_fpr_at_low_fpr"] = {"value": None, "numerator": None, "denominator": N, "status": "undefined",
                                    "population": EVAL_POP, "definition": "evaluation FPR at that threshold"}
    else:
        lp = np.array([bool(v) for v in low])
        ltp, lfp = int((lp & is_m).sum()), int((lp & ~is_m).sum())
        ci_f = wilson(lfp, N)
        m[f"{a}_tpr_at_low_fpr"] = {
            **ratio(ltp, P, "members called members at the low-FPR operating point / members", EVAL_POP),
            "target_fpr": target, "status": cal_low["status"] if cal_low else "undefined",
            "wilson_95": wilson(ltp, P),
            "note": "operating threshold calibrated on the attacker's reference models only; the evaluation FPR at "
                    "that threshold is reported next to it and is NOT guaranteed to equal the target"}
        m[f"{a}_fpr_at_low_fpr"] = {
            **ratio(lfp, N, "non-members called members at the low-FPR operating point / non-members", EVAL_POP),
            "wilson_95": ci_f, "target_fpr": target,
            "exceeds_target": bool(ci_f is not None and target is not None and ci_f[0] > target)}
    proj = []
    for pi in prevalences:
        den = None if tpr is None or fpr is None else tpr * pi + fpr * (1 - pi)
        ppv = None if not den else tpr * pi / den
        proj.append({"prevalence": pi, "precision": ppv, "false_discovery_rate": None if ppv is None else 1 - ppv})
    m[f"{a}_operational_precision"] = {
        "values": proj, "population": EVAL_POP,
        "definition": "PROJECTED precision TPR·π / (TPR·π + FPR·(1 − π)) and FDR at prevalence π, from the balanced "
                      "operating point's measured TPR and FPR (assumes the rates do not change with prevalence)"}
    m[f"{a}_roc_curve"] = {"points": roc_points(score[is_m], score[~is_m]), "population": EVAL_POP,
                           "definition": "(FPR, TPR) at every distinct score threshold of the EVALUATION population — "
                                         "descriptive only, never used to choose a threshold"}
    if calibration:
        pts = {p["name"]: {"calibration_tpr": p["calibration_tpr"], "calibration_fpr": p["calibration_fpr"],
                           "status": p["status"]} for p in calibration["operating_points"]}
        m[f"{a}_calibration"] = {"calibration_auc": calibration.get("calibration_auc"), **pts,
                                 "population": "the attacker's reference models on its calibration population "
                                               "(NOT the evaluation population)",
                                 "definition": "calibration-time rates; compare with the evaluation rates above"}
    if evaluation == "adaptive_membership_inference":
        und = [bool((_g(c, "undefended_correct") or {}).get(a)) for c in cands]
        cur = [bool(p) == mm for p, mm in zip(pred, member)]
        m[f"{a}_undefended_membership_accuracy"] = ratio(sum(und), n, "the SAME candidates decided correctly by the "
                                                         "parent attack against the undefended victim / candidates",
                                                         EVAL_POP)
        m[f"{a}_correctness_reduction"] = ratio(sum(und) - sum(cur), n, "(correct against the undefended victim − "
                                                "correct against the defended victim) / candidates — paired, signed",
                                                EVAL_POP)
        b = sum(u and not c for u, c in zip(und, cur))
        c2 = sum(c and not u for u, c in zip(und, cur))
        m[f"{a}_defense_reduction_p_value"] = {
            "value": mcnemar_one_sided_p(b, c2) if n else None, "only_undefended_correct": b,
            "only_defended_correct": c2, "population": EVAL_POP,
            "definition": "exact one-sided McNemar test that the defense reduces correct membership decisions on the "
                          "same candidates (balanced operating points, each recalibrated)"}
    return m


def _placebo_metrics(a: str, cands: Sequence[Any]) -> dict[str, Any]:
    """Population discrimination between placebo groups A and B (no candidate is a member)."""
    grp = np.array([_g(_g(c, "truth"), "placebo_group") == "A" for c in cands], dtype=bool)
    dec = [_g(c, a) for c in cands]
    score = np.array([_g(d, "score") for d in dec], dtype=np.float64)
    called = np.array([bool(_g(d, "member_balanced")) for d in dec])
    nA, nB = int(grp.sum()), int((~grp).sum())
    mw = auc_mann_whitney(score[grp], score[~grp])
    cA, cB = int((called & grp).sum()), int((called & ~grp).sum())
    m: dict[str, Any] = {}
    m[f"{a}_placebo_auc"] = {"value": mw["auc"], "n_group_a": nA, "n_group_b": nB, "population": PLACEBO_POP,
                             "definition": "P(score of a random group-A candidate > score of a random group-B "
                                           "candidate), ties ½ — population discrimination, NOT membership"}
    claimed = bool(mw["p_value"] is not None and mw["p_value"] < ALPHA / 2 and (mw["auc"] or 0) > 0.5)
    m[f"{a}_placebo_p_value"] = {"value": mw["p_value"], "z": mw["z"], "alpha_per_attack": ALPHA / 2,
                                 "discrimination_claimed": claimed, "population": PLACEBO_POP,
                                 "definition": "one-sided Mann–Whitney test of placebo AUC > 0.5 (same test and "
                                               "Bonferroni split as the membership test); a claim here would indicate "
                                               "a population difference or pipeline leak, never membership leakage"}
    m[f"{a}_placebo_advantage"] = {
        "value": (cA / nA - cB / nB) if nA and nB else None, "population": PLACEBO_POP,
        "definition": "rate at which group A is called 'member' − rate for group B, at the balanced operating point "
                      "(0 expected: neither group contains members)"}
    m[f"{a}_placebo_calls"] = {"group_a_called": cA, "group_a_not_called": nA - cA, "group_b_called": cB,
                               "group_b_not_called": nB - cB, "operating_point": "balanced", "population": PLACEBO_POP}
    m[f"{a}_placebo_roc_curve"] = {"points": roc_points(score[grp], score[~grp]), "population": PLACEBO_POP,
                                   "definition": "(rate group B called, rate group A called) at every score threshold"}
    return m


def derive_membership_metrics(evaluation: str, candidates: Sequence[Any], utility: Sequence[Any],
                              prevalences: Sequence[float], calibration: dict[str, Any] | None,
                              query_budget: int | None, disjoint: bool, contract: str | None = None) -> dict[str, Any]:
    """``contract=None`` reproduces schema-v1 metrics exactly; ``"membership"`` / ``"placebo"`` are schema v2."""
    if contract is None:
        member = [_g(_g(c, "truth"), "group") == "member" for c in candidates]
    else:
        member = [bool(_g(_g(c, "truth"), "in_training")) for c in candidates]
    n, P = len(member), sum(member)
    N = n - P
    m: dict[str, Any] = {"n_candidates": n, "n_members": P, "n_non_members": N,
                         "evaluation_prevalence": (P / n) if n else None}
    if contract is not None:
        m["evaluation_contract"] = contract
        m["actual_member_fraction"] = ratio(P, n, "candidates that are ACTUAL training members of this victim / "
                                            "candidates (0 under the placebo contract)", EVAL_POP)
    m["no_signal_accuracy"] = ratio(max(P, N), n, "accuracy of always guessing the larger group (no model output) / "
                                    "candidates", EVAL_POP)
    m["population_disjointness"] = {"value": 1.0 if disjoint else 0.0,
                                    "definition": "1 iff no record is duplicated or shared between any two "
                                                  "populations (victim train/test, shadow, calibration, evaluation "
                                                  "non-members, decoy), checked row by row"}
    m.update(_utility(utility))
    if evaluation == "clean":
        return m
    attacks = (calibration or {}).get("attacks", {})
    claimed = []
    for a in ATTACKS:
        m.update(_attack_metrics(a, candidates, member, prevalences, attacks.get(a), evaluation))
        if m[f"{a}_leakage_p_value"]["leakage_claimed"]:
            claimed.append(a)
    m["leakage_claimed"] = {"value": 1.0 if claimed else 0.0, "attacks": claimed, "alpha": ALPHA,
                            "definition": "1 if ANY attack's one-sided Mann–Whitney p < alpha/2 with AUC > 0.5 "
                                          "(Bonferroni over the two attacks), else 0"}
    if contract is not None and (P == 0 or N == 0):
        m["leakage_claimed"].update(value=None, status="undefined",
                                    reason="no membership test is possible: the candidates contain no actual "
                                           + ("training member" if P == 0 else "non-member"))
    if contract == "placebo":
        pc = []
        for a in ATTACKS:
            m.update(_placebo_metrics(a, candidates))
            if m[f"{a}_placebo_p_value"]["discrimination_claimed"]:
                pc.append(a)
        m["placebo_discrimination_claimed"] = {
            "value": 1.0 if pc else 0.0, "attacks": pc, "alpha": ALPHA, "population": PLACEBO_POP,
            "definition": "1 if ANY attack separates placebo groups A and B (one-sided Mann–Whitney p < alpha/2, AUC "
                          "> 0.5). Expected 0: a claim indicates a population difference or pipeline leak; it is "
                          "never a membership-leakage claim"}
    q = sum(_g(c, "observation") is not None for c in candidates)
    m["queries"] = {"total": q, "budget": query_budget, "per_candidate": 1,
                    "victim_queries_for_shadow_or_reference_models": 0,
                    "definition": "victim queries: one per candidate record (shadow and reference models are the "
                                  "attacker's own and never query the victim)"}
    m["query_budget_utilisation"] = ratio(q, query_budget or 0, "victim queries / query budget", EVAL_POP)
    return m
