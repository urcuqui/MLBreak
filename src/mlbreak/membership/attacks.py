"""ML04 attacker code: signals, threshold calibration and the threshold-based membership attack.

The attack functions receive an :class:`AttackerKnowledge` and a counting query function — never a membership flag
of the evaluation candidates, the victim model, its parameters or its training records. Everything the attacker
learns about membership comes from its OWN auxiliary populations: it trains reference models (and, in
:mod:`.shadow`, shadow models) with the victim's training recipe on records whose membership it controls.

Signals (oriented so that HIGHER = more member-like):

* ``true_label_confidence`` — released probability of the candidate's true label (label known);
* ``negative_log_likelihood`` — reported as ``log p_y`` = −NLL (label known). For one released probability this is
  a MONOTONE transform of the true-label confidence: same ranking, same ROC, same decisions — only the threshold
  scale differs. Both are offered because both appear in the literature;
* ``max_confidence`` — ``max(p, 1 − p)`` (label unknown). Under label-only output it is constant, i.e. no signal.

Calibration (:func:`calibrate`) is a pure function of the reference-model scores and the attacker-known membership
of those reference records. Integrity verification re-runs it on the stored calibration data. Two operating points:

* ``balanced`` — the threshold maximising Youden's J = TPR − FPR on the calibration data (ties → the HIGHEST such
  threshold); used for accuracy, precision, recall, advantage and the confusion matrix;
* ``low_fpr`` — the LOWEST threshold whose calibration FPR ≤ ``target_fpr``. "approximate" when the calibration
  non-members allow fewer than 10 false positives at that target (coarse FPR resolution); "undefined" when no such
  threshold calls any calibration member a member (the TPR at that FPR cannot be estimated).

Thresholds are never chosen on the final evaluation population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..attacks import _check_deadline
from .data import Population
from .defenses import apply_policy
from .metrics import auc_mann_whitney
from .models import FittedModel, Recipe, derive_state, fit

EPS = 1e-12
MIN_FALSE_POSITIVES = 10  # calibration non-member FPs allowed at target_fpr below which low_fpr is "approximate"


def signal(kind: str, released: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
    released = np.asarray(released, dtype=np.float64)
    if kind == "max_confidence":
        return np.maximum(released, 1.0 - released)
    if labels is None:
        raise ValueError(f"signal {kind!r} needs labels")
    conf = np.where(np.asarray(labels) == 1, released, 1.0 - released)
    if kind == "true_label_confidence":
        return conf
    if kind == "negative_log_likelihood":
        return np.log(np.maximum(conf, EPS))
    raise ValueError(f"unknown signal {kind!r}")


def class_keys(released: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
    """Class used by class-conditional thresholds: the true label when known, else the released decision."""
    if labels is not None:
        return np.asarray(labels, dtype=np.int64)
    return (np.asarray(released) >= 0.5).astype(np.int64)


def _sweep(scores: np.ndarray, is_member: np.ndarray):
    """(threshold, TP, FP) for every distinct threshold, from +inf (None: nobody is a member) downwards."""
    order = np.argsort(-scores, kind="stable")
    s, m = scores[order], is_member[order]
    out = [(None, 0, 0)]
    tp = fp = 0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            tp += int(m[j])
            fp += int(not m[j])
            j += 1
        out.append((float(s[i]), tp, fp))
        i = j
    return out


def youden_threshold(scores: np.ndarray, is_member: np.ndarray) -> float | None:
    P, N = int(is_member.sum()), int((~is_member).sum())
    best, best_t = None, None
    for t, tp, fp in _sweep(scores, is_member):
        j = tp * N - fp * P  # J · P · N, exact in integers
        if best is None or j > best:
            best, best_t = j, t
    return best_t


def low_fpr_threshold(scores: np.ndarray, is_member: np.ndarray, target: float) -> float | None:
    N = int((~is_member).sum())
    limit = math.floor(target * N + 1e-9)
    chosen = None
    for t, _tp, fp in _sweep(scores, is_member):
        if fp <= limit:
            chosen = t
        else:
            break
    return chosen


def decide(score: float, key: int, thresholds: dict[str, float | None]) -> bool:
    t = thresholds["all"] if "all" in thresholds else thresholds[str(int(key))]
    return t is not None and score >= t


def _rates(scores, keys, is_member, thresholds) -> tuple[float, float, int, int]:
    pred = np.array([decide(float(s), int(k), thresholds) for s, k in zip(scores, keys)], dtype=bool)
    P, N = int(is_member.sum()), int((~is_member).sum())
    tp, fp = int((pred & is_member).sum()), int((pred & ~is_member).sum())
    return (tp / P if P else 0.0), (fp / N if N else 0.0), tp, fp


def calibrate(scores, is_member, keys, mode: str, min_class: int, target_fpr: float) -> dict[str, Any]:
    """Operating points from attacker-side calibration data only (see module docstring)."""
    scores = np.asarray(scores, dtype=np.float64)
    is_member = np.asarray(is_member, dtype=bool)
    keys = np.asarray(keys, dtype=np.int64)
    groups: dict[str, np.ndarray] = {"all": np.ones(len(scores), dtype=bool)}
    class_modes: dict[str, str] = {}
    if mode == "class_conditional":
        groups = {}
        for c in (0, 1):
            idx = keys == c
            nm, nn = int((is_member & idx).sum()), int((~is_member & idx).sum())
            if nm >= min_class and nn >= min_class:
                groups[str(c)] = idx
                class_modes[str(c)] = f"own threshold ({nm} members / {nn} non-members of class {c} in calibration)"
            else:
                class_modes[str(c)] = (f"global threshold: only {nm} members / {nn} non-members of class {c} in "
                                       f"calibration (< {min_class})")
    points = []
    for name in ("balanced", "low_fpr"):
        g_thr = (youden_threshold(scores, is_member) if name == "balanced"
                 else low_fpr_threshold(scores, is_member, target_fpr))
        if mode == "class_conditional":
            thresholds = {}
            for c in ("0", "1"):
                if c in groups:
                    s, m = scores[groups[c]], is_member[groups[c]]
                    thresholds[c] = youden_threshold(s, m) if name == "balanced" else low_fpr_threshold(s, m, target_fpr)
                else:
                    thresholds[c] = g_thr
        else:
            thresholds = {"all": g_thr}
        tpr, fpr, tp, _fp = _rates(scores, keys, is_member, thresholds)
        n_neg = [int((~is_member & idx).sum()) for idx in groups.values()] or [int((~is_member).sum())]
        if name == "balanced":
            if tp == 0 or tpr <= fpr:
                status, note = "undefined", ("no threshold separates calibration members from non-members "
                                             "(Youden J ≤ 0): every candidate is called a non-member")
            else:
                status, note = "calibrated", "maximises TPR − FPR on the attacker's calibration data"
        else:
            if tp == 0:
                status, note = "undefined", (f"no threshold reaches calibration FPR ≤ {target_fpr:g} while calling "
                                             "any calibration member a member (scores too coarse or no signal)")
            elif min(math.floor(target_fpr * n + 1e-9) for n in n_neg) < MIN_FALSE_POSITIVES:
                status, note = "approximate", (f"calibration allows fewer than {MIN_FALSE_POSITIVES} false positives at "
                                               f"FPR {target_fpr:g} ({min(n_neg)} non-members): coarse FPR resolution")
            else:
                status, note = "calibrated", f"lowest threshold with calibration FPR ≤ {target_fpr:g}"
        points.append({"name": name, "thresholds": thresholds, "calibration_tpr": tpr, "calibration_fpr": fpr,
                       "status": status, "note": note})
    auc = auc_mann_whitney(scores[is_member], scores[~is_member])["auc"] if is_member.any() and (~is_member).any() \
        else None
    return {"class_modes": class_modes, "operating_points": points, "calibration_auc": auc}


# --------------------------------------------------------------------------
# Attacker knowledge and the reference models
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AttackerKnowledge:
    """Everything the adversary holds. No membership flag of an evaluation candidate is reachable from here."""

    candidate_ids: tuple[str, ...]
    X: np.ndarray  # candidate features
    y: np.ndarray | None  # candidate labels (only when the configuration grants them)
    recipe: Recipe  # the victim's training procedure (incl. a training-time defense)
    policy: dict[str, Any]  # the service's output policy
    n_train: int  # size of the victim's training set
    shadow: Population  # attacker auxiliary population (shadow models)
    calibration: Population  # attacker auxiliary population (reference models)
    seed: int


@dataclass
class ReferenceData:
    """Released outputs of the attacker's reference models on records whose membership the ATTACKER controls."""

    models: list[dict[str, Any]]
    released: np.ndarray
    labels: np.ndarray
    is_member: np.ndarray


def _split(n_pop: int, n_in: int, key: list[int]) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(key).permutation(n_pop)
    return np.sort(perm[:n_in]), np.sort(perm[n_in:2 * n_in])


def train_split_models(pop: Population, rec: Recipe, n_models: int, n_in: int, seed: int, purpose: int, role: str,
                       deadline: float | None) -> list[tuple[FittedModel, np.ndarray, np.ndarray]]:
    """``n_models`` models, each trained on a random ``n_in`` records of ``pop`` with ``n_in`` other records held out."""
    out = []
    for k in range(n_models):
        _check_deadline(deadline)
        a, b = _split(len(pop.y), n_in, [seed, purpose, k])
        m = fit(rec, pop.X[a], pop.y[a], derive_state(seed, purpose + 1, k), f"{role} {k}")
        out.append((m, a, b))
    return out


def reference_data(k: AttackerKnowledge, n_models: int, deadline: float | None) -> ReferenceData:
    n_in = min(k.n_train, len(k.calibration.y) // 2)
    rel, lab, mem, desc = [], [], [], []
    for m, a, b in train_split_models(k.calibration, k.recipe, n_models, n_in, k.seed, 4201, "attacker reference model",
                                      deadline):
        idx = np.r_[a, b]
        rel.append(apply_policy(k.policy, m.proba(k.calibration.X[idx])))
        lab.append(k.calibration.y[idx])
        mem.append(np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)])
        desc.append({**m.describe(), "n_in": int(len(a)), "n_out": int(len(b)),
                     "in_rows_sha256": m.training_rows_sha256,
                     "population": "calibration (attacker auxiliary data)"})
    return ReferenceData(desc, np.concatenate(rel), np.concatenate(lab), np.concatenate(mem))


def threshold_attack(k: AttackerKnowledge, released: np.ndarray, ref: ReferenceData, cfg: dict[str, Any]
                     ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calibrate on the reference models, then decide for every candidate. Returns (calibration, decisions)."""
    labels_ref = ref.labels if cfg["label_known"] else None
    s_ref = signal(cfg["threshold_signal"], ref.released, labels_ref)
    k_ref = class_keys(ref.released, labels_ref)
    cal = calibrate(s_ref, ref.is_member, k_ref, cfg["threshold_mode"], cfg["min_class_calibration"], cfg["target_fpr"])
    s = signal(cfg["threshold_signal"], released, k.y)
    keys = class_keys(released, k.y)
    calibration = {"attack": "threshold", "signal": cfg["threshold_signal"], "mode": cfg["threshold_mode"],
                   "target_fpr": cfg["target_fpr"], "scores": s_ref.tolist(), "is_member": ref.is_member.tolist(),
                   "class_keys": k_ref.tolist(), **cal}
    return calibration, decisions(s, keys, cal["operating_points"], cfg["threshold_mode"])


def decisions(scores: np.ndarray, keys: np.ndarray, points: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    bal = next(p for p in points if p["name"] == "balanced")
    low = next(p for p in points if p["name"] == "low_fpr")
    out = []
    for s, c in zip(scores, keys):
        out.append({"score": float(s), "class_key": int(c) if mode == "class_conditional" else None,
                    "member_balanced": decide(float(s), int(c), bal["thresholds"]),
                    "member_low_fpr": None if low["status"] == "undefined" else decide(float(s), int(c),
                                                                                         low["thresholds"])})
    return out


def query_candidates(k: AttackerKnowledge, query: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """The attacker's only victim interaction: one query per candidate record, in candidate order."""
    return np.asarray(query(k.X), dtype=np.float64)
