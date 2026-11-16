"""Shadow-model membership inference (after Shokri et al., 2017), with traditional ML only.

1. The attacker trains ``n_shadow_models`` shadow models with the victim's training recipe, each on a random
   ``n_train`` records of its OWN shadow population, holding out ``n_train`` other records. The member / non-member
   label of every shadow record is known to the attacker by construction — it is never the victim's membership.
2. Each shadow model's released outputs (through the service's output policy) become attack-training examples:
   features of the released output (and the true label when known) → shadow membership label.
3. A gradient-boosting meta-classifier learns membership from those examples.
4. Its operating thresholds are calibrated on the attacker's REFERENCE models (:func:`.attacks.reference_data`,
   a disjoint auxiliary population), exactly like the threshold attack, so both attacks are compared at operating
   points chosen the same way and never on the evaluation population.
5. The meta-classifier scores the victim's released outputs for the evaluation candidates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from ..attacks import _check_deadline
from ..hashing import sha256_arrays, sha256_json
from .attacks import (EPS, AttackerKnowledge, ReferenceData, calibrate, class_keys, decisions, signal,
                      train_split_models)
from .defenses import apply_policy
from .models import derive_state

META_PARAMS: dict[str, Any] = {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0}
FEATURES_LABEL_KNOWN = ("released_p1", "true_label", "true_label_confidence", "log_true_label_confidence",
                        "max_confidence")
FEATURES_LABEL_UNKNOWN = ("released_p1", "max_confidence")


def meta_features(released: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
    released = np.asarray(released, dtype=np.float64)
    mx = np.maximum(released, 1.0 - released)
    if labels is None:
        return np.c_[released, mx]
    conf = signal("true_label_confidence", released, labels)
    return np.c_[released, np.asarray(labels, dtype=np.float64), conf, np.log(np.maximum(conf, EPS)), mx]


def _meta_sha256(meta: GradientBoostingClassifier) -> str:
    trees = [a for t in meta.estimators_.ravel() for a in (t.tree_.children_left, t.tree_.children_right,
                                                            t.tree_.feature, t.tree_.threshold, t.tree_.value)]
    prior = getattr(getattr(meta, "init_", None), "class_prior_", None)
    return sha256_json({"params": META_PARAMS, "trees": sha256_arrays(*trees),
                        "prior": None if prior is None else sha256_arrays(np.asarray(prior))})


def shadow_attack(k: AttackerKnowledge, released: np.ndarray, ref: ReferenceData, cfg: dict[str, Any],
                  deadline: float | None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Returns (shadow provenance, calibration, per-candidate decisions)."""
    label_known = cfg["label_known"]
    n_in = min(k.n_train, len(k.shadow.y) // 2)
    F, L, models = [], [], []
    for m, a, b in train_split_models(k.shadow, k.recipe, cfg["n_shadow_models"], n_in, k.seed, 4301, "shadow model",
                                      deadline):
        for idx, member in ((a, True), (b, False)):
            rel = apply_policy(k.policy, m.proba(k.shadow.X[idx]))
            F.append(meta_features(rel, k.shadow.y[idx] if label_known else None))
            L.append(np.full(len(idx), member))
        models.append({**m.describe(), "n_in": int(len(a)), "n_out": int(len(b)),
                       "in_rows_sha256": m.training_rows_sha256,
                       "out_rows_sha256": sha256_arrays(k.shadow.X[b], k.shadow.y[b])})
    _check_deadline(deadline)
    X_meta, y_meta = np.vstack(F), np.concatenate(L)
    meta = GradientBoostingClassifier(**META_PARAMS, random_state=derive_state(k.seed, 4303)).fit(X_meta, y_meta)
    _check_deadline(deadline)
    ref_labels = ref.labels if label_known else None
    s_ref = meta.predict_proba(meta_features(ref.released, ref_labels))[:, 1]
    k_ref = class_keys(ref.released, ref_labels)
    cal = calibrate(s_ref, ref.is_member, k_ref, cfg["threshold_mode"], cfg["min_class_calibration"], cfg["target_fpr"])
    s = meta.predict_proba(meta_features(released, k.y))[:, 1]
    provenance = {
        "population": "shadow (attacker auxiliary data, disjoint from every victim and evaluation population)",
        "population_sha256": k.shadow.sha256(), "n_shadow_models": len(models), "records_in_per_model": n_in,
        "membership_labels": "known to the attacker by construction of each shadow split; never the victim's",
        "models": models,
        "meta_classifier": {"estimator": "sklearn.ensemble.GradientBoostingClassifier",
                            "hyperparameters": {**META_PARAMS, "random_state": derive_state(k.seed, 4303)},
                            "features": list(FEATURES_LABEL_KNOWN if label_known else FEATURES_LABEL_UNKNOWN),
                            "n_training_examples": int(len(y_meta)), "n_member_examples": int(y_meta.sum()),
                            "training_examples_sha256": sha256_arrays(X_meta, y_meta.astype(np.int64)),
                            "artifact_sha256": _meta_sha256(meta)},
    }
    calibration = {"attack": "shadow", "signal": "meta-classifier P(member)", "mode": cfg["threshold_mode"],
                   "target_fpr": cfg["target_fpr"], "scores": s_ref.tolist(), "is_member": ref.is_member.tolist(),
                   "class_keys": k_ref.tolist(), **cal}
    return provenance, calibration, decisions(s, class_keys(released, k.y), cal["operating_points"],
                                              cfg["threshold_mode"])
