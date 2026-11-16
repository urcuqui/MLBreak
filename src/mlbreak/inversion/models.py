"""Victim and auxiliary models for the ML03 lab (trained in-process; never loaded from pickle/joblib).

* ``logreg`` — the victim: logistic regression on ``[public features, private attribute]`` → elevated payment risk.
  The private attribute is a genuine input feature (its learned weight is recorded in the model description).
* ``logreg-public-only`` — the NEGATIVE CONTROL victim: same service interface (it still receives the private
  attribute in every record) but the model is trained on the public features only, so its output carries no
  information about the attribute by construction.
* the auxiliary attacker model — logistic regression ``public features → private attribute`` fitted on the
  attacker's auxiliary population; it never sees a victim output.

Scikit-learn fits the parameters (lbfgs, fixed ``random_state``, deterministic); predictions are computed with
:func:`~.data.linear` so they are bit-identical for a single row and for a batch. The artifact hash covers the
fitted parameters and the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..hashing import sha256_arrays, sha256_json
from .data import PRIVATE_ATTRIBUTE, PUBLIC_FEATURES, linear, load_inversion_dataset

LOGREG_PARAMS: dict[str, Any] = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
VICTIM_MODELS = {
    "logreg": "Logistic regression on 4 public features + the private attribute (victim).",
    "logreg-public-only": "Logistic regression on the 4 public features only; the private attribute is accepted "
                          "by the service but ignored (negative control).",
}


@dataclass(frozen=True)
class LinearModel:
    model_id: str
    role: str
    features: tuple[str, ...]
    weights: np.ndarray
    intercept: float
    seed: int
    n_train: int
    training: str

    def proba(self, X: np.ndarray) -> np.ndarray:
        z = linear(X, self.weights, self.intercept)
        return 1.0 / (1.0 + np.exp(-z))

    def artifact_sha256(self) -> str:
        return sha256_json({"model_id": self.model_id, "features": list(self.features), "params": LOGREG_PARAMS,
                            "parameters": sha256_arrays(self.weights, np.array([self.intercept]))})

    def describe(self) -> dict[str, Any]:
        out = {"model_id": self.model_id, "role": self.role, "estimator": "sklearn.linear_model.LogisticRegression",
               "hyperparameters": {**LOGREG_PARAMS, "random_state": self.seed}, "features": list(self.features),
               "training": self.training, "n_train": self.n_train, "artifact_sha256": self.artifact_sha256(),
               "artifact_format": "in-memory weights + intercept (no pickle); hash over parameters and configuration"}
        if self.model_id == "logreg":
            # Evaluation-side evidence that the private attribute is genuinely used. Not given to the attacker.
            out["private_attribute_weight"] = float(self.weights[-1])
        return out


def _fit(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    clf = LogisticRegression(random_state=seed, **LOGREG_PARAMS).fit(X, y)
    return np.ascontiguousarray(clf.coef_[0], dtype=np.float64), float(clf.intercept_[0])


@lru_cache(maxsize=16)
def victim_model(model_id: str, seed: int) -> LinearModel:
    ds = load_inversion_dataset(seed)
    if model_id == "logreg":
        w, b = _fit(np.c_[ds.X_train, ds.s_train], ds.y_train, seed)
        feats = (*PUBLIC_FEATURES, PRIVATE_ATTRIBUTE)
        note = "private training split: public features + private attribute → label"
    elif model_id == "logreg-public-only":
        w0, b = _fit(ds.X_train, ds.y_train, seed)
        w = np.r_[w0, 0.0]  # the service passes the attribute through; the model gives it no weight
        feats = (*PUBLIC_FEATURES, PRIVATE_ATTRIBUTE)
        note = "private training split: public features only → label (attribute ignored)"
    else:
        raise KeyError(f"unknown inversion victim model {model_id!r}")
    w.setflags(write=False)
    return LinearModel(model_id, "victim", feats, w, b, seed, len(ds.y_train), note)


@lru_cache(maxsize=16)
def public_only_reference(seed: int) -> LinearModel:
    """Reference model WITHOUT the private attribute, to measure how relevant the attribute is to the label."""
    return victim_model("logreg-public-only", seed)


@lru_cache(maxsize=16)
def auxiliary_model(seed: int) -> LinearModel:
    """The attacker's auxiliary-only model: P(s = 1 | public features), fitted on the auxiliary population."""
    ds = load_inversion_dataset(seed)
    w, b = _fit(ds.aux.X, ds.aux.s, seed)
    w.setflags(write=False)
    return LinearModel("aux-logreg", "attacker auxiliary model", PUBLIC_FEATURES, w, b, seed, len(ds.aux.s),
                       "auxiliary population: public features → private attribute (no victim outputs)")


@lru_cache(maxsize=32)
def auxiliary_model_sized(seed: int, n: int) -> LinearModel | None:
    """A WEAKER auxiliary attacker for sensitivity studies: fitted on the first ``n`` auxiliary rows only.

    ``n = 0`` (no auxiliary data) returns None: the attacker then has no prior (P(s=1) = 0.5 for everyone). A sample
    containing a single class also returns None — logistic regression cannot be fitted on it.
    """
    if n <= 0:
        return None
    ds = load_inversion_dataset(seed)
    X, s = ds.aux.X[:n], ds.aux.s[:n]
    if len(set(s.tolist())) < 2:
        return None
    w, b = _fit(X, s, seed)
    w.setflags(write=False)
    return LinearModel(f"aux-logreg-n{n}", "attacker auxiliary model (reduced)", PUBLIC_FEATURES, w, b, seed, n,
                       f"first {n} auxiliary rows: public features → private attribute")
