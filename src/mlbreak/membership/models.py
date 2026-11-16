"""Victim, reference and shadow models of the ML04 lab (trained in-process; never loaded from pickle/joblib).

A model is defined by a :class:`Recipe` (family + hyperparameters) and the records it is trained on. The same recipe
code trains the victim, the attacker's reference models (threshold calibration) and its shadow models, so a
recalibrated attack against a defended victim uses exactly the defended recipe (the attacker is assumed to know the
training procedure — Kerckhoffs' principle; see docs/ml04-membership-inference.md §2).

Victims (identical prediction interface, see :class:`~.defenses.MembershipService`):

* ``random-forest-overfit`` — 100 fully grown trees (``min_samples_leaf = 1``, no depth limit) on the noisy-label
  training records: intentionally overfitting (training accuracy 1.0).
* ``logreg-l2`` — L2-regularised logistic regression (C = 1.0): a lower-capacity comparison. It is NOT assumed to be
  leak-free; its leakage is measured like any other victim's.
* ``random-forest-decoy`` — NEGATIVE CONTROL: the overfitting recipe trained on the disjoint ``decoy`` population, so
  NO evaluation candidate is a member of its training set. Any measured "leakage" is unsupported by construction.

Artifact hashes cover every fitted parameter (all tree arrays / coefficients) and the recipe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..hashing import sha256_arrays, sha256_json
from ..inversion.data import linear
from .data import load_membership_dataset

RF_OVERFIT: dict[str, Any] = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt",
                              "bootstrap": True}
LOGREG_L2: dict[str, Any] = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000}
VICTIMS: dict[str, dict[str, Any]] = {
    "random-forest-overfit": {"family": "random_forest", "hyperparameters": RF_OVERFIT, "population": "victim_train",
                              "description": "Random forest, fully grown trees: intentionally overfitting victim."},
    "logreg-l2": {"family": "logistic_regression", "hyperparameters": LOGREG_L2, "population": "victim_train",
                  "description": "L2-regularised logistic regression: lower-capacity comparison victim."},
    "random-forest-decoy": {"family": "random_forest", "hyperparameters": RF_OVERFIT, "population": "decoy",
                            "description": "Negative control: the overfitting recipe trained on a disjoint decoy "
                                           "population; no evaluation candidate is a training member."},
}


@dataclass(frozen=True)
class Recipe:
    family: str  # "random_forest" | "logistic_regression"
    hyperparameters_json: str  # canonical JSON (hashable)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return json.loads(self.hyperparameters_json)

    def describe(self) -> dict[str, Any]:
        est = {"random_forest": "sklearn.ensemble.RandomForestClassifier",
               "logistic_regression": "sklearn.linear_model.LogisticRegression"}[self.family]
        return {"family": self.family, "estimator": est, "hyperparameters": self.hyperparameters}


def recipe(family: str, hyper: dict[str, Any]) -> Recipe:
    return Recipe(family, json.dumps(hyper, sort_keys=True))


def victim_recipe(model_id: str) -> Recipe:
    v = VICTIMS[model_id]
    return recipe(v["family"], v["hyperparameters"])


def apply_defense(base: Recipe, defense: dict[str, Any] | None) -> tuple[Recipe, dict[str, Any]]:
    """Recipe of the defended victim and a description of what was applied. Raises ValueError if not applicable."""
    if not defense or defense["kind"] == "output_restriction":
        return base, {"training_changed": False}
    hyper = base.hyperparameters
    if defense["kind"] == "regularization":
        if base.family == "random_forest":
            hyper["min_samples_leaf"] = int(defense["min_samples_leaf"])
            applied = {"parameter": "min_samples_leaf", "value": hyper["min_samples_leaf"]}
        else:
            hyper["C"] = float(defense["C"])
            applied = {"parameter": "C", "value": hyper["C"]}
    elif defense["kind"] == "complexity_reduction":
        if base.family != "random_forest":
            raise ValueError("complexity_reduction applies to tree ensembles only; logistic regression is already a "
                             "linear model (use regularization)")
        hyper["max_depth"] = int(defense["max_depth"])
        hyper["n_estimators"] = int(defense["n_estimators"])
        applied = {"parameter": "max_depth, n_estimators", "value": [hyper["max_depth"], hyper["n_estimators"]]}
    else:  # pragma: no cover - schema rejects other kinds
        raise ValueError(f"unknown defense {defense['kind']!r}")
    return recipe(base.family, hyper), {"training_changed": True, **applied}


def derive_state(*key: int) -> int:
    """Deterministic scikit-learn ``random_state`` for a (seed, purpose, index) key."""
    return int(np.random.default_rng(list(key)).integers(2**31 - 1))


@dataclass(frozen=True)
class FittedModel:
    role: str
    recipe: Recipe
    random_state: int
    n_train: int
    estimator: Any  # in-memory scikit-learn estimator (never serialised)
    training_rows_sha256: str

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.recipe.family == "logistic_regression":
            # explicit column sum: row-independent and identical for one row or a batch
            z = linear(X, self.estimator.coef_[0], float(self.estimator.intercept_[0]))
            return 1.0 / (1.0 + np.exp(-z))
        return np.ascontiguousarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.proba(X) >= 0.5).astype(np.int64)

    def artifact_sha256(self) -> str:
        est = self.estimator
        if self.recipe.family == "logistic_regression":
            params = sha256_arrays(est.coef_, est.intercept_)
        else:
            params = sha256_arrays(*[a for t in est.estimators_ for a in (
                t.tree_.children_left, t.tree_.children_right, t.tree_.feature, t.tree_.threshold, t.tree_.value)])
        return sha256_json({"recipe": self.recipe.describe(), "random_state": self.random_state, "parameters": params})

    def describe(self) -> dict[str, Any]:
        out = {"role": self.role, **self.recipe.describe(), "random_state": self.random_state, "n_train": self.n_train,
               "training_rows_sha256": self.training_rows_sha256, "artifact_sha256": self.artifact_sha256(),
               "artifact_format": "in-memory estimator (no pickle); hash over every fitted parameter and the recipe"}
        if self.recipe.family == "random_forest":
            depths = [int(t.tree_.max_depth) for t in self.estimator.estimators_]
            leaves = [int(t.tree_.n_leaves) for t in self.estimator.estimators_]
            out["complexity"] = {"trees": len(depths), "max_depth": max(depths), "mean_leaves": float(np.mean(leaves))}
        return out


def fit(rec: Recipe, X: np.ndarray, y: np.ndarray, random_state: int, role: str) -> FittedModel:
    h = rec.hyperparameters
    if rec.family == "random_forest":
        est = RandomForestClassifier(**h, random_state=random_state, n_jobs=1)
    else:
        est = LogisticRegression(**{k: v for k, v in h.items() if k != "penalty"}, random_state=random_state)
    est.fit(np.asarray(X, dtype=np.float64), np.asarray(y))
    return FittedModel(role, rec, random_state, len(y), est, sha256_arrays(np.asarray(X), np.asarray(y)))


@lru_cache(maxsize=32)
def victim_model(model_id: str, seed: int, rec: Recipe | None = None) -> FittedModel:
    """The victim (optionally retrained with a defended recipe) on its training population."""
    ds = load_membership_dataset(seed)
    pop = ds[VICTIMS[model_id]["population"]]
    return fit(rec or victim_recipe(model_id), pop.X, pop.y, derive_state(seed, 4200), f"victim ({model_id})")
