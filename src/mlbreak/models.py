"""Approved model configurations and a differentiable view of fitted models.

The white-box attacks need the gradient of the model's logit with respect to
the *input*. scikit-learn does not expose that, so :class:`DifferentiableClassifier`
re-implements the forward pass from the fitted parameters and back-propagates
analytically. ``tests/test_models.py`` checks that the forward pass reproduces
``predict_proba`` exactly and that gradients match finite differences.

Models are never pickled. The "artifact" is the set of fitted parameter arrays,
and the artifact hash is a SHA-256 over those arrays. Replays retrain from the
approved configuration and compare hashes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .hashing import sha256_arrays


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    estimator: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


MODELS: dict[str, ModelSpec] = {
    "mlp-tanh-16x16": ModelSpec(
        model_id="mlp-tanh-16x16",
        family="mlp",
        estimator="sklearn.neural_network.MLPClassifier",
        params={
            "hidden_layer_sizes": (16, 16),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.5,
            "max_iter": 5000,
        },
        description="Two-hidden-layer tanh MLP (non-linear decision boundary).",
    ),
    "logreg": ModelSpec(
        model_id="logreg",
        family="linear",
        estimator="sklearn.linear_model.LogisticRegression",
        params={"C": 1.0, "max_iter": 1000},
        description="Logistic regression (linear decision boundary) for contrast.",
    ),
}

_ACTS = {
    "tanh": (np.tanh, lambda a, z: 1.0 - a**2),
    "relu": (lambda z: np.maximum(z, 0.0), lambda a, z: (z > 0).astype(float)),
    "logistic": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a, z: a * (1.0 - a)),
    "identity": (lambda z: z, lambda a, z: np.ones_like(z)),
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic


class DifferentiableClassifier:
    """Binary classifier exposing logit, probability and input-gradient."""

    def __init__(self, spec: ModelSpec, estimator: Any, seed: int, training_note: str = "clean"):
        if list(getattr(estimator, "classes_", [])) != [0, 1]:
            raise ValueError("only binary classifiers with classes [0, 1] are supported")
        self.spec = spec
        self.estimator = estimator
        self.seed = seed
        self.training_note = training_note
        self.n_iter = int(np.max(getattr(estimator, "n_iter_", 0)))

    # -- forward -------------------------------------------------------------
    def logit(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=float))[0]

    def proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of class 1."""
        return sigmoid(self.logit(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn predicts class 1 iff p > 0.5, i.e. logit > 0.
        return (self.logit(X) > 0).astype(np.int64)

    def grad_logit(self, X: np.ndarray) -> np.ndarray:
        """d logit / d x, shape (n, n_features)."""
        X = np.asarray(X, dtype=float)
        est = self.estimator
        if self.spec.family == "linear":
            return np.tile(est.coef_[0], (len(X), 1))
        _, acts, pre = self._forward(X, keep=True)
        d_act = _ACTS[est.activation][1]
        g = np.tile(est.coefs_[-1][:, 0], (len(X), 1))
        for W, a, z in zip(reversed(est.coefs_[:-1]), reversed(acts), reversed(pre)):
            g = (g * d_act(a, z)) @ W.T
        return g

    def _forward(self, X: np.ndarray, keep: bool = False):
        est = self.estimator
        if self.spec.family == "linear":
            return (X @ est.coef_[0] + est.intercept_[0],)
        act = _ACTS[est.activation][0]
        a, acts, pre = X, [], []
        for W, b in zip(est.coefs_[:-1], est.intercepts_[:-1]):
            z = a @ W + b
            a = act(z)
            pre.append(z)
            acts.append(a)
        out = (a @ est.coefs_[-1] + est.intercepts_[-1])[:, 0]
        return (out, acts, pre) if keep else (out,)

    # -- provenance ----------------------------------------------------------
    def parameter_arrays(self) -> list[np.ndarray]:
        est = self.estimator
        if self.spec.family == "linear":
            return [est.coef_, est.intercept_]
        return [*est.coefs_, *est.intercepts_]

    def artifact_sha256(self) -> str:
        return sha256_arrays(*self.parameter_arrays())

    def describe(self) -> dict[str, Any]:
        params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.spec.params.items()}
        return {
            "model_id": self.spec.model_id,
            "family": self.spec.family,
            "estimator": self.spec.estimator,
            "hyperparameters": {**params, "random_state": self.seed},
            "training": self.training_note,
            "n_iter": self.n_iter,
            "artifact_sha256": self.artifact_sha256(),
            "artifact_format": "in-memory fitted parameters (no pickle); hash over float64 parameter arrays",
        }


def build_estimator(spec: ModelSpec, seed: int):
    if spec.family == "mlp":
        return MLPClassifier(random_state=seed, **spec.params)
    if spec.family == "linear":
        return LogisticRegression(random_state=seed, **spec.params)
    raise ValueError(f"unsupported family {spec.family}")


def train_model(model_id: str, X: np.ndarray, y: np.ndarray, seed: int,
                training_note: str = "clean") -> DifferentiableClassifier:
    spec = MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"unknown model {model_id!r}")
    est = build_estimator(spec, seed)
    with warnings.catch_warnings():
        # Convergence is reported through n_iter in the model description.
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(np.asarray(X, dtype=float), np.asarray(y))
    return DifferentiableClassifier(spec, est, seed, training_note)
