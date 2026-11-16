"""Deterministic synthetic data for the ML03 lab: a FICTIONAL account-risk scoring service.

No real person, account, medical or financial record is used or imitated. Every value is drawn from the
generator below with a seeded NumPy ``default_rng``; the same seed always produces the same three splits.

Variables (per fictional account holder):

* public features ``x ∈ R⁴`` (standardised, attacker-known): ``account_tenure``, ``monthly_activity``,
  ``service_usage``, ``region_index`` — i.i.d. N(0, 1);
* private attribute ``s ∈ {0, 1}`` (attacker-unknown): enrolment in a confidential hardship-support programme,
  ``P(s = 1 | x) = σ(−0.2 + 0.9·x₀ − 0.7·x₂)`` — partly predictable from public data, so an auxiliary-only
  attacker beats a coin flip (that is why every inversion result is compared with the auxiliary baseline);
* target label ``y ∈ {0, 1}`` (what the model predicts): the account is flagged for elevated payment risk,
  ``P(y = 1 | x, s) = σ(−0.9 + 0.5·x₀ + 0.8·x₁ − 0.6·x₃ + 1.8·s)`` — the private attribute is genuinely relevant
  to the prediction, but ``y`` and ``s`` are different variables.

Splits (disjoint, drawn i.i.d. from the same generator):

* ``train`` (1200) — the victim's PRIVATE training records (x, s, y); never visible to the attacker;
* ``auxiliary`` (600) — an independent population sample of (x, s) WITHOUT labels, standing in for public
  survey-style knowledge the attacker may hold; no victim output is attached to it;
* ``evaluation`` (400) — held-out records whose private attribute the attacker tries to reconstruct. They are
  NOT training members: inversion here does not rely on membership (that would be ML04).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..hashing import sha256_arrays

DATASET_ID = "synthetic-account-risk-v1"
PUBLIC_FEATURES = ("account_tenure", "monthly_activity", "service_usage", "region_index")
PRIVATE_ATTRIBUTE = "confidential_support_programme"
TARGET_LABEL = "elevated_payment_risk"
N_TRAIN, N_AUX, N_EVAL = 1200, 600, 400
GENERATOR: dict[str, Any] = {
    "public": "x ~ N(0, I_4)",
    "private": {"intercept": -0.2, "weights": [0.9, 0.0, -0.7, 0.0], "link": "logistic"},
    "label": {"intercept": -0.9, "weights": [0.5, 0.8, 0.0, -0.6], "private_weight": 1.8, "link": "logistic"},
    "rng": "numpy.random.default_rng([seed, 3003]); x, then u_s, then u_y, each one draw per row",
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def linear(X: np.ndarray, weights, intercept: float = 0.0) -> np.ndarray:
    """``intercept + X · w`` as an explicit column sum (no BLAS): row-independent and bit-identical whether rows
    are evaluated one at a time or in a batch — the candidate-matching attack compares such outputs exactly."""
    X = np.asarray(X, dtype=np.float64)
    out = np.full(X.shape[0], float(intercept))
    for j, w in enumerate(weights):
        out = out + X[:, j] * float(w)
    return out


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class AuxiliaryData:
    """The attacker's auxiliary population: public features and private attribute, no labels, no victim outputs."""

    X: np.ndarray
    s: np.ndarray


@dataclass(frozen=True)
class InversionDataset:
    seed: int
    X_train: np.ndarray
    s_train: np.ndarray
    y_train: np.ndarray
    aux: AuxiliaryData
    X_eval: np.ndarray
    s_eval: np.ndarray
    y_eval: np.ndarray

    @property
    def split_id(self) -> str:
        return f"{DATASET_ID}/seed={self.seed}/train={N_TRAIN}/aux={N_AUX}/eval={N_EVAL}"

    def fingerprint(self) -> dict[str, str]:
        return {"train_sha256": sha256_arrays(self.X_train, self.s_train, self.y_train),
                "auxiliary_sha256": sha256_arrays(self.aux.X, self.aux.s),
                "test_sha256": sha256_arrays(self.X_eval, self.s_eval, self.y_eval)}

    def describe(self) -> dict[str, Any]:
        return {
            "dataset_id": DATASET_ID, "synthetic": True,
            "description": "fictional account-risk records generated in-process; no real personal data",
            "public_features": list(PUBLIC_FEATURES), "private_attribute": PRIVATE_ATTRIBUTE,
            "target_label": TARGET_LABEL, "generator": GENERATOR, "split_id": self.split_id,
            "n_train": N_TRAIN, "n_auxiliary": N_AUX, "n_test": N_EVAL,
            "private_rate_train": float(self.s_train.mean()), "private_rate_eval": float(self.s_eval.mean()),
            "label_rate_train": float(self.y_train.mean()), "label_rate_eval": float(self.y_eval.mean()),
            "splits": "train = victim's private training records; auxiliary = attacker's (x, s) population sample "
                      "without labels; test = held-out evaluation records (inversion targets, not training members)",
            **self.fingerprint(),
        }


@lru_cache(maxsize=16)
def load_inversion_dataset(seed: int) -> InversionDataset:
    rng = np.random.default_rng([seed, 3003])
    n = N_TRAIN + N_AUX + N_EVAL
    X = rng.standard_normal((n, len(PUBLIC_FEATURES)))
    p = GENERATOR["private"]
    s = (rng.random(n) < _sigmoid(linear(X, p["weights"], p["intercept"]))).astype(np.int64)
    lab = GENERATOR["label"]
    y = (rng.random(n) < _sigmoid(linear(X, lab["weights"], lab["intercept"]) + lab["private_weight"] * s)
         ).astype(np.int64)
    a, b = N_TRAIN, N_TRAIN + N_AUX
    return InversionDataset(
        seed=seed, X_train=_readonly(X[:a]), s_train=_readonly(s[:a]), y_train=_readonly(y[:a]),
        aux=AuxiliaryData(X=_readonly(X[a:b]), s=_readonly(s[a:b])),
        X_eval=_readonly(X[b:]), s_eval=_readonly(s[b:]), y_eval=_readonly(y[b:]))
