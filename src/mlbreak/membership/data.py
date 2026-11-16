"""Deterministic synthetic data for the ML04 lab: a FICTIONAL transaction-risk classification service.

No real person, account or transaction is used or imitated. Every value comes from a seeded NumPy ``default_rng``;
the same seed always produces the same populations.

Record: 8 standardised features ``x ~ N(0, I_8)`` and a binary label ``y`` ("high risk") with
``P(y = 1 | x) = σ(−0.2 + x·w + 0.7·x₂·x₃)``. The label is NOISY by construction (the Bayes-optimal classifier is
far from perfect), so a model that fits its training labels perfectly has memorised individual records — the
controlled overfitting that makes training membership observable.

Populations (disjoint draws; no record appears in two of them, verified row by row):

=========================  =====  ==========================================================================
population                 size   role
=========================  =====  ==========================================================================
``victim_train``            600   the victim's training records = the MEMBERS (evaluator-only knowledge)
``victim_test``             600   held-out records for victim utility and the generalisation gap (never attacked)
``shadow``                 2400   attacker's auxiliary population for shadow models (known member/non-member splits)
``calibration``            1200   attacker's auxiliary population for reference models that calibrate thresholds
``evaluation_non_members``  600   non-member pool of the final membership evaluation
``decoy``                   600   training set of the NEGATIVE-CONTROL victim (``random-forest-decoy``) only
=========================  =====  ==========================================================================

The final membership-evaluation candidates are ``n_members`` records of ``victim_train`` and ``n_non_members`` of
``evaluation_non_members`` (see :class:`~.schemas.CandidateSelection`). The attacker's populations (``shadow``,
``calibration``) come from the same distribution as the victim's data but share no record with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..hashing import sha256_arrays
from ..inversion.data import linear

DATASET_ID = "synthetic-transaction-risk-v1"
FEATURES = ("account_age", "transaction_volume", "transaction_velocity", "chargeback_ratio", "device_changes",
            "login_failures", "geo_distance", "support_contacts")
LABEL = "high_risk"
N_TRAIN, N_TEST, N_SHADOW, N_CALIBRATION, N_EVAL_NON_MEMBERS, N_DECOY = 600, 600, 2400, 1200, 600, 600
POPULATIONS = ("victim_train", "victim_test", "shadow", "calibration", "evaluation_non_members", "decoy")
SIZES = dict(zip(POPULATIONS, (N_TRAIN, N_TEST, N_SHADOW, N_CALIBRATION, N_EVAL_NON_MEMBERS, N_DECOY)))
GENERATOR: dict[str, Any] = {
    "features": "x ~ N(0, I_8)",
    "label": {"intercept": -0.2, "weights": [1.0, -0.8, 0.6, 0.0, 0.5, 0.0, -0.4, 0.0],
              "interaction": "0.7 · x2 · x3", "link": "logistic"},
    "rng": "numpy.random.default_rng([seed, 4004]); all features, then one uniform per row for the label",
    "populations": "consecutive slices in the order " + ", ".join(POPULATIONS),
}


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def row_hashes(X: np.ndarray) -> list[str]:
    """Per-record fingerprint (features only) used for duplicate and overlap checks."""
    return [sha256_arrays(np.ascontiguousarray(r)) for r in np.asarray(X, dtype=np.float64)]


@dataclass(frozen=True)
class Population:
    name: str
    X: np.ndarray
    y: np.ndarray

    def sha256(self) -> str:
        return sha256_arrays(self.X, self.y)


@dataclass(frozen=True)
class MembershipDataset:
    seed: int
    pops: dict[str, Population]

    def __getitem__(self, name: str) -> Population:
        return self.pops[name]

    @property
    def split_id(self) -> str:
        return f"{DATASET_ID}/seed={self.seed}/" + "/".join(f"{k}={v}" for k, v in SIZES.items())

    def fingerprint(self) -> dict[str, str]:
        return {f"{k}_sha256": p.sha256() for k, p in self.pops.items()}

    def disjointness(self) -> dict[str, Any]:
        """Duplicate records anywhere, and records shared between any two populations (both must be 0)."""
        hashes = {k: row_hashes(p.X) for k, p in self.pops.items()}
        all_rows = [h for v in hashes.values() for h in v]
        overlaps = {}
        names = list(hashes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlaps[f"{a}∩{b}"] = len(set(hashes[a]) & set(hashes[b]))
        return {"duplicate_records": len(all_rows) - len(set(all_rows)), "pairwise_overlaps": overlaps,
                "disjoint": len(all_rows) == len(set(all_rows)),
                "method": "SHA-256 of each record's float64 feature vector, across every population"}

    def describe(self) -> dict[str, Any]:
        return {
            "dataset_id": DATASET_ID, "synthetic": True,
            "description": "fictional transaction-risk records generated in-process; no real personal data",
            "features": list(FEATURES), "label": LABEL, "generator": GENERATOR, "split_id": self.split_id,
            "population_sizes": dict(SIZES),
            "label_rates": {k: float(p.y.mean()) for k, p in self.pops.items()},
            "roles": {"victim_train": "victim training records (members; evaluator-only)",
                      "victim_test": "held-out victim utility records (never attacked)",
                      "shadow": "attacker auxiliary data for shadow models",
                      "calibration": "attacker auxiliary data for threshold-calibration reference models",
                      "evaluation_non_members": "non-member pool of the final membership evaluation",
                      "decoy": "training set of the negative-control victim only"},
            **self.fingerprint(), "disjointness": _disjointness(self.seed),
        }


@lru_cache(maxsize=16)
def _disjointness(seed: int) -> dict[str, Any]:
    return load_membership_dataset(seed).disjointness()


@lru_cache(maxsize=16)
def load_membership_dataset(seed: int) -> MembershipDataset:
    rng = np.random.default_rng([seed, 4004])
    n = sum(SIZES.values())
    X = rng.standard_normal((n, len(FEATURES)))
    g = GENERATOR["label"]
    # explicit column sum (no BLAS): identical labels in every numerical environment
    z = linear(X, g["weights"], g["intercept"]) + 0.7 * X[:, 2] * X[:, 3]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)
    pops, start = {}, 0
    for name in POPULATIONS:
        stop = start + SIZES[name]
        pops[name] = Population(name, _readonly(X[start:stop]), _readonly(y[start:stop]))
        start = stop
    return MembershipDataset(seed=seed, pops=pops)


def select_candidates(seed: int, n_members: int, n_non_members: int) -> tuple[list[int], list[int]]:
    """Indices into ``victim_train`` (members) and ``evaluation_non_members``; deterministic, nested in the counts."""
    m = np.random.default_rng([seed, 4104]).permutation(N_TRAIN)[:n_members]
    n = np.random.default_rng([seed, 4105]).permutation(N_EVAL_NON_MEMBERS)[:n_non_members]
    return [int(i) for i in m], [int(i) for i in n]
