"""Bundled, approved datasets.

Datasets are *generated* deterministically from a seed — nothing is
downloaded. Arrays returned from :func:`load_dataset` are read-only so a
caller cannot corrupt the cached copy shared between experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from .hashing import sha256_arrays


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    generator: str
    n_samples: int
    noise: float
    test_size: float
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    # Valid feature domain (min, max) per feature. Attacks must stay inside it.
    domain: tuple[tuple[float, float], ...]


DATASETS: dict[str, DatasetSpec] = {
    "moons-v1": DatasetSpec(
        dataset_id="moons-v1",
        generator="sklearn.datasets.make_moons",
        n_samples=500,
        noise=0.25,
        test_size=0.3,
        feature_names=("x1", "x2"),
        class_names=("class_0", "class_1"),
        domain=((-2.5, 3.5), (-2.0, 2.5)),
    )
}


@dataclass(frozen=True)
class Dataset:
    spec: DatasetSpec
    seed: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def domain_low(self) -> np.ndarray:
        return np.array([d[0] for d in self.spec.domain], dtype=float)

    @property
    def domain_high(self) -> np.ndarray:
        return np.array([d[1] for d in self.spec.domain], dtype=float)

    @property
    def split_id(self) -> str:
        return f"{self.spec.dataset_id}/seed={self.seed}/test_size={self.spec.test_size}/stratified"

    def fingerprint(self) -> dict[str, str]:
        return {
            "train_sha256": sha256_arrays(self.X_train, self.y_train),
            "test_sha256": sha256_arrays(self.X_test, self.y_test),
        }

    def describe(self) -> dict:
        s = self.spec
        return {
            "dataset_id": s.dataset_id,
            "generator": s.generator,
            "generator_params": {"n_samples": s.n_samples, "noise": s.noise, "random_state": self.seed},
            "feature_names": list(s.feature_names),
            "class_names": list(s.class_names),
            "domain": [list(d) for d in s.domain],
            "split_id": self.split_id,
            "n_train": int(len(self.y_train)),
            "n_test": int(len(self.y_test)),
            **self.fingerprint(),
        }


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


@lru_cache(maxsize=16)
def load_dataset(dataset_id: str, seed: int) -> Dataset:
    spec = DATASETS.get(dataset_id)
    if spec is None:
        raise KeyError(f"unknown dataset {dataset_id!r}")
    X, y = make_moons(n_samples=spec.n_samples, noise=spec.noise, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=spec.test_size, random_state=seed, stratify=y
    )
    lo = np.array([d[0] for d in spec.domain])
    hi = np.array([d[1] for d in spec.domain])
    if not (np.all(X >= lo) and np.all(X <= hi)):
        # The domain must contain every clean sample, otherwise the attack's
        # domain projection would itself modify clean inputs.
        raise ValueError(f"dataset {dataset_id} seed={seed} falls outside its declared domain")
    return Dataset(spec, seed, _readonly(X_tr), _readonly(y_tr.astype(np.int64)),
                   _readonly(X_te), _readonly(y_te.astype(np.int64)))
