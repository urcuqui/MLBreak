"""Defensive controls and the *deployed system* abstraction.

A :class:`DeployedSystem` is what an attacker actually interacts with: a model
plus (optionally) a decision policy. Two real controls are provided:

* **Adversarial training** (model-level). The model is retrained on the clean
  training set plus PGD adversarial copies generated against the current model,
  for ``rounds`` rounds (Madry et al., 2018, approximated with a convex
  learner's full retraining in place of SGD minibatches).
* **Confidence rejection** (system-level). The system abstains when the
  top-class probability is below a threshold. Abstention is a safe outcome for
  the attacker's goal (no wrong decision is emitted) but costs coverage on
  clean inputs.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from .attacks import ABSTAIN, _check_deadline, gradient_attack
from .data import Dataset, load_dataset
from .models import DifferentiableClassifier, train_model
from .schemas import AdversarialTrainingDefense, AttackConfig, ConfidenceRejectionDefense, DefenseConfig

_DEFENSE_ADAPTER = TypeAdapter(DefenseConfig)


def logit_of(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


@dataclass
class Decision:
    labels: np.ndarray  # 0/1, or ABSTAIN (-1)
    p1: np.ndarray
    confidence: np.ndarray


@dataclass
class DeployedSystem:
    model: DifferentiableClassifier
    defense: dict[str, Any] | None
    reject_threshold: float | None = None

    def decide(self, X: np.ndarray) -> Decision:
        p1 = self.model.proba(X)
        conf = np.maximum(p1, 1.0 - p1)
        labels = (self.model.logit(X) > 0).astype(np.int64)
        if self.reject_threshold is not None:
            labels = np.where(conf >= self.reject_threshold, labels, ABSTAIN)
        return Decision(labels, p1, conf)

    def describe(self) -> dict[str, Any]:
        return {
            "defense": self.defense,
            "model_artifact_sha256": self.model.artifact_sha256(),
            "decision_policy": (
                f"abstain if max(p) < {self.reject_threshold}" if self.reject_threshold else "argmax"
            ),
            "intentionally_vulnerable": self.defense is None,
        }


@lru_cache(maxsize=16)
def base_model(model_id: str, dataset_id: str, seed: int) -> DifferentiableClassifier:
    ds = load_dataset(dataset_id, seed)
    return train_model(model_id, ds.X_train, ds.y_train, seed)


def adversarially_train(model_id: str, ds: Dataset, seed: int,
                        cfg: AdversarialTrainingDefense, deadline: float | None = None) -> DifferentiableClassifier:
    """Adversarial training under a cooperative wall-clock ``deadline``.

    The deadline is checked before every round, inside the PGD example generation (per step) and
    after every model fit. A single scikit-learn ``fit`` cannot be interrupted, so the worst-case
    overshoot is one fit (bounded by ``max_iter`` and the fixed training-set size).
    """
    attack = AttackConfig(attack="pgd", norm=cfg.norm, epsilon=cfg.epsilon, steps=cfg.steps)
    lo, hi = ds.domain_low, ds.domain_high
    model = base_model(model_id, ds.spec.dataset_id, seed)
    _check_deadline(deadline)
    X_aug, y_aug = [ds.X_train], [ds.y_train]
    for r in range(cfg.rounds):
        _check_deadline(deadline)
        rngs = [np.random.default_rng([seed, r, i]) for i in range(len(ds.X_train))]
        out = gradient_attack(model, ds.X_train, ds.y_train, attack, lo, hi, rngs, deadline)
        X_aug.append(out.x_adv)
        y_aug.append(ds.y_train)
        _check_deadline(deadline)
        model = train_model(
            model_id, np.vstack(X_aug), np.concatenate(y_aug), seed,
            training_note=f"adversarial training round {r + 1}/{cfg.rounds} "
                          f"(PGD {cfg.norm} ε={cfg.epsilon}, {cfg.steps} steps)",
        )
        _check_deadline(deadline)
    return model


_DEFENDED_CACHE_SIZE = 32
_defended_cache: "OrderedDict[tuple, DifferentiableClassifier]" = OrderedDict()
_defended_lock = threading.Lock()


def defended_model(model_id: str, dataset_id: str, seed: int, cfg: AdversarialTrainingDefense,
                   deadline: float | None = None) -> DifferentiableClassifier:
    """Adversarially trained model, cached ONLY after training completed within the deadline.

    An interrupted training (ExperimentTimeout) leaves the cache untouched, so a later request never
    receives a partially trained model.
    """
    key = (model_id, dataset_id, seed, json.dumps(cfg.model_dump(), sort_keys=True))
    with _defended_lock:
        if key in _defended_cache:
            _defended_cache.move_to_end(key)
            return _defended_cache[key]
    model = adversarially_train(model_id, load_dataset(dataset_id, seed), seed, cfg, deadline)
    with _defended_lock:
        _defended_cache[key] = model
        _defended_cache.move_to_end(key)
        while len(_defended_cache) > _DEFENDED_CACHE_SIZE:
            _defended_cache.popitem(last=False)
    return model


def clear_model_caches() -> None:
    """Drop cached trained models (tests and memory pressure). Datasets stay cached."""
    base_model.cache_clear()
    with _defended_lock:
        _defended_cache.clear()


def build_system(model_id: str, dataset_id: str, seed: int,
                 defense: AdversarialTrainingDefense | ConfidenceRejectionDefense | None,
                 deadline: float | None = None) -> DeployedSystem:
    model = base_model(model_id, dataset_id, seed)
    _check_deadline(deadline)  # a cold base-model fit may already have used the budget
    if defense is None:
        return DeployedSystem(model, None)
    dump = defense.model_dump()
    if isinstance(defense, AdversarialTrainingDefense):
        return DeployedSystem(defended_model(model_id, dataset_id, seed, defense, deadline), dump)
    if isinstance(defense, ConfidenceRejectionDefense):
        return DeployedSystem(model, dump, defense.threshold)
    raise TypeError(type(defense))


def parse_defense(data: dict[str, Any] | None):
    return None if data is None else _DEFENSE_ADAPTER.validate_python(data)
