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
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from .attacks import ABSTAIN, gradient_attack
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
                        cfg: AdversarialTrainingDefense) -> DifferentiableClassifier:
    attack = AttackConfig(attack="pgd", norm=cfg.norm, epsilon=cfg.epsilon, steps=cfg.steps)
    lo, hi = ds.domain_low, ds.domain_high
    model = base_model(model_id, ds.spec.dataset_id, seed)
    X_aug, y_aug = [ds.X_train], [ds.y_train]
    for r in range(cfg.rounds):
        rngs = [np.random.default_rng([seed, r, i]) for i in range(len(ds.X_train))]
        out = gradient_attack(model, ds.X_train, ds.y_train, attack, lo, hi, rngs)
        X_aug.append(out.x_adv)
        y_aug.append(ds.y_train)
        model = train_model(
            model_id, np.vstack(X_aug), np.concatenate(y_aug), seed,
            training_note=f"adversarial training round {r + 1}/{cfg.rounds} "
                          f"(PGD {cfg.norm} ε={cfg.epsilon}, {cfg.steps} steps)",
        )
    return model


@lru_cache(maxsize=32)
def _defended_model(model_id: str, dataset_id: str, seed: int, cfg_json: str) -> DifferentiableClassifier:
    cfg = AdversarialTrainingDefense.model_validate_json(cfg_json)
    return adversarially_train(model_id, load_dataset(dataset_id, seed), seed, cfg)


def build_system(model_id: str, dataset_id: str, seed: int,
                 defense: AdversarialTrainingDefense | ConfidenceRejectionDefense | None) -> DeployedSystem:
    if defense is None:
        return DeployedSystem(base_model(model_id, dataset_id, seed), None)
    dump = defense.model_dump()
    if isinstance(defense, AdversarialTrainingDefense):
        cfg_json = json.dumps(dump, sort_keys=True)
        return DeployedSystem(_defended_model(model_id, dataset_id, seed, cfg_json), dump)
    if isinstance(defense, ConfidenceRejectionDefense):
        return DeployedSystem(base_model(model_id, dataset_id, seed), dump, defense.threshold)
    raise TypeError(type(defense))


def parse_defense(data: dict[str, Any] | None):
    return None if data is None else _DEFENSE_ADAPTER.validate_python(data)
