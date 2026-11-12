"""Deployed spam-filter systems: undefended, text normalisation, adversarial data augmentation.

Leakage rules (tested): the normaliser's spell-snap vocabulary and the adversarial augmentation use the
TRAINING split only; every system is evaluated on the same held-out test split.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..attacks import _check_deadline
from .attacks import deepwordbug_attack
from .data import load_text_dataset
from .models import TextClassifier, train_text_model
from .normalize import Normalizer
from .schemas import TextAdversarialTrainingDefense, TextAttackConfig, TextNormalizationDefense


@dataclass
class TextSystem:
    model: TextClassifier
    defense: dict[str, Any] | None
    details: dict[str, Any]

    def p_spam(self, texts):
        return self.model.p_spam(texts)

    def describe(self) -> dict[str, Any]:
        return {"defense": self.defense, "model_artifact_sha256": self.model.artifact_sha256(),
                "decision_policy": "spam if P(spam) ≥ 0.5", "intentionally_vulnerable": self.defense is None,
                "model": self.model.describe(), **self.details}


@lru_cache(maxsize=8)
def base_text_model(model_id: str, seed: int) -> TextClassifier:
    ds = load_text_dataset(seed)
    return train_text_model(model_id, ds.train_texts, ds.train_labels, seed)


@lru_cache(maxsize=16)
def _normalized_model(model_id: str, seed: int, cfg_json: str) -> TextClassifier:
    cfg = TextNormalizationDefense.model_validate_json(cfg_json)
    ds = load_text_dataset(seed)
    norm = Normalizer(cfg, ds.train_texts)  # vocabulary from TRAINING messages only
    return train_text_model(model_id, ds.train_texts, ds.train_labels, seed, normalizer=norm,
                            training_note="trained and served on normalised text")


_aug_cache: "OrderedDict[tuple, tuple[TextClassifier, dict]]" = OrderedDict()
_aug_lock = threading.Lock()


def augmentation_attack_config(cfg: TextAdversarialTrainingDefense) -> TextAttackConfig:
    return TextAttackConfig(goal="spam_to_ham", query_budget=cfg.query_budget, max_edit_distance=cfg.max_edit_distance,
                            max_char_ratio=cfg.max_char_ratio, candidates_per_word=cfg.candidates_per_word)


def adversarially_augmented_model(model_id: str, seed: int, cfg: TextAdversarialTrainingDefense,
                                  deadline: float | None = None) -> tuple[TextClassifier, dict]:
    """Attack the first ``max_train_samples`` TRAINING spam messages with the undefended model, add every
    successful adversarial message (label spam) to the training set and retrain. Cached only once complete."""
    key = (model_id, seed, json.dumps(cfg.model_dump(), sort_keys=True))
    with _aug_lock:
        if key in _aug_cache:
            _aug_cache.move_to_end(key)
            return _aug_cache[key]
    ds = load_text_dataset(seed)
    base = base_text_model(model_id, seed)
    acfg = augmentation_attack_config(cfg)
    spam_idx = [i for i, y in enumerate(ds.train_labels) if y == 1][:cfg.max_train_samples]
    adv_texts, outcomes = [], {"evaded": 0, "failed": 0, "budget_exhausted": 0, "skipped_misclassified": 0}
    for i in spam_idx:
        _check_deadline(deadline)
        r = deepwordbug_attack(ds.train_texts[i], 1, base.p_spam, acfg, np.random.default_rng([seed, 4242, i]), deadline)
        outcomes[r.outcome] += 1
        if r.outcome == "evaded":
            adv_texts.append(r.text)
    model = train_text_model(model_id, list(ds.train_texts) + adv_texts, list(ds.train_labels) + [1] * len(adv_texts),
                             seed, training_note=f"clean training split + {len(adv_texts)} adversarial training spam")
    _check_deadline(deadline)
    details = {"augmentation": {"source": "training split only (spam messages, in split order)",
                                "attacked": len(spam_idx), "adversarial_examples_added": len(adv_texts),
                                "outcomes": outcomes, "attack_config": acfg.model_dump()}}
    with _aug_lock:
        _aug_cache[key] = (model, details)
        while len(_aug_cache) > 8:
            _aug_cache.popitem(last=False)
    return model, details


def clear_text_caches() -> None:
    base_text_model.cache_clear()
    _normalized_model.cache_clear()
    with _aug_lock:
        _aug_cache.clear()


def build_text_system(model_id: str, seed: int, defense, deadline: float | None = None) -> TextSystem:
    base = base_text_model(model_id, seed)
    _check_deadline(deadline)
    if defense is None:
        return TextSystem(base, None, {})
    dump = defense.model_dump()
    if isinstance(defense, TextNormalizationDefense):
        return TextSystem(_normalized_model(model_id, seed, json.dumps(dump, sort_keys=True)), dump, {})
    model, details = adversarially_augmented_model(model_id, seed, defense, deadline)
    return TextSystem(model, dump, details)
