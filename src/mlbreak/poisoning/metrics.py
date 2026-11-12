"""ML02 poisoning metrics, derived ONLY from per-sample evidence.

:func:`derive_metrics` is used both to build a record and to verify it later
(:mod:`mlbreak.poisoning.integrity`), so a stored metric can always be recomputed from the stored
training rows (true/observed/sanitised labels, flags, predictions) and test rows (predictions).

Every ratio carries numerator, denominator, population and definition; an empty denominator gives
``value: None`` (undefined), never 0.

Test-sample outcome (reference = clean model, other = poisoned or defended model):

=================  =========================================================
``kept``           correct under both
``broken``         correct under the clean model, wrong under the other one
``fixed``          wrong under the clean model, correct under the other one
``still_wrong``    wrong under both
=================  =========================================================
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from ..metrics import ratio

TEST_OUTCOMES = ("kept", "broken", "fixed", "still_wrong")

# Headline metrics that regression invariants may reference (all flat ratios in record.metrics).
INVARIANT_METRICS = (
    "test_accuracy", "clean_test_accuracy", "train_accuracy", "target_class_recall", "attack_success_rate",
    "collateral_damage_rate", "undefended_attack_success_rate", "detection_recall", "detection_precision",
    "false_positive_rate", "recovery_rate", "clean_data_test_accuracy", "clean_data_flag_rate", "poison_rate",
    "labels_repaired", "labels_corrupted",
)


def classify_test_outcome(y: int, ref: int, other: int) -> str:
    if ref == y:
        return "kept" if other == y else "broken"
    return "fixed" if other == y else "still_wrong"


def _g(s: Any, k: str):
    return s.get(k) if isinstance(s, dict) else getattr(s, k)


def _counts(outcomes: Sequence[str]) -> dict[str, int]:
    c = Counter(outcomes)
    return {k: int(c.get(k, 0)) for k in TEST_OUTCOMES}


def model_block(test, train, pred_key: str, train_pred_key: str, observed_key: str | None,
                train_mask: Sequence[bool] | None = None) -> dict[str, Any]:
    """Performance of one model: test accuracy, per-class recall, training accuracy."""
    n = len(test)
    correct = sum(_g(s, pred_key) == _g(s, "y_true") for s in test)
    out: dict[str, Any] = {
        "test_accuracy": ratio(correct, n, "test samples predicted as their true class / test samples",
                               "all test-split samples (clean labels, clean inputs)"),
    }
    for c in (0, 1):
        pop = [s for s in test if _g(s, "y_true") == c]
        out[f"recall_class_{c}"] = ratio(sum(_g(s, pred_key) == c for s in pop), len(pop),
                                         f"class-{c} test samples predicted as class {c} / class-{c} test samples",
                                         f"test-split samples whose true class is {c}")
    out["train_accuracy"] = ratio(
        sum(_g(s, train_pred_key) == _g(s, "y_true") for s in train), len(train),
        "training samples predicted as their TRUE (clean) label / training samples",
        "all training-split samples")
    if observed_key is not None:
        rows = [s for s, m in zip(train, train_mask or [True] * len(train)) if m]
        out["train_accuracy_observed"] = ratio(
            sum(_g(s, train_pred_key) == _g(s, observed_key) for s in rows), len(rows),
            "training samples predicted as the label the learner was GIVEN / training samples the learner "
            "was trained on (shows how far the model fitted the manipulated labels)",
            "training samples used to fit this model")
    return out


def _attack_population(test, target_class: int | None):
    if target_class is None:
        return test, "all test-split samples"
    return [s for s in test if _g(s, "y_true") == target_class], f"test-split samples of target class {target_class}"


def success_rate(test, other_key: str, target_class: int | None, *, label: str) -> dict[str, Any]:
    pop, desc = _attack_population(test, target_class)
    eligible = [s for s in pop if _g(s, "pred_clean") == _g(s, "y_true")]
    broken = sum(_g(s, other_key) != _g(s, "y_true") for s in eligible)
    return ratio(broken, len(eligible),
                 f"{desc} that the CLEAN model classifies correctly and the {label} model misclassifies / "
                 f"{desc} that the clean model classifies correctly",
                 f"{desc}, restricted to those the clean model classifies correctly")


def derive_metrics(evaluation: str, train, test, poison_cfg: dict | None,
                   defense_cfg: dict | None) -> dict[str, Any]:
    target = (poison_cfg or {}).get("target_class")
    m: dict[str, Any] = {"by_model": {"clean": model_block(test, train, "pred_clean", "pred_clean", None)}}
    clean = m["by_model"]["clean"]
    m["test_accuracy"] = clean["test_accuracy"]
    m["train_accuracy"] = clean["train_accuracy"]
    m["recall_class_0"], m["recall_class_1"] = clean["recall_class_0"], clean["recall_class_1"]
    if evaluation == "clean_training":
        return m

    n_train = len(train)
    poisoned = [s for s in train if _g(s, "poisoned")]
    m["poison_rate"] = ratio(len(poisoned), n_train, "training labels flipped by the attacker / training samples",
                             "all training-split samples")
    m["poisoned_by_true_class"] = {str(c): sum(_g(s, "y_true") == c for s in poisoned) for c in (0, 1)}
    m["by_model"]["poisoned"] = model_block(test, train, "pred_poisoned", "pred_poisoned", "y_observed")
    pm = m["by_model"]["poisoned"]
    m["clean_test_accuracy"] = clean["test_accuracy"]
    m["test_accuracy"] = pm["test_accuracy"]
    m["train_accuracy"] = pm["train_accuracy"]
    m["train_accuracy_observed"] = pm["train_accuracy_observed"]
    m["recall_class_0"], m["recall_class_1"] = pm["recall_class_0"], pm["recall_class_1"]
    m["test_accuracy_delta"] = {
        "value": (pm["test_accuracy"]["value"] - clean["test_accuracy"]["value"]) if len(test) else None,
        "definition": "poisoned-model test accuracy − clean-model test accuracy (negative = damage)"}
    if target is not None:
        m["target_class"] = target
        m["clean_target_class_recall"] = clean[f"recall_class_{target}"]
        m["target_class_recall"] = pm[f"recall_class_{target}"]
    attack_sr = success_rate(test, "pred_poisoned", target, label="poisoned")
    m["test_outcomes"] = {
        "reference": "clean model", "compared": "poisoned model", "population": "all test-split samples",
        "counts": _counts([classify_test_outcome(_g(s, "y_true"), _g(s, "pred_clean"), _g(s, "pred_poisoned"))
                           for s in test]),
    }
    if target is not None:
        other = 1 - target
        m["collateral_damage_rate"] = success_rate(test, "pred_poisoned", other, label="poisoned")
    if evaluation == "poisoning":
        m["attack_success_rate"] = attack_sr
        return m

    # ---- defensive replay ---------------------------------------------------------------------------
    m["undefended_attack_success_rate"] = attack_sr
    kept_mask = [not _g(s, "removed") for s in train]
    m["by_model"]["defended"] = model_block(test, train, "pred_defended", "pred_defended", "y_sanitized", kept_mask)
    # Training predictions of the clean-sanitised model are not stored, so its block is test-only.
    m["by_model"]["clean_sanitized"] = _clean_sanitized_block(test)
    dm = m["by_model"]["defended"]
    m["test_accuracy"] = dm["test_accuracy"]
    m["train_accuracy"] = dm["train_accuracy"]
    m["train_accuracy_observed"] = dm["train_accuracy_observed"]
    m["recall_class_0"], m["recall_class_1"] = dm["recall_class_0"], dm["recall_class_1"]
    m["undefended_test_accuracy"] = pm["test_accuracy"]
    if target is not None:
        m["undefended_target_class_recall"] = pm[f"recall_class_{target}"]
        m["target_class_recall"] = dm[f"recall_class_{target}"]
        m["collateral_damage_rate"] = success_rate(test, "pred_defended", 1 - target, label="defended")
    m["attack_success_rate"] = success_rate(test, "pred_defended", target, label="defended")
    m["defended_test_outcomes"] = {
        "reference": "clean model", "compared": "defended model", "population": "all test-split samples",
        "counts": _counts([classify_test_outcome(_g(s, "y_true"), _g(s, "pred_clean"), _g(s, "pred_defended"))
                           for s in test]),
    }
    broken = [s for s in test if _g(s, "pred_clean") == _g(s, "y_true") != _g(s, "pred_poisoned")]
    m["recovery_rate"] = ratio(
        sum(_g(s, "pred_defended") == _g(s, "y_true") for s in broken), len(broken),
        "test samples broken by the poisoning (clean model right, poisoned model wrong) that the defended "
        "model classifies correctly / test samples broken by the poisoning",
        "test-split samples broken by the poisoning")

    flagged = [s for s in train if _g(s, "flagged")]
    tp = sum(_g(s, "poisoned") for s in flagged)
    clean_rows = n_train - len(poisoned)
    m["flagged"] = ratio(len(flagged), n_train, "training samples whose label the defense flagged as suspicious / "
                         "training samples", "all training-split samples (poisoned labels)")
    m["detection_recall"] = ratio(tp, len(poisoned), "poisoned training samples that were flagged / poisoned "
                                  "training samples", "training samples whose label was flipped")
    m["detection_precision"] = ratio(tp, len(flagged), "flagged training samples that were actually poisoned / "
                                     "flagged training samples", "training samples flagged by the defense")
    m["false_positive_rate"] = ratio(len(flagged) - tp, clean_rows, "clean (unpoisoned) training samples that "
                                     "were flagged / clean training samples", "training samples whose label was not "
                                     "flipped")
    m["detection_counts"] = {k: sum(_g(s, "detection") == k for s in train)
                             for k in ("true_positive", "false_positive", "false_negative", "true_negative")}
    if (defense_cfg or {}).get("action") == "remove":
        m["removed"] = ratio(sum(bool(_g(s, "removed")) for s in train), n_train,
                             "training samples removed before retraining / training samples",
                             "all training-split samples")
    else:
        m["labels_repaired"] = ratio(
            sum(_g(s, "y_sanitized") == _g(s, "y_true") for s in poisoned), len(poisoned),
            "poisoned training samples whose sanitised label equals the true label / poisoned training samples",
            "training samples whose label was flipped")
        m["labels_corrupted"] = ratio(
            sum(_g(s, "y_sanitized") != _g(s, "y_true") for s in train if not _g(s, "poisoned")), clean_rows,
            "clean training samples whose sanitised label differs from the true label / clean training samples",
            "training samples whose label was not flipped")
    cs = m["by_model"]["clean_sanitized"]
    m["clean_data_test_accuracy"] = cs["test_accuracy"]
    m["clean_data_flag_rate"] = ratio(
        sum(bool(_g(s, "flagged_on_clean")) for s in train), n_train,
        "training samples flagged when the same defense is applied to the CLEAN labels / training samples "
        "(the defense's cost when there is no attack)", "all training-split samples (clean labels)")
    return m


def _clean_sanitized_block(test) -> dict[str, Any]:
    n = len(test)
    out = {"test_accuracy": ratio(sum(_g(s, "pred_clean_sanitized") == _g(s, "y_true") for s in test), n,
                                  "test samples predicted correctly by a model trained on the CLEAN labels after "
                                  "the same sanitisation / test samples",
                                  "all test-split samples (clean labels, clean inputs)")}
    for c in (0, 1):
        pop = [s for s in test if _g(s, "y_true") == c]
        out[f"recall_class_{c}"] = ratio(sum(_g(s, "pred_clean_sanitized") == c for s in pop), len(pop),
                                         f"class-{c} test samples predicted as class {c} / class-{c} test samples",
                                         f"test-split samples whose true class is {c}")
    return out


def consistency_errors(evaluation: str, train, test) -> list[str]:
    """Per-sample invariants that must hold in every ML02 record."""
    errs: list[str] = []
    for s in train:
        i = _g(s, "train_index")
        if _g(s, "poisoned") != (_g(s, "y_observed") != _g(s, "y_true")):
            errs.append(f"train {i}: poisoned flag ≠ (observed label ≠ true label)")
        if evaluation == "poisoning_replay":
            removed, y_san, flagged = _g(s, "removed"), _g(s, "y_sanitized"), _g(s, "flagged")
            if removed and (y_san is not None or not flagged):
                errs.append(f"train {i}: removed sample must be flagged and have no sanitised label")
            if not removed and y_san is None:
                errs.append(f"train {i}: kept sample has no sanitised label")
            if not flagged and not removed and y_san != _g(s, "y_observed"):
                errs.append(f"train {i}: unflagged label was changed by the defense")
            expect = {(True, True): "true_positive", (True, False): "false_positive",
                      (False, True): "false_negative", (False, False): "true_negative"}[(bool(flagged),
                                                                                          bool(_g(s, "poisoned")))]
            if _g(s, "detection") != expect:
                errs.append(f"train {i}: detection label {_g(s, 'detection')} ≠ {expect}")
        if errs and len(errs) > 20:
            break
    for s in test:
        if evaluation != "clean_training":
            exp = classify_test_outcome(_g(s, "y_true"), _g(s, "pred_clean"), _g(s, "pred_poisoned"))
            if _g(s, "outcome") != exp:
                errs.append(f"test {_g(s, 'test_index')}: outcome {_g(s, 'outcome')} ≠ {exp}")
        if evaluation == "poisoning_replay":
            exp = classify_test_outcome(_g(s, "y_true"), _g(s, "pred_clean"), _g(s, "pred_defended"))
            if _g(s, "defended_outcome") != exp:
                errs.append(f"test {_g(s, 'test_index')}: defended_outcome ≠ {exp}")
        if len(errs) > 20:
            break
    return errs
