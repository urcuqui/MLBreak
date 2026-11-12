"""Metrics for spam-filter evasion, derived ONLY from stored per-sample evidence.

Every selected message has exactly one outcome:

=========================  ===================================================================================
``evaded``                 clean decision correct; the adversarial message is decided as the TARGET class with
                           probability ≥ ``min_target_confidence`` AND satisfies every declared constraint
``failed``                 clean decision correct; the search ended within its constraints without success
``budget_exhausted``       clean decision correct; the per-sample query budget ran out before success
``skipped_misclassified``  the clean message is already misclassified — nothing to evade (TextAttack: SKIPPED)
``not_perturbed``          replay only: clean decision correct, but the parent produced no adversarial message
=========================  ===================================================================================

Attacked (eligible) messages = evaded + failed + budget_exhausted. A label change alone is not success: an
adversarial message that changes the label but violates a constraint is a FAILED EXPERIMENT (the engine
raises), never an ``evaded`` outcome.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Sequence

from ..metrics import ratio

OUTCOMES = ("evaded", "failed", "budget_exhausted", "skipped_misclassified", "not_perturbed")
ATTACKED = ("evaded", "failed", "budget_exhausted")

INVARIANT_METRICS = (
    "attack_success_rate", "accuracy_under_attack", "clean_accuracy", "spam_recall", "spam_precision",
    "ham_false_positive_rate", "budget_exhaustion_rate", "constraint_violation_rate",
)


def _g(s: Any, k: str):
    return s.get(k) if isinstance(s, dict) else getattr(s, k)


def clean_metrics(preds: Sequence[Any]) -> dict[str, Any]:
    pop = "all test-split messages (clean)"
    n = len(preds)
    correct = sum(_g(p, "pred") == _g(p, "y_true") for p in preds)
    spam = [p for p in preds if _g(p, "y_true") == 1]
    ham = [p for p in preds if _g(p, "y_true") == 0]
    flagged = [p for p in preds if _g(p, "pred") == 1]
    return {
        "clean_accuracy": ratio(correct, n, "messages classified as their true label / test messages", pop),
        "spam_recall": ratio(sum(_g(p, "pred") == 1 for p in spam), len(spam),
                             "spam messages classified as spam / spam messages", "test-split spam messages"),
        "spam_precision": ratio(sum(_g(p, "y_true") == 1 for p in flagged), len(flagged),
                                "messages classified as spam that are spam / messages classified as spam",
                                "test-split messages classified as spam"),
        "ham_false_positive_rate": ratio(sum(_g(p, "pred") == 1 for p in ham), len(ham),
                                         "ham messages classified as spam / ham messages", "test-split ham messages"),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(mean(xs)) if xs else None


def derive_text_metrics(samples: Sequence[Any], test_predictions: Sequence[Any]) -> dict[str, Any]:
    m = clean_metrics(test_predictions)
    if not samples:
        return m
    c = Counter(_g(s, "outcome") for s in samples)
    counts = {k: int(c.get(k, 0)) for k in OUTCOMES}
    n = len(samples)
    attacked = sum(counts[k] for k in ATTACKED)
    sel = "selected test messages (the goal's source class)"
    att = "attacked messages (clean decision correct, adversarial message evaluated)"
    m["outcomes"] = {"population": sel, "n_selected": n, "attacked": attacked, "counts": counts,
                     "definition": "each selected message has exactly one outcome; counts sum to n_selected"}
    m["clean_accuracy_selected"] = ratio(attacked + counts["not_perturbed"], n,
                                         "selected messages whose clean decision is correct / selected messages", sel)
    m["attack_success_rate"] = ratio(counts["evaded"], attacked,
                                     "attacked messages evaded within ALL constraints / attacked messages", att)
    m["accuracy_under_attack"] = ratio(
        counts["failed"] + counts["budget_exhausted"] + counts["not_perturbed"], n,
        "selected messages still classified correctly after the attack / selected messages (TextAttack's "
        "'accuracy under attack'; skipped messages count as incorrect)", sel)
    m["budget_exhaustion_rate"] = ratio(counts["budget_exhausted"], attacked,
                                        "attacked messages whose query budget ran out / attacked messages", att)
    with_text = [s for s in samples if _g(s, "text_adversarial") is not None and _g(s, "constraints") is not None]
    m["constraint_violation_rate"] = ratio(
        sum(not all(_g(s, "constraints").values()) for s in with_text), len(with_text),
        "adversarial messages violating at least one declared constraint / adversarial messages (must be 0 for "
        "evaded messages; the engine refuses to record an 'evaded' outcome that violates a constraint)",
        "selected messages with an adversarial message")
    evaded = [s for s in samples if _g(s, "outcome") == "evaded"]
    attacked_s = [s for s in samples if _g(s, "outcome") in ATTACKED]
    m["effort"] = {
        "mean_queries_attacked": _mean(_g(s, "queries") for s in attacked_s),
        "mean_queries_evaded": _mean(_g(s, "queries") for s in evaded),
        "max_queries": max((_g(s, "queries") or 0 for s in samples), default=0),
        "mean_edit_distance_evaded": _mean(_g(s, "edit_distance") for s in evaded),
        "mean_char_ratio_evaded": _mean(_g(s, "char_ratio") for s in evaded),
        "mean_modified_word_ratio_evaded": _mean(
            (_g(s, "words_modified") / _g(s, "words_modifiable")) if _g(s, "words_modifiable") else None for s in evaded),
        "definition": "means over the stated population; None when the population is empty",
    }
    return m


def consistency_errors(samples: Sequence[Any], min_conf: float, target: int) -> list[str]:
    errs = []
    for s in samples:
        o, i = _g(s, "outcome"), _g(s, "test_index")
        if o in ATTACKED:
            if _g(s, "pred_original") != _g(s, "y_true"):
                errs.append(f"test {i}: attacked although the clean decision was wrong")
            p = _g(s, "p_spam_adversarial")
            if p is None or _g(s, "text_adversarial") is None:
                errs.append(f"test {i}: attacked message without adversarial evidence")
                continue
            if _g(s, "pred_adversarial") != int(p >= 0.5):
                errs.append(f"test {i}: adversarial prediction inconsistent with its probability")
            reached = (p if target == 1 else 1 - p) >= min_conf
            ok = all((_g(s, "constraints") or {"missing": False}).values())
            if (o == "evaded") != (reached and ok):
                errs.append(f"test {i}: outcome {o} inconsistent with goal/constraints")
        elif o == "skipped_misclassified" and _g(s, "pred_original") == _g(s, "y_true"):
            errs.append(f"test {i}: skipped although the clean decision was correct")
        elif o == "not_perturbed" and _g(s, "pred_original") != _g(s, "y_true"):
            errs.append(f"test {i}: not_perturbed requires a correct clean decision")
        if len(errs) > 20:
            break
    return errs
