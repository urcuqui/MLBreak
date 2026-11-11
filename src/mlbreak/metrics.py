"""Evaluation metrics for evasion experiments, derived from per-sample evidence.

Every selected sample is assigned exactly ONE outcome (a partition of the selected population):

=========================  =====================================================================
outcome                    meaning
=========================  =====================================================================
``evaded``                 clean decision correct; adversarial decision is a *different class*
                           (not an abstention). The attack succeeded.
``abstained_adversarial``  clean decision correct; the system *abstained* on the adversarial
                           input. Neither a success nor a correct prediction.
``robust``                 clean decision correct; adversarial decision is still the true class.
``clean_misclassified``    clean decision is the wrong class; the sample is not attacked.
``clean_abstained``        the system abstained on the clean input; the sample is not attacked.
``not_perturbed``          clean decision correct but no adversarial input was evaluated (replay
                           only: the parent attack did not perturb this sample). Its decision
                           under the evaluation is its clean decision.
=========================  =====================================================================

The *attack-eligible* population is ``evaded + abstained_adversarial + robust`` (samples that were
correctly decided on their clean input and for which an adversarial input was evaluated).

Abstentions are never counted as correct predictions and never counted as attack successes.
Every ratio carries its numerator, denominator, population and definition; an empty denominator
yields ``value: None`` (undefined), never 0.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .attacks import ABSTAIN

OUTCOMES = ("evaded", "abstained_adversarial", "robust", "clean_misclassified", "clean_abstained", "not_perturbed")

# Metric keys whose value is fully determined by the per-sample outcomes (used by integrity checks).
OUTCOME_METRIC_KEYS = (
    "clean_accuracy_selected", "adversarial_accuracy", "robust_accuracy", "attack_success_rate",
    "abstention_rate_adversarial", "attack_resisted_rate", "abstention_rate_clean_selected",
    "abstained_on_adversarial", "adversarial_coverage", "adversarial_selective_accuracy",
)


class InconsistentSample(ValueError):
    """A per-sample record is internally contradictory (e.g. attacked although misclassified)."""


def ratio(num: int, den: int, definition: str, population: str) -> dict[str, Any]:
    return {"numerator": int(num), "denominator": int(den), "value": (num / den) if den else None,
            "population": population, "definition": definition}


def classify(s: Any) -> str:
    """Outcome of one :class:`~mlbreak.schemas.SampleResult` (works on models or plain dicts)."""
    get = s.get if isinstance(s, dict) else lambda k: getattr(s, k)
    y, clean, adv = get("y_true"), get("pred_original"), get("pred_adversarial")
    if adv is not None:
        if clean != y:
            raise InconsistentSample(f"test_index {get('test_index')}: adversarial result recorded for a sample "
                                     "whose clean decision was not correct")
        if adv == ABSTAIN:
            return "abstained_adversarial"
        return "robust" if adv == y else "evaded"
    if clean == ABSTAIN:
        return "clean_abstained"
    return "not_perturbed" if clean == y else "clean_misclassified"


def outcome_counts(samples: Iterable[Any]) -> dict[str, int]:
    c = Counter(classify(s) for s in samples)
    return {k: int(c.get(k, 0)) for k in OUTCOMES}


def outcome_metrics(samples: list[Any], can_abstain: bool) -> dict[str, Any]:
    """Selected-population metrics of an attack / replay evaluation, from per-sample outcomes."""
    c = outcome_counts(samples)
    n = len(samples)
    eligible = c["evaded"] + c["abstained_adversarial"] + c["robust"]
    final_correct = c["robust"] + c["not_perturbed"]
    clean_correct = eligible + c["not_perturbed"]
    sel = "selected test samples"
    elig = "attack-eligible samples (clean decision correct and an adversarial input was evaluated)"
    m: dict[str, Any] = {
        "outcomes": {
            "population": sel, "n_selected": n, "attack_eligible": eligible, "counts": c,
            "definition": "each selected sample has exactly one outcome; counts sum to n_selected",
        },
        "clean_accuracy_selected": ratio(clean_correct, n, "selected samples whose clean decision is the true class "
                                         "/ selected samples (abstentions are not correct)", sel),
        "adversarial_accuracy": ratio(
            final_correct, n,
            "selected samples whose decision under the evaluation is the true class / selected samples. "
            "The decision under the evaluation is the adversarial decision for attack-eligible samples and the "
            "clean decision otherwise. Abstentions and misclassifications are not correct.", sel),
        "robust_accuracy": ratio(
            c["robust"], n,
            "selected samples decided correctly on the clean input AND on an evaluated adversarial input / "
            "selected samples. Samples without an evaluated adversarial input are not counted as robust.", sel),
        "attack_success_rate": ratio(
            c["evaded"], eligible,
            "attack-eligible samples whose adversarial input is decided as a different class (not an abstention) "
            "/ attack-eligible samples", elig),
        "abstention_rate_adversarial": ratio(
            c["abstained_adversarial"], eligible,
            "attack-eligible samples on whose adversarial input the system abstained / attack-eligible samples", elig),
        "attack_resisted_rate": ratio(
            c["robust"], eligible,
            "attack-eligible samples whose adversarial input is still decided as the true class / attack-eligible "
            "samples. attack_success_rate + abstention_rate_adversarial + attack_resisted_rate = 1", elig),
        "abstention_rate_clean_selected": ratio(
            c["clean_abstained"], n, "selected samples on whose clean input the system abstained / selected samples",
            sel),
    }
    if can_abstain:
        accepted = n - c["clean_abstained"] - c["abstained_adversarial"]
        m["abstained_on_adversarial"] = {**m["abstention_rate_adversarial"],
                                         "definition": "alias of abstention_rate_adversarial (kept for API "
                                                       "compatibility)"}
        m["adversarial_coverage"] = ratio(
            accepted, n, "selected samples whose decision under the evaluation is not an abstention / selected samples",
            sel)
        m["adversarial_selective_accuracy"] = ratio(
            final_correct, accepted,
            "selected samples decided correctly under the evaluation / selected samples whose decision under the "
            "evaluation is not an abstention", "selected samples with a non-abstained decision under the evaluation")
    return m


def clean_metrics(labels, y_true, can_abstain: bool) -> dict[str, Any]:
    """Clean metrics over the FULL test split (all test samples, clean inputs)."""
    n = len(y_true)
    correct = int(sum(int(a) == int(b) for a, b in zip(labels, y_true)))
    abstained = int(sum(int(a) == ABSTAIN for a in labels))
    pop = "all test-split samples (clean inputs)"
    out = {
        "clean_accuracy": ratio(correct, n, "correct decisions / all test samples (abstentions count as not correct)",
                                pop),
        "abstention_rate_clean": ratio(abstained, n, "clean inputs the system abstained on / all test samples", pop),
    }
    if can_abstain:
        out["coverage"] = ratio(n - abstained, n, "non-abstained decisions / all test samples", pop)
        out["selective_accuracy"] = ratio(correct, n - abstained, "correct decisions / non-abstained decisions",
                                          "test-split samples with a non-abstained clean decision")
    return out


def consistency_errors(metrics: dict[str, Any]) -> list[str]:
    """Arithmetic invariants every outcome-metric block must satisfy."""
    errs: list[str] = []
    o = metrics.get("outcomes")
    if not o:
        return errs
    c, n, e = o["counts"], o["n_selected"], o["attack_eligible"]
    if sum(c.values()) != n:
        errs.append("outcome counts do not sum to n_selected")
    if c["evaded"] + c["abstained_adversarial"] + c["robust"] != e:
        errs.append("attack-eligible count does not match its outcomes")
    parts = [metrics[k]["numerator"] for k in ("attack_success_rate", "abstention_rate_adversarial", "attack_resisted_rate")]
    if sum(parts) != e:
        errs.append("success + adversarial abstention + resisted ≠ attack-eligible")
    if metrics["robust_accuracy"]["numerator"] > metrics["adversarial_accuracy"]["numerator"]:
        errs.append("robust accuracy exceeds adversarial accuracy")
    if metrics["adversarial_accuracy"]["numerator"] > metrics["clean_accuracy_selected"]["numerator"]:
        errs.append("adversarial accuracy exceeds clean accuracy on the selected samples")
    return errs
