"""Metric definitions: misclassification vs abstention vs correct prediction, and empty populations."""

import numpy as np
import pytest

from mlbreak import experiments as exp_mod
from mlbreak.attacks import AttackOutput
from mlbreak.defenses import build_system
from mlbreak.data import load_dataset
from mlbreak.metrics import InconsistentSample, classify, clean_metrics, consistency_errors, outcome_metrics
from mlbreak.schemas import (AttackConfig, AttackRequest, ConfidenceRejectionDefense, DefendRequest, SampleResult,
                             SampleSelection)

A = -1  # ABSTAIN


def S(y, clean, adv=None):
    return SampleResult(test_index=0, y_true=y, x_original=[0.0, 0.0], pred_original=clean, p1_original=0.5,
                        confidence_original=0.5, clean_correct=clean == y,
                        pred_adversarial=adv, x_adversarial=None if adv is None else [0.1, 0.1],
                        attack_success=None if adv is None else (adv != y and adv != A))


# One sample of every outcome (true class 1 throughout).
EVERY = [S(1, 1, 0), S(1, 1, A), S(1, 1, 1), S(1, 0), S(1, A), S(1, 1)]


def val(m, k):
    return (m[k]["numerator"], m[k]["denominator"])


def test_outcome_partition_covers_every_case():
    assert [classify(s) for s in EVERY] == ["evaded", "abstained_adversarial", "robust", "clean_misclassified",
                                            "clean_abstained", "not_perturbed"]
    with pytest.raises(InconsistentSample):
        classify(S(1, 0, 1))  # adversarial result on a misclassified clean input is contradictory


def test_exact_metric_values_with_every_outcome():
    m = outcome_metrics(EVERY, can_abstain=True)
    assert m["outcomes"]["counts"] == {k: 1 for k in m["outcomes"]["counts"]}
    assert m["outcomes"]["attack_eligible"] == 3
    assert val(m, "clean_accuracy_selected") == (4, 6)       # evaded, abstained_adv, robust, not_perturbed
    assert val(m, "adversarial_accuracy") == (2, 6)          # robust + not_perturbed
    assert val(m, "robust_accuracy") == (1, 6)               # robust only
    assert val(m, "attack_success_rate") == (1, 3)           # evaded / eligible
    assert val(m, "abstention_rate_adversarial") == (1, 3)
    assert val(m, "attack_resisted_rate") == (1, 3)
    assert val(m, "abstention_rate_clean_selected") == (1, 6)
    assert val(m, "adversarial_coverage") == (4, 6)          # 6 − clean abstention − adversarial abstention
    assert val(m, "adversarial_selective_accuracy") == (2, 4)
    assert val(m, "abstained_on_adversarial") == val(m, "abstention_rate_adversarial")
    for k, v in m.items():
        if isinstance(v, dict) and "numerator" in v:
            assert v["population"] and v["definition"], k
    assert consistency_errors(m) == []


def test_abstention_is_neither_success_nor_correct():
    """Regression for the original bug: (clean correct − successes) counted abstentions as correct."""
    samples = [S(1, 1, A)] * 3 + [S(0, 0, 1)] + [S(0, 0, 0)]
    m = outcome_metrics(samples, can_abstain=True)
    assert val(m, "attack_success_rate") == (1, 5)
    assert val(m, "abstention_rate_adversarial") == (3, 5)
    assert val(m, "adversarial_accuracy") == (1, 5)
    buggy = sum(s.clean_correct for s in samples) - sum(bool(s.attack_success) for s in samples)
    assert buggy == 4 != m["adversarial_accuracy"]["numerator"]


def test_misclassification_counts_as_success_and_not_correct():
    m = outcome_metrics([S(0, 0, 1), S(1, 1, 0)], can_abstain=False)
    assert val(m, "attack_success_rate") == (2, 2)
    assert val(m, "adversarial_accuracy") == (0, 2) and val(m, "robust_accuracy") == (0, 2)
    assert "adversarial_coverage" not in m  # only reported for systems that can abstain


def test_zero_eligible_samples_gives_undefined_rates_not_zero():
    m = outcome_metrics([S(1, A), S(0, 1), S(1, A)], can_abstain=True)
    assert m["outcomes"]["attack_eligible"] == 0
    for k in ("attack_success_rate", "abstention_rate_adversarial", "attack_resisted_rate"):
        assert m[k]["denominator"] == 0 and m[k]["value"] is None
    assert m["adversarial_accuracy"]["value"] == 0.0 and m["robust_accuracy"]["value"] == 0.0
    assert consistency_errors(m) == []


def test_clean_metrics_distinguish_abstention_from_error():
    c = clean_metrics([1, A, 0, 0], [1, 1, 1, 0], can_abstain=True)
    assert val(c, "clean_accuracy") == (2, 4)
    assert val(c, "abstention_rate_clean") == (1, 4)
    assert val(c, "coverage") == (3, 4) and val(c, "selective_accuracy") == (2, 3)


def test_correct_classification_after_perturbation(engine, monkeypatch):
    """An attack that changes nothing must yield ASR 0 and every eligible sample robust."""
    def null_attack(model, X0, y, cfg, *a, **k):
        return AttackOutput(np.array(X0, dtype=float), np.zeros(len(X0), bool), np.ones(len(X0), np.int64), 0.0)

    monkeypatch.setattr(exp_mod, "gradient_attack", null_attack)
    rec = engine.run_attack(AttackRequest(samples=SampleSelection(max_samples=40)))
    m = rec.metrics
    assert rec.status == "completed"
    assert m["attack_success_rate"]["numerator"] == 0 and m["attack_success_rate"]["value"] == 0.0
    assert m["robust_accuracy"]["numerator"] == m["adversarial_accuracy"]["numerator"] \
        == m["clean_accuracy_selected"]["numerator"] == m["outcomes"]["attack_eligible"]
    assert all(s.outcome in ("robust", "clean_misclassified") for s in rec.samples)


def test_replay_against_rejection_reports_abstentions_separately(engine, attack_record):
    rec = engine.run_defend(DefendRequest(parent_experiment_id=attack_record.experiment_id,
                                          defense=ConfidenceRejectionDefense(threshold=0.8)))
    m, c = rec.metrics, rec.metrics["outcomes"]["counts"]
    assert c["abstained_adversarial"] > 0
    assert m["adversarial_accuracy"]["numerator"] == c["robust"] + c["not_perturbed"]
    assert m["attack_success_rate"]["numerator"] == c["evaded"]
    # The old formula would have counted every adversarial abstention as a correct prediction.
    buggy = m["clean_accuracy_selected"]["numerator"] - m["attack_success_rate"]["numerator"]
    assert buggy - m["adversarial_accuracy"]["numerator"] == c["abstained_adversarial"]
    assert m["abstained_on_adversarial"]["numerator"] == c["abstained_adversarial"]
    # Coverage under the replay: selected samples whose decision (adversarial if evaluated, clean otherwise)
    # is not an abstention.
    n = m["outcomes"]["n_selected"]
    cov = m["adversarial_coverage"]
    assert cov["denominator"] == n
    assert cov["numerator"] == n - c["clean_abstained"] - c["abstained_adversarial"]
    # The parent attacked the whole test split, so the selected population IS the clean-coverage population:
    # the clean coverage of these samples must agree with the full-split `coverage` metric ...
    assert n == m["coverage"]["denominator"]
    assert n - c["clean_abstained"] == m["coverage"]["numerator"]
    # ... and adversarial abstentions (> 0, asserted above) strictly reduce coverage below it.
    assert cov["numerator"] < m["coverage"]["numerator"]
    assert cov["value"] < m["coverage"]["value"]
    assert consistency_errors(m) == []


def test_zero_eligible_samples_end_to_end(engine):
    """Every selected clean input is abstained on: the experiment completes with undefined rates."""
    ds = load_dataset("moons-v1", 7)
    system = build_system("mlp-tanh-16x16", "moons-v1", 7, ConfidenceRejectionDefense(threshold=0.99))
    abstained = [int(i) for i in np.flatnonzero(system.decide(ds.X_test).labels == A)][:6]
    assert abstained
    rec = engine.run_attack(AttackRequest(attack=AttackConfig(random_start=True),
                                          samples=SampleSelection(indices=abstained),
                                          target_defense=ConfidenceRejectionDefense(threshold=0.99)))
    m = rec.metrics
    assert rec.status == "completed"
    assert m["outcomes"]["counts"]["clean_abstained"] == len(abstained)
    assert m["attack_success_rate"] == {**m["attack_success_rate"], "numerator": 0, "denominator": 0, "value": None}
    assert m["adversarial_accuracy"]["value"] == 0.0
    assert m["perturbation"]["attacked"] == {"n": 0}
    bb = engine.run_attack(AttackRequest(attack=AttackConfig(attack="random_search", query_budget=50),
                                         samples=SampleSelection(indices=abstained),
                                         target_defense=ConfidenceRejectionDefense(threshold=0.99)))
    assert bb.status == "completed" and bb.metrics["queries"]["total"] == 0
