"""Regression tests for the EFFECT of defenses (MLB-REG-002 root cause, see docs/regression-analysis-mlb-reg-002.md).

These tests catch a defense that silently stops working, which the former single-bound invariant
("defended success ≤ 30%") could not: that bound passed for a defense that changed nothing whenever the
undefended rate was already low, and failed for a working defense when numerical noise moved the number.
"""

import pytest
from pydantic import ValidationError

from mlbreak import defenses
from mlbreak.config import PROJECT_ROOT
from mlbreak.defenses import clear_model_caches
from mlbreak.regression import Invariant, PoisoningInvariant, load_cases, run_case
from mlbreak.schemas import AdversarialTrainingDefense, AttackConfig, AttackRequest, SampleResult
from mlbreak.stats import mcnemar_one_sided_p, paired_comparison

CASES = PROJECT_ROOT / "security_cases"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_mcnemar_exact_values():
    # P(X ≤ c), X ~ Bin(b + c, 1/2)
    assert mcnemar_one_sided_p(15, 3) == pytest.approx(0.0037689208984375)  # MLB-REG-002 on the CI kernel
    assert mcnemar_one_sided_p(2, 0) == pytest.approx(0.25)
    assert mcnemar_one_sided_p(1, 0) == pytest.approx(0.5)
    assert mcnemar_one_sided_p(0, 0) == 1.0
    assert mcnemar_one_sided_p(5, 5) == pytest.approx(0.623046875)  # no evidence of reduction
    assert mcnemar_one_sided_p(0, 4) == 1.0                         # defense made it worse


class _Rec:
    """Minimal ML01 record stand-in for paired_comparison."""
    family = None

    def __init__(self, outcomes):
        self.samples = [SampleResult(test_index=i, y_true=1, x_original=[0.0, 0.0], pred_original=1, p1_original=0.9,
                                     confidence_original=0.9, clean_correct=True, outcome=o,
                                     pred_adversarial=0 if o == "evaded" else 1)
                        for i, o in enumerate(outcomes)]


def test_paired_comparison_on_identical_samples():
    und = _Rec(["evaded"] * 6 + ["robust"] * 4)
    dfd = _Rec(["evaded"] * 2 + ["robust"] * 7 + ["evaded"])
    c = paired_comparison(und, dfd)
    assert c["n_paired"] == 10 and c["undefended_successes"] == 6 and c["defended_successes"] == 3
    assert (c["only_undefended_succeeded_b"], c["only_defended_succeeded_c"]) == (4, 1)
    assert c["absolute_reduction"] == pytest.approx(0.3) and c["relative_reduction"] == pytest.approx(0.5)
    assert c["mcnemar_p_value"] == pytest.approx(mcnemar_one_sided_p(4, 1))


def test_single_bound_cannot_tell_a_useless_defense_from_a_useful_one():
    """The old MLB-REG-002 invariant only bounded the defended rate. A defense that changes NOTHING passes it
    whenever the undefended rate is already ≤ 30%; the paired statistics do not."""
    same = ["evaded"] * 28 + ["robust"] * 72
    c = paired_comparison(_Rec(same), _Rec(same))
    assert c["defended_rate"] <= 0.30                      # old invariant: PASS
    assert c["absolute_reduction"] == 0.0                  # new: FAIL (≥ 0.05 required)
    assert c["mcnemar_p_value"] == 1.0                     # new: FAIL (≤ 0.05 required)


def test_undefined_statistics_are_none_not_zero():
    c = paired_comparison(_Rec(["robust"] * 5), _Rec(["robust"] * 5))
    assert c["relative_reduction"] is None  # undefended success rate is 0
    empty = paired_comparison(_Rec([]), _Rec([]))
    assert empty["n_paired"] == 0 and empty["absolute_reduction"] is None and empty["mcnemar_p_value"] is None


@pytest.mark.parametrize("cls", [Invariant, PoisoningInvariant])
def test_paired_invariant_contract(cls):
    base = {"id": "x", "kind": "security", "metric": "attack_success_rate", "op": ">=", "value": 0.05,
            "description": "d"}
    cls(**base, posture="paired", statistic="absolute_reduction")
    with pytest.raises(ValidationError):
        cls(**base, posture="paired")                                     # statistic missing
    with pytest.raises(ValidationError):
        cls(**base, posture="defended", statistic="absolute_reduction")  # statistic without pairing
    with pytest.raises(ValidationError):
        cls(**{**base, "metric": "clean_accuracy" if cls is Invariant else "test_accuracy"},
            posture="paired", statistic="mcnemar_p_value")             # only attack success is paired
    with pytest.raises(ValidationError):
        cls(**base, posture="paired", statistic="median_of_vibes")


# ---------------------------------------------------------------------------
# MLB-REG-002 detects a defense that stops working
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_models():
    clear_model_caches()
    yield
    clear_model_caches()


def _case(case_id):
    return next(c for c in load_cases(CASES) if c.id == case_id)


def test_mlb_reg_002_passes_with_the_real_defense(engine, fresh_models):
    r = run_case(engine, _case("MLB-REG-002"))
    assert r["passed"], [(i["id"], i["observed"]) for i in r["invariants"] if not i["passed"]]
    paired = [i for i in r["invariants"] if i["posture"] == "paired"]
    assert len(paired) == 3 and all(i["details"]["n_paired"] > 100 for i in paired)


def test_mlb_reg_002_fails_when_adversarial_training_is_neutralised(engine, fresh_models, monkeypatch):
    """Adversarial training that returns the undefended model must fail every comparative invariant, while the
    attack canary still passes (so the failure is attributed to the defense, not to a broken attack)."""
    monkeypatch.setattr(defenses, "adversarially_train",
                        lambda model_id, ds, seed, cfg, deadline=None: defenses.base_model(model_id, ds.spec.dataset_id, seed))
    r = run_case(engine, _case("MLB-REG-002"))
    by_id = {i["id"]: i for i in r["invariants"]}
    assert not r["passed"]
    assert by_id["attack-still-effective"]["passed"]
    for k in ("adaptive-reduction-significant", "adaptive-absolute-reduction", "adaptive-relative-reduction"):
        assert not by_id[k]["passed"], k
    assert by_id["adaptive-absolute-reduction"]["observed"] == 0.0
    assert by_id["adaptive-reduction-significant"]["observed"] == 1.0


def test_mlb_reg_002_fails_when_the_defense_is_too_weak(engine, fresh_models, monkeypatch):
    """A much weaker adversarial training (1 round, tiny ε, 1 PGD step) must not satisfy the case."""
    real = defenses.adversarially_train

    def weak(model_id, ds, seed, cfg, deadline=None):
        return real(model_id, ds, seed, AdversarialTrainingDefense(epsilon=0.01, steps=1, rounds=1), deadline)

    monkeypatch.setattr(defenses, "adversarially_train", weak)
    r = run_case(engine, _case("MLB-REG-002"))
    failed = {i["id"] for i in r["invariants"] if not i["passed"]}
    assert not r["passed"] and failed & {"adaptive-reduction-significant", "adaptive-absolute-reduction",
                                         "adaptive-relative-reduction"}


# ---------------------------------------------------------------------------
# The effect is not specific to one seed (measured: docs/regression-analysis-mlb-reg-002.md)
# ---------------------------------------------------------------------------
def test_adversarial_training_reduces_adaptive_success_across_seeds(engine):
    """Seeds 0–9, MLB-REG-002 configuration. Measured on macOS/Accelerate and Linux/OpenBLAS-Haswell:
    reduction > 0 in 20/20 seed×environment runs, mean absolute reduction 0.082–0.106, significant at
    α = 0.05 in 9/10 seeds per environment. The bounds below leave margin for other numerical stacks."""
    case = _case("MLB-REG-002")
    reductions, significant = [], 0
    for seed in range(10):
        req = AttackRequest(model_id=case.model_id, seed=seed, attack=AttackConfig(**case.attack.model_dump()),
                            samples=case.samples)
        u = engine.run_attack(req)
        a = engine.run_attack(req.model_copy(update={"target_defense": case.defense,
                                                     "parent_experiment_id": u.experiment_id}))
        c = paired_comparison(u, a)
        reductions.append(c["absolute_reduction"])
        significant += c["mcnemar_p_value"] <= 0.05
    assert all(r > 0 for r in reductions), reductions
    assert sum(reductions) / len(reductions) >= 0.05, reductions
    assert significant >= 7, (significant, reductions)

