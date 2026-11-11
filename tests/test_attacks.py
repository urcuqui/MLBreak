import numpy as np
import pytest
from pydantic import ValidationError

from mlbreak.attacks import (BudgetViolation, enforce_constraints, gradient_attack, lp_norm,
                             random_search_attack)
from mlbreak.data import load_dataset
from mlbreak.defenses import base_model
from mlbreak.schemas import AttackConfig

DS = load_dataset("moons-v1", 7)
LO, HI = DS.domain_low, DS.domain_high


def _rngs(n, seed=7):
    return [np.random.default_rng([seed, i]) for i in range(n)]


def _run(cfg, model_id="mlp-tanh-16x16"):
    m = base_model(model_id, "moons-v1", 7)
    X, y = DS.X_test, DS.y_test
    if cfg.attack == "random_search":
        return random_search_attack(m.predict, X, y, cfg, LO, HI, _rngs(len(X)))
    return gradient_attack(m, X, y, cfg, LO, HI, _rngs(len(X)))


CONFIGS = [
    AttackConfig(attack="fgsm", epsilon=0.25),
    AttackConfig(attack="pgd", norm="linf", epsilon=0.3),
    AttackConfig(attack="pgd", norm="l2", epsilon=0.4, random_start=True),
    AttackConfig(attack="pgd", norm="linf", epsilon=0.5, step_size=0.5),  # large step -> projection matters
    AttackConfig(attack="random_search", norm="linf", epsilon=0.3, query_budget=200),
    AttackConfig(attack="random_search", norm="l2", epsilon=0.3, query_budget=200),
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.attack}-{c.norm}-{c.epsilon}")
def test_perturbation_budget_and_domain_enforced(cfg):
    out = _run(cfg)
    norms = lp_norm(out.x_adv - DS.X_test, cfg.norm)
    assert np.all(norms <= cfg.epsilon + 1e-9)
    assert np.all(out.x_adv >= LO) and np.all(out.x_adv <= HI)
    assert enforce_constraints(DS.X_test, out.x_adv, cfg.epsilon, cfg.norm, LO, HI) <= 1 + 1e-9


def test_domain_projection_near_boundary():
    # A sample on the domain edge must not be pushed outside it.
    m = base_model("mlp-tanh-16x16", "moons-v1", 7)
    X = np.array([[LO[0], LO[1]], [HI[0], HI[1]]])
    out = gradient_attack(m, X, np.array([0, 1]), AttackConfig(epsilon=1.0), LO, HI, _rngs(2))
    assert np.all(out.x_adv >= LO) and np.all(out.x_adv <= HI)


def test_enforce_constraints_rejects_violations():
    x0 = np.zeros((2, 2))
    with pytest.raises(BudgetViolation):
        enforce_constraints(x0, x0 + 0.31, 0.3, "linf", LO, HI)
    with pytest.raises(BudgetViolation):
        enforce_constraints(x0, x0 + np.array([0.3, 0.3]), 0.3, "l2", LO, HI)
    with pytest.raises(BudgetViolation):
        enforce_constraints(np.array([[3.4, 0.0]]), np.array([[3.6, 0.0]]), 0.3, "linf", LO, HI)
    with pytest.raises(BudgetViolation):
        enforce_constraints(x0, np.full((2, 2), np.nan), 0.3, "linf", LO, HI)


def test_attack_config_constraints():
    with pytest.raises(ValidationError):
        AttackConfig(attack="fgsm", norm="l2")
    with pytest.raises(ValidationError):
        AttackConfig(epsilon=0)
    with pytest.raises(ValidationError):
        AttackConfig(epsilon=5.0)
    with pytest.raises(ValidationError):
        AttackConfig(steps=10_000)
    with pytest.raises(ValidationError):
        AttackConfig(query_budget=10**6)
    with pytest.raises(ValidationError):
        AttackConfig(attack="eval('x')")
    fgsm = AttackConfig(attack="fgsm", epsilon=0.2, steps=50)
    assert fgsm.steps == 1 and fgsm.step_size == 0.2


def test_query_budget_respected():
    cfg = AttackConfig(attack="random_search", epsilon=0.3, query_budget=120)
    out = _run(cfg)
    assert out.queries.max() <= 120


def test_black_box_only_sees_labels():
    calls = []

    def oracle(X):
        calls.append(len(X))
        return base_model("mlp-tanh-16x16", "moons-v1", 7).predict(X)

    cfg = AttackConfig(attack="random_search", epsilon=0.3, query_budget=100)
    out = random_search_attack(oracle, DS.X_test[:5], DS.y_test[:5], cfg, LO, HI, _rngs(5))
    assert sum(calls) == int(out.queries.sum())


def test_attacks_are_deterministic():
    for cfg in CONFIGS:
        a, b = _run(cfg), _run(cfg)
        assert np.array_equal(a.x_adv, b.x_adv)


def test_linear_model_attack_respects_theoretical_bound():
    """For a linear model the minimal L∞ evasion is |logit| / ||w||_1.

    Samples whose minimal perturbation exceeds ε cannot be evaded (any reported
    success would be fabricated); those comfortably within ε must be evaded.
    """
    m = base_model("logreg", "moons-v1", 7)
    X, y = DS.X_test, DS.y_test
    correct = m.predict(X) == y
    X, y = X[correct], y[correct]
    eps = 0.3
    out = gradient_attack(m, X, y, AttackConfig(epsilon=eps, steps=100), LO, HI, _rngs(len(X)))
    flipped = m.predict(out.x_adv) != y
    needed = np.abs(m.logit(X)) / np.abs(m.estimator.coef_[0]).sum()
    assert not np.any(flipped[needed > eps + 1e-9])
    inside = needed < 0.9 * eps
    assert inside.any() and np.all(flipped[inside])


def test_attack_failure_is_reported_not_faked():
    out = _run(AttackConfig(epsilon=1e-4))
    m = base_model("mlp-tanh-16x16", "moons-v1", 7)
    flipped = m.predict(out.x_adv) != DS.y_test
    clean_wrong = m.predict(DS.X_test) != DS.y_test
    # a negligible budget flips (almost) nothing beyond already-misclassified inputs
    assert (flipped & ~clean_wrong).sum() <= 1
    assert np.array_equal(out.reached_goal, m.predict(out.x_adv) != DS.y_test)


def test_larger_budget_is_at_least_as_effective():
    m = base_model("mlp-tanh-16x16", "moons-v1", 7)
    rates = []
    for eps in (0.05, 0.15, 0.3, 0.5):
        out = _run(AttackConfig(epsilon=eps))
        rates.append(np.mean(m.predict(out.x_adv) != DS.y_test))
    assert rates == sorted(rates)
