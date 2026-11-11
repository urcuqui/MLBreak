"""Execution safety: deadlines over training, concurrency gates and attack budgets."""

import time

import numpy as np
import pytest

from mlbreak import defenses
from mlbreak.attacks import ExperimentTimeout
from mlbreak.config import LIMITS, PROJECT_ROOT
from mlbreak.data import load_dataset
from mlbreak.experiments import EngineBusy, ExperimentEngine
from mlbreak.regression import run_suite
from mlbreak.schemas import (AdversarialTrainingDefense, AttackConfig, AttackRequest, BaselineRequest,
                             BoundaryRequest, ConfidenceRejectionDefense, SampleSelection)

AT = AdversarialTrainingDefense(epsilon=0.2, rounds=2, steps=10)


def _key(cfg=AT):
    import json
    return ("mlp-tanh-16x16", "moons-v1", 7, json.dumps(cfg.model_dump(), sort_keys=True))


def test_adversarial_training_honours_deadline_and_never_caches_partial_models():
    defenses.clear_model_caches()
    ds = load_dataset("moons-v1", 7)
    with pytest.raises(ExperimentTimeout):
        defenses.adversarially_train("mlp-tanh-16x16", ds, 7, AT, deadline=time.monotonic() - 1)
    with pytest.raises(ExperimentTimeout):
        defenses.defended_model("mlp-tanh-16x16", "moons-v1", 7, AT, deadline=time.monotonic() - 1)
    assert _key() not in defenses._defended_cache
    m = defenses.defended_model("mlp-tanh-16x16", "moons-v1", 7, AT, deadline=time.monotonic() + 60)
    assert _key() in defenses._defended_cache and "round 2/2" in m.training_note


def test_adversarial_training_timeout_marks_experiment_failed_then_recovers(store):
    defenses.clear_model_caches()
    defenses.base_model("mlp-tanh-16x16", "moons-v1", 7)  # warm only the base model
    t0 = time.monotonic()
    rec = ExperimentEngine(store, time_limit=0.05).run_baseline(BaselineRequest(defense=AT))
    assert rec.status == "failed" and "ExperimentTimeout" in rec.error
    assert time.monotonic() - t0 < 10, "the deadline must interrupt adversarial training"
    assert _key() not in defenses._defended_cache
    ok = ExperimentEngine(store).run_baseline(BaselineRequest(defense=AT))
    assert ok.status == "completed"


def test_timeout_in_attack_and_in_defended_attack(store):
    eng = ExperimentEngine(store, time_limit=-1.0)
    for req in (AttackRequest(samples=SampleSelection(max_samples=10)),
                AttackRequest(samples=SampleSelection(max_samples=10), target_defense=AT)):
        rec = eng.run_attack(req)
        assert rec.status == "failed" and "ExperimentTimeout" in rec.error and rec.samples == []
        assert rec.reproducibility["results_sha256"] is None


def test_boundary_and_dataset_are_gated_and_time_limited(store, client):
    eng = ExperimentEngine(store, max_concurrent=1)
    assert eng._gate.acquire(blocking=False)
    try:
        with pytest.raises(EngineBusy):
            eng.boundary(BoundaryRequest(resolution=20))
        with pytest.raises(EngineBusy):
            eng.dataset_view("ml01-moons-evasion", "logreg", 7)
    finally:
        eng._gate.release()
    with pytest.raises(ExperimentTimeout):
        ExperimentEngine(store, time_limit=-1.0).boundary(BoundaryRequest(resolution=20, defense=AT))
    # API mapping: busy → 429, timeout → 504.
    engine = client.app.state.engine
    holders = [engine._gate.acquire(blocking=False) for _ in range(LIMITS.max_concurrent_experiments)]
    try:
        assert all(holders)
        assert client.post("/api/boundary", json={"resolution": 20}).status_code == 429
        assert client.get("/api/scenarios/ml01-moons-evasion/dataset").status_code == 429
        assert client.post("/api/experiments/baseline", json={}).status_code == 429
    finally:
        for _ in holders:
            engine._gate.release()
    engine.time_limit = -1.0
    r = client.post("/api/boundary", json={"resolution": 20})
    assert r.status_code == 504 and "wall-clock" in r.json()["detail"]


def test_only_one_regression_suite_at_a_time(engine, client):
    assert engine.suite_lock.acquire(blocking=False)
    try:
        with pytest.raises(EngineBusy):
            run_suite(engine, PROJECT_ROOT / "security_cases")
    finally:
        engine.suite_lock.release()
    eng = client.app.state.engine
    assert eng.suite_lock.acquire(blocking=False)
    try:
        assert client.post("/api/regression/run").status_code == 429
    finally:
        eng.suite_lock.release()


def test_suite_deadline_fails_cases_instead_of_passing_them(engine):
    report = run_suite(engine, PROJECT_ROOT / "security_cases", time_limit=-1.0)
    assert report["ok"] is False and report["failed"] == report["total"] >= 4
    assert all("wall-clock" in c["error"] for c in report["cases"])
    assert all(not i["passed"] and i["observed"] is None for c in report["cases"] for i in c["invariants"])


def test_black_box_query_budget_is_enforced_per_sample(engine):
    rec = engine.run_attack(AttackRequest(attack=AttackConfig(attack="random_search", query_budget=60),
                                          samples=SampleSelection(max_samples=40)))
    assert rec.status == "completed"
    assert all(s.queries <= 60 for s in rec.samples if s.queries is not None)
    assert rec.metrics["queries"]["total"] <= 60 * rec.metrics["outcomes"]["attack_eligible"]


def test_white_box_budget_and_step_limits(engine):
    cfg = AttackConfig(attack="pgd", epsilon=0.25, steps=15, random_start=True)
    rec = engine.run_attack(AttackRequest(attack=cfg, samples=SampleSelection(max_samples=40)))
    for s in rec.samples:
        if s.x_adversarial is not None:
            assert s.perturbation_linf <= 0.25 + 1e-9
            assert s.queries <= 1 + 2 * 15
    lo, hi = load_dataset("moons-v1", 7).domain_low, load_dataset("moons-v1", 7).domain_high
    adv = np.array([s.x_adversarial for s in rec.samples if s.x_adversarial is not None])
    assert np.all(adv >= lo - 1e-9) and np.all(adv <= hi + 1e-9)
    assert rec.metrics["perturbation"]["max_budget_ratio"] <= 1.0


def test_request_limits_are_enforced_by_schema(client):
    over = {"attack": {"epsilon": LIMITS.max_epsilon * 2}}
    assert client.post("/api/experiments/attack", json=over).status_code == 422
    assert client.post("/api/experiments/attack", json={"attack": {"steps": LIMITS.max_steps + 1}}).status_code == 422
    assert client.post("/api/experiments/attack",
                       json={"attack": {"query_budget": LIMITS.max_query_budget + 1}}).status_code == 422
    assert client.post("/api/experiments/baseline", json={
        "defense": {"kind": "adversarial_training", "rounds": LIMITS.max_adv_training_rounds + 1}}).status_code == 422
    assert client.post("/api/boundary", json={"resolution": LIMITS.max_grid_resolution + 1}).status_code == 422
    rej = ConfidenceRejectionDefense(threshold=0.9)
    assert rej.threshold == 0.9
