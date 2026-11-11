import json

import numpy as np
import pytest

from mlbreak import experiments as exp_mod
from mlbreak.attacks import AttackOutput
from mlbreak.data import load_dataset
from mlbreak.evidence import ExperimentNotFound
from mlbreak.experiments import EngineBusy, ExperimentEngine, InvalidExperiment
from mlbreak.schemas import (AdversarialTrainingDefense, AttackConfig, AttackRequest, BaselineRequest,
                             ConfidenceRejectionDefense, DefendRequest, ExperimentRecord, SampleSelection)

REQUIRED = ["experiment_id", "scenario_id", "owasp", "attacker_capabilities", "model", "dataset", "seed",
            "attack_config", "defense_config", "samples", "metrics", "status", "reproducibility"]


def test_baseline_evidence(engine):
    rec = engine.run_baseline(BaselineRequest())
    assert rec.status == "completed" and rec.evaluation == "clean" and rec.intentionally_vulnerable
    assert len(rec.samples) == 150
    acc = rec.metrics["clean_accuracy"]
    assert acc["denominator"] == 150 and acc["numerator"] == sum(s.clean_correct for s in rec.samples)


def test_attack_evidence_contains_required_fields(engine, attack_record, store):
    data = json.loads(store.json_path(attack_record.experiment_id).read_text())
    for key in REQUIRED:
        assert key in data, key
    assert data["owasp"]["id"] == "ML01:2023" and data["owasp"]["taxonomy_version"] == "2023"
    assert data["model"]["artifact_sha256"] and data["dataset"]["split_id"]
    assert data["dataset"]["train_sha256"] and data["dataset"]["test_sha256"]
    assert data["attacker_capabilities"]["knowledge"] == "white-box"
    for s in data["samples"]:
        assert s["x_original"] is not None and s["pred_original"] in (0, 1)
    m = data["metrics"]
    for key in ("clean_accuracy", "adversarial_accuracy", "attack_success_rate", "perturbation"):
        assert key in m
    assert m["attack_success_rate"]["definition"]
    assert data["reproducibility"]["results_sha256"] and data["reproducibility"]["numpy"]


def test_evidence_roundtrip_and_markdown(attack_record, store):
    loaded = store.load(attack_record.experiment_id)
    assert loaded == ExperimentRecord.model_validate_json(loaded.model_dump_json())
    assert loaded.model_dump() == attack_record.model_dump()
    md = store.markdown(attack_record.experiment_id)
    assert "ML01:2023" in md and "attack_success_rate" in md and attack_record.experiment_id in md
    csv = store.samples_csv(attack_record.experiment_id)
    assert csv.splitlines()[0].startswith("test_index") and len(csv.splitlines()) == 151


def test_success_rate_matches_samples(attack_record):
    s = attack_record.samples
    asr = attack_record.metrics["attack_success_rate"]
    assert asr["denominator"] == sum(x.clean_correct for x in s)
    assert asr["numerator"] == sum(bool(x.attack_success) for x in s)
    for x in s:
        if x.attack_success:
            assert x.clean_correct and x.pred_adversarial != x.y_true and x.pred_adversarial != -1
        if x.x_adversarial is not None:
            assert x.perturbation_linf <= 0.3 + 1e-9
    adv = attack_record.metrics["adversarial_accuracy"]
    assert adv["numerator"] == asr["denominator"] - asr["numerator"]


def test_replay_is_identical(engine, attack_record):
    r = engine.replay(attack_record.experiment_id)
    assert r["identical"], r["checks"]
    new = engine.store.load(r["replay_experiment_id"])
    assert new.replay_of == attack_record.experiment_id
    assert [s.x_adversarial for s in new.samples] == [s.x_adversarial for s in attack_record.samples]


def test_replay_of_baseline_and_black_box(engine):
    b = engine.run_baseline(BaselineRequest(model_id="logreg", seed=3))
    assert engine.replay(b.experiment_id)["identical"]
    bb = engine.run_attack(AttackRequest(attack=AttackConfig(attack="random_search", query_budget=100),
                                         samples=SampleSelection(max_samples=40)))
    assert engine.replay(bb.experiment_id)["identical"]


def test_defend_replays_exact_parent_perturbations(engine, attack_record):
    d = engine.run_defend(DefendRequest(parent_experiment_id=attack_record.experiment_id,
                                        defense=AdversarialTrainingDefense()))
    assert d.mode == "defend" and d.evaluation == "replay" and not d.intentionally_vulnerable
    assert d.parent_experiment_id == attack_record.experiment_id
    assert d.attacker_capabilities["knows_defense"] is False
    parent = {s.test_index: s.x_adversarial for s in attack_record.samples}
    replayed = [s for s in d.samples if s.x_adversarial is not None]
    assert replayed
    for s in replayed:
        assert s.x_adversarial == parent[s.test_index]  # unchanged perturbation
    assert d.target_system["model_artifact_sha256"] != attack_record.model["artifact_sha256"]
    cmp = d.metrics["comparison"]["attack_success_rate"]
    assert cmp["defended"]["value"] < cmp["undefended"]["value"]
    assert engine.replay(d.experiment_id)["identical"]


def test_adaptive_attack_is_separate_experiment(engine, attack_record):
    a = engine.run_attack(AttackRequest(target_defense=AdversarialTrainingDefense(),
                                        parent_experiment_id=attack_record.experiment_id))
    assert a.evaluation == "adaptive_attack" and a.attacker_capabilities["knows_defense"]
    assert a.experiment_id != attack_record.experiment_id
    assert a.parent_experiment_id == attack_record.experiment_id
    new = {s.test_index: s.x_adversarial for s in a.samples}
    old = {s.test_index: s.x_adversarial for s in attack_record.samples}
    assert any(new[k] != old[k] for k in new if new[k] and old[k])  # freshly generated


def test_rejection_replay_vs_adaptive(engine, attack_record):
    rep = engine.run_defend(DefendRequest(parent_experiment_id=attack_record.experiment_id,
                                          defense=ConfidenceRejectionDefense(threshold=0.8)))
    ada = engine.run_attack(AttackRequest(target_defense=ConfidenceRejectionDefense(threshold=0.8)))
    assert rep.metrics["coverage"]["value"] < 1.0
    assert ada.attacker_capabilities["effective_confidence_kappa"] > 0
    # The adaptive attacker targets confidence above τ, so its successes are accepted, not rejected.
    for s in ada.samples:
        if s.attack_success:
            assert s.confidence_adversarial >= 0.8
    assert ada.metrics["attack_success_rate"]["value"] > rep.metrics["attack_success_rate"]["value"]


def test_defend_requires_undefended_attack_parent(engine, attack_record):
    b = engine.run_baseline(BaselineRequest())
    with pytest.raises(InvalidExperiment):
        engine.run_defend(DefendRequest(parent_experiment_id=b.experiment_id,
                                        defense=ConfidenceRejectionDefense()))
    ada = engine.run_attack(AttackRequest(target_defense=ConfidenceRejectionDefense()))
    with pytest.raises(InvalidExperiment):
        engine.run_defend(DefendRequest(parent_experiment_id=ada.experiment_id,
                                        defense=ConfidenceRejectionDefense()))
    with pytest.raises(ExperimentNotFound):
        engine.run_defend(DefendRequest(parent_experiment_id="MLB-EXP-000000000000",
                                        defense=ConfidenceRejectionDefense()))


def test_explicit_sample_selection(engine):
    rec = engine.run_attack(AttackRequest(samples=SampleSelection(indices=[3, 10, 42])))
    assert [s.test_index for s in rec.samples] == [3, 10, 42]
    ds = load_dataset("moons-v1", 7)
    assert rec.samples[1].x_original == ds.X_test[10].tolist()
    with pytest.raises(InvalidExperiment):
        engine.run_attack(AttackRequest(samples=SampleSelection(indices=[150])))


def test_budget_violation_marks_experiment_failed(engine, monkeypatch):
    def cheating_attack(model, X0, y, cfg, *a, **k):
        return AttackOutput(X0 + 2 * cfg.epsilon, np.ones(len(X0), bool), np.zeros(len(X0), int), 0.0)

    monkeypatch.setattr(exp_mod, "gradient_attack", cheating_attack)
    rec = engine.run_attack(AttackRequest())
    assert rec.status == "failed" and "BudgetViolation" in rec.error
    assert rec.metrics == {} and rec.samples == []
    assert engine.store.load(rec.experiment_id).status == "failed"


def test_timeout_marks_experiment_failed(store):
    eng = ExperimentEngine(store, time_limit=-1.0)
    rec = eng.run_attack(AttackRequest())
    assert rec.status == "failed" and "ExperimentTimeout" in rec.error


def test_concurrency_limit(store):
    eng = ExperimentEngine(store, max_concurrent=1)
    assert eng._gate.acquire(blocking=False)
    try:
        with pytest.raises(EngineBusy):
            eng.run_baseline(BaselineRequest())
    finally:
        eng._gate.release()
    assert eng.run_baseline(BaselineRequest()).status == "completed"


def test_defended_baseline(engine):
    rec = engine.run_baseline(BaselineRequest(defense=ConfidenceRejectionDefense(threshold=0.9)))
    assert not rec.intentionally_vulnerable
    assert rec.metrics["coverage"]["value"] < 1.0
    assert any(s.pred_original == -1 for s in rec.samples)


def test_boundary_grid(engine):
    from mlbreak.schemas import BoundaryRequest
    g = engine.boundary(BoundaryRequest(resolution=30, defense=ConfidenceRejectionDefense()))
    assert len(g["x"]) == 30 and len(g["p1"]) == 30 and len(g["p1"][0]) == 30
    assert any(any(row) for row in g["abstain"])
