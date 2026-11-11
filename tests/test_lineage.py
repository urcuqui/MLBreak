"""Experiment lineage: parent validation for adaptive attacks and defensive replay."""

import json

import pytest

from mlbreak.experiments import InvalidExperiment
from mlbreak.schemas import (AdversarialTrainingDefense, AttackConfig, AttackRequest, BaselineRequest,
                             ConfidenceRejectionDefense, DefendRequest, SampleSelection)

REJ = ConfidenceRejectionDefense(threshold=0.8)
SMALL = SampleSelection(max_samples=30)


@pytest.fixture
def parent(engine):
    return engine.run_attack(AttackRequest(samples=SMALL))


def adaptive(parent, **over):
    base = dict(samples=SMALL, target_defense=REJ, parent_experiment_id=parent.experiment_id)
    return AttackRequest(**{**base, **over})


def test_adaptive_and_replay_record_validated_provenance(engine, parent):
    a = engine.run_attack(adaptive(parent))
    d = engine.run_defend(DefendRequest(parent_experiment_id=parent.experiment_id, defense=REJ))
    for rec, relation in ((a, "adaptive_attack_of"), (d, "replay_against_defense_of")):
        lin = rec.lineage
        assert lin["relation"] == relation and lin["parent_experiment_id"] == parent.experiment_id
        assert lin["parent_results_sha256"] == parent.reproducibility["results_sha256"]
        assert lin["parent_record_sha256"] == parent.reproducibility["record_sha256"]
        assert lin["shared"]["base_model_artifact_sha256"] == parent.model["artifact_sha256"]
        assert lin["shared"]["dataset_test_sha256"] == parent.dataset["test_sha256"]
        assert "parent evidence integrity" in lin["checks_passed"]
    assert a.evaluation == "adaptive_attack" and a.attacker_capabilities["knows_defense"] is True
    assert d.evaluation == "replay" and d.attacker_capabilities["knows_defense"] is False
    assert parent.lineage is None


def test_adaptive_vs_non_adaptive_against_rejection(engine):
    """Same parent, same samples: replay is non-adaptive (abstains), adaptive targets confidence ≥ τ."""
    p = engine.run_attack(AttackRequest())
    rep = engine.run_defend(DefendRequest(parent_experiment_id=p.experiment_id, defense=REJ))
    ada = engine.run_attack(AttackRequest(target_defense=REJ, parent_experiment_id=p.experiment_id))
    assert rep.metrics["outcomes"]["n_selected"] == ada.metrics["outcomes"]["n_selected"]
    assert rep.metrics["attack_success_rate"]["value"] < ada.metrics["attack_success_rate"]["value"]
    assert rep.metrics["abstention_rate_adversarial"]["value"] > ada.metrics["abstention_rate_adversarial"]["value"]
    assert all(s.confidence_adversarial >= 0.8 for s in ada.samples if s.outcome == "evaded")


@pytest.mark.parametrize("over, fragment", [
    ({"seed": 8}, "seed"),
    ({"model_id": "logreg"}, "model"),
    ({"attack": AttackConfig(epsilon=0.2)}, "epsilon"),
    ({"attack": AttackConfig(attack="pgd", norm="l2")}, "norm"),
    ({"samples": SampleSelection(max_samples=20)}, "selected test samples"),
])
def test_incompatible_adaptive_parent_is_rejected(engine, parent, over, fragment):
    with pytest.raises(InvalidExperiment, match=fragment):
        engine.run_attack(adaptive(parent, **over))


def test_adaptive_attack_type_may_differ_but_is_recorded(engine, parent):
    rec = engine.run_attack(adaptive(parent, attack=AttackConfig(attack="random_search", query_budget=50)))
    assert rec.lineage["attack_differs"] is True


def test_invalid_parent_types_are_rejected(engine, parent):
    base = engine.run_baseline(BaselineRequest())
    ada = engine.run_attack(adaptive(parent))
    rep = engine.run_defend(DefendRequest(parent_experiment_id=parent.experiment_id, defense=REJ))
    for bad in (base, ada, rep):
        with pytest.raises(InvalidExperiment, match="non-adaptive attack"):
            engine.run_defend(DefendRequest(parent_experiment_id=bad.experiment_id, defense=REJ))
        with pytest.raises(InvalidExperiment, match="non-adaptive attack"):
            engine.run_attack(adaptive(bad))


def test_failed_parent_is_rejected(store):
    from mlbreak.experiments import ExperimentEngine
    failed = ExperimentEngine(store, time_limit=-1.0).run_attack(AttackRequest(samples=SMALL))
    assert failed.status == "failed"
    eng = ExperimentEngine(store)
    with pytest.raises(InvalidExperiment, match="did not complete"):
        eng.run_defend(DefendRequest(parent_experiment_id=failed.experiment_id, defense=REJ))


def test_parent_without_target_defense_is_rejected(engine, parent):
    with pytest.raises(InvalidExperiment, match="only valid for adaptive"):
        engine.run_attack(AttackRequest(samples=SMALL, parent_experiment_id=parent.experiment_id))


def test_tampered_parent_is_rejected(engine, parent, store):
    path = store.json_path(parent.experiment_id)
    data = json.loads(path.read_text())
    s = next(s for s in data["samples"] if s["x_adversarial"] is not None)
    s["x_adversarial"][0] += 0.05  # forged perturbation (still a valid-looking record)
    path.write_text(json.dumps(data))
    with pytest.raises(InvalidExperiment, match="integrity"):
        engine.run_defend(DefendRequest(parent_experiment_id=parent.experiment_id,
                                        defense=AdversarialTrainingDefense()))
    with pytest.raises(InvalidExperiment, match="integrity"):
        engine.run_attack(adaptive(parent))


def test_api_rejects_incompatible_parent_with_422(client):
    p = client.post("/api/experiments/attack", json={"samples": {"max_samples": 20}}).json()
    r = client.post("/api/experiments/attack", json={"samples": {"max_samples": 20}, "seed": 9,
                                                     "target_defense": {"kind": "confidence_rejection"},
                                                     "parent_experiment_id": p["experiment_id"]})
    assert r.status_code == 422 and "seed" in r.json()["detail"]
