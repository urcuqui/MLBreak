"""Phase-7 evidence hardening and ML02 scientific validity checks.

* lineage integrity: a child record is invalidated when its parent is edited or deleted;
* numerical-environment provenance in every record;
* ML02: poisoning touches TRAINING labels only, exact budgets, seed reproducibility, every model trained on
  its documented label vector, test-set independence, undefined metrics stay undefined.
"""

import json

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mlbreak.data import load_dataset
from mlbreak.experiments import EvidenceIntegrityError, environment_differences
from mlbreak.models import train_model
from mlbreak.poisoning import attacks as pattacks
from mlbreak.poisoning.metrics import derive_metrics
from mlbreak.poisoning.schemas import (KnnLabelSanitization, LabelFlipConfig, PoisonAttackRequest,
                                       PoisonBaselineRequest, PoisonDefendRequest, TestSampleResult,
                                       TrainingSampleResult)
from mlbreak.schemas import ConfidenceRejectionDefense, DefendRequest, SampleSelection, AttackRequest

DS = load_dataset("moons-v1", 7)
TARGET10 = LabelFlipConfig(strategy="targeted", fraction=0.1, target_class=1, selection="random")


def _edit(store, exp_id, fn):
    path = store.json_path(exp_id)
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def _status(report, check):
    return next(c["status"] for c in report["checks"] if c["check"] == check)


# ---------------------------------------------------------------------------
# Lineage integrity
# ---------------------------------------------------------------------------
@pytest.fixture
def ml01_pair(engine):
    a = engine.run_attack(AttackRequest(samples=SampleSelection(max_samples=30)))
    d = engine.run_defend(DefendRequest(parent_experiment_id=a.experiment_id, defense=ConfidenceRejectionDefense()))
    return a, d


@pytest.fixture
def ml02_pair(engine):
    a = engine.poisoning.run_attack(PoisonAttackRequest(poison=TARGET10))
    d = engine.poisoning.run_defend(PoisonDefendRequest(parent_experiment_id=a.experiment_id))
    return a, d


@pytest.mark.parametrize("pair", ["ml01_pair", "ml02_pair"])
def test_lineage_verifies_for_fresh_children(engine, pair, request):
    _, child = request.getfixturevalue(pair)
    rep = engine.verify(child.experiment_id)
    assert rep["ok"] and _status(rep, "lineage") == "pass"


@pytest.mark.parametrize("pair", ["ml01_pair", "ml02_pair"])
def test_editing_the_parent_invalidates_the_child(engine, store, pair, request):
    parent, child = request.getfixturevalue(pair)
    _edit(store, parent.experiment_id, lambda d: d.update(created_at="2000-01-01T00:00:00+00:00"))
    rep = engine.verify(child.experiment_id)
    assert not rep["ok"] and _status(rep, "lineage") == "fail"
    # The child itself is untouched: only lineage fails.
    assert all(c["status"] != "fail" for c in rep["checks"] if c["check"] != "lineage")
    with pytest.raises(EvidenceIntegrityError):
        engine.replay(child.experiment_id)


@pytest.mark.parametrize("pair", ["ml01_pair", "ml02_pair"])
def test_deleted_parent_invalidates_the_child(engine, store, pair, request):
    parent, child = request.getfixturevalue(pair)
    store.json_path(parent.experiment_id).unlink()
    rep = engine.verify(child.experiment_id)
    assert _status(rep, "lineage") == "fail" and "not found" in next(
        c["detail"] for c in rep["checks"] if c["check"] == "lineage")


def test_unlinked_records_have_no_lineage_check(engine, ml01_pair):
    parent, _ = ml01_pair
    assert "lineage" not in {c["check"] for c in engine.verify(parent.experiment_id)["checks"]}


def test_ml02_boundary_refuses_a_child_of_a_tampered_parent(engine, store, ml02_pair):
    parent, child = ml02_pair
    _edit(store, parent.experiment_id, lambda d: d["metrics"]["poison_rate"].update(numerator=1))
    with pytest.raises(EvidenceIntegrityError):
        engine.poisoning.boundary(child.experiment_id, "defended", 20)


# ---------------------------------------------------------------------------
# Numerical environment provenance
# ---------------------------------------------------------------------------
def test_records_capture_the_numerical_environment(ml02_pair, ml01_pair):
    for rec in (*ml01_pair, *ml02_pair):
        r = rec.reproducibility
        for dep in ("numpy", "scipy", "scikit-learn", "pandas", "pydantic", "threadpoolctl"):
            assert r["dependencies"][dep], dep
        assert r["numerical_backend"]["numpy_blas_build"]["name"]
        assert "openblas_coretype_env" in r["numerical_backend"]
        assert len(r["numerical_environment_sha256"]) == 64


def test_environment_differences_ignore_keys_older_records_lack():
    old = {"python": "3.12.2", "numpy": "2.5.4"}
    new = {"python": "3.12.2", "numpy": "2.5.4", "scipy": "1.18.1", "numerical_environment_sha256": "x"}
    assert environment_differences(old, new) == {}
    assert set(environment_differences({**old, "numpy": "2.4.6"}, new)) == {"numpy"}


def test_same_environment_replay_reports_no_differences(engine, ml02_pair):
    r = engine.replay(ml02_pair[0].experiment_id)
    assert r["identical"] and r["environment_differences"] == {} and r["notes"] == []


# ---------------------------------------------------------------------------
# ML02: what poisoning may and may not touch
# ---------------------------------------------------------------------------
def test_poisoning_changes_training_labels_only(engine, ml02_pair):
    attack, defend = ml02_pair
    X, y = make_moons(n_samples=500, noise=0.25, random_state=7)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    for rec in (attack, defend):
        assert [s.y_true for s in rec.test_samples] == y_te.tolist()          # test labels are the clean labels
        assert np.array_equal(np.array([s.x for s in rec.test_samples]), X_te)  # test inputs untouched
        assert [s.y_true for s in rec.training_samples] == y_tr.tolist()
        assert all(0 <= i < 350 for i in rec.poisoning["poisoned_indices"])
    # The shared, cached dataset was not mutated by any experiment.
    assert np.array_equal(DS.y_train, y_tr) and np.array_equal(DS.y_test, y_te)


def test_test_set_is_independent_of_training_and_of_the_defense(engine, monkeypatch):
    """No test row appears in the training split, and the defense only ever sees training data."""
    train_rows = {tuple(r) for r in DS.X_train.round(12).tolist()}
    assert not any(tuple(r) in train_rows for r in DS.X_test.round(12).tolist())
    seen = []
    real = pattacks.knn_label_check
    from mlbreak.poisoning import engine as peng

    def spy(X, y, cfg):
        seen.append(np.array(X))
        return real(X, y, cfg)

    monkeypatch.setattr(peng, "knn_label_check", spy)
    a = engine.poisoning.run_attack(PoisonAttackRequest(poison=TARGET10))
    engine.poisoning.run_defend(PoisonDefendRequest(parent_experiment_id=a.experiment_id))
    assert seen and all(np.array_equal(X, DS.X_train) for X in seen)


@pytest.mark.parametrize("fraction", [0.01, 0.05, 0.105, 0.2, 1 / 3, 0.5])
def test_exact_contamination_budget(engine, fraction):
    rec = engine.poisoning.run_attack(PoisonAttackRequest(poison=LabelFlipConfig(strategy="random", fraction=fraction)))
    flips = sum(s.poisoned for s in rec.training_samples)
    assert flips == int(round(fraction * 350)) == rec.poisoning["n_flips"] == rec.poisoning["budget"]
    assert rec.metrics["poison_rate"] == {**rec.metrics["poison_rate"], "numerator": flips, "denominator": 350}


@pytest.mark.parametrize("cfg", [LabelFlipConfig(strategy="random", fraction=0.2), TARGET10,
                                 LabelFlipConfig(strategy="targeted", fraction=0.1, target_class=0, selection="boundary")])
def test_fixed_seed_reproduces_the_attack_exactly(engine, cfg):
    a = engine.poisoning.run_attack(PoisonAttackRequest(poison=cfg, seed=11))
    b = engine.poisoning.run_attack(PoisonAttackRequest(poison=cfg, seed=11))
    assert a.poisoning["poisoned_indices"] == b.poisoning["poisoned_indices"]
    assert a.models["poisoned"]["artifact_sha256"] == b.models["poisoned"]["artifact_sha256"]
    assert a.reproducibility["results_sha256"] == b.reproducibility["results_sha256"]
    if cfg.strategy == "targeted" and cfg.selection == "random" or cfg.strategy == "random":
        c = engine.poisoning.run_attack(PoisonAttackRequest(poison=cfg, seed=12))
        assert c.poisoning["poisoned_indices"] != a.poisoning["poisoned_indices"]


def test_every_model_is_trained_on_its_documented_label_vector(engine):
    for action in ("relabel", "remove"):
        a = engine.poisoning.run_attack(PoisonAttackRequest(poison=TARGET10))
        cfg = KnnLabelSanitization(action=action)
        d = engine.poisoning.run_defend(PoisonDefendRequest(parent_experiment_id=a.experiment_id, defense=cfg))
        rows = d.training_samples
        X = np.array([r.x for r in rows])
        y_true = np.array([r.y_true for r in rows])
        y_obs = np.array([r.y_observed for r in rows])
        keep = np.array([not r.removed for r in rows])
        y_san = np.array([-1 if r.y_sanitized is None else r.y_sanitized for r in rows])
        assert train_model("mlp-tanh-16x16", X, y_true, 7).artifact_sha256() == d.models["clean"]["artifact_sha256"]
        assert train_model("mlp-tanh-16x16", X, y_obs, 7).artifact_sha256() == d.models["poisoned"]["artifact_sha256"]
        assert train_model("mlp-tanh-16x16", X[keep], y_san[keep], 7).artifact_sha256() == \
            d.models["defended"]["artifact_sha256"]
        _, fl, maj = pattacks.knn_label_check(X, y_true, cfg)
        kc, yc = pattacks.sanitize(y_true, fl, maj, cfg.action)
        assert train_model("mlp-tanh-16x16", X[kc], yc[kc], 7).artifact_sha256() == \
            d.models["clean_sanitized"]["artifact_sha256"]
        # Stored predictions are those of the documented models.
        assert [r.pred_defended for r in d.test_samples] == \
            train_model("mlp-tanh-16x16", X[keep], y_san[keep], 7).predict(DS.X_test).tolist()


def test_undefined_ml02_metrics_stay_undefined():
    """With no test sample broken by the poisoning, recovery_rate is undefined (None), never 0 or 1."""
    train = [TrainingSampleResult(train_index=0, x=[0, 0], y_true=1, y_observed=0, poisoned=True, pred_clean=1,
                                  pred_poisoned=1, pred_defended=1, knn_agreement=0.0, flagged=True, y_sanitized=1,
                                  removed=False, detection="true_positive", flagged_on_clean=False)]
    test = [TestSampleResult(test_index=0, x=[0, 0], y_true=1, pred_clean=1, p1_clean=0.9, pred_poisoned=1,
                             p1_poisoned=0.9, pred_defended=1, p1_defended=0.9, pred_clean_sanitized=1,
                             outcome="kept", defended_outcome="kept")]
    cfg = {"strategy": "targeted", "fraction": 0.5, "target_class": 0, "selection": "random"}
    m = derive_metrics("poisoning_replay", train, test, cfg, KnnLabelSanitization().model_dump())
    assert m["recovery_rate"]["denominator"] == 0 and m["recovery_rate"]["value"] is None
    # target class 0 has no test samples at all here → every target-class ratio is undefined
    assert m["target_class_recall"]["value"] is None and m["attack_success_rate"]["value"] is None
    assert m["detection_precision"]["value"] == 1.0 and m["false_positive_rate"]["value"] is None


def test_ml02_cases_cover_both_working_and_bypassed_sanitisation():
    from mlbreak.config import PROJECT_ROOT
    from mlbreak.regression import load_cases
    ml02 = [c for c in load_cases(PROJECT_ROOT / "security_cases") if c.owasp_id == "ML02:2023"]
    works = [c for c in ml02 if any(i.kind == "security" for i in c.invariants)]
    bypassed = [c for c in ml02 if c.title.startswith("Known limitation")
                and all(i.kind == "attack_canary" for i in c.invariants)]
    assert len(works) >= 2 and len(bypassed) >= 2
    # Each limitation case pins, with a paired statistic, that the defense does NOT reduce success.
    for c in bypassed:
        assert any(i.posture == "paired" for i in c.invariants), c.id
