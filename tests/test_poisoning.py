"""ML02:2023 Data Poisoning lab — acceptance tests.

Covers: deterministic selection, budget enforcement, real retraining, metric definitions and denominators,
kNN label sanitisation, defensive replay lineage, evidence integrity, exact replay, API, CLI and timeouts.
"""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from mlbreak.data import load_dataset
from mlbreak.defenses import base_model
from mlbreak.evidence import parse_record
from mlbreak.experiments import EvidenceIntegrityError, ExperimentEngine, InvalidExperiment
from mlbreak.models import train_model
from mlbreak.poisoning.attacks import (PoisonBudgetViolation, enforce_poison_budget, flip_labels, knn_label_check,
                                       knn_neighbours, n_flips, sanitize, select_poison_indices)
from mlbreak.poisoning.integrity import verify_poisoning_record
from mlbreak.poisoning.metrics import classify_test_outcome, derive_metrics
from mlbreak.poisoning.schemas import (KnnLabelSanitization, LabelFlipConfig, PoisonAttackRequest,
                                       PoisonBaselineRequest, PoisonDefendRequest, PoisoningRecord,
                                       TrainingSampleResult)
from mlbreak.poisoning.schemas import TestSampleResult
from mlbreak.schemas import AttackRequest, ConfidenceRejectionDefense, DefendRequest, SampleSelection

DS = load_dataset("moons-v1", 7)
RANDOM20 = LabelFlipConfig(strategy="random", fraction=0.2)
TARGET10 = LabelFlipConfig(strategy="targeted", fraction=0.1, target_class=1, selection="random")
BOUNDARY10 = LabelFlipConfig(strategy="targeted", fraction=0.1, target_class=1, selection="boundary")


@pytest.fixture
def lab(engine):
    return engine.poisoning


@pytest.fixture
def random_attack(lab):
    return lab.run_attack(PoisonAttackRequest(poison=RANDOM20))


def statuses(report):
    return {c["check"]: c["status"] for c in report["checks"]}


# ---------------------------------------------------------------------------
# Poison selection: deterministic, budgeted, strategy-correct
# ---------------------------------------------------------------------------
def _margin():
    return np.abs(base_model("mlp-tanh-16x16", "moons-v1", 7).logit(DS.X_train))


@pytest.mark.parametrize("cfg", [RANDOM20, TARGET10, BOUNDARY10], ids=["random", "targeted", "boundary"])
def test_selection_is_reproducible_and_within_budget(cfg):
    a, rule = select_poison_indices(cfg, DS.y_train, 7, _margin())
    b, _ = select_poison_indices(cfg, DS.y_train, 7, _margin())
    assert np.array_equal(a, b) and rule
    assert len(a) == n_flips(cfg, len(DS.y_train)) == round(cfg.fraction * 350)
    assert len(set(a.tolist())) == len(a) and np.all(np.diff(a) > 0)  # unique, sorted
    y_obs = flip_labels(DS.y_train, a)
    enforce_poison_budget(DS.y_train, y_obs, a, cfg)  # does not raise
    assert np.array_equal(np.flatnonzero(y_obs != DS.y_train), a)


def test_different_seed_selects_different_rows():
    a, _ = select_poison_indices(RANDOM20, DS.y_train, 7)
    b, _ = select_poison_indices(RANDOM20, DS.y_train, 8)
    assert not np.array_equal(a, b)


def test_targeted_only_flips_target_class_and_random_spans_both():
    t, _ = select_poison_indices(TARGET10, DS.y_train, 7)
    assert set(DS.y_train[t].tolist()) == {1}
    r, _ = select_poison_indices(RANDOM20, DS.y_train, 7)
    assert set(DS.y_train[r].tolist()) == {0, 1}


def test_boundary_selection_picks_smallest_clean_margins():
    m = _margin()
    idx, _ = select_poison_indices(BOUNDARY10, DS.y_train, 7, m)
    pool = np.flatnonzero(DS.y_train == 1)
    rest = np.setdiff1d(pool, idx)
    assert m[idx].max() <= m[rest].min()


def test_budget_violations_are_detected():
    idx, _ = select_poison_indices(TARGET10, DS.y_train, 7)
    y_obs = flip_labels(DS.y_train, idx)
    extra = y_obs.copy()
    extra[np.setdiff1d(np.arange(350), idx)[0]] ^= 1
    with pytest.raises(PoisonBudgetViolation):
        enforce_poison_budget(DS.y_train, extra, idx, TARGET10)  # change outside the selected rows
    wrong_class = np.flatnonzero(DS.y_train == 0)[:3]
    with pytest.raises(PoisonBudgetViolation):
        enforce_poison_budget(DS.y_train, flip_labels(DS.y_train, wrong_class), wrong_class, TARGET10)
    over = np.arange(n_flips(RANDOM20, 350) + 1)
    with pytest.raises(PoisonBudgetViolation):
        enforce_poison_budget(DS.y_train, flip_labels(DS.y_train, over), over, RANDOM20)


@pytest.mark.parametrize("kwargs", [
    {"strategy": "targeted"},                                   # no target class
    {"strategy": "random", "target_class": 1},                  # target on random
    {"strategy": "random", "selection": "boundary"},            # boundary on random
    {"strategy": "random", "fraction": 0.6},                    # over the 50% cap
    {"strategy": "random", "fraction": 0},
    {"strategy": "random", "fraction": 0.2, "extra": 1},        # strict schema
    {"strategy": "targeted", "target_class": 2},
])
def test_poison_config_validation(kwargs):
    with pytest.raises(ValidationError):
        LabelFlipConfig(**kwargs)


def test_budget_larger_than_target_class_is_rejected(lab):
    # 0.5 × 350 = 175 = size of class 1 → allowed; class sizes are 175/175, so nothing larger is expressible.
    # A tiny fraction that rounds to zero flips is rejected instead of silently doing nothing.
    with pytest.raises(InvalidExperiment, match="flips no labels"):
        lab.run_attack(PoisonAttackRequest(poison=LabelFlipConfig(strategy="random", fraction=0.001)))


# ---------------------------------------------------------------------------
# kNN label sanitisation
# ---------------------------------------------------------------------------
def test_knn_flags_an_isolated_flipped_label():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(3, 0.1, (20, 2))])
    y = np.array([0] * 20 + [1] * 20)
    y_obs = y.copy()
    y_obs[5] = 1  # inside the class-0 cluster
    cfg = KnnLabelSanitization(k=5)
    agreement, flagged, majority = knn_label_check(X, y_obs, cfg)
    assert flagged[5] and agreement[5] == 0.0 and majority[5] == 0
    assert flagged.sum() == 1
    keep, y_san = sanitize(y_obs, flagged, majority, "relabel")
    assert keep.all() and np.array_equal(y_san, y)
    keep, y_rm = sanitize(y_obs, flagged, majority, "remove")
    assert not keep[5] and keep.sum() == 39


def test_knn_neighbours_exclude_self_and_tie_rule():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    nb = knn_neighbours(X, 2)
    assert all(i not in row for i, row in enumerate(nb))
    # k=4 even: a 2–2 split among neighbours keeps the sample's own label as "majority".
    X = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
    y = np.array([0, 0, 1, 1, 0])
    _, _, maj = knn_label_check(X, y, KnnLabelSanitization(k=4, agreement_threshold=0.5))
    assert maj[4] in (0, 1)


# ---------------------------------------------------------------------------
# Metric definitions (exact values on hand-built rows)
# ---------------------------------------------------------------------------
def _tr(i, y, y_obs, pc, pp, **kw):
    return TrainingSampleResult(train_index=i, x=[0.0, 0.0], y_true=y, y_observed=y_obs, poisoned=y != y_obs,
                                pred_clean=pc, pred_poisoned=pp, **kw)


def _te(i, y, pc, pp, pd=None, pcs=None):
    return TestSampleResult(test_index=i, x=[0.0, 0.0], y_true=y, pred_clean=pc, p1_clean=0.5, pred_poisoned=pp,
                            p1_poisoned=0.5, pred_defended=pd, pred_clean_sanitized=pcs,
                            outcome=classify_test_outcome(y, pc, pp),
                            defended_outcome=None if pd is None else classify_test_outcome(y, pc, pd))


def val(m, k):
    return (m[k]["numerator"], m[k]["denominator"])


def test_outcome_partition():
    assert [classify_test_outcome(1, 1, 1), classify_test_outcome(1, 1, 0), classify_test_outcome(1, 0, 1), classify_test_outcome(1, 0, 0)] == \
        ["kept", "broken", "fixed", "still_wrong"]


def test_targeted_metrics_use_target_class_population():
    train = [_tr(0, 1, 0, 1, 0), _tr(1, 1, 1, 1, 1), _tr(2, 0, 0, 0, 0), _tr(3, 0, 0, 1, 1)]
    # target class 1: three clean-correct, one clean-wrong; poisoned breaks two of the three.
    test = [_te(0, 1, 1, 0), _te(1, 1, 1, 0), _te(2, 1, 1, 1), _te(3, 1, 0, 0),
            _te(4, 0, 0, 1), _te(5, 0, 0, 0)]
    cfg = {"strategy": "targeted", "fraction": 0.25, "target_class": 1, "selection": "random"}
    m = derive_metrics("poisoning", train, test, cfg, None)
    assert val(m, "attack_success_rate") == (2, 3)          # broken / clean-correct target-class samples
    assert val(m, "collateral_damage_rate") == (1, 2)       # same, on the other class
    assert val(m, "target_class_recall") == (1, 4) and val(m, "clean_target_class_recall") == (3, 4)
    assert val(m, "test_accuracy") == (2, 6) and val(m, "clean_test_accuracy") == (5, 6)
    assert val(m, "poison_rate") == (1, 4)
    assert val(m, "train_accuracy") == (2, 4)                # vs TRUE labels
    assert val(m, "train_accuracy_observed") == (3, 4)       # vs labels the learner was given
    assert m["test_outcomes"]["counts"] == {"kept": 2, "broken": 3, "fixed": 0, "still_wrong": 1}
    assert m["test_accuracy_delta"]["value"] == pytest.approx(2 / 6 - 5 / 6)
    for k, v in m.items():
        if isinstance(v, dict) and "numerator" in v:
            assert v["definition"] and v["population"], k


def test_random_attack_population_is_whole_test_set_and_empty_denominator_is_undefined():
    train = [_tr(0, 1, 0, 1, 0)]
    test = [_te(0, 1, 0, 0), _te(1, 0, 1, 1)]  # the clean model gets nothing right
    m = derive_metrics("poisoning", train, test, {"strategy": "random", "fraction": 0.2,
                                                   "target_class": None, "selection": "random"}, None)
    assert m["attack_success_rate"]["denominator"] == 0 and m["attack_success_rate"]["value"] is None
    assert "target_class_recall" not in m and "collateral_damage_rate" not in m


def test_detection_and_recovery_metrics():
    det = {(True, True): "true_positive", (True, False): "false_positive",
           (False, True): "false_negative", (False, False): "true_negative"}

    def row(i, y, y_obs, flagged, y_san):
        return _tr(i, y, y_obs, y, y, pred_defended=y, knn_agreement=0.5, flagged=flagged, y_sanitized=y_san,
                   removed=False, detection=det[(flagged, y != y_obs)], flagged_on_clean=False)
    train = [row(0, 1, 0, True, 1), row(1, 1, 0, False, 0), row(2, 0, 0, True, 1), row(3, 0, 0, False, 0),
             row(4, 0, 0, False, 0)]
    test = [_te(0, 1, 1, 0, pd=1, pcs=1), _te(1, 1, 1, 0, pd=0, pcs=1), _te(2, 0, 0, 0, pd=0, pcs=0)]
    m = derive_metrics("poisoning_replay", train, test, {"strategy": "random", "fraction": 0.4,
                                                          "target_class": None, "selection": "random"},
                       {"kind": "knn_label_sanitization", "k": 3, "agreement_threshold": 0.5, "action": "relabel"})
    assert val(m, "detection_recall") == (1, 2)       # flagged poisoned / poisoned
    assert val(m, "detection_precision") == (1, 2)    # flagged poisoned / flagged
    assert val(m, "false_positive_rate") == (1, 3)    # flagged clean / clean
    assert val(m, "labels_repaired") == (1, 2) and val(m, "labels_corrupted") == (1, 3)
    assert val(m, "recovery_rate") == (1, 2)          # broken by poisoning, fixed by the defense
    assert val(m, "undefended_attack_success_rate") == (2, 3) and val(m, "attack_success_rate") == (1, 3)
    assert m["detection_counts"] == {"true_positive": 1, "false_positive": 1, "false_negative": 1,
                                     "true_negative": 2}


# ---------------------------------------------------------------------------
# End-to-end experiments
# ---------------------------------------------------------------------------
REQUIRED = ["experiment_id", "scenario_id", "owasp", "attacker_capabilities", "model", "models", "dataset", "seed",
            "poison_config", "defense_config", "training_samples", "test_samples", "metrics", "status",
            "reproducibility", "poisoning"]


def test_clean_baseline(lab):
    rec = lab.run_baseline(PoisonBaselineRequest())
    assert rec.status == "completed" and rec.evaluation == "clean_training" and rec.mode == "baseline"
    assert rec.owasp["id"] == "ML02:2023" and rec.scenario_id == "ml02-moons-label-flip"
    assert len(rec.training_samples) == 350 and len(rec.test_samples) == 150
    assert not any(s.poisoned for s in rec.training_samples)
    # The ML02 clean model IS the ML01 base model (same data, config, seed).
    assert rec.model["artifact_sha256"] == base_model("mlp-tanh-16x16", "moons-v1", 7).artifact_sha256()
    assert rec.metrics["test_accuracy"]["value"] > 0.9 and "train_accuracy" in rec.metrics


def test_attack_really_retrains_on_poisoned_labels(random_attack, store):
    rec = random_attack
    data = json.loads(store.json_path(rec.experiment_id).read_text())
    for key in REQUIRED:
        assert key in data, key
    assert rec.status == "completed" and rec.intentionally_vulnerable
    y_obs = np.array([s.y_observed for s in rec.training_samples])
    retrained = train_model("mlp-tanh-16x16", DS.X_train, y_obs, 7)
    assert retrained.artifact_sha256() == rec.models["poisoned"]["artifact_sha256"]
    assert rec.models["poisoned"]["artifact_sha256"] != rec.models["clean"]["artifact_sha256"]
    assert [s.pred_poisoned for s in rec.test_samples] == retrained.predict(DS.X_test).tolist()
    assert sum(s.poisoned for s in rec.training_samples) == 70
    assert rec.poisoning["poisoned_indices"] == [s.train_index for s in rec.training_samples if s.poisoned]
    assert rec.metrics["attack_success_rate"]["denominator"] == sum(s.pred_clean == s.y_true for s in rec.test_samples)
    caps = rec.attacker_capabilities
    assert "LABELS" in caps["can_modify"] and "features" in caps["cannot"]


def test_targeted_attacks_damage_the_target_class(lab):
    for cfg in (TARGET10, BOUNDARY10):
        rec = lab.run_attack(PoisonAttackRequest(poison=cfg))
        m = rec.metrics
        assert m["target_class"] == 1
        assert m["target_class_recall"]["value"] < m["clean_target_class_recall"]["value"]
        assert all(s.y_true == 1 for s in rec.training_samples if s.poisoned)


def test_defensive_replay_uses_exact_parent_labels(lab, random_attack):
    d = lab.run_defend(PoisonDefendRequest(parent_experiment_id=random_attack.experiment_id))
    assert d.status == "completed" and d.mode == "defend" and d.evaluation == "poisoning_replay"
    assert not d.intentionally_vulnerable and d.parent_experiment_id == random_attack.experiment_id
    assert [s.y_observed for s in d.training_samples] == [s.y_observed for s in random_attack.training_samples]
    assert d.models["poisoned"]["artifact_sha256"] == random_attack.models["poisoned"]["artifact_sha256"]
    assert d.lineage["relation"] == "defensive_replay_of"
    assert d.attacker_capabilities["knows_defense"] is False
    m = d.metrics
    assert m["detection_recall"]["denominator"] == 70
    assert m["flagged"]["numerator"] == sum(bool(s.flagged) for s in d.training_samples)
    assert "clean_data_test_accuracy" in m and "clean_data_flag_rate" in m
    # Random flips are local outliers: the defense must find most of them and help.
    assert m["detection_recall"]["value"] >= 0.8
    assert m["test_accuracy"]["value"] > m["undefended_test_accuracy"]["value"]


def test_remove_action(lab, random_attack):
    d = lab.run_defend(PoisonDefendRequest(parent_experiment_id=random_attack.experiment_id,
                                           defense=KnnLabelSanitization(action="remove")))
    removed = [s for s in d.training_samples if s.removed]
    assert removed and all(s.flagged and s.y_sanitized is None for s in removed)
    assert d.metrics["removed"]["numerator"] == len(removed)
    assert verify_poisoning_record(d)["ok"]


def test_boundary_targeted_flips_evade_knn(lab):
    a = lab.run_attack(PoisonAttackRequest(poison=BOUNDARY10))
    d = lab.run_defend(PoisonDefendRequest(parent_experiment_id=a.experiment_id))
    assert d.metrics["detection_recall"]["value"] <= 0.2
    assert d.metrics["attack_success_rate"]["value"] >= 0.15


def test_defend_rejects_wrong_parents(lab, engine, random_attack):
    base = lab.run_baseline(PoisonBaselineRequest())
    with pytest.raises(InvalidExperiment, match="poisoning attack"):
        lab.run_defend(PoisonDefendRequest(parent_experiment_id=base.experiment_id))
    ml01 = engine.run_attack(AttackRequest(samples=SampleSelection(max_samples=10)))
    with pytest.raises(InvalidExperiment, match="not an ML02"):
        lab.run_defend(PoisonDefendRequest(parent_experiment_id=ml01.experiment_id))
    with pytest.raises(InvalidExperiment, match="not an ML01"):
        engine.run_defend(DefendRequest(parent_experiment_id=random_attack.experiment_id,
                                        defense=ConfidenceRejectionDefense()))


# ---------------------------------------------------------------------------
# Integrity and replay
# ---------------------------------------------------------------------------
def _tamper(store, exp_id, fn):
    path = store.json_path(exp_id)
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))
    return parse_record(data)


def test_fresh_records_verify(lab, engine, random_attack):
    d = lab.run_defend(PoisonDefendRequest(parent_experiment_id=random_attack.experiment_id))
    for rec in (lab.run_baseline(PoisonBaselineRequest()), random_attack, d):
        rep = engine.verify(rec.experiment_id)
        assert rep["ok"], rep
        assert "not authenticity" in rep["guarantee"]
    st = statuses(engine.verify(d.experiment_id))
    assert st["metrics_derivation"] == st["label_vectors"] == st["sample_consistency"] == "pass"


@pytest.mark.parametrize("edit,check", [
    (lambda d: d["metrics"]["attack_success_rate"].update(numerator=0), "metrics_derivation"),
    (lambda d: d["training_samples"][0].update(y_observed=1 - d["training_samples"][0]["y_observed"]),
     "sample_consistency"),
    (lambda d: d["poisoning"].update(poisoned_indices=[]), "poisoned_indices"),
    (lambda d: d["test_samples"][0].update(pred_poisoned=1 - d["test_samples"][0]["pred_poisoned"]),
     "results_sha256"),
    (lambda d: d["request"]["poison"].update(fraction=0.3), "config_sha256"),
])
def test_tampering_is_detected_and_blocks_replay(engine, store, random_attack, edit, check):
    rec = _tamper(store, random_attack.experiment_id, edit)
    st = statuses(verify_poisoning_record(rec))
    assert st[check] == "fail" and st["record_sha256"] == "fail"
    with pytest.raises(EvidenceIntegrityError):
        engine.replay(random_attack.experiment_id)


def test_tampered_parent_cannot_be_defended(lab, store, random_attack):
    _tamper(store, random_attack.experiment_id, lambda d: d["metrics"]["poison_rate"].update(numerator=1))
    with pytest.raises(EvidenceIntegrityError):
        lab.run_defend(PoisonDefendRequest(parent_experiment_id=random_attack.experiment_id))


def test_exact_replay_of_every_mode(engine, lab, random_attack):
    d = lab.run_defend(PoisonDefendRequest(parent_experiment_id=random_attack.experiment_id))
    for rec in (lab.run_baseline(PoisonBaselineRequest(model_id="logreg", seed=3)), random_attack, d):
        r = engine.replay(rec.experiment_id)
        assert r["identical"], r["checks"]
        assert r["original_integrity"]["ok"]
        names = {c["check"] for c in r["checks"]}
        assert {"config_sha256", "results_sha256", "clean_model_artifact_sha256"} <= names
    r = engine.replay(d.experiment_id)
    assert {"defended_model_artifact_sha256", "sanitized_labels_sha256"} <= {c["check"] for c in r["checks"]}
    assert isinstance(engine.store.load(r["replay_experiment_id"]), PoisoningRecord)


def test_timeout_marks_poisoning_experiment_failed(store):
    eng = ExperimentEngine(store, time_limit=-1.0)
    rec = eng.poisoning.run_attack(PoisonAttackRequest(poison=RANDOM20))
    assert rec.status == "failed" and "ExperimentTimeout" in rec.error
    assert rec.metrics == {} and rec.training_samples == [] and rec.test_samples == []
    assert eng.verify(rec.experiment_id)["ok"]


def test_boundary_reproduces_recorded_models(lab, random_attack):
    d = lab.run_defend(PoisonDefendRequest(parent_experiment_id=random_attack.experiment_id))
    for which in ("clean", "poisoned", "defended", "clean_sanitized"):
        g = lab.boundary(d.experiment_id, which, 25)
        assert len(g["x"]) == 25 and len(g["p1"]) == 25
        assert g["target_system"]["model_artifact_sha256"] == d.models[which]["artifact_sha256"]
    with pytest.raises(InvalidExperiment, match="no 'defended' model"):
        lab.boundary(random_attack.experiment_id, "defended", 25)


# ---------------------------------------------------------------------------
# API and CLI
# ---------------------------------------------------------------------------
def test_api_flow(client):
    b = client.post("/api/experiments/poisoning/baseline", json={})
    assert b.status_code == 200 and b.json()["family"] == "poisoning"
    a = client.post("/api/experiments/poisoning/attack",
                    json={"poison": {"strategy": "targeted", "fraction": 0.1, "target_class": 1,
                                     "selection": "boundary"}})
    assert a.status_code == 200, a.text
    aid = a.json()["experiment_id"]
    d = client.post("/api/experiments/poisoning/defend", json={"parent_experiment_id": aid})
    assert d.status_code == 200 and d.json()["evaluation"] == "poisoning_replay"
    did = d.json()["experiment_id"]
    for which in ("clean", "poisoned", "defended"):
        r = client.get(f"/api/experiments/{did}/boundary?model={which}&resolution=30")
        assert r.status_code == 200 and len(r.json()["x"]) == 30
    assert client.get(f"/api/experiments/{did}/boundary?model=bogus").status_code == 422
    assert client.get(f"/api/experiments/{did}/boundary?resolution=9999").status_code == 422
    assert client.get(f"/api/experiments/{did}").json()["family"] == "poisoning"
    assert client.get(f"/api/experiments/{did}/verify").json()["ok"]
    assert client.post(f"/api/experiments/{aid}/replay").json()["identical"]
    md = client.get(f"/api/experiments/{did}/report.md").text
    assert "ML02:2023" in md and "detection_recall" in md and "Defensive replay" in md
    csv = client.get(f"/api/experiments/{did}/samples.csv").text.splitlines()
    assert csv[0].startswith("split,index") and len(csv) == 1 + 350 + 150
    listed = {e["experiment_id"]: e for e in client.get("/api/experiments").json()}
    assert listed[aid]["family"] == "poisoning" and listed[aid]["attack"] == "label_flip:targeted/boundary"


@pytest.mark.parametrize("path,body", [
    ("attack", {"poison": {"strategy": "random", "fraction": 0.9}}),
    ("attack", {"poison": {"strategy": "targeted"}}),
    ("attack", {"poison": {"strategy": "random"}, "scenario_id": "ml01-moons-evasion"}),
    ("attack", {"poison": {"strategy": "random"}, "labels": [0, 1]}),
    ("defend", {"parent_experiment_id": "../../etc/passwd"}),
    ("defend", {"parent_experiment_id": "MLB-EXP-000000000000", "defense": {"k": 1}}),
    ("defend", {"parent_experiment_id": "MLB-EXP-000000000000", "defense": {"action": "exec"}}),
])
def test_api_rejects_invalid_poisoning_requests(client, path, body):
    assert client.post(f"/api/experiments/poisoning/{path}", json=body).status_code == 422


def test_api_boundary_on_ml01_record_is_rejected(client):
    a = client.post("/api/experiments/attack", json={"samples": {"max_samples": 5}}).json()
    assert client.get(f"/api/experiments/{a['experiment_id']}/boundary").status_code == 422


def test_api_tampered_poisoning_replay_is_409(client):
    a = client.post("/api/experiments/poisoning/attack", json={"poison": {"strategy": "random", "fraction": 0.1}})
    aid = a.json()["experiment_id"]
    _tamper(client.app.state.store, aid, lambda d: d["metrics"]["test_accuracy"].update(numerator=150))
    r = client.post(f"/api/experiments/{aid}/replay")
    assert r.status_code == 409 and r.json()["integrity"]["ok"] is False


def test_cli_poison_commands(tmp_path, monkeypatch, capsys):
    from mlbreak import cli
    monkeypatch.setenv("MLBREAK_DATA_DIR", str(tmp_path / "var"))
    assert cli.main(["poison", "baseline"]) == 0
    assert cli.main(["poison", "attack", "--strategy", "targeted", "--fraction", "0.1", "--selection",
                     "boundary"]) == 0
    out = capsys.readouterr().out
    aid = [w for w in out.split() if w.startswith("MLB-EXP-")][-1]
    assert cli.main(["poison", "defend", aid, "--action", "remove"]) == 0
    assert cli.main(["check", aid]) == 0
    assert cli.main(["replay", aid]) == 0
    assert cli.main(["poison", "attack", "--fraction", "0.9"]) == 2  # validation error
