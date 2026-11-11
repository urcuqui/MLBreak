"""Evidence integrity, deterministic replay and provenance."""

import json

from mlbreak.hashing import sha256_json
from mlbreak.integrity import record_sha256, results_fingerprint, samples_sha256, verify_record
from mlbreak.schemas import (SCHEMA_V1, SCHEMA_V2, AttackConfig, AttackRequest, BaselineRequest,
                             ConfidenceRejectionDefense, DefendRequest, ExperimentRecord, SampleSelection)

SMALL = SampleSelection(max_samples=30)


def statuses(report):
    return {c["check"]: c["status"] for c in report["checks"]}


def test_fresh_records_verify(engine):
    a = engine.run_attack(AttackRequest(samples=SMALL))
    recs = [engine.run_baseline(BaselineRequest()), a,
            engine.run_defend(DefendRequest(parent_experiment_id=a.experiment_id,
                                            defense=ConfidenceRejectionDefense())),
            engine.run_attack(AttackRequest(samples=SMALL, target_defense=ConfidenceRejectionDefense(),
                                            parent_experiment_id=a.experiment_id)),
            engine.run_attack(AttackRequest(attack=AttackConfig(attack="random_search", query_budget=40),
                                            samples=SMALL))]
    for r in recs:
        rep = engine.verify(r.experiment_id)
        assert rep["ok"], rep
        assert r.schema_version == SCHEMA_V2
        # Baseline (clean-only) samples have no adversarial evaluation, hence no outcome.
        assert all((s.outcome is None) == (r.mode == "baseline") for s in r.samples)
    assert statuses(engine.verify(a.experiment_id))["metrics_derivation"] == "pass"


def _tamper(store, exp_id, fn):
    path = store.json_path(exp_id)
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))
    return ExperimentRecord.model_validate(data)


def test_edited_metric_is_detected(engine, store, attack_record):
    def inflate(d):
        d["metrics"]["adversarial_accuracy"]["numerator"] += 10
    rec = _tamper(store, attack_record.experiment_id, inflate)
    st = statuses(verify_record(rec))
    assert st["results_sha256"] == "fail" and st["metrics_derivation"] == "fail" and st["record_sha256"] == "fail"


def test_metrics_recomputed_consistently_still_fail_record_hash(engine, store, attack_record):
    """Even a forger who also fixes results_sha256 is caught by record_sha256."""
    def forge(d):
        s = next(s for s in d["samples"] if s["outcome"] == "evaded")
        s["pred_adversarial"], s["attack_success"], s["outcome"] = s["y_true"], False, "robust"
        rec = ExperimentRecord.model_validate(d)
        from mlbreak.metrics import outcome_metrics
        d["metrics"].update(outcome_metrics(rec.samples, can_abstain=False))
        rec = ExperimentRecord.model_validate(d)
        d["reproducibility"]["results_sha256"] = results_fingerprint(rec.samples, rec.metrics)
    rec = _tamper(store, attack_record.experiment_id, forge)
    st = statuses(verify_record(rec))
    assert st["results_sha256"] == "pass" and st["metrics_derivation"] == "pass"
    assert st["record_sha256"] == "fail"


def test_timestamp_edit_only_breaks_record_hash(store, attack_record):
    rec = _tamper(store, attack_record.experiment_id, lambda d: d.update(created_at="2000-01-01T00:00:00+00:00"))
    st = statuses(verify_record(rec))
    assert st["record_sha256"] == "fail" and st["results_sha256"] == "pass"


def test_inconsistent_sample_is_detected(store, attack_record):
    def contradict(d):
        s = next(s for s in d["samples"] if s["pred_adversarial"] is not None)
        s["pred_original"] = 1 - s["y_true"]  # adversarial result on a misclassified clean input
    st = statuses(verify_record(_tamper(store, attack_record.experiment_id, contradict)))
    assert st["sample_consistency"] == "fail"


def test_api_and_cli_verify(client, tmp_path, monkeypatch):
    a = client.post("/api/experiments/attack", json={"samples": {"max_samples": 10}}).json()
    r = client.get(f"/api/experiments/{a['experiment_id']}/verify")
    assert r.status_code == 200 and r.json()["ok"]
    assert client.get("/api/experiments/MLB-EXP-000000000000/verify").status_code == 404
    from mlbreak import cli
    monkeypatch.setenv("MLBREAK_DATA_DIR", str(tmp_path / "var"))
    assert cli.main(["check", a["experiment_id"]]) == 0


def test_replay_compares_every_reproducible_field(engine, attack_record):
    r = engine.replay(attack_record.experiment_id)
    assert r["identical"], r["checks"]
    names = {c["check"] for c in r["checks"]}
    assert {"config_sha256", "samples_sha256", "metrics_sha256", "results_sha256", "model_artifact_sha256",
            "target_system_artifact_sha256", "train_split_sha256", "test_split_sha256"} <= names
    assert r["environment_differences"] == {}
    rep = attack_record.reproducibility
    assert rep["fields"]["bitwise"] and rep["fields"]["runtime"]
    for k in ("python", "numpy", "scikit_learn", "platform", "config_sha256", "record_sha256"):
        assert rep[k]


def test_v1_record_is_readable_verifiable_and_replayable(engine, store, attack_record):
    """Records written before schema v2 (no outcome, old adversarial_accuracy) still verify and replay."""
    def downgrade(d):
        d["schema_version"] = SCHEMA_V1
        for s in d["samples"]:
            s.pop("outcome")
        d.pop("lineage")
        for k in [k for k in d["metrics"] if k not in ("clean_accuracy", "adversarial_accuracy",
                                                       "attack_success_rate", "perturbation", "queries")]:
            d["metrics"].pop(k)
        rep = d["reproducibility"]
        for k in ("record_sha256", "samples_sha256", "metrics_sha256"):
            rep.pop(k)
        rec = ExperimentRecord.model_validate(d)
        rep["results_sha256"] = sha256_json({"samples": [s.model_dump(exclude={"outcome"}) for s in rec.samples],
                                             "metrics": rec.metrics})
    v1 = _tamper(store, attack_record.experiment_id, downgrade)
    st = statuses(verify_record(v1))
    assert st["results_sha256"] == "pass" and st["record_sha256"] == "skipped"
    assert st["metrics_derivation"] == "skipped"
    r = engine.replay(attack_record.experiment_id)
    assert r["identical"] and any("schema v1" in n for n in r["notes"])
    assert samples_sha256(v1.samples) == samples_sha256(attack_record.samples)


def test_record_hash_covers_lineage(engine, attack_record):
    d = engine.run_defend(DefendRequest(parent_experiment_id=attack_record.experiment_id,
                                        defense=ConfidenceRejectionDefense()))
    assert d.reproducibility["record_sha256"] == record_sha256(d)
    forged = d.model_copy(update={"lineage": {**d.lineage, "parent_experiment_id": "MLB-EXP-000000000000"}})
    assert record_sha256(forged) != d.reproducibility["record_sha256"]


# ---------------------------------------------------------------------------
# Replay requires evidence integrity
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

from mlbreak.experiments import EvidenceIntegrityError  # noqa: E402
from mlbreak.integrity import metrics_sha256  # noqa: E402


def _n_records(store):
    return len(store.list())


@pytest.mark.parametrize("edit", [
    pytest.param(lambda d: d["metrics"]["attack_success_rate"].update(numerator=0), id="metric"),
    pytest.param(lambda d: d["request"]["attack"].update(epsilon=0.9), id="request"),
    pytest.param(lambda d: d.update(created_at="2000-01-01T00:00:00+00:00"), id="timestamp-only"),
    pytest.param(lambda d: d["samples"][0].update(x_original=[9.0, 9.0]), id="sample"),
])
def test_replay_refuses_tampered_evidence_without_executing(engine, store, attack_record, edit):
    _tamper(store, attack_record.experiment_id, edit)
    before = _n_records(store)
    with pytest.raises(EvidenceIntegrityError) as exc:
        engine.replay(attack_record.experiment_id)
    assert exc.value.report["ok"] is False
    assert any(c["status"] == "fail" for c in exc.value.report["checks"])
    assert _n_records(store) == before  # nothing was re-executed or stored


def test_replay_reports_original_integrity(engine, attack_record):
    r = engine.replay(attack_record.experiment_id)
    assert r["original_integrity"]["ok"] is True
    assert r["original_integrity"]["experiment_id"] == attack_record.experiment_id


def test_replay_integrity_gate_api_and_cli(client, tmp_path, monkeypatch):
    a = client.post("/api/experiments/attack", json={"samples": {"max_samples": 10}}).json()
    store = client.app.state.store
    _tamper(store, a["experiment_id"], lambda d: d["metrics"]["adversarial_accuracy"].update(numerator=0))
    r = client.post(f"/api/experiments/{a['experiment_id']}/replay")
    assert r.status_code == 409
    body = r.json()
    assert "integrity" in body["detail"] and body["integrity"]["ok"] is False
    from mlbreak import cli
    monkeypatch.setenv("MLBREAK_DATA_DIR", str(tmp_path / "var"))
    assert cli.main(["replay", a["experiment_id"]]) == 3


def test_fully_reforged_record_passes_integrity_but_not_replay(engine, store, attack_record):
    """Integrity is not authenticity: a forger who recomputes EVERY fingerprint passes verify_record.

    The replay gate therefore accepts it, and only re-execution exposes the fabricated results. This test
    pins that documented limitation (docs/reproducibility.md, "Integrity is not authenticity").
    """
    def reforge(d):
        s = next(s for s in d["samples"] if s["outcome"] == "evaded")
        s["pred_adversarial"], s["attack_success"], s["outcome"] = s["y_true"], False, "robust"
        rec = ExperimentRecord.model_validate(d)
        from mlbreak.metrics import outcome_metrics
        d["metrics"].update(outcome_metrics(rec.samples, can_abstain=False))
        rec = ExperimentRecord.model_validate(d)
        rep = d["reproducibility"]
        rep["samples_sha256"] = samples_sha256(rec.samples)
        rep["metrics_sha256"] = metrics_sha256(rec.metrics)
        rep["results_sha256"] = results_fingerprint(rec.samples, rec.metrics)
        rep["record_sha256"] = record_sha256(ExperimentRecord.model_validate(d))
    forged = _tamper(store, attack_record.experiment_id, reforge)
    assert verify_record(forged)["ok"] is True
    r = engine.replay(attack_record.experiment_id)
    assert r["identical"] is False
    failed = {c["check"] for c in r["checks"] if not c["match"]}
    assert {"samples_sha256", "metrics_sha256", "results_sha256"} <= failed
