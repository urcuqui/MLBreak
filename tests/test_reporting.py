"""Cross-laboratory evidence: one report structure, live integrity, honest limitations, schema stability."""

import json

import pytest

from mlbreak.experiments import ExperimentEngine
from mlbreak.inversion.schemas import (InversionAttackRequest, InversionBaselineRequest, InversionDefendRequest,
                                       InversionRecord, LabelOnly, TargetSelection)
from mlbreak.poisoning.schemas import (LabelFlipConfig, PoisonAttackRequest, PoisonBaselineRequest,
                                       PoisonDefendRequest, PoisoningRecord)
from mlbreak.reporting import SECTIONS, SEMANTIC_DISCLAIMER
from mlbreak.schemas import (AttackRequest, BaselineRequest, ConfidenceRejectionDefense, DefendRequest,
                             ExperimentRecord, SampleSelection)
from mlbreak.text.schemas import (TextAttackRequest, TextDefendRequest, TextEvasionRecord, TextNormalizationDefense,
                                  TextSampleSelection)


@pytest.fixture
def records(engine: ExperimentEngine):
    a = engine.run_attack(AttackRequest(samples=SampleSelection(max_samples=20)))
    t = engine.text.run_attack(TextAttackRequest(samples=TextSampleSelection(max_samples=20)))
    p = engine.poisoning.run_attack(PoisonAttackRequest(poison=LabelFlipConfig(strategy="targeted", fraction=0.1,
                                                                                  target_class=1, selection="boundary")))
    i = engine.inversion.run_attack(InversionAttackRequest(targets=TargetSelection(max_targets=30)))
    return {
        "2d-baseline": engine.run_baseline(BaselineRequest()),
        "2d-attack": a,
        "2d-replay": engine.run_defend(DefendRequest(parent_experiment_id=a.experiment_id,
                                                     defense=ConfidenceRejectionDefense())),
        "spam-attack": t,
        "spam-replay": engine.text.run_defend(TextDefendRequest(parent_experiment_id=t.experiment_id,
                                                                defense=TextNormalizationDefense())),
        "ml02-baseline": engine.poisoning.run_baseline(PoisonBaselineRequest()),
        "ml02-attack": p,
        "ml02-replay": engine.poisoning.run_defend(PoisonDefendRequest(parent_experiment_id=p.experiment_id)),
        "ml03-baseline": engine.inversion.run_baseline(InversionBaselineRequest(targets=TargetSelection(max_targets=30))),
        "ml03-attack": i,
        "ml03-adaptive": engine.inversion.run_defend(InversionDefendRequest(parent_experiment_id=i.experiment_id,
                                                                            output_policy=LabelOnly())),
    }


def test_every_report_has_the_seven_sections_in_order(engine, records):
    for name, rec in records.items():
        md = engine.report(rec.experiment_id)
        positions = [md.find(f"## {s}") for s in SECTIONS]
        assert all(p >= 0 for p in positions), (name, [s for s, p in zip(SECTIONS, positions) if p < 0])
        assert positions == sorted(positions), name
        assert "Result at" in md and "PASS" in md, name                     # live integrity verification
        assert "Integrity is not authenticity" in md, name
        assert f"mlbreak replay {rec.experiment_id}" in md, name


def test_reports_state_scenario_threat_model_and_limitations(engine, records):
    md = {k: engine.report(r.experiment_id) for k, r in records.items()}
    assert "input → model" in md["2d-attack"] and "label source → training pipeline" in md["ml02-attack"]
    assert "message sender → classifier" in md["spam-attack"]
    assert "None — this is a clean evaluation" in md["2d-baseline"]
    for k in ("2d-replay", "spam-replay", "ml02-replay"):
        assert "Non-adaptive replay" in md[k], k
    assert "MLB-REG-004" in md["2d-replay"]                  # confidence rejection bypassed adaptively
    assert "MLB-REG-007" in md["ml02-replay"]                # kNN blind to boundary flips
    assert "MLB-REG-009" in md["spam-replay"]                # spell-snap bypassed by real-word edits
    for k in ("spam-attack", "spam-replay"):
        assert md[k].count(SEMANTIC_DISCLAIMER) >= 2, k      # next to the messages AND in the limitations
    assert SEMANTIC_DISCLAIMER not in md["2d-attack"]
    assert "prediction service → output consumer" in md["ml03-attack"]
    assert "MLB-REG-016" in md["ml03-adaptive"] and "not a formal privacy guarantee" in md["ml03-adaptive"]
    for k in ("ml03-attack", "ml03-adaptive"):
        assert "Reconstruction accuracy alone is not evidence of model-induced leakage" in md[k], k
        assert "leakage claimed" in md[k] or "leakage NOT claimed" in md[k], k


def test_live_report_reveals_tampering_that_the_snapshot_hides(engine, store, records):
    rec = records["spam-attack"]
    snapshot_before = store.markdown(rec.experiment_id)
    path = store.json_path(rec.experiment_id)
    data = json.loads(path.read_text())
    data["metrics"]["attack_success_rate"]["numerator"] = 0
    path.write_text(json.dumps(data))
    live = engine.report(rec.experiment_id)
    assert "FAIL — do not rely on this evidence" in live
    assert store.markdown(rec.experiment_id) == snapshot_before and "FAIL" not in snapshot_before


def test_api_and_cli_serve_the_live_report(client, tmp_path, monkeypatch, capsys):
    a = client.post("/api/experiments/text/attack", json={"samples": {"max_samples": 5}}).json()
    md = client.get(f"/api/experiments/{a['experiment_id']}/report.md").text
    assert all(f"## {s}" in md for s in SECTIONS)
    from mlbreak import cli
    monkeypatch.setenv("MLBREAK_DATA_DIR", str(tmp_path / "var"))
    assert cli.main(["text", "attack", "--samples", "3"]) == 0
    eid = [w for w in capsys.readouterr().out.split() if w.startswith("MLB-EXP-")][-1]
    assert cli.main(["export", eid, "--format", "md"]) == 0
    assert "## 7. Integrity verification" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Schema stability: record_sha256 hashes model_dump(), so adding a field to a record model silently breaks the
# verification of every record stored before the change. Field changes require a NEW schema version.
# ---------------------------------------------------------------------------
PINNED_FIELDS = {
    ExperimentRecord: {"schema_version", "experiment_id", "created_at", "mode", "evaluation", "status", "error",
                       "intentionally_vulnerable", "scenario_id", "owasp", "attacker_capabilities", "model",
                       "target_system", "dataset", "seed", "attack_config", "defense_config", "parent_experiment_id",
                       "replay_of", "lineage", "request", "metrics", "samples", "reproducibility", "duration_ms"},
    PoisoningRecord: {"schema_version", "family", "experiment_id", "created_at", "mode", "evaluation", "status",
                      "error", "intentionally_vulnerable", "scenario_id", "owasp", "attacker_capabilities", "model",
                      "models", "target_system", "dataset", "seed", "poison_config", "poisoning", "defense_config",
                      "parent_experiment_id", "replay_of", "lineage", "request", "metrics", "training_samples",
                      "test_samples", "reproducibility", "duration_ms"},
    TextEvasionRecord: {"schema_version", "family", "experiment_id", "created_at", "mode", "evaluation", "status",
                        "error", "intentionally_vulnerable", "scenario_id", "owasp", "attacker_capabilities", "model",
                        "target_system", "dataset", "seed", "attack_config", "defense_config", "parent_experiment_id",
                        "replay_of", "lineage", "request", "metrics", "samples", "test_predictions",
                        "reproducibility", "duration_ms"},
    InversionRecord: {"schema_version", "family", "experiment_id", "created_at", "mode", "evaluation", "status",
                      "error", "intentionally_vulnerable", "scenario_id", "owasp", "attacker_capabilities", "model",
                      "auxiliary_model", "target_system", "dataset", "seed", "output_policy", "attack_config",
                      "defense_config", "target_selection", "parent_experiment_id", "replay_of", "lineage", "request",
                      "metrics", "targets", "service", "reproducibility", "duration_ms"},
}


@pytest.mark.parametrize("model", list(PINNED_FIELDS), ids=lambda m: m.__name__)
def test_record_schemas_are_frozen_per_version(model):
    assert set(model.model_fields) == PINNED_FIELDS[model], (
        f"{model.__name__} fields changed: this breaks record_sha256 of stored evidence. Introduce a new "
        "schema_version (and keep the old one loadable) instead of editing this one.")
