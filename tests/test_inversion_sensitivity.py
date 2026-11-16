"""ML03 sensitivity study: baseline comparison on identical targets, negative control, evidence, replay."""

import json

import numpy as np
import pytest

from mlbreak.experiments import EvidenceIntegrityError
from mlbreak.inversion.data import load_inversion_dataset
from mlbreak.inversion.defenses import PredictionService
from mlbreak.inversion.models import victim_model
from mlbreak.inversion.sensitivity import FACTORS, SensitivityRequest, SensitivityStore


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    from mlbreak.evidence import EvidenceStore
    from mlbreak.experiments import ExperimentEngine
    eng = ExperimentEngine(EvidenceStore(tmp_path_factory.mktemp("sens") / "var"))
    return eng, {m: eng.inversion.sensitivity(SensitivityRequest(model_id=m)) for m in ("logreg", "logreg-public-only")}


def by_label(rep, prefix):
    return next(s for s in rep.results if s.label.startswith(prefix))


def test_every_factor_runs_on_identical_targets_with_a_baseline(reports):
    eng, reps = reports
    rep = reps["logreg"]
    assert {s.factor for s in rep.results} == set(FACTORS)
    n = len(rep.target_indices)
    for s in rep.results:
        assert len(s.inversion_correct) == len(s.baseline_correct) == n and s.assumptions
        m = s.metrics
        assert m["inference_advantage"]["numerator"] == m["attribute_inference_accuracy"]["numerator"] - \
            m["baseline_inference_accuracy"]["numerator"]
    assert eng.inversion.verify_sensitivity(rep.report_id)["ok"]


def test_negative_control_never_claims_leakage(reports):
    _, reps = reports
    for s in reps["logreg-public-only"].results:
        assert s.metrics["inference_advantage"]["value"] <= 0, s.label
        assert not s.metrics["leakage_p_value"]["leakage_claimed"], s.label


def test_reference_attack_depends_on_exact_knowledge_and_adapted_attacker_still_leaks(reports):
    rep = reports[1]["logreg"]
    adv = lambda label: by_label(rep, label).metrics["inference_advantage"]["value"]  # noqa: E731
    claimed = lambda label: by_label(rep, label).metrics["leakage_p_value"]["leakage_claimed"]  # noqa: E731
    assert adv("reference") > 0.25 and claimed("reference")
    # with knowledge errors the exact matcher's elimination rule is confidently wrong (assumption 2 violated) …
    assert adv("feature noise σ=0.5 · exact") < 0 and not claimed("feature noise σ=0.5 · exact")
    # … while a nearest-response attacker still extracts model-induced information
    assert adv("feature noise σ=1.0 · nearest") > 0 and claimed("feature noise σ=1.0 · nearest")


def test_baseline_uses_exactly_the_attackers_knowledge(reports):
    rep = reports[1]["logreg"]
    ref = by_label(rep, "reference").baseline_correct
    assert by_label(rep, "observed rounded to 1").baseline_correct == ref
    assert by_label(rep, "feature noise σ=1.0 · exact").baseline_correct != ref          # noisy knowledge, noisy baseline
    none_known = by_label(rep, "no feature known · nearest")
    assert len(set(none_known.baseline_correct)) <= 2 and none_known.metrics["baseline_inference_accuracy"]["value"] == \
        np.mean([load_inversion_dataset(7).s_eval[i] == 0 for i in rep.target_indices])  # constant guess


def test_weaker_auxiliary_attacker_inflates_advantage_not_inversion(reports):
    rep = reports[1]["logreg"]
    strong, none = by_label(rep, "auxiliary attacker on 600 rows · full"), by_label(rep, "auxiliary attacker on 0 rows · full")
    acc = lambda s: s.metrics["attribute_inference_accuracy"]["value"]  # noqa: E731
    adv = lambda s: s.metrics["inference_advantage"]["value"]  # noqa: E731
    assert acc(strong) == acc(none) == 1.0 and adv(none) > adv(strong)


def test_policies_apply_identically_to_observation_and_queries():
    ds, v = load_inversion_dataset(7), victim_model("logreg", 7)
    for pol in ({"kind": "full_probability"}, {"kind": "quantized_probability", "bins": 4}, {"kind": "label_only"}):
        svc = PredictionService(v, pol)
        observed = svc.respond(ds.X_eval, ds.s_eval)
        own = np.array([svc.respond(ds.X_eval[i:i + 1], ds.s_eval[i:i + 1])[0] for i in range(len(ds.s_eval))])
        assert np.array_equal(observed, own), pol  # the real record's own candidate reproduces it bit for bit


def test_tampered_report_is_rejected_and_replay_refused(reports):
    eng, reps = reports
    rep = reps["logreg"]
    store = SensitivityStore(eng.store.root)
    path = store.path(rep.report_id)
    original = path.read_text()
    data = json.loads(original)
    data["results"][0]["inversion_correct"][0] = not data["results"][0]["inversion_correct"][0]
    path.write_text(json.dumps(data))
    check = eng.inversion.verify_sensitivity(rep.report_id)
    failed = {c["check"] for c in check["checks"] if c["status"] == "fail"}
    assert not check["ok"] and {"metrics_derivation", "results_sha256", "record_sha256"} <= failed
    with pytest.raises(EvidenceIntegrityError):
        eng.inversion.replay_sensitivity(rep.report_id)
    path.write_text(original)
    assert eng.inversion.verify_sensitivity(rep.report_id)["ok"]


def test_exact_replay(reports):
    eng, reps = reports
    r = eng.inversion.replay_sensitivity(reps["logreg-public-only"].report_id)
    assert r["identical"], r["checks"]


def test_sensitivity_api(client):
    r = client.post("/api/inversion/sensitivity", json={"factors": ["reference", "tie_handling"],
                                                          "targets": {"max_targets": 30}})
    assert r.status_code == 200, r.text
    rid = r.json()["report_id"]
    assert client.get(f"/api/inversion/sensitivity/{rid}").json()["report_id"] == rid
    assert client.get(f"/api/inversion/sensitivity/{rid}/verify").json()["ok"]
    assert client.get("/api/inversion/sensitivity/MLB-SEN-xyz").status_code == 404
    assert client.post("/api/inversion/sensitivity", json={"factors": ["reference", "reference"]}).status_code == 422
    assert client.post("/api/inversion/sensitivity", json={"factors": ["membership"]}).status_code == 422
    assert client.post("/api/inversion/sensitivity", json={"noise": 3}).status_code == 422


def test_docs_explain_why_output_restriction_is_not_differential_privacy():
    from mlbreak.config import PROJECT_ROOT
    doc = (PROJECT_ROOT / "docs" / "ml03-model-inversion.md").read_text()
    assert "not differential privacy" in doc and "ε = ∞" in doc and "inference-time input" in doc
    assert "## 13. Sensitivity study" in doc and "## 12. Interactive attack sessions" in doc
