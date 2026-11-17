"""ML04 membership inference: data separation, hidden ground truth, calibration independence, statistics, defenses,
budgets, evidence integrity and replay."""

import inspect
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from mlbreak.api import ATTACKER_ROUTES, create_app
from mlbreak.config import PROJECT_ROOT
from mlbreak.evidence import EvidenceStore
from mlbreak.experiments import EvidenceIntegrityError, ExperimentEngine, InvalidExperiment
from mlbreak.membership import attacks as attack_mod
from mlbreak.membership import shadow as shadow_mod
from mlbreak.membership.attacks import AttackerKnowledge, calibrate
from mlbreak.membership.data import POPULATIONS, SIZES, load_membership_dataset, select_candidates
from mlbreak.membership.defenses import MembershipService, counting_oracle
from mlbreak.membership.metrics import auc_mann_whitney, fisher_one_sided, wilson
from mlbreak.membership.models import victim_model, victim_recipe
from mlbreak.membership.schemas import (CandidateSelection, ComplexityReduction, LabelOnly, MembershipAttackConfig,
                                        MembershipAttackRequest, MembershipBaselineRequest, MembershipDefendRequest,
                                        MembershipRecord, OutputRestriction, Regularization)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    eng = ExperimentEngine(EvidenceStore(tmp_path_factory.mktemp("ml04") / "var"))
    atk = eng.membership.run_attack(MembershipAttackRequest())
    return eng, {
        "baseline": eng.membership.run_baseline(MembershipBaselineRequest()),
        "attack": atk,
        "decoy": eng.membership.run_attack(MembershipAttackRequest(model_id="random-forest-decoy")),
        "regularization": eng.membership.run_defend(MembershipDefendRequest(parent_experiment_id=atk.experiment_id,
                                                                            defense=Regularization())),
        "label_only": eng.membership.run_defend(MembershipDefendRequest(
            parent_experiment_id=atk.experiment_id, defense=OutputRestriction(policy=LabelOnly()))),
    }


def v(rec, key):
    return rec.metrics[key]["value"]


# ------------------------------------------------------------------ data
def test_populations_are_disjoint_deterministic_and_sized():
    ds = load_membership_dataset(7)
    assert {k: len(ds[k].y) for k in POPULATIONS} == SIZES
    d = ds.disjointness()
    assert d["disjoint"] and d["duplicate_records"] == 0 and not any(d["pairwise_overlaps"].values())
    load_membership_dataset.cache_clear()
    assert load_membership_dataset(7).fingerprint() == ds.fingerprint()
    assert load_membership_dataset(8).fingerprint() != ds.fingerprint()
    m, n = select_candidates(7, 250, 250)
    assert len(set(m)) == 250 and len(set(n)) == 250
    assert select_candidates(7, 300, 10)[0][:250] == m  # a larger count extends a smaller one


def test_every_victim_exposes_the_same_prediction_interface():
    ds = load_membership_dataset(7)
    X = ds["victim_test"].X[:20]
    for mid in ("random-forest-overfit", "logreg-l2", "random-forest-decoy"):
        out = MembershipService(victim_model(mid, 7, victim_recipe(mid)), {"kind": "full_probability"}).respond(X)
        assert out.shape == (20,) and out.dtype == np.float64 and ((0 <= out) & (out <= 1)).all()
        one = MembershipService(victim_model(mid, 7, victim_recipe(mid)), {"kind": "full_probability"}).respond(X[:1])
        assert one[0] == out[0]  # row-independent: same value for one record or a batch
    a = victim_model("random-forest-overfit", 7, victim_recipe("random-forest-overfit")).artifact_sha256()
    victim_model.cache_clear()
    assert victim_model("random-forest-overfit", 7, victim_recipe("random-forest-overfit")).artifact_sha256() == a


def test_counting_oracle_refuses_before_evaluating():
    svc = MembershipService(victim_model("logreg-l2", 7, victim_recipe("logreg-l2")), {"kind": "label_only"})
    query, used = counting_oracle(svc, 5)
    X = load_membership_dataset(7)["victim_test"].X
    assert set(query(X[:3]).tolist()) <= {0.0, 1.0} and used() == 3
    with pytest.raises(Exception, match="budget"):
        query(X[:3])
    assert used() == 3


# ------------------------------------------------------------------ ground truth stays with the evaluator
def test_attack_code_never_receives_membership_flags(lab):
    fields = set(AttackerKnowledge.__dataclass_fields__)
    assert fields == {"candidate_ids", "X", "y", "recipe", "policy", "n_train", "shadow", "calibration", "seed"}
    for fn in (attack_mod.threshold_attack, attack_mod.reference_data, shadow_mod.shadow_attack):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"truth", "member", "is_member", "group", "in_training"}, fn
    eng, recs = lab
    rec = recs["attack"]
    groups = [c.truth.group for c in rec.candidates]
    assert groups[:250].count("member") not in (0, 250)  # candidate order does not reveal the group
    ind = eng.membership.attacker_independence(rec)
    assert ind["ok"] and all(c["ok"] for c in ind["checks"]), ind


def test_thresholds_come_from_calibration_data_only(lab):
    _, recs = lab
    rec = recs["attack"]
    cal = rec.calibration
    assert cal["population_sha256"] == rec.dataset["calibration_sha256"]
    for a in ("threshold", "shadow"):
        c = cal["attacks"][a]
        again = calibrate(c["scores"], c["is_member"], c["class_keys"], c["mode"], 50, c["target_fpr"])
        assert again["operating_points"] == c["operating_points"]
        # the calibration population is the attacker's: 2 reference models x (600 in + 600 out)
        assert len(c["scores"]) == 2400 and sum(c["is_member"]) == 1200
    m = rec.metrics
    assert m["threshold_calibration"]["population"].startswith("the attacker's reference models")


# ------------------------------------------------------------------ measurement
def test_overfit_victim_leaks_and_metrics_are_consistent(lab):
    _, recs = lab
    rec = recs["attack"]
    assert v(rec, "generalization_gap") >= 0.2 and v(rec, "train_accuracy") == 1.0
    for a in ("threshold", "shadow"):
        member = np.array([c.truth.group == "member" for c in rec.candidates])
        score = np.array([getattr(c, a).score for c in rec.candidates])
        assert v(rec, f"{a}_roc_auc") == pytest.approx(roc_auc_score(member, score), abs=1e-12)
        cm = rec.metrics[f"{a}_confusion"]
        assert cm["tp"] + cm["fn"] == 250 and cm["fp"] + cm["tn"] == 250
        tpr, fpr = cm["tp"] / 250, cm["fp"] / 250
        assert v(rec, f"{a}_membership_advantage") == pytest.approx(tpr - fpr)
        assert v(rec, f"{a}_balanced_accuracy") == pytest.approx((tpr + 1 - fpr) / 2)
        proj = {p["prevalence"]: p for p in rec.metrics[f"{a}_operational_precision"]["values"]}
        assert proj[0.01]["precision"] == pytest.approx(tpr * 0.01 / (tpr * 0.01 + fpr * 0.99))
        assert proj[0.01]["precision"] < proj[0.5]["precision"]  # rare members => many false discoveries
        assert rec.metrics[f"{a}_leakage_p_value"]["leakage_claimed"]
    assert v(rec, "leakage_claimed") == 1.0 and rec.metrics["queries"]["total"] == 500


def test_decoy_is_an_explicit_placebo_with_no_actual_member(lab):
    _, recs = lab
    rec = recs["decoy"]
    assert rec.schema_version == "mlbreak.membership/v2" and rec.evaluation_contract == "placebo"
    assert not any(c.truth.in_training for c in rec.candidates)                       # zero actual members
    assert v(rec, "actual_member_fraction") == 0.0 and rec.metrics["n_members"] == 0
    assert {c.truth.placebo_group for c in rec.candidates} == {"A", "B"}
    assert all((c.truth.placebo_group == "A") == (c.truth.population == "victim_train") for c in rec.candidates)
    for a in ("threshold", "shadow"):
        # positive-class membership metrics are UNDEFINED, never computed from placebo group labels
        for k in ("roc_auc", "membership_recall", "membership_advantage", "tpr_at_low_fpr", "leakage_p_value",
                  "balanced_accuracy", "operating_point_p_value"):
            assert v(rec, f"{a}_{k}") is None, (a, k)
        assert rec.metrics[f"{a}_roc_curve"]["points"] is None
        assert all(p["precision"] is None for p in rec.metrics[f"{a}_operational_precision"]["values"])
        # false accusations of real non-members ARE defined
        assert rec.metrics[f"{a}_false_positive_rate"]["denominator"] == 500
        # the placebo analysis is separate and labelled as such
        pl = rec.metrics[f"{a}_placebo_auc"]
        assert abs(pl["value"] - 0.5) < 0.1 and pl["population"].startswith("PLACEBO")
    # no unsupported claim: the membership claim is undefined and placebo discrimination is not claimed
    assert rec.metrics["leakage_claimed"]["value"] is None and rec.metrics["leakage_claimed"]["status"] == "undefined"
    assert v(rec, "placebo_discrimination_claimed") == 0.0 and v(rec, "generalization_gap") >= 0.2


def test_ordinary_membership_metrics_use_actual_membership(lab):
    from mlbreak.membership.metrics import derive_membership_metrics
    _, recs = lab
    rec = recs["attack"]
    assert rec.evaluation_contract == "membership" and v(rec, "actual_member_fraction") == 0.5
    args = (rec.evaluation, None, rec.utility, [0.5, 0.1, 0.01], rec.calibration, 1200, True)
    # relabel the DESIGNATED groups while keeping actual membership: v2 metrics must not move, v1 semantics would
    flipped = [c.model_copy(update={"truth": c.truth.model_copy(update={"group": "non_member" if c.truth.group == "member"
                                                                        else "member"})}) for c in rec.candidates]
    a2 = list(args)
    a2[1] = flipped
    assert derive_membership_metrics(*a2, contract="membership") == rec.metrics
    assert derive_membership_metrics(*a2, contract=None)["threshold_roc_auc"]["value"] < 0.2  # v1 rule follows groups


def test_low_leakage_reference_has_real_members_and_is_measured(engine):
    rec = engine.membership.run_attack(MembershipAttackRequest(
        model_id="logreg-l2", candidates=CandidateSelection(n_members=80, n_non_members=80),
        attack=MembershipAttackConfig(n_shadow_models=2)))
    assert rec.evaluation_contract == "membership" and v(rec, "actual_member_fraction") == 0.5
    assert all(c.truth.in_training == (c.truth.population == "victim_train") for c in rec.candidates)
    for a in ("threshold", "shadow"):   # measured, not assumed: the values exist and are near chance here
        assert v(rec, f"{a}_roc_auc") is not None and v(rec, f"{a}_leakage_p_value") is not None
    assert rec.metrics["leakage_claimed"]["value"] in (0.0, 1.0)


def test_baseline_has_no_attacker_output_and_reports_no_signal(lab):
    _, recs = lab
    rec = recs["baseline"]
    assert all(c.observation is None and c.threshold is None for c in rec.candidates)
    assert v(rec, "no_signal_accuracy") == 0.5 and "threshold_roc_auc" not in rec.metrics


def test_statistics_helpers():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(0.3, 1, 80), rng.normal(0, 1, 120)
    assert auc_mann_whitney(pos, neg)["auc"] == pytest.approx(roc_auc_score(np.r_[np.ones(80), np.zeros(120)],
                                                                            np.r_[pos, neg]))
    same = auc_mann_whitney([1.0] * 10, [1.0] * 10)
    assert same["auc"] == 0.5 and same["p_value"] == 1.0
    assert auc_mann_whitney(np.arange(50) + 100, np.arange(50))["p_value"] < 1e-15
    from math import comb
    brute = sum(comb(10, x) * comb(10, 8 - x) for x in range(6, 9)) / comb(20, 8)
    assert fisher_one_sided(6, 4, 2, 8) == pytest.approx(brute)
    lo, hi = wilson(1, 250)
    assert 0 < lo < 0.004 < hi < 0.03 and wilson(0, 0) is None


# ------------------------------------------------------------------ defenses
def test_regularization_reduces_leakage_under_recalibrated_attack(lab):
    eng, recs = lab
    d, u = recs["regularization"], recs["attack"]
    assert d.evaluation == "adaptive_membership_inference" and d.lineage["parent_experiment_id"] == u.experiment_id
    assert d.defense_config["kind"] == "regularization"
    assert d.model["hyperparameters"]["min_samples_leaf"] == 25 and d.model["artifact_sha256"] != u.model["artifact_sha256"]
    # recalibrated: the attacker's reference and shadow models use the defended recipe
    assert all(m["hyperparameters"]["min_samples_leaf"] == 25 for m in d.calibration["reference_models"])
    assert all(m["hyperparameters"]["min_samples_leaf"] == 25 for m in d.shadow["models"])
    assert [c.candidate_id for c in d.candidates] == [c.candidate_id for c in u.candidates]
    for a in ("threshold", "shadow"):
        assert v(d, f"{a}_roc_auc") < v(u, f"{a}_roc_auc") - 0.2
        assert v(d, f"{a}_defense_reduction_p_value") < 1e-6
    assert "test_accuracy" in d.metrics and v(d, "generalization_gap") < v(u, "generalization_gap")
    assert eng.verify(d.experiment_id)["ok"]


def test_label_only_policy_is_applied_to_reference_shadow_and_victim(lab):
    _, recs = lab
    d = recs["label_only"]
    assert d.output_policy == {"kind": "label_only"}
    assert {c.observation.released_output for c in d.candidates} <= {0.0, 1.0}
    assert {r.released for r in d.utility} <= {0.0, 1.0}
    assert set(d.calibration["attacks"]["threshold"]["scores"]) <= {0.0, 1.0}  # reference outputs restricted too
    assert d.metrics["threshold_tpr_at_low_fpr"]["status"] == "undefined"  # cannot be estimated, not invented
    assert v(d, "threshold_membership_advantage") > 0.2  # documented failure: correctness still leaks


def test_label_unaware_attack_against_label_only_has_no_signal(engine):
    req = MembershipAttackRequest(candidates=CandidateSelection(n_members=60, n_non_members=60),
                                  attack=MembershipAttackConfig(label_known=False, threshold_signal="max_confidence",
                                                                n_shadow_models=2))
    atk = engine.membership.run_attack(req)
    d = engine.membership.run_defend(MembershipDefendRequest(parent_experiment_id=atk.experiment_id,
                                                             defense=OutputRestriction(policy=LabelOnly())))
    assert v(d, "threshold_roc_auc") == 0.5
    ops = {p["name"]: p for p in d.calibration["attacks"]["threshold"]["operating_points"]}
    assert ops["balanced"]["status"] == "undefined" and ops["balanced"]["thresholds"] == {"all": None}
    assert all(c.observation.label is None for c in d.candidates)


def test_complexity_reduction_refused_for_logistic_regression(engine):
    atk = engine.membership.run_attack(MembershipAttackRequest(
        model_id="logreg-l2", candidates=CandidateSelection(n_members=40, n_non_members=40),
        attack=MembershipAttackConfig(n_shadow_models=1, n_reference_models=1)))
    with pytest.raises(InvalidExperiment, match="tree ensembles"):
        engine.membership.run_defend(MembershipDefendRequest(parent_experiment_id=atk.experiment_id,
                                                             defense=ComplexityReduction()))
    reg = engine.membership.run_defend(MembershipDefendRequest(parent_experiment_id=atk.experiment_id,
                                                               defense=Regularization(C=0.001)))
    assert reg.lineage["defense_applied"] == {"training_changed": True, "parameter": "C", "value": 0.001}


def test_calibration_status_reports_approximate_and_fallbacks(engine):
    rec = engine.membership.run_attack(MembershipAttackRequest(
        candidates=CandidateSelection(n_members=50, n_non_members=50),
        attack=MembershipAttackConfig(n_reference_models=1, n_shadow_models=1, threshold_mode="class_conditional",
                                      min_class_calibration=1000)))
    cal = rec.calibration["attacks"]["threshold"]
    assert all(m.startswith("global threshold") for m in cal["class_modes"].values())  # not enough data per class
    low = next(p for p in cal["operating_points"] if p["name"] == "low_fpr")
    assert low["status"] == "approximate"  # 600 calibration non-members allow < 10 false positives at 1%
    assert engine.verify(rec.experiment_id)["ok"]


# ------------------------------------------------------------------ budgets, integrity, replay
def test_query_budget_is_enforced(engine):
    rec = engine.membership.run_attack(MembershipAttackRequest(
        candidates=CandidateSelection(n_members=30, n_non_members=30), attack=MembershipAttackConfig(query_budget=59)))
    assert rec.status == "failed" and rec.error.startswith("QueryBudgetExceeded")
    assert engine.verify(rec.experiment_id)["ok"]


@pytest.mark.parametrize("edit", [
    lambda d: d["candidates"][0]["truth"].update(group="non_member" if d["candidates"][0]["truth"]["group"] == "member"
                                                 else "member"),
    lambda d: d["candidates"][3]["threshold"].update(member_balanced=not d["candidates"][3]["threshold"]["member_balanced"]),
    lambda d: d["calibration"]["attacks"]["threshold"]["is_member"].__setitem__(0, False),
    lambda d: d["metrics"]["threshold_roc_auc"].update(value=0.99),
    lambda d: d["candidates"][5]["observation"].update(released_output=0.123456),
    lambda d: d["candidates"][2]["truth"].update(in_training=not d["candidates"][2]["truth"]["in_training"]),
    lambda d: d["candidates"][7]["observation"]["features"].__setitem__(0, 0.5),
    lambda d: d.update(evaluation_contract="placebo"),
])
def test_tampered_evidence_is_rejected_and_not_replayed(lab, edit):
    eng, recs = lab
    rid = recs["attack"].experiment_id
    path = eng.store.json_path(rid)
    original = path.read_text()
    try:
        data = json.loads(original)
        edit(data)
        path.write_text(json.dumps(data))
        assert not eng.verify(rid)["ok"]
        with pytest.raises(EvidenceIntegrityError):
            eng.replay(rid)
    finally:
        path.write_text(original)
    assert eng.verify(rid)["ok"]


def test_exact_replay(lab):
    eng, recs = lab
    r = eng.replay(recs["regularization"].experiment_id)
    assert r["identical"], [c for c in r["checks"] if not c["match"]]


def test_requests_are_strict():
    from pydantic import ValidationError
    for bad in ({"attack": {"label_known": False}},                                   # label signal without label
                {"candidates": {"n_members": 0}}, {"candidates": {"n_members": 601}},
                {"attack": {"n_shadow_models": 17}}, {"attack": {"query_budget": 5000}},
                {"membership": [1, 0]}, {"candidates": {"operational_prevalences": [0.1, 0.1]}}):
        with pytest.raises(ValidationError):
            MembershipAttackRequest.model_validate(bad)


# ------------------------------------------------------------------ API and roles
def test_membership_api_is_evaluator_only(tmp_path):
    for method, rx in ATTACKER_ROUTES:
        assert not rx.fullmatch("/api/experiments/membership/attack") and not rx.fullmatch("/api/membership/study")
    c = TestClient(create_app(root=tmp_path / "var", frontend_dist=None, evaluator_token="s3cret"))
    body = {"candidates": {"n_members": 20, "n_non_members": 20}, "attack": {"n_shadow_models": 1,
                                                                           "n_reference_models": 1}}
    assert c.post("/api/experiments/membership/attack", json=body).status_code == 401
    r = c.post("/api/experiments/membership/attack", json=body, headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200 and r.json()["family"] == "membership"
    eid = r.json()["experiment_id"]
    assert c.get(f"/api/experiments/{eid}").status_code == 401
    assert c.get(f"/api/membership/independence/{eid}").status_code == 401
    assert c.get(f"/api/membership/independence/{eid}", headers={"Authorization": "Bearer s3cret"}).json()["ok"]
    tags = {t for p in c.get("/openapi.json").json()["paths"].values() for m in p.values() for t in m.get("tags", [])}
    assert "ml04" in tags


def test_membership_api_contract(client):
    body = {"candidates": {"n_members": 30, "n_non_members": 30}, "attack": {"n_shadow_models": 1}}
    r = client.post("/api/experiments/membership/attack", json=body)
    assert r.status_code == 200
    eid = r.json()["experiment_id"]
    d = client.post("/api/experiments/membership/defend", json={"parent_experiment_id": eid,
                                                                "defense": {"kind": "complexity_reduction"}})
    assert d.status_code == 200 and d.json()["evaluation"] == "adaptive_membership_inference"
    assert client.post("/api/experiments/membership/defend", json={"parent_experiment_id": "MLB-EXP-000000000000"}
                       ).status_code == 404
    assert client.post("/api/experiments/membership/attack", json={"seed": -1}).status_code == 422
    assert "## 7. Integrity verification" in client.get(f"/api/experiments/{eid}/report.md").text
    csv = client.get(f"/api/experiments/{eid}/samples.csv").text.splitlines()[0]
    assert "threshold_score" in csv and "eval_in_training" in csv and "eval_designated_group" in csv
    lst = {e["experiment_id"]: e for e in client.get("/api/experiments").json()}
    assert lst[eid]["family"] == "membership" and lst[eid]["membership_auc"] is not None
    assert client.get("/api/membership/study/MLB-MIS-../../x").status_code == 404
    cats = {c["code"]: c for c in client.get("/api/owasp").json()["categories"]}
    assert cats["ML04"]["lab_status"] == "implemented" and cats["ML05"]["lab_status"] == "not_implemented"


def test_seed_study_is_verifiable_and_replays(tmp_path):
    from mlbreak.membership.study import CONFIGS, StudyRequest
    eng = ExperimentEngine(EvidenceStore(tmp_path / "var"))
    req = StudyRequest(seeds=[3], attack=MembershipAttackConfig(n_shadow_models=1, n_reference_models=1))
    rep = eng.membership.study(req)
    assert {r["config"] for r in rep.rows} == set(CONFIGS) and eng.membership.verify_study(rep.report_id)["ok"]
    assert rep.summary["decoy"]["seeds"] == 1 and "defense_failures" in rep.summary["quantized_4"]
    path = eng.store.root / "membership_studies" / f"{rep.report_id}.json"
    original = path.read_text()
    data = json.loads(original)
    data["rows"][0]["threshold_roc_auc"] = 0.0
    path.write_text(json.dumps(data))
    assert not eng.membership.verify_study(rep.report_id)["ok"]
    with pytest.raises(EvidenceIntegrityError):
        eng.membership.replay_study(rep.report_id)
    path.write_text(original)
    assert eng.membership.replay_study(rep.report_id)["identical"]


def test_record_family_round_trips_and_older_families_still_parse(lab):
    from mlbreak.evidence import parse_record
    eng, recs = lab
    data = json.loads(eng.store.json_path(recs["attack"].experiment_id).read_text())
    assert isinstance(parse_record(data), MembershipRecord) and data["schema_version"] == "mlbreak.membership/v2"
    assert data["evaluation_contract"] == "membership"


# ------------------------------------------------------------------ UI and docs
def _top_level_fields(body: str) -> set[str]:
    import re
    flat, depth = [], 0
    for ch in body:
        depth += ch in "{([<"
        depth -= ch in "})]>"
        flat.append(ch if depth == 0 else " ")
    return set(re.findall(r"(?:^|[;\n])\s*([A-Za-z_][A-Za-z0-9_]*)\??\s*:", "".join(flat)))


def test_frontend_calls_existing_routes_and_types_match_records(client):
    from mlbreak.membership.schemas import CandidateResultV2, MembershipRecordV2, MembershipTruthV2
    api = (PROJECT_ROOT / "frontend" / "src" / "api.ts").read_text()
    routes = {r.path for r in client.app.routes}
    for path in ("/api/experiments/membership/baseline", "/api/experiments/membership/attack",
                 "/api/experiments/membership/defend"):
        assert path in api and path in routes, path
    types = (PROJECT_ROOT / "frontend" / "src" / "types.ts").read_text()
    for model, iface in ((MembershipRecordV2, "MembershipRecord"), (CandidateResultV2, "MembershipCandidate"),
                         (MembershipTruthV2, "MembershipTruth")):
        body = types.split(f"export interface {iface} {{")[1].split("\n}")[0]
        fields = _top_level_fields(body)
        assert fields and fields <= set(model.model_fields), (iface, fields - set(model.model_fields))
    page = (PROJECT_ROOT / "frontend" / "src" / "pages" / "MembershipWorkspace.tsx").read_text()
    assert "evaluator" in page.lower() and "not differential privacy" in page
    comp = (PROJECT_ROOT / "frontend" / "src" / "components" / "membership.tsx").read_text()
    assert "placebo" in comp.lower() and "evaluation_contract" in comp


def test_docs_cover_the_methodology():
    doc = (PROJECT_ROOT / "docs" / "ml04-membership-inference.md").read_text()
    for needle in ("OWASP ML04", "## 2. Threat model", "calibration", "shadow", "Placebo control", "placebo contract",
                   "Low-leakage reference", "Root cause", "### Schema versions and compatibility", "MLB-REG-026",
                   "not differential privacy", "ε = ∞", "MLB-REG-019", "MLB-REG-025", "Responsible use",
                   "Known limitations", "privacy–utility"):
        assert needle in doc, needle


# ------------------------------------------------------------------ historical evidence and reproducibility
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "membership_v1"


@pytest.fixture
def v1_engine(tmp_path):
    import shutil
    store = EvidenceStore(tmp_path / "var")
    for f in FIXTURES.glob("MLB-EXP-*.json"):
        shutil.copy(f, store.dir / f.name)
    return ExperimentEngine(store), json.loads((FIXTURES / "manifest.json").read_text())


def test_historical_v1_records_load_and_verify_under_their_schema(v1_engine):
    from mlbreak.integrity import record_sha256
    from mlbreak.membership.schemas import MembershipRecordV2
    eng, man = v1_engine
    for name, m in man.items():
        rec = eng.store.load(m["experiment_id"])
        assert rec.schema_version == "mlbreak.membership/v1" and not isinstance(rec, MembershipRecordV2)
        assert record_sha256(rec) == m["record_sha256"] and rec.reproducibility["results_sha256"] == m["results_sha256"]
        report = eng.verify(m["experiment_id"])
        assert report["ok"], [c for c in report["checks"] if c["status"] == "fail"]
        contract = next(c for c in report["checks"] if c["check"] == "evaluation_contract")
        assert contract["status"] == "skipped" and ("DECOY" in contract["detail"]) == (name == "decoy_attack")
    assert "Schema v1 decoy record" in eng.report(man["decoy_attack"]["experiment_id"])


def test_historical_v1_replay_and_defend(v1_engine):
    eng, man = v1_engine
    r = eng.replay(man["decoy_attack"]["experiment_id"])   # verified first: refused if the evidence fails
    checks = {c["check"]: c["match"] for c in r["checks"]}
    assert "metrics_sha256" not in checks and "v1" in r["notes"][0]
    # schema- and environment-independent parts of a replay must always match
    assert all(checks[k] for k in ("status", "config_sha256", "dataset_sha256", "ground_truth_sha256")), r["checks"]
    if "numerical_environment_sha256" not in r["environment_differences"]:
        assert r["identical"], r["checks"]          # same numerical environment as the fixture: bit-identical
    else:
        # fixtures were written on macOS/arm64; model training is only bit-reproducible within one numerical
        # environment (docs/reproducibility.md), and the replay must SAY so rather than claim identity
        assert any("environment" in n.lower() for n in r["notes"]), r["notes"]
    new = eng.store.load(r["replay_experiment_id"])
    assert new.schema_version == "mlbreak.membership/v2" and new.evaluation_contract == "placebo"
    with pytest.raises(InvalidExperiment, match="v1"):
        eng.membership.run_defend(MembershipDefendRequest(parent_experiment_id=man["rf_attack"]["experiment_id"]))


def test_results_are_reproducible_across_seeds(engine):
    def run(seed, model="random-forest-overfit"):
        return engine.membership.run_attack(MembershipAttackRequest(
            model_id=model, seed=seed, candidates=CandidateSelection(n_members=60, n_non_members=60),
            attack=MembershipAttackConfig(n_shadow_models=1, n_reference_models=1)))
    for seed in (3, 11):
        a, b = run(seed), run(seed)
        assert a.reproducibility["results_sha256"] == b.reproducibility["results_sha256"]
        assert v(a, "threshold_roc_auc") > 0.7                      # the overfit victim leaks on every seed tried
        d = run(seed, "random-forest-decoy")                         # the placebo never yields a membership claim
        assert d.metrics["leakage_claimed"]["value"] is None and d.evaluation_contract == "placebo"
    assert run(3).reproducibility["results_sha256"] != run(11).reproducibility["results_sha256"]


# ------------------------------------------------------------------ cross-platform verification of historical evidence
FIXTURE_SHA256 = {  # committed bytes of the historical fixtures: never regenerated or re-hashed
    "MLB-EXP-1caef42a85e2.json": "3a3c7920bd9310d468aeda58cb7aeea438fe0b8ca704d9a281e7d92fbc663bd6",
    "MLB-EXP-b1e77a5a68cb.json": "575c412584c2e57e3a12a41beef0ea9c9ab2694b8d912b29a5d717b9f2405e70",
    "MLB-EXP-cc6efc5522f0.json": "8a5a8c2475fe9786193d54deceeca6c285644c3490434d27a56a316734c277ad",
    "manifest.json": "901482d7800b3d4b5d20d62553f31b992f211a1b66d845965b68f98ddc51d6de",
}


def _fixture(name):
    from mlbreak.evidence import parse_record
    return parse_record(json.loads((FIXTURES / name).read_text()))


def test_historical_fixtures_are_byte_identical():
    import hashlib
    assert {p.name for p in FIXTURES.iterdir()} == set(FIXTURE_SHA256)
    for name, digest in FIXTURE_SHA256.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest, name


def test_v1_metrics_keep_v1_semantics_and_are_not_reinterpreted():
    from mlbreak.membership.integrity import metric_differences
    from mlbreak.membership.metrics import derive_membership_metrics
    rec = _fixture("MLB-EXP-cc6efc5522f0.json")                                   # v1 decoy
    args = (rec.evaluation, rec.candidates, rec.utility, rec.candidate_selection["operational_prevalences"],
            rec.calibration, rec.attack_config["query_budget"], True)
    assert rec.metrics["threshold_roc_auc"]["value"] is not None                 # v1: designated-group labels
    assert metric_differences(rec.metrics, derive_membership_metrics(*args, None)) == []
    v2 = derive_membership_metrics(*args, "membership")       # v2 rule (actual membership: none) would differ
    assert v2["threshold_roc_auc"]["value"] is None and metric_differences(rec.metrics, v2)


def test_libm_rounding_differences_are_tolerated_but_material_changes_are_not():
    """Simulates the macOS-libm vs glibc difference seen in CI: erfc output off by a few ULP."""
    from mlbreak.membership.integrity import metric_differences
    m = {"threshold_leakage_p_value": {"value": 0.5484553232697535, "z": -0.12175967182733022,
                                       "leakage_claimed": False, "alpha": 0.05, "definition": "d"}}
    ulp2 = np.nextafter(np.nextafter(0.5484553232697535, 1.0), 1.0)
    same = {"threshold_leakage_p_value": {**m["threshold_leakage_p_value"], "value": float(ulp2)}}
    assert metric_differences(m, same) == []                                       # 2 ULP: platform rounding
    for field, val in (("value", 0.5484553232697535 * (1 + 1e-9)), ("value", 0.01), ("value", None),
                       ("z", float(np.nextafter(-0.12175967182733022, 0.0))),     # z is exact: 1 ULP fails
                       ("leakage_claimed", True), ("definition", "other")):
        bad = {"threshold_leakage_p_value": {**m["threshold_leakage_p_value"], field: val}}
        assert metric_differences(m, bad), (field, val)
    other = {"threshold_roc_auc": {"value": 0.49}}                                # non-libm values stay exact
    assert metric_differences(other, {"threshold_roc_auc": {"value": float(np.nextafter(0.49, 1.0))}})


def test_fixtures_verify_under_a_different_platform_erfc(monkeypatch, v1_engine):
    import mlbreak.membership.metrics as mm
    orig = mm.math.erfc
    eng, man = v1_engine
    monkeypatch.setattr(mm.math, "erfc", lambda x: orig(x) * (1 + 4.4e-16))      # ~2 ULP, as on glibc
    for m in man.values():
        assert eng.verify(m["experiment_id"])["ok"]
    monkeypatch.setattr(mm.math, "erfc", lambda x: orig(x) * (1 + 1e-9))         # a different statistic
    report = eng.verify(man["decoy_attack"]["experiment_id"])
    failed = {c["check"] for c in report["checks"] if c["status"] == "fail"}
    assert not report["ok"] and "metrics_derivation" in failed


@pytest.mark.parametrize("edit", [
    lambda d: d["metrics"]["threshold_leakage_p_value"].update(value=d["metrics"]["threshold_leakage_p_value"]["value"] * 1.01),
    lambda d: d["metrics"]["threshold_leakage_p_value"].update(value=float(np.nextafter(
        d["metrics"]["threshold_leakage_p_value"]["value"], 1.0))),               # even 1 ULP: record hash
    lambda d: d["metrics"]["threshold_leakage_p_value"].update(leakage_claimed=True),
    lambda d: d["candidates"][4]["threshold"].update(score=0.999),
    lambda d: d["candidates"][1]["truth"].update(in_training=not d["candidates"][1]["truth"]["in_training"]),
    lambda d: d.update(schema_version="mlbreak.membership/v2"),
])
def test_tampered_historical_evidence_is_rejected_and_not_replayed(v1_engine, edit):
    eng, man = v1_engine
    rid = man["decoy_attack"]["experiment_id"]
    path = eng.store.json_path(rid)
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    try:
        ok = eng.verify(rid)["ok"]
    except Exception:  # an edit that no longer parses (e.g. v2 without a contract) is rejected outright
        ok = False
    assert not ok
    with pytest.raises(Exception):
        eng.replay(rid)


def test_nll_scores_tolerate_platform_log_but_not_edits():
    from mlbreak.membership.integrity import _libm_close
    x = float(np.log(0.37))
    assert _libm_close(x, float(np.nextafter(x, 0.0))) and not _libm_close(x, x * (1 + 1e-9))
    assert not _libm_close(x, None) and _libm_close(None, None)
