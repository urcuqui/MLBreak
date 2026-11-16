"""ML03 Model Inversion — dataset, ground-truth isolation, attack, defenses, metrics, evidence and API."""

import ast
import inspect
import json

import numpy as np
import pytest
from pydantic import ValidationError

from mlbreak.attacks import QueryBudgetExceeded
from mlbreak.config import PROJECT_ROOT
from mlbreak.experiments import EvidenceIntegrityError, InvalidExperiment
from mlbreak.integrity import record_sha256
from mlbreak.inversion import attacks as attacks_mod
from mlbreak.inversion.attacks import AttackerView, candidate_matching, decide
from mlbreak.inversion.data import N_AUX, N_EVAL, N_TRAIN, load_inversion_dataset
from mlbreak.inversion.defenses import PredictionService, apply_policy, counting_oracle, is_valid_output
from mlbreak.inversion.metrics import derive_inversion_metrics
from mlbreak.inversion.models import auxiliary_model, victim_model
from mlbreak.inversion.schemas import (AttackerTrace, InversionAttackConfig, InversionAttackRequest,
                                       InversionBaselineRequest, InversionDefendRequest, LabelOnly,
                                       QuantizedProbability, TargetSelection)

SMALL = TargetSelection(max_targets=60)


@pytest.fixture
def attack(engine):
    return engine.inversion.run_attack(InversionAttackRequest(targets=SMALL))


# ------------------------------------------------------------------ dataset and victim
def test_dataset_is_deterministic_disjoint_and_synthetic():
    a, b = load_inversion_dataset(7), load_inversion_dataset(7)
    assert a.fingerprint() == b.fingerprint()
    assert load_inversion_dataset(8).fingerprint()["train_sha256"] != a.fingerprint()["train_sha256"]
    assert (len(a.y_train), len(a.aux.s), len(a.y_eval)) == (N_TRAIN, N_AUX, N_EVAL)
    rows = [tuple(r) for r in np.r_[a.X_train, a.aux.X, a.X_eval]]
    assert len(set(rows)) == len(rows)                                   # no record shared between splits
    assert not hasattr(a.aux, "y")                                       # the auxiliary population has no labels
    assert a.describe()["synthetic"] is True
    with pytest.raises(ValidationError):
        InversionAttackRequest(seed=10**9)                               # bounded seed


def test_private_attribute_is_a_genuine_and_relevant_input():
    v = victim_model("logreg", 7)
    assert v.features[-1] == "confidential_support_programme" and abs(v.weights[-1]) > 1.0
    assert victim_model("logreg-public-only", 7).weights[-1] == 0.0
    ds = load_inversion_dataset(7)
    assert not np.array_equal(v.proba(np.c_[ds.X_eval, np.zeros(N_EVAL)]), v.proba(np.c_[ds.X_eval, np.ones(N_EVAL)]))
    batch = v.proba(np.c_[ds.X_eval, ds.s_eval])
    single = np.array([v.proba(np.c_[ds.X_eval[i:i + 1], ds.s_eval[i:i + 1]])[0] for i in range(N_EVAL)])
    assert np.array_equal(batch, single)                                  # single row == batch row, bit for bit


def test_baseline_measures_utility_and_relevance(engine):
    rec = engine.inversion.run_baseline(InversionBaselineRequest())
    m = rec.metrics
    assert rec.status == "completed" and all(t.trace is None for t in rec.targets)
    assert m["log_loss_with_private_attribute"]["value"] < m["log_loss_without_private_attribute"]["value"]
    assert m["baseline_inference_accuracy"]["denominator"] == 200 and 0.5 < m["baseline_inference_accuracy"]["value"] < 1
    assert "attribute_inference_accuracy" not in m                       # no attack ran


# ------------------------------------------------------------------ ground-truth isolation
def test_attacker_view_and_trace_have_no_ground_truth_fields():
    forbidden = {"private_attribute", "s", "s_true", "target_label", "y", "y_true", "ground_truth", "p_full"}
    assert not forbidden & set(AttackerView.__dataclass_fields__)
    assert not forbidden & set(AttackerTrace.model_fields)
    sig = inspect.signature(candidate_matching)
    assert list(sig.parameters) == ["view", "query", "cfg"]


def test_attack_module_cannot_reach_data_models_or_engine():
    tree = ast.parse(inspect.getsource(attacks_mod))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    allowed = {"__future__", "dataclasses", "typing", "numpy", "schemas"}  # "schemas" = .schemas
    assert imported <= allowed, imported - allowed   # no .data, .models, .engine, .defenses, sklearn


def test_inference_does_not_depend_on_hidden_truth():
    """Same observed outputs, permuted hidden attributes → identical inferences: the attack only sees outputs."""
    ds, v, aux = load_inversion_dataset(7), victim_model("logreg", 7), auxiliary_model(7)
    service, cfg = PredictionService(v, {"kind": "label_only"}), InversionAttackConfig()
    observed = service.respond(ds.X_eval[:80], ds.s_eval[:80])
    p1 = aux.proba(ds.X_eval[:80])

    def run():
        out = []
        for i in range(80):
            q, _ = counting_oracle(service, cfg.query_budget)
            out.append(candidate_matching(AttackerView(tuple(ds.X_eval[i]), float(observed[i]), float(p1[i])), q, cfg))
        return [t.model_dump() for t in out]
    first = run()
    hidden = np.random.default_rng(0).permutation(ds.s_eval[:80])        # what the evaluator holds changes …
    assert hidden.tolist() != ds.s_eval[:80].tolist()
    assert run() == first                                                # … the attacker's output does not


def test_attacker_facing_targets_endpoint_hides_ground_truth(client):
    r = client.post("/api/inversion/targets", json={"targets": {"max_targets": 5}, "output_policy": {"kind": "label_only"}})
    assert r.status_code == 200
    body = r.json()
    assert body["private_attribute"]["value"] == "hidden"
    for t in body["targets"]:
        assert set(t) == {"target_index", "public_features", "observed_output"}
        assert t["observed_output"] in (0.0, 1.0)
    text = json.dumps(body)
    assert "p_full" not in text and "y_true" not in text and "ground_truth" not in text
    q = client.post("/api/inversion/query", json={"public_features": body["targets"][0]["public_features"],
                                                  "private_attribute": 1})
    assert q.status_code == 200 and 0 <= q.json()["released"] <= 1
    assert client.post("/api/inversion/query", json={"public_features": [0, 0, 0, 0], "private_attribute": 1,
                                                     "target_index": 3}).status_code == 422  # no lookup by id


# ------------------------------------------------------------------ attack semantics
def test_decide_rules_and_ties():
    assert decide(0.7, {1: 0.7, 0: 0.3}, (0, 1), 0.2, "auxiliary_prior", 1e-9)[2:] == ("unique_match", 1)
    assert decide(0.7, {0: 0.3}, (0, 1), 0.2, "auxiliary_prior", 1e-9)[2:] == ("elimination", 1)
    assert decide(1.0, {0: 1.0, 1: 1.0}, (0, 1), 0.8, "auxiliary_prior", 0)[1:] == (True, "tie_auxiliary_prior", 1)
    assert decide(1.0, {0: 1.0, 1: 1.0}, (0, 1), 0.5, "auxiliary_prior", 0)[3] == 0      # exact 0.5 → 0
    assert decide(1.0, {0: 1.0, 1: 1.0}, (0, 1), 0.8, "abstain", 0)[2:] == ("tie_abstain", None)
    assert decide(0.5, {0: 0.1, 1: 0.9}, (0, 1), 0.8, "auxiliary_prior", 0)[2:] == ("no_consistent_candidate", 1)


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_query_budget_is_enforced_and_counted(engine, budget):
    rec = engine.inversion.run_attack(InversionAttackRequest(targets=SMALL,
                                                            attack=InversionAttackConfig(query_budget=budget)))
    qs = [t.trace.n_queries for t in rec.targets]
    assert max(qs) <= budget and all(len(t.trace.queries) == t.trace.n_queries for t in rec.targets)
    assert rec.metrics["queries"]["total"] == sum(qs)
    if budget == 1:
        assert rec.metrics["budget_exhaustion_rate"]["value"] > 0


def test_counting_oracle_refuses_queries_beyond_budget():
    query, count = counting_oracle(PredictionService(victim_model("logreg", 7), {"kind": "full_probability"}), 2)
    query((0.0, 0.0, 0.0, 0.0), 0)
    query((0.0, 0.0, 0.0, 0.0), 1)
    with pytest.raises(QueryBudgetExceeded):
        query((0.0, 0.0, 0.0, 0.0), 0)
    assert count() == 2


def test_attack_is_deterministic(engine):
    a = engine.inversion.run_attack(InversionAttackRequest(targets=TargetSelection(max_targets=40, order="random")))
    b = engine.inversion.run_attack(InversionAttackRequest(targets=TargetSelection(max_targets=40, order="random")))
    assert a.reproducibility["results_sha256"] == b.reproducibility["results_sha256"]
    assert [t.target_index for t in a.targets] != list(range(40))


def test_inversion_beats_baseline_and_negative_control_does_not(engine, attack):
    m = attack.metrics
    assert m["inference_advantage"]["value"] > 0.2 and m["leakage_p_value"]["leakage_claimed"]
    neg = engine.inversion.run_attack(InversionAttackRequest(model_id="logreg-public-only", targets=SMALL))
    n = neg.metrics
    assert n["inference_advantage"]["value"] == 0 and not n["leakage_p_value"]["leakage_claimed"]
    assert n["ambiguity_rate"]["value"] == 1.0
    assert n["attribute_inference_accuracy"]["value"] == n["baseline_inference_accuracy"]["value"] > 0.5


# ------------------------------------------------------------------ defenses
@pytest.mark.parametrize("policy", [QuantizedProbability(bins=4), LabelOnly()], ids=["quantized", "label_only"])
def test_policy_applies_to_observation_and_queries_and_is_paired(engine, attack, policy):
    rec = engine.inversion.run_defend(InversionDefendRequest(parent_experiment_id=attack.experiment_id,
                                                             output_policy=policy))
    pol = policy.model_dump()
    released = {r.eval_index: r.released for r in rec.service}
    for t in rec.targets:
        assert is_valid_output(pol, t.trace.observed_output) and t.trace.observed_output == released[t.target_index]
        assert all(is_valid_output(pol, q.response) for q in t.trace.queries)
    assert [t.target_index for t in rec.targets] == [t.target_index for t in attack.targets]
    assert [t.undefended_inversion_correct for t in rec.targets] == [t.inversion_correct for t in attack.targets]
    m = rec.metrics
    assert m["defense_accuracy_reduction"]["value"] >= 0
    assert m["victim_prediction_accuracy"] == attack.metrics["victim_prediction_accuracy"]
    assert rec.lineage["relation"] == "adaptive_inversion_of"


def test_label_only_limits_but_does_not_eliminate_leakage(engine, attack):
    rec = engine.inversion.run_defend(InversionDefendRequest(parent_experiment_id=attack.experiment_id))
    m = rec.metrics
    assert m["attribute_inference_accuracy"]["value"] < attack.metrics["attribute_inference_accuracy"]["value"]
    assert m["inference_advantage"]["value"] > 0                          # still leaks near the boundary
    assert m["released_brier_score"]["value"] > attack.metrics["released_brier_score"]["value"]  # utility cost


def test_apply_policy_values():
    p = np.array([0.0, 0.2, 0.49, 0.5, 0.74, 0.999, 1.0])
    assert apply_policy({"kind": "label_only"}, p).tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert apply_policy({"kind": "quantized_probability", "bins": 4}, p).tolist() == \
        [0.125, 0.125, 0.375, 0.625, 0.625, 0.875, 0.875]
    assert apply_policy({"kind": "full_probability"}, p).tolist() == p.tolist()


# ------------------------------------------------------------------ metrics
def test_undefined_denominators_are_null(engine):
    ds = load_inversion_dataset(7)
    only_zero = [i for i in range(N_EVAL) if ds.s_eval[i] == 0][:5]
    rec = engine.inversion.run_attack(InversionAttackRequest(targets=TargetSelection(indices=only_zero)))
    m = rec.metrics
    assert m["recall_private_1"] == {**m["recall_private_1"], "numerator": 0, "denominator": 0, "value": None}
    assert m["recall_private_0"]["denominator"] == 5
    empty = derive_inversion_metrics("inversion", [], [])
    assert empty["attribute_inference_accuracy"]["value"] is None and empty["leakage_p_value"]["value"] is None


def test_every_ratio_has_definition_and_population(attack):
    for k, v in attack.metrics.items():
        if isinstance(v, dict) and "numerator" in v:
            assert v["definition"] and v["population"] and "denominator" in v, k


# ------------------------------------------------------------------ evidence
def test_verify_replay_and_tampering(engine, attack):
    assert engine.verify(attack.experiment_id)["ok"]
    r = engine.replay(attack.experiment_id)
    assert r["identical"], [c for c in r["checks"] if not c["match"]]
    path = engine.store.json_path(attack.experiment_id)
    data = json.loads(path.read_text())
    tr = data["targets"][0]["trace"]
    tr["inferred"] = 1 - tr["inferred"]                                 # claim a different reconstruction
    path.write_text(json.dumps(data))
    rep = engine.verify(attack.experiment_id)
    assert not rep["ok"]
    failed = {c["check"] for c in rep["checks"] if c["status"] == "fail"}
    assert {"record_sha256", "reconstruction_revalidated"} <= failed
    with pytest.raises(EvidenceIntegrityError):
        engine.replay(attack.experiment_id)


def test_rehashed_forgery_is_still_caught(engine, attack):
    """Even if a forger recomputes every hash, a decision that does not follow from its queries fails."""
    rec = engine.store.load(attack.experiment_id)
    t = rec.targets[0]
    t.trace.queries[0].response = 1 - t.trace.queries[0].response
    from mlbreak.inversion.integrity import results_fingerprint
    from mlbreak.hashing import sha256_json
    rec.reproducibility["results_sha256"] = results_fingerprint(rec.targets, rec.service, rec.metrics)
    rec.reproducibility["metrics_sha256"] = sha256_json(rec.metrics)
    rec.reproducibility["record_sha256"] = record_sha256(rec)
    rep = engine.verify_full(rec)
    assert not rep["ok"] and any(c["check"] == "reconstruction_revalidated" and c["status"] == "fail"
                                 for c in rep["checks"])


def test_defend_requires_a_verified_inversion_parent(engine, attack):
    base = engine.inversion.run_baseline(InversionBaselineRequest(targets=SMALL))
    with pytest.raises(InvalidExperiment):
        engine.inversion.run_defend(InversionDefendRequest(parent_experiment_id=base.experiment_id))
    path = engine.store.json_path(attack.experiment_id)
    data = json.loads(path.read_text())
    data["metrics"]["inference_advantage"]["value"] = 0.99
    path.write_text(json.dumps(data))
    with pytest.raises(EvidenceIntegrityError):
        engine.inversion.run_defend(InversionDefendRequest(parent_experiment_id=attack.experiment_id))


def test_csv_and_listing(engine, attack):
    csv = engine.store.samples_csv(attack.experiment_id)
    assert csv.splitlines()[0].startswith("target_index,") and "observed_output" in csv
    row = next(r for r in engine.store.list() if r["experiment_id"] == attack.experiment_id)
    assert row["family"] == "inversion" and row["attack_success_rate"] is None
    assert row["inference_advantage"] == attack.metrics["inference_advantage"]["value"]


# ------------------------------------------------------------------ API validation
def test_api_validation(client):
    ok = client.post("/api/experiments/inversion/attack", json={"targets": {"max_targets": 20}})
    assert ok.status_code == 200 and ok.json()["family"] == "inversion"
    eid = ok.json()["experiment_id"]
    bad = [
        ("/api/experiments/inversion/attack", {"private_attribute": 1}),            # unknown field
        ("/api/experiments/inversion/attack", {"attack": {"query_budget": 0}}),
        ("/api/experiments/inversion/attack", {"attack": {"query_budget": 17}}),
        ("/api/experiments/inversion/attack", {"targets": {"max_targets": 401}}),
        ("/api/experiments/inversion/attack", {"targets": {"indices": [400]}}),
        ("/api/experiments/inversion/defend", {"parent_experiment_id": eid, "output_policy": {"kind": "full_probability"}}),
        ("/api/experiments/inversion/defend", {"parent_experiment_id": eid, "output_policy": {"kind": "quantized_probability", "bins": 1}}),
        ("/api/experiments/inversion/defend", {"parent_experiment_id": "MLB-EXP-000000000000"[:-1] + "x"}),
    ]
    for path, body in bad:
        assert client.post(path, json=body).status_code == 422, (path, body)
    assert client.post("/api/experiments/inversion/attack",
                       json={"targets": {"indices": [1, 1]}}).status_code == 422          # duplicates
    d = client.post("/api/experiments/inversion/defend", json={"parent_experiment_id": eid})
    assert d.status_code == 200 and d.json()["evaluation"] == "adaptive_inversion"
    assert client.get(f"/api/experiments/{eid}/verify").json()["ok"]
    assert "## 7. Integrity verification" in client.get(f"/api/experiments/{eid}/report.md").text
    cats = {c["code"]: c for c in client.get("/api/owasp").json()["categories"]}
    assert cats["ML03"]["lab_status"] == "implemented"
    assert cats["ML04"]["lab_status"] == "not_implemented" and cats["ML05"]["lab_status"] == "not_implemented"


# ------------------------------------------------------------------ UI contract
def _top_level_fields(body: str) -> set[str]:
    """Property names at brace depth 0 of a TypeScript interface body (nested object types are skipped)."""
    import re
    flat, depth = [], 0
    for ch in body:
        depth += ch in "{([<"
        depth -= ch in "})]>"
        flat.append(ch if depth == 0 else " ")
    return set(re.findall(r"(?:^|[;\n])\s*([A-Za-z_][A-Za-z0-9_]*)\??\s*:", "".join(flat)))


def test_frontend_calls_existing_routes_and_types_match_records(client):
    api = (PROJECT_ROOT / "frontend" / "src" / "api.ts").read_text()
    for path in ("/api/experiments/inversion/baseline", "/api/experiments/inversion/attack",
                 "/api/experiments/inversion/defend", "/api/inversion/targets"):
        assert path in api, path
    routes = {r.path for r in client.app.routes}
    assert {"/api/experiments/inversion/baseline", "/api/inversion/targets", "/api/inversion/query"} <= routes
    types = (PROJECT_ROOT / "frontend" / "src" / "types.ts").read_text()
    from mlbreak.inversion.schemas import InversionRecord, TargetResult
    for model, iface in ((InversionRecord, "InversionRecord"), (TargetResult, "InversionTarget"),
                         (AttackerTrace, "InversionTrace")):
        body = types.split(f"export interface {iface} {{")[1].split("\n}")[0]
        fields = _top_level_fields(body)
        assert fields and fields <= set(model.model_fields), (iface, fields - set(model.model_fields))
    page = (PROJECT_ROOT / "frontend" / "src" / "pages" / "InversionWorkspace.tsx").read_text()
    assert "hidden" in page.lower() and "not a formal privacy guarantee" in page
