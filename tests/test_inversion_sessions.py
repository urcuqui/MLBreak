"""ML03 attack sessions and attacker/evaluator separation: budget bypass attempts, scope, ground-truth isolation."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from mlbreak.api import create_app, is_attacker_route
from mlbreak.config import LIMITS, PROJECT_ROOT
from mlbreak.inversion.data import load_inversion_dataset

TOKEN = "evaluator-secret-for-tests"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
# Names that must never appear in an attacker-facing response schema.
FORBIDDEN = {"private_attribute", "ground_truth", "target_label", "y_true", "y", "s", "p_full", "weights",
             "intercept", "coef", "private_attribute_weight", "artifact_sha256", "training_samples", "metrics",
             "correct", "inversion_correct", "baseline_correct", "attribute_inference_accuracy", "inference_advantage",
             "leakage_p_value", "auxiliary_p1", "service", "targets_truth"}


def _app(tmp_path, token=None):
    return TestClient(create_app(root=tmp_path / "var", cases=PROJECT_ROOT / "security_cases", frontend_dist=None,
                                 evaluator_token=token))


def _create(c, headers=None, **kw):
    body = {"targets": {"max_targets": 4}, **kw}
    r = c.post("/api/inversion/sessions", json=body, headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------------------------------- budget accounting
def test_session_counts_server_side_and_refuses_when_exhausted(client):
    s = _create(client, query_budget_per_target=2, query_budget_total=5)
    sid, t0 = s["session_id"], s["targets"][0]["target_index"]
    assert s["budget"] == {"per_target": 2, "total": 5, "used_total": 0, "remaining_total": 5,
                           "used_per_target": {str(t["target_index"]): 0 for t in s["targets"]}}
    r1 = client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t0, "candidate": 0}).json()
    r2 = client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t0, "candidate": 0}).json()
    assert (r1["query_number"], r2["query_number"]) == (1, 2) and r1["released"] == r2["released"]  # repeats count
    over = client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t0, "candidate": 1})
    assert over.status_code == 403 and over.json()["error"] == "query_budget_exhausted"
    assert over.json()["budget"]["used_per_target"][str(t0)] == 2
    others = [t["target_index"] for t in s["targets"][1:]]
    for t in others[:2]:
        for cand in (0, 1):
            client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t, "candidate": cand})
    total = client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": others[2], "candidate": 0})
    assert total.status_code == 403 and total.json()["budget"]["used_total"] == 5          # total cap binds
    assert client.get(f"/api/inversion/sessions/{sid}").json()["budget"]["remaining_total"] == 0


def test_budget_cannot_be_reset_by_resubmitting_or_restarting(client, tmp_path):
    s = _create(client, query_budget_per_target=1)
    sid, t0 = s["session_id"], s["targets"][0]["target_index"]
    assert client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t0, "candidate": 1}).status_code == 200
    # client-chosen ids are rejected; re-reading the session does not reset it
    assert client.post("/api/inversion/sessions", json={"session_id": sid}).status_code == 422
    assert client.get(f"/api/inversion/sessions/{sid}").json()["budget"]["used_total"] == 1
    # a fresh app on the same evidence root (a "restart") keeps the count
    restarted = TestClient(create_app(root=tmp_path / "var", cases=PROJECT_ROOT / "security_cases",
                                      frontend_dist=None))
    again = restarted.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t0, "candidate": 0})
    assert again.status_code == 403 and again.json()["error"] == "query_budget_exhausted"


def test_concurrent_queries_never_exceed_the_budget(client):
    s = _create(client, query_budget_per_target=3)
    sid, t0 = s["session_id"], s["targets"][0]["target_index"]
    codes = []

    def hit():
        codes.append(client.post(f"/api/inversion/sessions/{sid}/query",
                                 json={"target_index": t0, "candidate": 0}).status_code)
    threads = [threading.Thread(target=hit) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes.count(200) == 3 and codes.count(403) == 9
    assert client.get(f"/api/inversion/sessions/{sid}").json()["budget"]["used_total"] == 3


def test_session_ids_are_validated_and_queries_are_scoped(client):
    s = _create(client)
    sid = s["session_id"]
    for bad in ("MLB-SES-123", "../../etc/passwd", "MLB-SES-" + "g" * 32, "MLB-EXP-0123456789ab"):
        assert client.get(f"/api/inversion/sessions/{bad}").status_code == 404
    assert client.get("/api/inversion/sessions/MLB-SES-" + "0" * 32).status_code == 404       # well-formed, unknown
    out_of_scope = next(i for i in range(400) if i not in {t["target_index"] for t in s["targets"]})
    r = client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": out_of_scope, "candidate": 0})
    assert r.status_code == 403 and r.json()["error"] == "target_out_of_scope"
    # no way to supply features, a policy, a model or a seed per query
    t0 = s["targets"][0]["target_index"]
    for extra in ({"public_features": [0, 0, 0, 0]}, {"output_policy": {"kind": "full_probability"}},
                  {"model_id": "logreg"}, {"seed": 8}):
        assert client.post(f"/api/inversion/sessions/{sid}/query",
                           json={"target_index": t0, "candidate": 0, **extra}).status_code == 422
    assert client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t0, "candidate": 2}).status_code == 422


def test_policy_is_fixed_by_the_session_and_applied_to_observation_and_queries(client):
    s = _create(client, output_policy={"kind": "label_only"}, query_budget_per_target=2)
    ds = load_inversion_dataset(7)
    for t in s["targets"]:
        assert t["observed_output"] in (0.0, 1.0)
        own = int(ds.s_eval[t["target_index"]])  # test-side knowledge: the true candidate reproduces the observation
        r = client.post(f"/api/inversion/sessions/{s['session_id']}/query",
                        json={"target_index": t["target_index"], "candidate": own}).json()
        assert r["released"] == t["observed_output"]


def test_submit_once_then_closed_and_scored_only_after_submission(client):
    s = _create(client)
    sid, idx = s["session_id"], [t["target_index"] for t in s["targets"]]
    assert client.get(f"/api/inversion/sessions/{sid}/score").status_code == 409            # not before submission
    assert client.post(f"/api/inversion/sessions/{sid}/submit",
                       json={"guesses": [{"target_index": idx[0], "inferred": 1}]}).json()["error"] == "invalid_submission"
    receipt = client.post(f"/api/inversion/sessions/{sid}/submit",
                          json={"guesses": [{"target_index": i, "inferred": 0} for i in idx]}).json()
    assert receipt["status"] == "submitted" and set(receipt) & FORBIDDEN == set()
    assert client.post(f"/api/inversion/sessions/{sid}/submit",
                       json={"guesses": [{"target_index": i, "inferred": 1} for i in idx]}).json()["error"] == "already_submitted"
    closed = client.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": idx[0], "candidate": 0})
    assert closed.status_code == 403 and closed.json()["error"] == "session_closed"
    score = client.get(f"/api/inversion/sessions/{sid}/score").json()
    ds = load_inversion_dataset(7)
    assert score["attribute_inference_accuracy"]["numerator"] == sum(int(ds.s_eval[i]) == 0 for i in idx)
    assert {"inference_advantage", "baseline_inference_accuracy", "leakage_p_value"} <= set(score)


def test_session_limit_is_enforced(client, monkeypatch):
    import mlbreak.inversion.sessions as sess
    monkeypatch.setattr(sess, "LIMITS", type(LIMITS)(max_inversion_sessions=2))
    _create(client)
    _create(client)
    r = client.post("/api/inversion/sessions", json={})
    assert r.status_code == 403 and r.json()["error"] == "session_limit_reached"


# ------------------------------------------------------------------------- attacker / evaluator separation
def _schema_names(spec, schema, seen=None) -> set[str]:
    seen = set() if seen is None else seen
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        if name in seen:
            return set()
        seen.add(name)
        schema = spec["components"]["schemas"][name]
    out = set(schema.get("properties", {}))
    for sub in [*schema.get("properties", {}).values(), *schema.get("anyOf", []), *schema.get("allOf", []),
                schema.get("items", {}), schema.get("additionalProperties") or {}]:
        if isinstance(sub, dict) and sub:
            out |= _schema_names(spec, sub, seen)
    return out


def test_attacker_facing_response_schemas_cannot_carry_ground_truth(client):
    spec = client.app.openapi()
    attacker_ops = [(path, m, op) for path, ops in spec["paths"].items() for m, op in ops.items()
                    if "ml03-attacker" in op.get("tags", [])]
    assert {(p, m) for p, m, _ in attacker_ops} == {
        ("/api/inversion/sessions/{session_id}", "get"), ("/api/inversion/sessions/{session_id}/query", "post"),
        ("/api/inversion/sessions/{session_id}/submit", "post")}
    for path, m, op in attacker_ops:
        schema = op["responses"]["200"]["content"]["application/json"]["schema"]
        names = _schema_names(spec, schema)
        assert names and not names & FORBIDDEN, (path, names & FORBIDDEN)
        assert is_attacker_route(m.upper(), path.replace("{session_id}", "MLB-SES-" + "0" * 32))
    # the session view, as actually served, contains nothing beyond its schema
    s = _create(client, output_policy={"kind": "full_probability"})
    assert set(s) == {"session_id", "status", "model_id", "seed", "output_policy", "public_feature_names",
                      "hidden_attribute", "targets", "budget"}
    assert s["hidden_attribute"] == {"name": "confidential_support_programme", "domain": [0, 1], "value": "hidden"}


def test_evaluator_token_mode_blocks_evaluator_surface_but_not_the_attack_session(tmp_path):
    c = _app(tmp_path, TOKEN)
    assert c.post("/api/inversion/sessions", json={}).status_code == 401                  # only evaluators create
    assert c.post("/api/inversion/sessions", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    s = _create(c, headers=AUTH)
    sid, t0 = s["session_id"], s["targets"][0]["target_index"]
    assert c.get(f"/api/inversion/sessions/{sid}").status_code == 200                     # attacker, no token
    assert c.post(f"/api/inversion/sessions/{sid}/query", json={"target_index": t0, "candidate": 1}).status_code == 200
    eid = c.post("/api/experiments/inversion/attack", json={"targets": {"max_targets": 5}}, headers=AUTH).json()["experiment_id"]
    for method, path in (("get", f"/api/experiments/{eid}"), ("get", f"/api/experiments/{eid}/evidence.json"),
                         ("get", f"/api/experiments/{eid}/report.md"), ("get", f"/api/experiments/{eid}/samples.csv"),
                         ("get", "/api/experiments"), ("get", f"/api/inversion/sessions/{sid}/score"),
                         ("post", "/api/inversion/targets"), ("post", "/api/experiments/inversion/attack"),
                         ("get", "/api/dashboard"), ("get", "/api/regression/latest")):
        r = getattr(c, method)(path, **({"json": {}} if method == "post" else {}))
        assert r.status_code == 401 and r.json()["error"] == "evaluator_token_required", (method, path)
    full = c.get(f"/api/experiments/{eid}", headers=AUTH).json()                          # evaluator keeps everything
    assert full["targets"][0]["ground_truth"]["private_attribute"] in (0, 1)
    assert c.get("/api/health").status_code == 200


def test_default_mode_keeps_the_local_research_workflow(client):
    eid = client.post("/api/experiments/inversion/attack", json={"targets": {"max_targets": 5}}).json()["experiment_id"]
    assert client.get(f"/api/experiments/{eid}").status_code == 200


def test_session_files_hold_no_ground_truth(client, tmp_path):
    s = _create(client)
    path = tmp_path / "var" / "inversion_sessions" / f"{s['session_id']}.json"
    state = json.loads(path.read_text())
    assert not {"ground_truth", "private_attribute", "target_label", "y_true"} & set(json.dumps(state).replace(
        '"name": "confidential_support_programme"', "").split('"'))


def test_session_responses_are_deterministic(tmp_path):
    c1, c2 = _app(tmp_path / "a"), _app(tmp_path / "b")
    a, b = _create(c1), _create(c2)
    assert a["session_id"] != b["session_id"] and a["targets"] == b["targets"]
    t0 = a["targets"][0]["target_index"]
    ra = c1.post(f"/api/inversion/sessions/{a['session_id']}/query", json={"target_index": t0, "candidate": 1}).json()
    rb = c2.post(f"/api/inversion/sessions/{b['session_id']}/query", json={"target_index": t0, "candidate": 1}).json()
    assert ra["released"] == rb["released"] and ra["query_number"] == rb["query_number"] == 1
