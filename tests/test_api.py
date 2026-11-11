import pytest


def test_health_and_headers(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"


def test_owasp_registry_is_honest(client):
    cats = client.get("/api/owasp").json()["categories"]
    assert [c["id"] for c in cats] == [f"ML{i:02d}:2023" for i in range(1, 11)]
    implemented = [c["id"] for c in cats if c["lab_status"] == "implemented"]
    assert implemented == ["ML01:2023", "ML02:2023"]
    for c in cats[2:]:  # ML03–ML10: metadata only, no scenarios, no placeholder attacks
        assert c["scenario_ids"] == [] and c["lab_status"] == "not_implemented"


def test_dataset_endpoint(client):
    d = client.get("/api/scenarios/ml01-moons-evasion/dataset").json()
    assert len(d["test"]["X"]) == 150 and len(d["train"]["X"]) == 350
    assert d["model"]["artifact_sha256"]
    assert client.get("/api/scenarios/nope/dataset").status_code == 404
    assert client.get("/api/scenarios/ml01-moons-evasion/dataset?model_id=evil").status_code == 404


def test_full_flow_via_api(client):
    b = client.post("/api/experiments/baseline", json={})
    assert b.status_code == 200 and b.json()["status"] == "completed"
    a = client.post("/api/experiments/attack", json={"attack": {"attack": "pgd", "epsilon": 0.3},
                                                     "samples": {"indices": [0, 1, 2, 3, 4, 5]}})
    assert a.status_code == 200, a.text
    aid = a.json()["experiment_id"]
    d = client.post("/api/experiments/defend", json={"parent_experiment_id": aid,
                                                     "defense": {"kind": "confidence_rejection"}})
    assert d.status_code == 200 and d.json()["evaluation"] == "replay"
    # An adaptive attack must be comparable with its parent: same ε/norm and the same test samples.
    mismatched = client.post("/api/experiments/attack", json={"target_defense": {"kind": "adversarial_training"},
                                                              "parent_experiment_id": aid})
    assert mismatched.status_code == 422 and "selected test samples" in mismatched.json()["detail"]
    ad = client.post("/api/experiments/attack", json={"attack": {"attack": "pgd", "epsilon": 0.3},
                                                      "samples": {"indices": [0, 1, 2, 3, 4, 5]},
                                                      "target_defense": {"kind": "adversarial_training"},
                                                      "parent_experiment_id": aid})
    assert ad.status_code == 200 and ad.json()["evaluation"] == "adaptive_attack"
    assert ad.json()["lineage"]["relation"] == "adaptive_attack_of"
    r = client.post(f"/api/experiments/{aid}/replay")
    assert r.status_code == 200 and r.json()["identical"]
    ids = [e["experiment_id"] for e in client.get("/api/experiments").json()]
    assert aid in ids
    ej = client.get(f"/api/experiments/{aid}/evidence.json")
    assert ej.status_code == 200 and ej.json()["experiment_id"] == aid
    md = client.get(f"/api/experiments/{aid}/report.md")
    assert md.status_code == 200 and "# MLBreak experiment" in md.text
    assert client.get(f"/api/experiments/{aid}/samples.csv").status_code == 200
    dash = client.get("/api/dashboard").json()
    assert dash["experiments_total"] >= 5 and dash["owasp_implemented"] == 2


@pytest.mark.parametrize("body", [
    {"attack": {"epsilon": 5}},
    {"attack": {"epsilon": -0.1}},
    {"attack": {"attack": "fgsm", "norm": "l2"}},
    {"attack": {"attack": "exec"}},
    {"attack": {"steps": 100000}},
    {"attack": {"query_budget": 10**7}},
    {"samples": {"max_samples": 100000}},
    {"samples": {"indices": [1, 1]}},
    {"samples": {"indices": list(range(400))}},
    {"model_id": "../../etc/passwd"},
    {"scenario_id": "ml02-poisoning"},
    {"code": "import os"},
    {"target_defense": {"kind": "confidence_rejection", "threshold": 0.3}},
    {"target_defense": {"kind": "adversarial_training", "rounds": 99}},
    {"seed": -1},
])
def test_api_rejects_invalid_requests(client, body):
    assert client.post("/api/experiments/attack", json=body).status_code == 422


def test_out_of_range_index_is_422(client):
    r = client.post("/api/experiments/attack", json={"samples": {"indices": [999]}})
    assert r.status_code == 422


@pytest.mark.parametrize("bad", ["../../etc/passwd", "MLB-EXP-xyz", "MLB-EXP-000000000000", "%2e%2e%2fsecret"])
def test_unknown_or_malicious_ids_are_404(client, bad):
    assert client.get(f"/api/experiments/{bad}").status_code == 404
    assert client.get(f"/api/experiments/{bad}/report.md").status_code == 404


def test_defend_with_unknown_parent_is_404(client):
    r = client.post("/api/experiments/defend", json={"parent_experiment_id": "MLB-EXP-000000000000",
                                                     "defense": {"kind": "confidence_rejection"}})
    assert r.status_code == 404


def test_boundary_limits(client):
    assert client.post("/api/boundary", json={"resolution": 40}).status_code == 200
    assert client.post("/api/boundary", json={"resolution": 5000}).status_code == 422


def test_no_upload_or_code_routes(client):
    paths = {r.path for r in client.app.routes}
    assert not any(k in p for p in paths for k in ("upload", "exec", "eval", "pickle", "model/load"))
