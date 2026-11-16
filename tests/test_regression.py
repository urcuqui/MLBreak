import json

from mlbreak.config import PROJECT_ROOT
from mlbreak.regression import load_cases, run_suite


def test_declared_cases_load():
    cases = load_cases(PROJECT_ROOT / "security_cases")
    assert len(cases) >= 4
    assert {c.defended_evaluation for c in cases} == {"replay", "adaptive", "none"}
    assert {c.owasp_id for c in cases} == {"ML01:2023", "ML02:2023", "ML03:2023"}
    ml03 = sorted(c.id for c in cases if c.owasp_id == "ML03:2023")
    assert ml03 == ["MLB-REG-014", "MLB-REG-015", "MLB-REG-016", "MLB-REG-017", "MLB-REG-018"]


def test_ml03_case_contract_is_validated():
    from pydantic import ValidationError
    from mlbreak.regression import parse_case
    case = json.loads((PROJECT_ROOT / "security_cases" / "MLB-REG-016.json").read_text())
    bad = dict(case, defense=None)                                   # defended invariants without a policy
    try:
        parse_case(json.dumps(bad))
        raise AssertionError("accepted a defended invariant without a defense")
    except ValidationError:
        pass
    bad = dict(case, invariants=[dict(case["invariants"][0], metric="attack_success_rate")])
    try:
        parse_case(json.dumps(bad))                                  # ML01 semantics are not valid for ML03
        raise AssertionError("accepted an ML01 metric in an ML03 case")
    except ValidationError:
        pass


def test_security_regression_suite_passes(engine, tmp_path):
    report = run_suite(engine, PROJECT_ROOT / "security_cases", tmp_path / "reg")
    failures = [(c["id"], i["id"], i["observed"]) for c in report["cases"] for i in c["invariants"]
                if not i["passed"]]
    assert report["ok"], failures
    assert (tmp_path / "reg" / "security_regression_report.md").is_file()


def test_violated_invariant_is_detected(engine, tmp_path):
    case = json.loads((PROJECT_ROOT / "security_cases" / "MLB-REG-001.json").read_text())
    case["id"] = "MLB-REG-999"
    case["invariants"] = [{"id": "impossible", "kind": "security", "posture": "defended",
                           "metric": "attack_success_rate", "op": "<=", "value": 0.0,
                           "description": "deliberately unsatisfiable"}]
    case["defended_evaluation"] = "adaptive"
    d = tmp_path / "cases"
    d.mkdir()
    (d / "MLB-REG-999.json").write_text(json.dumps(case))
    report = run_suite(engine, d)
    assert not report["ok"] and report["cases"][0]["invariants"][0]["observed"] > 0


def test_empty_suite_is_not_ok(engine, tmp_path):
    assert run_suite(engine, tmp_path)["ok"] is False


def test_implemented_categories_are_backed_by_scenarios_and_regression_cases():
    """A category may only be marked implemented if it has a registered scenario for that category and at
    least one declared security regression case (which test_security_regression_suite_passes runs)."""
    from mlbreak.experiments import SCENARIOS
    from mlbreak.owasp import CATEGORIES
    cases = load_cases(PROJECT_ROOT / "security_cases")
    for cat in CATEGORIES:
        if cat.lab_status != "implemented":
            assert not cat.scenario_ids
            assert not [c for c in cases if c.owasp_id == cat.id]
            continue
        assert cat.scenario_ids, cat.id
        for sid in cat.scenario_ids:
            assert SCENARIOS[sid].owasp_id == cat.id
        assert [c for c in cases if c.owasp_id == cat.id], f"{cat.id} has no regression case"
    assert {s.owasp_id for s in SCENARIOS.values()} == {c.id for c in CATEGORIES if c.lab_status == "implemented"}
