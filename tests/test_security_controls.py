"""Static checks for the platform's own security requirements."""

import ast
from pathlib import Path

import mlbreak

SRC = Path(mlbreak.__file__).parent
FORBIDDEN_IMPORTS = {"pickle", "joblib", "dill", "cloudpickle", "socket", "requests", "urllib",
                     "http.client", "httpx", "subprocess", "ftplib", "smtplib"}
# uvicorn/fastapi are only used to *serve* the UI; experiment code must not touch the network.
EXPERIMENT_MODULES = ["attacks", "data", "defenses", "experiments", "models", "evidence", "regression",
                      "schemas", "hashing", "owasp", "metrics", "integrity"]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_experiment_code_has_no_network_or_unsafe_deserialisation():
    for mod in EXPERIMENT_MODULES:
        imported = _imports(SRC / f"{mod}.py")
        bad = {i for i in imported if i in FORBIDDEN_IMPORTS or i.split(".")[0] in FORBIDDEN_IMPORTS}
        assert not bad, f"{mod}.py imports {bad}"


def test_no_dynamic_code_execution():
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "compile", "__import__"}, path


def test_server_binds_localhost_by_default():
    from mlbreak.cli import build_parser
    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
