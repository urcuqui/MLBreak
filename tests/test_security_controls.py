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


# Lab sub-packages are experiment code in their entirety (ML02 poisoning, ML01 spam-filter evasion).
EXPERIMENT_PACKAGES = ["poisoning", "text"]


def _experiment_files() -> list[Path]:
    files = [SRC / f"{mod}.py" for mod in EXPERIMENT_MODULES]
    for pkg in EXPERIMENT_PACKAGES:
        files += sorted((SRC / pkg).glob("*.py"))
    return files


def test_experiment_code_has_no_network_or_unsafe_deserialisation():
    files = _experiment_files()
    assert any(f.parent.name == "text" for f in files)
    for path in files:
        imported = _imports(path)
        bad = {i for i in imported if i in FORBIDDEN_IMPORTS or i.split(".")[0] in FORBIDDEN_IMPORTS}
        assert not bad, f"{path.relative_to(SRC)} imports {bad}"


def test_bundled_text_corpus_is_local_and_hash_pinned():
    """The spam corpus ships with the package (no download) and is verified before use."""
    from mlbreak.text import data
    assert data.CORPUS_FILE.is_file() and data.CORPUS_FILE.is_relative_to(SRC)
    assert (data.DATA_DIR / "NOTICE.md").is_file() and len(data.CORPUS_SHA256) == 64


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
