"""Repository-quality guards: API documentation, documentation links, licensing consistency, attribution."""

import json
import re
import tomllib

from mlbreak.api import create_app
from mlbreak.config import PROJECT_ROOT


def test_every_api_route_is_tagged_and_documented():
    spec = create_app(frontend_dist=None).openapi()
    tags = {t["name"] for t in spec["tags"]}
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("tags") and set(op["tags"]) <= tags, f"{method.upper()} {path} has no known tag"
    for code in ("404", "409", "422", "429", "504"):
        assert f"| {code} |" in spec["info"]["description"]


def test_relative_links_in_readme_and_docs_resolve():
    files = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
    broken = []
    for f in files:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", f.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (f.parent / target).exists():
                broken.append(f"{f.relative_to(PROJECT_ROOT)} → {target}")
    assert not broken, broken


def test_license_status_is_consistent():
    """Until a maintainer chooses a license, nothing may claim one; once LICENSE exists, metadata must declare it."""
    has_license = (PROJECT_ROOT / "LICENSE").exists()
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]
    package = json.loads((PROJECT_ROOT / "frontend" / "package.json").read_text())
    readme = (PROJECT_ROOT / "README.md").read_text()
    if has_license:
        assert pyproject.get("license") and package.get("license"), "declare the chosen license in package metadata"
    else:
        assert "license" not in pyproject and "license" not in package
        assert "No license has been chosen yet" in readme
        assert (PROJECT_ROOT / "docs" / "licensing.md").is_file()


def test_bundled_dataset_attribution_is_documented():
    notice = (PROJECT_ROOT / "src" / "mlbreak" / "text" / "data" / "sms_spam_collection" / "NOTICE.md").read_text()
    datasets = (PROJECT_ROOT / "docs" / "datasets.md").read_text()
    for text in (notice, datasets):
        assert "CC BY 4.0" in text and "10.24432/C5CC84" in text
