import pytest
from fastapi.testclient import TestClient

from mlbreak.api import create_app
from mlbreak.config import PROJECT_ROOT
from mlbreak.evidence import EvidenceStore
from mlbreak.experiments import ExperimentEngine
from mlbreak.schemas import AttackRequest


@pytest.fixture
def store(tmp_path):
    return EvidenceStore(tmp_path / "var")


@pytest.fixture
def engine(store):
    return ExperimentEngine(store)


@pytest.fixture
def client(tmp_path):
    app = create_app(root=tmp_path / "var", cases=PROJECT_ROOT / "security_cases", frontend_dist=None)
    return TestClient(app)


@pytest.fixture
def attack_record(engine):
    return engine.run_attack(AttackRequest())
