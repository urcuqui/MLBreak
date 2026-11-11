import numpy as np
import pytest

from mlbreak.data import load_dataset
from mlbreak.models import train_model

MODEL_IDS = ["mlp-tanh-16x16", "logreg"]


@pytest.fixture(scope="module")
def ds():
    return load_dataset("moons-v1", 7)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_training_is_reproducible(ds, model_id):
    a = train_model(model_id, ds.X_train, ds.y_train, 7)
    b = train_model(model_id, ds.X_train, ds.y_train, 7)
    assert a.artifact_sha256() == b.artifact_sha256()


def test_different_seed_changes_mlp_artifact(ds):
    a = train_model("mlp-tanh-16x16", ds.X_train, ds.y_train, 7)
    b = train_model("mlp-tanh-16x16", ds.X_train, ds.y_train, 8)
    assert a.artifact_sha256() != b.artifact_sha256()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_forward_pass_matches_sklearn(ds, model_id):
    m = train_model(model_id, ds.X_train, ds.y_train, 7)
    np.testing.assert_allclose(m.proba(ds.X_test), m.estimator.predict_proba(ds.X_test)[:, 1], atol=1e-10)
    assert np.array_equal(m.predict(ds.X_test), m.estimator.predict(ds.X_test))


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_input_gradient_matches_finite_differences(ds, model_id):
    m = train_model(model_id, ds.X_train, ds.y_train, 7)
    X = ds.X_test[:25]
    g = m.grad_logit(X)
    h = 1e-6
    for j in range(X.shape[1]):
        e = np.zeros_like(X)
        e[:, j] = h
        fd = (m.logit(X + e) - m.logit(X - e)) / (2 * h)
        np.testing.assert_allclose(g[:, j], fd, rtol=1e-5, atol=1e-6)


def test_prediction_consistency_repeated_calls(ds):
    m = train_model("mlp-tanh-16x16", ds.X_train, ds.y_train, 7)
    assert np.array_equal(m.predict(ds.X_test), m.predict(ds.X_test.copy()))


def test_baseline_model_is_accurate(ds):
    m = train_model("mlp-tanh-16x16", ds.X_train, ds.y_train, 7)
    assert (m.predict(ds.X_test) == ds.y_test).mean() >= 0.9


def test_unknown_model_rejected(ds):
    with pytest.raises(KeyError):
        train_model("pickle-upload", ds.X_train, ds.y_train, 7)
