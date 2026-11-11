import numpy as np
import pytest

from mlbreak.data import load_dataset


def test_same_seed_same_data():
    load_dataset.cache_clear()
    a = load_dataset("moons-v1", 7)
    load_dataset.cache_clear()
    b = load_dataset("moons-v1", 7)
    assert a is not b
    assert a.fingerprint() == b.fingerprint()
    assert np.array_equal(a.X_test, b.X_test) and np.array_equal(a.y_train, b.y_train)


def test_different_seed_different_data():
    assert load_dataset("moons-v1", 7).fingerprint() != load_dataset("moons-v1", 8).fingerprint()


def test_split_shapes_and_stratification():
    ds = load_dataset("moons-v1", 7)
    assert ds.X_train.shape == (350, 2) and ds.X_test.shape == (150, 2)
    assert abs(ds.y_test.mean() - 0.5) < 0.01


def test_arrays_are_read_only():
    ds = load_dataset("moons-v1", 7)
    with pytest.raises(ValueError):
        ds.X_test[0, 0] = 99.0


def test_data_inside_declared_domain():
    ds = load_dataset("moons-v1", 7)
    for X in (ds.X_train, ds.X_test):
        assert np.all(X >= ds.domain_low) and np.all(X <= ds.domain_high)


def test_unknown_dataset_rejected():
    with pytest.raises(KeyError):
        load_dataset("not-approved", 7)
