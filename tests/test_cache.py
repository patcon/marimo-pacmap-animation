"""Unit tests for the on-disk fit cache (pacmap_cli/cache.py).

These use synthetic arrays rather than real fits - the cache doesn't care
what produced the arrays, only that it round-trips them exactly and that its
key changes whenever anything that would change a fit changes.
"""

import json

import numpy as np
import pytest

from _loader import cli


PARAMS = {
    "algorithm": "localmap",
    "n_neighbors": 10,
    "mn_ratio": 0.5,
    "fp_ratio": 2.0,
    "num_iters": (10, 10, 20),
    "seed": 42,
    "n_components": 2,
    "low_dist_thres": 10.0,
    "pacmap_version": "0.9.1",
}


@pytest.fixture
def X():
    return np.random.RandomState(0).rand(20, 8).astype(np.float32)


@pytest.fixture
def fit_result():
    """A stand-in fit_trace() return tuple with a multi-checkpoint FP history
    whose checkpoints deliberately differ in row count."""
    rs = np.random.RandomState(1)
    trace = rs.rand(5, 20, 2).astype(np.float32)
    pair_neighbors = rs.randint(0, 20, size=(40, 2))
    pair_MN = rs.randint(0, 20, size=(10, 2))
    pair_FP_history = [
        (0, rs.randint(0, 20, size=(30, 2))),
        (11, rs.randint(0, 20, size=(28, 2))),
        (21, rs.randint(0, 20, size=(25, 2))),
    ]
    return trace, pair_neighbors, pair_MN, pair_FP_history


def test_fit_key_is_stable_for_identical_inputs(X):
    assert cli.cache.fit_key(X, PARAMS) == cli.cache.fit_key(X.copy(), dict(PARAMS))


@pytest.mark.parametrize("key,value", [
    ("algorithm", "pacmap"),
    ("n_neighbors", 11),
    ("mn_ratio", 0.6),
    ("fp_ratio", 3.0),
    ("num_iters", (10, 10, 21)),
    ("seed", 43),
    ("n_components", 3),
    ("low_dist_thres", 3.0),
    ("pacmap_version", "0.9.2"),
])
def test_fit_key_changes_when_any_param_changes(X, key, value):
    assert cli.cache.fit_key(X, {**PARAMS, key: value}) != cli.cache.fit_key(X, PARAMS)


def test_fit_key_changes_when_data_changes(X):
    other = X.copy()
    other[3, 4] += 0.5
    assert cli.cache.fit_key(other, PARAMS) != cli.cache.fit_key(X, PARAMS)


def test_load_fit_returns_none_on_miss(tmp_path):
    assert cli.cache.load_fit(tmp_path, "localmap", "deadbeef") is None


def test_save_then_load_round_trips_arrays_exactly(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    loaded = cli.cache.load_fit(tmp_path, "localmap", "abc123")

    trace, pair_neighbors, pair_MN, pair_FP_history = loaded
    assert np.array_equal(trace, fit_result[0])
    assert np.array_equal(pair_neighbors, fit_result[1])
    assert np.array_equal(pair_MN, fit_result[2])
    assert len(pair_FP_history) == 3


def test_save_then_load_preserves_fp_checkpoint_frames_and_shapes(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    _, _, _, pair_FP_history = cli.cache.load_fit(tmp_path, "localmap", "abc123")

    assert [f for f, _ in pair_FP_history] == [0, 11, 21]
    for (_, got), (_, want) in zip(pair_FP_history, fit_result[3]):
        assert np.array_equal(got, want)


def test_load_fit_of_a_different_key_is_a_miss(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    assert cli.cache.load_fit(tmp_path, "localmap", "different") is None


def test_load_fit_of_a_different_algorithm_is_a_miss(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    assert cli.cache.load_fit(tmp_path, "pacmap", "abc123") is None


def test_corrupt_entry_is_a_miss_rather_than_an_error(tmp_path, fit_result):
    path = cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    path.write_bytes(b"not an npz")

    assert cli.cache.load_fit(tmp_path, "localmap", "abc123") is None


def test_truncated_entry_is_a_miss_rather_than_an_error(tmp_path, fit_result):
    path = cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    assert cli.cache.load_fit(tmp_path, "localmap", "abc123") is None


def test_save_writes_a_readable_params_sidecar(tmp_path, fit_result):
    path = cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    sidecar = path.with_suffix(".json")

    meta = json.loads(sidecar.read_text())
    assert meta["algorithm"] == "localmap"
    assert meta["seed"] == 42
    assert meta["low_dist_thres"] == 10.0


def test_save_leaves_no_temp_files_behind(tmp_path, fit_result):
    cache_dir = tmp_path / "fits"
    cli.cache.save_fit(cache_dir, "localmap", "abc123", PARAMS, fit_result)
    assert sorted(p.suffix for p in cache_dir.iterdir()) == [".json", ".npz"]


def test_save_creates_the_cache_directory(tmp_path, fit_result):
    nested = tmp_path / "does" / "not" / "exist"
    cli.cache.save_fit(nested, "localmap", "abc123", PARAMS, fit_result)
    assert cli.cache.load_fit(nested, "localmap", "abc123") is not None
