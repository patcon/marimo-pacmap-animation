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
    """A stand-in fit_trace() return tuple whose FP checkpoints share a source
    column and differ only in their target endpoint - what LocalMAP's resample
    actually produces, and what the cache's dedup relies on."""
    rs = np.random.RandomState(1)
    trace = rs.rand(5, 20, 2).astype(np.float32)
    pair_neighbors = rs.randint(0, 20, size=(40, 2)).astype(np.int32)
    pair_MN = rs.randint(0, 20, size=(10, 2)).astype(np.int32)
    sources = np.repeat(np.arange(20, dtype=np.int32), 3)
    pair_FP_history = [
        (frame, np.column_stack([sources, rs.randint(0, 20, size=60).astype(np.int32)]))
        for frame in (0, 11, 21)
    ]
    return trace, pair_neighbors, pair_MN, pair_FP_history


@pytest.fixture
def ragged_fit_result(fit_result):
    """A history whose checkpoints do NOT share a source column, forcing the
    cache off its dedup path."""
    trace, pair_neighbors, pair_MN, history = fit_result
    rs = np.random.RandomState(2)
    history = [(frame, arr.copy()) for frame, arr in history]
    history[1][1][:, 0] = rs.randint(0, 20, size=60)
    return trace, pair_neighbors, pair_MN, history


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
    trace, pair_neighbors, pair_MN, pair_FP_history = cli.cache.load_fit(tmp_path, "localmap", "abc123")

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
        assert got.shape == want.shape


def test_fp_history_round_trips_when_source_columns_differ(tmp_path, ragged_fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, ragged_fit_result)
    _, _, _, pair_FP_history = cli.cache.load_fit(tmp_path, "localmap", "abc123")

    assert [f for f, _ in pair_FP_history] == [0, 11, 21]
    for (_, got), (_, want) in zip(pair_FP_history, ragged_fit_result[3]):
        assert np.array_equal(got, want)


def test_shared_fp_source_column_is_stored_only_once(tmp_path, fit_result):
    """The dedup that makes an entry ~25% smaller: LocalMAP's resamples only
    change each far pair's target endpoint, so the source column is one array
    for the whole history rather than one per checkpoint."""
    path = cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)

    assert (path / "fp_sources.npy").exists()
    assert not list(path.glob("fp_0.npy"))


def test_dedup_makes_the_entry_smaller_than_the_ragged_fallback(tmp_path, fit_result, ragged_fit_result):
    shared = cli.cache.save_fit(tmp_path, "localmap", "shared", PARAMS, fit_result)
    ragged = cli.cache.save_fit(tmp_path, "localmap", "ragged", PARAMS, ragged_fit_result)

    def total(p):
        return sum(f.stat().st_size for f in p.rglob("*"))

    assert total(shared) < total(ragged)


def test_fp_history_supports_indexing_iteration_and_len(tmp_path, fit_result):
    """render.py indexes it, iterates it, and takes its length - a lazily
    reconstructed history has to behave like the plain list a fresh fit
    returns."""
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    _, _, _, history = cli.cache.load_fit(tmp_path, "localmap", "abc123")

    assert len(history) == 3
    assert history[0][0] == 0
    assert [f for f, _arr in history] == [0, 11, 21]
    assert np.array_equal(history[-1][1], fit_result[3][-1][1])


def test_large_arrays_are_memory_mapped_rather_than_read_into_memory(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    trace, pair_neighbors, pair_MN, _ = cli.cache.load_fit(tmp_path, "localmap", "abc123")

    assert isinstance(trace, np.memmap)
    assert isinstance(pair_neighbors, np.memmap)
    assert isinstance(pair_MN, np.memmap)


def test_memory_mapped_trace_supports_the_reads_the_pipeline_does(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    trace, _, _, history = cli.cache.load_fit(tmp_path, "localmap", "abc123")

    assert np.percentile(np.linalg.norm(trace[0], axis=-1), 99.5) >= 0
    assert trace[2].shape == (20, 2)
    assert history[0][1][np.array([0, 5, 9])].shape == (3, 2)


def test_load_fit_of_a_different_key_is_a_miss(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    assert cli.cache.load_fit(tmp_path, "localmap", "different") is None


def test_load_fit_of_a_different_algorithm_is_a_miss(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    assert cli.cache.load_fit(tmp_path, "pacmap", "abc123") is None


def test_corrupt_entry_is_a_miss_rather_than_an_error(tmp_path, fit_result):
    path = cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    (path / "trace.npy").write_bytes(b"not an npy")

    assert cli.cache.load_fit(tmp_path, "localmap", "abc123") is None


def test_entry_missing_an_array_is_a_miss_rather_than_an_error(tmp_path, fit_result):
    path = cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)
    (path / "pair_MN.npy").unlink()

    assert cli.cache.load_fit(tmp_path, "localmap", "abc123") is None


def test_save_writes_a_readable_params_sidecar(tmp_path, fit_result):
    path = cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)

    meta = json.loads((path / "meta.json").read_text())
    assert meta["algorithm"] == "localmap"
    assert meta["seed"] == 42
    assert meta["low_dist_thres"] == 10.0


def test_save_leaves_no_temp_directories_behind(tmp_path, fit_result):
    cache_dir = tmp_path / "fits"
    cli.cache.save_fit(cache_dir, "localmap", "abc123", PARAMS, fit_result)

    assert [p.name for p in cache_dir.iterdir()] == ["localmap_abc123"]


def test_saving_over_an_existing_entry_replaces_it(tmp_path, fit_result):
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, fit_result)

    replacement = (fit_result[0] + 1.0, *fit_result[1:])
    cli.cache.save_fit(tmp_path, "localmap", "abc123", PARAMS, replacement)
    trace, _, _, _ = cli.cache.load_fit(tmp_path, "localmap", "abc123")

    assert np.array_equal(trace, fit_result[0] + 1.0)


def test_save_creates_the_cache_directory(tmp_path, fit_result):
    nested = tmp_path / "does" / "not" / "exist"
    cli.cache.save_fit(nested, "localmap", "abc123", PARAMS, fit_result)
    assert cli.cache.load_fit(nested, "localmap", "abc123") is not None
