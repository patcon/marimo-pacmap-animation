"""fit_trace()'s caching layer: when does it refit, and when does it reuse?

The fit itself is stubbed out with a counting fake - what's under test is
the cache lookup/store around it, not pacmap.
"""

import numpy as np
import pytest

from _loader import cli


FIT_KWARGS = dict(
    n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0, num_iters=(2, 2, 2),
    seed=0, n_components=2, low_dist_thres=10.0,
)


@pytest.fixture
def X():
    return np.random.RandomState(0).rand(20, 8).astype(np.float32)


@pytest.fixture
def fit_calls(monkeypatch):
    """Replaces the real fit with a fake returning deterministic arrays, and
    returns the list of kwargs it was called with."""
    calls = []

    def fake_fit(X, algorithm, **kwargs):
        calls.append({"algorithm": algorithm, **kwargs})
        rs = np.random.RandomState(len(X))
        return (
            rs.rand(sum(kwargs["num_iters"]) + 1, len(X), kwargs["n_components"]).astype(np.float32),
            rs.randint(0, len(X), size=(10, 2)),
            rs.randint(0, len(X), size=(5, 2)),
            [(0, rs.randint(0, len(X), size=(8, 2))), (3, rs.randint(0, len(X), size=(7, 2)))],
        )

    monkeypatch.setattr(cli.fit, "_fit_uncached", fake_fit)
    return calls


def test_second_identical_fit_is_served_from_cache(tmp_path, X, fit_calls):
    first = cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)
    second = cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)

    assert len(fit_calls) == 1
    assert np.array_equal(first[0], second[0])


def test_cached_fit_round_trips_pairs_and_fp_history(tmp_path, X, fit_calls):
    first = cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)
    second = cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)

    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert [f for f, _ in second[3]] == [0, 3]
    for (_, want), (_, got) in zip(first[3], second[3]):
        assert np.array_equal(want, got)


def test_no_cache_dir_never_reuses_and_writes_nothing(tmp_path, X, fit_calls):
    cache_dir = tmp_path / "fits"

    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=None)
    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=None)

    assert len(fit_calls) == 2
    assert not cache_dir.exists()


@pytest.mark.parametrize("key,value", [
    ("n_neighbors", 6),
    ("mn_ratio", 0.6),
    ("fp_ratio", 3.0),
    ("num_iters", (2, 2, 3)),
    ("seed", 1),
    ("n_components", 3),
    ("low_dist_thres", 3.0),
])
def test_changing_any_fit_param_misses_the_cache(tmp_path, X, fit_calls, key, value):
    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)
    cli.fit_trace(X, "localmap", **{**FIT_KWARGS, key: value}, cache_dir=tmp_path)

    assert len(fit_calls) == 2


def test_changing_algorithm_misses_the_cache(tmp_path, X, fit_calls):
    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)
    cli.fit_trace(X, "pacmap", **FIT_KWARGS, cache_dir=tmp_path)

    assert len(fit_calls) == 2


def test_changing_the_data_misses_the_cache(tmp_path, X, fit_calls):
    other = X.copy()
    other[2, 3] += 1.0

    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)
    cli.fit_trace(other, "localmap", **FIT_KWARGS, cache_dir=tmp_path)

    assert len(fit_calls) == 2


def test_a_pacmap_upgrade_invalidates_existing_entries(tmp_path, X, fit_calls, monkeypatch):
    import pacmap

    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)
    monkeypatch.setattr(pacmap, "__version__", "99.0.0")
    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)

    assert len(fit_calls) == 2


def test_low_dist_thres_is_dropped_from_the_pacmap_cache_key(tmp_path, X, fit_calls):
    """PaCMAP ignores low_dist_thres, so varying it must not force a refit -
    otherwise `--algorithm both` runs re-fit the pacmap half for nothing."""
    cli.fit_trace(X, "pacmap", **FIT_KWARGS, cache_dir=tmp_path)
    cli.fit_trace(X, "pacmap", **{**FIT_KWARGS, "low_dist_thres": 3.0}, cache_dir=tmp_path)

    assert len(fit_calls) == 1


def test_a_corrupt_entry_causes_a_refit(tmp_path, X, fit_calls):
    cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)
    for path in tmp_path.glob("*.npz"):
        path.write_bytes(b"garbage")

    result = cli.fit_trace(X, "localmap", **FIT_KWARGS, cache_dir=tmp_path)

    assert len(fit_calls) == 2
    assert result[0].shape == (7, 20, 2)
