"""Driving a real fit from a prebuilt schedule.

Unlike test_fit_cache.py these run pacmap for real (on tiny data), because
what's under test is that the monkey-patch actually reaches the optimizer.
"""

import numpy as np
import pytest

from _loader import cli


FIT_KWARGS = dict(
    n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0, seed=0, n_components=2,
    low_dist_thres=10.0,
)


@pytest.fixture
def X():
    return np.random.RandomState(0).rand(60, 10).astype(np.float32)


def test_vanilla_schedule_reproduces_an_unpatched_fit(X):
    """The safety property the whole feature rests on: driving a fit from the
    vanilla array is indistinguishable from not patching at all."""
    num_iters = (2, 2, 2)
    unpatched, *_ = cli.fit._fit_uncached(X, "pacmap", num_iters=num_iters, **FIT_KWARGS)
    driven, *_ = cli.fit._fit_uncached(
        X, "pacmap", num_iters=num_iters,
        schedule=cli.build_schedule("vanilla", num_iters), **FIT_KWARGS,
    )
    assert np.array_equal(driven, unpatched)


def test_a_different_schedule_actually_changes_the_trace(X):
    """Proves the patch reaches the optimizer rather than being computed and
    quietly discarded - the failure mode that would make cycle a silent no-op."""
    num_iters = (2, 2, 2)
    unpatched, *_ = cli.fit._fit_uncached(X, "pacmap", num_iters=num_iters, **FIT_KWARGS)
    driven, *_ = cli.fit._fit_uncached(
        X, "pacmap", num_iters=num_iters,
        schedule=cli.build_schedule("cycle", num_iters, period=4), **FIT_KWARGS,
    )
    assert not np.array_equal(driven, unpatched)


def test_localmap_fit_accepts_a_schedule_and_still_captures_fp_history(X):
    """The schedule patch and capture_fp_history() are both active during a
    LocalMAP fit; neither may disturb the other."""
    num_iters = (0, 0, 30)
    trace, _, _, pair_FP_history = cli.fit._fit_uncached(
        X, "localmap", num_iters=num_iters,
        schedule=cli.build_schedule("cycle", num_iters, period=10), **FIT_KWARGS,
    )
    assert trace.shape[0] == sum(num_iters) + 1
    # phase 3 spans the whole run here, so LocalMAP resamples at 10 and 20.
    assert [frame for frame, _ in pair_FP_history] == [0, 11, 21]


def test_cycle_over_an_all_phase_three_run(X):
    """--num-iters 0,0,N is the regime cycling is aimed at: it puts LocalMAP's
    contrastive far-pair resampling in effect for the entire run."""
    num_iters = (0, 0, 30)
    trace, *_ = cli.fit._fit_uncached(
        X, "localmap", num_iters=num_iters,
        schedule=cli.build_schedule("cycle", num_iters, period=10), **FIT_KWARGS,
    )
    assert np.isfinite(trace).all()


def test_find_weight_is_restored_after_a_schedule_driven_fit(X):
    import pacmap

    original = pacmap.pacmap.find_weight
    num_iters = (2, 2, 2)
    cli.fit._fit_uncached(
        X, "pacmap", num_iters=num_iters,
        schedule=cli.build_schedule("cycle", num_iters, period=4), **FIT_KWARGS,
    )
    assert pacmap.pacmap.find_weight is original


def test_find_weight_is_restored_when_a_schedule_driven_fit_raises(X):
    """An --algorithm both run fits twice; a leaked patch would corrupt the
    second fit rather than failing visibly."""
    import pacmap

    original = pacmap.pacmap.find_weight
    # A schedule built for different num_iters than the fit runs.
    with pytest.raises(ValueError):
        cli.fit._fit_uncached(
            X, "pacmap", num_iters=(2, 2, 2),
            schedule=cli.build_schedule("cycle", (5, 5, 5)), **FIT_KWARGS,
        )
    assert pacmap.pacmap.find_weight is original
