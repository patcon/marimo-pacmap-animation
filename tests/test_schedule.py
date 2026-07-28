import numpy as np
import pytest

from _loader import cli


NUM_ITERS_CASES = [(100, 100, 250), (7, 13, 29), (0, 0, 50), (0, 0, 251)]


@pytest.mark.parametrize("num_iters", NUM_ITERS_CASES)
def test_vanilla_matches_find_weight_exactly(num_iters):
    """The bit-identity anchor: the vanilla preset must reproduce pacmap's own
    schedule exactly, so driving a fit with it is indistinguishable from not
    patching at all. Includes zero-length phases, which pacmap handles fine
    (the `itr/phase_1_iters` division is unreachable when phase 1 is empty)."""
    from pacmap.pacmap import find_weight

    expected = np.array([find_weight(1000.0, i, num_iters=num_iters) for i in range(sum(num_iters))])
    assert np.array_equal(cli.build_schedule("vanilla", num_iters), expected)


def test_vanilla_has_one_row_per_iteration_and_three_columns():
    num_iters = (10, 10, 30)
    assert cli.build_schedule("vanilla", num_iters).shape == (sum(num_iters), 3)


def test_vanilla_columns_are_w_mn_w_nb_w_fp():
    """Column order is a contract the renderer's strip and overlay depend on.
    Pinned against pacmap's documented phase behavior rather than against
    find_weight, so it stays meaningful independent of the implementation."""
    S = cli.build_schedule("vanilla", (100, 100, 250))
    assert S[0, 0] == 1000.0  # phase 1 starts w_MN at w_MN_init
    assert S[-1, 0] == 0.0  # phase 3 drops w_MN to zero
    assert S[150, 1] == 3.0  # phase 2 holds w_neighbors at 3
    assert np.all(S[:, 2] == 1.0)  # w_FP is fixed at 1 throughout


def test_unknown_preset_raises_valueerror_naming_valid_presets():
    with pytest.raises(ValueError) as excinfo:
        cli.build_schedule("bogus", (10, 10, 10))
    assert "vanilla" in str(excinfo.value)


def test_weight_schedule_delegates_to_vanilla_preset():
    """weight_schedule() drives the overlay and the schedule strip. It must be
    the same array the fit is driven by (plus the init row), not a second
    implementation that could silently drift out of agreement with it."""
    num_iters = (10, 10, 30)
    S = cli.build_schedule("vanilla", num_iters)
    assert np.array_equal(cli.weight_schedule(num_iters), np.vstack([S[0], S]))
