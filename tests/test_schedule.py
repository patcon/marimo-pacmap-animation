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


def test_cycle_has_one_row_per_iteration_and_three_columns():
    num_iters = (0, 0, 400)
    assert cli.build_schedule("cycle", num_iters).shape == (sum(num_iters), 3)


def test_cycle_starts_at_mn_max():
    """The cos anchor: a run opens on the global-structure phase, matching
    vanilla's spirit of starting w_MN high so the random init gets pulled into
    a coherent global arrangement before local detail is refined."""
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, mn_min=0.05, mn_max=100.0)
    assert S[0, 0] == pytest.approx(100.0)


def test_cycle_reaches_mn_min_at_half_period():
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, mn_min=0.05, mn_max=100.0)
    assert S[50, 0] == pytest.approx(0.05)


def test_cycle_is_periodic():
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, mn_min=0.05, mn_max=100.0)
    assert S[100, 0] == pytest.approx(S[0, 0])
    assert S[300, 0] == pytest.approx(S[200, 0])


def test_cycle_stays_within_mn_bounds():
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, mn_min=0.05, mn_max=100.0)
    assert np.all(S[:, 0] >= 0.05 - 1e-12)
    assert np.all(S[:, 0] <= 100.0 + 1e-12)


def test_cycle_sweeps_w_mn_in_log_space():
    """w_MN is a scale parameter, so an even sweep is multiplicative: the
    quarter-period midpoint is the *geometric* mean of the bounds, not the
    arithmetic one."""
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, mn_min=0.05, mn_max=100.0)
    assert S[25, 0] == pytest.approx(np.sqrt(0.05 * 100.0))


def test_cycle_holds_w_nb_and_w_fp_constant():
    """Only ratios between the three forces matter, so one moving knob is
    enough to sweep local<->global."""
    S = cli.build_schedule("cycle", (0, 0, 400))
    assert np.all(S[:, 1] == 2.0)
    assert np.all(S[:, 2] == 1.0)


def test_cycle_defaults_are_period_100_bounds_0p05_to_100():
    S_default = cli.build_schedule("cycle", (0, 0, 400))
    S_explicit = cli.build_schedule("cycle", (0, 0, 400), period=100, mn_min=0.05, mn_max=100.0)
    assert np.array_equal(S_default, S_explicit)


@pytest.mark.parametrize("bad", [{"mn_min": 0.0}, {"mn_min": -1.0}, {"mn_max": 0.0}])
def test_cycle_rejects_nonpositive_mn_bounds(bad):
    """The sweep is log-spaced, so a nonpositive bound has no meaning - and
    w_MN = 0 is the boundary at infinite log-ratio distance."""
    with pytest.raises(ValueError):
        cli.build_schedule("cycle", (0, 0, 400), **bad)


def test_cycle_rejects_nonpositive_period():
    with pytest.raises(ValueError):
        cli.build_schedule("cycle", (0, 0, 400), period=0)


def test_cycle_still_holds_w_fp_at_1_by_default():
    """Regression: adding the repulsion channel must not change what --schedule-preset
    cycle already does, since its cached fits are keyed on those params."""
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, mn_min=0.05, mn_max=100.0)
    assert np.all(S[:, 2] == 1.0)


def test_cycle_can_sweep_w_fp_too():
    """Cycling both ratios at once: the attractive and repulsive channels each
    get their own bounds."""
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, fp_min=0.2, fp_max=5.0, fp_phase=0.0)
    assert S[0, 2] == pytest.approx(5.0)
    assert S[50, 2] == pytest.approx(0.2)


def test_w_fp_sweeps_in_log_space_like_w_mn():
    S = cli.build_schedule("cycle", (0, 0, 400), period=100, fp_min=0.2, fp_max=5.0, fp_phase=0.0)
    assert S[25, 2] == pytest.approx(np.sqrt(0.2 * 5.0))


def test_fp_phase_half_puts_repulsion_in_antiphase_with_attraction():
    """The default: when global attraction peaks, repulsion troughs, so the
    condense and expand halves of a cycle reinforce instead of cancelling."""
    S = cli.build_schedule("cycle", (0, 0, 400), period=100,
                           mn_min=0.05, mn_max=100.0, fp_min=0.2, fp_max=5.0, fp_phase=0.5)
    assert S[0, 0] == pytest.approx(100.0)  # w_MN at its max
    assert S[0, 2] == pytest.approx(0.2)    # w_FP simultaneously at its min
    assert S[50, 0] == pytest.approx(0.05)
    assert S[50, 2] == pytest.approx(5.0)


def test_fp_phase_defaults_to_antiphase():
    explicit = cli.build_schedule("cycle", (0, 0, 400), fp_min=0.2, fp_max=5.0, fp_phase=0.5)
    default = cli.build_schedule("cycle", (0, 0, 400), fp_min=0.2, fp_max=5.0)
    assert np.array_equal(default, explicit)


def test_breathe_sweeps_repulsion_and_holds_attraction():
    """The repulsive channel is the visually legible one: an inhale that spreads
    and untangles, an exhale that lets attraction re-condense."""
    S = cli.build_schedule("breathe", (0, 0, 400), period=100)
    assert len(set(np.round(S[:, 0], 9))) == 1  # w_MN held
    assert S[:, 2].max() > S[:, 2].min()        # w_FP sweeps


def test_breathe_holds_w_mn_at_vanillas_settled_value():
    S = cli.build_schedule("breathe", (0, 0, 400))
    assert np.all(S[:, 0] == pytest.approx(3.0))


@pytest.mark.parametrize("bad", [{"fp_min": 0.0}, {"fp_min": -1.0}, {"fp_max": 0.0}])
def test_cycle_rejects_nonpositive_fp_bounds(bad):
    with pytest.raises(ValueError):
        cli.build_schedule("cycle", (0, 0, 400), **bad)


def test_both_channels_can_cycle_at_different_amplitudes():
    S = cli.build_schedule("cycle", (0, 0, 400), period=100,
                           mn_min=1.0, mn_max=1000.0, fp_min=0.5, fp_max=2.0, fp_phase=0.5)
    assert S[:, 0].max() / S[:, 0].min() == pytest.approx(1000.0)
    assert S[:, 2].max() / S[:, 2].min() == pytest.approx(4.0)


def _sentinel_schedule(total):
    """A schedule whose every value is distinguishable from every other, so a
    test can tell exactly which row was served."""
    return np.arange(total * 3, dtype=float).reshape(total, 3)


def test_override_serves_the_row_for_each_iteration():
    import pacmap

    W = _sentinel_schedule(30)
    with cli.override_weight_schedule(W):
        for itr in (0, 7, 29):
            got = pacmap.pacmap.find_weight(1000.0, itr, num_iters=(10, 10, 10))
            assert tuple(got) == tuple(W[itr])


def test_override_returns_three_unpackable_floats():
    """pacmap unpacks the result as `w_MN, w_neighbors, w_FP = find_weight(...)`
    and passes each into a numba-jitted gradient, so the shape and the element
    types both matter."""
    import pacmap

    with cli.override_weight_schedule(_sentinel_schedule(30)):
        w_MN, w_NB, w_FP = pacmap.pacmap.find_weight(1000.0, 3, num_iters=(10, 10, 10))
    assert all(isinstance(w, float) for w in (w_MN, w_NB, w_FP))


def test_override_restores_the_original_function_on_exit():
    import pacmap

    original = pacmap.pacmap.find_weight
    with cli.override_weight_schedule(_sentinel_schedule(30)):
        assert pacmap.pacmap.find_weight is not original
    assert pacmap.pacmap.find_weight is original


def test_override_restores_the_original_function_when_the_body_raises():
    """A `both` run fits twice; a patch leaking out of a failed first fit would
    silently corrupt the second."""
    import pacmap

    original = pacmap.pacmap.find_weight
    with pytest.raises(RuntimeError):
        with cli.override_weight_schedule(_sentinel_schedule(30)):
            raise RuntimeError("boom")
    assert pacmap.pacmap.find_weight is original


def test_override_rejects_a_schedule_of_the_wrong_length():
    """Catches a schedule built for different num_iters than the fit is running
    - otherwise it would silently IndexError deep inside the fit, or worse,
    serve the wrong rows."""
    import pacmap

    with cli.override_weight_schedule(_sentinel_schedule(30)):
        with pytest.raises(ValueError):
            pacmap.pacmap.find_weight(1000.0, 0, num_iters=(10, 10, 50))


def test_override_raises_if_find_weights_signature_changed(monkeypatch):
    """A pacmap upgrade that reshapes find_weight should fail loudly here
    rather than silently driving the fit with the wrong arguments."""
    monkeypatch.setattr("pacmap.pacmap.find_weight", lambda w_MN_init, itr: None)
    with pytest.raises(RuntimeError):
        with cli.override_weight_schedule(_sentinel_schedule(30)):
            pass


def test_weight_schedule_delegates_to_vanilla_preset():
    """weight_schedule() drives the overlay and the schedule strip. It must be
    the same array the fit is driven by (plus the init row), not a second
    implementation that could silently drift out of agreement with it."""
    num_iters = (10, 10, 30)
    S = cli.build_schedule("vanilla", num_iters)
    assert np.array_equal(cli.weight_schedule(num_iters), np.vstack([S[0], S]))
