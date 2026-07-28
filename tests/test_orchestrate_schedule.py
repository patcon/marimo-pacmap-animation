"""Wiring the schedule through run_algorithm(): one array drives the fit and
feeds the display, and the two can't disagree."""

import numpy as np
import pytest

from _loader import cli


NUM_ITERS = (2, 2, 2)


@pytest.fixture
def harness(monkeypatch):
    """Runs main() with the fit and the renderer stubbed out, capturing the
    kwargs each was handed."""
    captured = {"fit": [], "render": []}

    def fake_load_mnist(n=None, seed=0):
        rs = np.random.RandomState(seed)
        n_points = 60 if n is None else int(n)
        return rs.rand(n_points, 784).astype(np.float32), rs.randint(0, 10, size=n_points), rs

    def fake_fit_trace(X, algorithm, **kwargs):
        captured["fit"].append({"algorithm": algorithm, **kwargs})
        rs = np.random.RandomState(0)
        total = sum(kwargs["num_iters"])
        return (
            rs.rand(total + 1, len(X), kwargs["n_components"]).astype(np.float32),
            rs.randint(0, len(X), size=(10, 2)),
            rs.randint(0, len(X), size=(5, 2)),
            [(0, rs.randint(0, len(X), size=(8, 2)))],
        )

    def fake_backend():
        def write(out_path=None, **kwargs):
            captured["render"].append(kwargs)
            with open(out_path, "wb") as f:
                f.write(b"fake")
            return out_path
        return {"animation": write, "frame": write}

    monkeypatch.setattr(cli.orchestrate, "load_mnist", fake_load_mnist)
    monkeypatch.setattr(cli.orchestrate, "fit_trace", fake_fit_trace)
    monkeypatch.setitem(cli.render.RENDERERS, "matplotlib", fake_backend)
    return captured


def _run(tmp_path, *extra):
    cli.main([
        "--algorithm", "pacmap", "--n", "60", "--n-neighbors", "5",
        "--num-iters", ",".join(map(str, NUM_ITERS)), "--n-lines", "5",
        "--output-dir", str(tmp_path / "run"), *extra,
    ])


def test_vanilla_leaves_the_fit_unpatched(tmp_path, harness):
    """The safety property: a default run passes no schedule at all, so the
    vanilla code path is structurally unchanged rather than merely equal."""
    _run(tmp_path)
    assert harness["fit"][0]["schedule"] is None


def test_cycle_passes_the_cycle_schedule_to_the_fit(tmp_path, harness):
    _run(tmp_path, "--schedule-preset", "cycle", "--schedule-period", "4")
    expected = cli.build_schedule("cycle", NUM_ITERS, period=4, mn_min=0.05, mn_max=100.0)
    assert np.array_equal(harness["fit"][0]["schedule"], expected)


def test_schedule_params_reach_the_cache_key(tmp_path, harness):
    _run(tmp_path, "--schedule-preset", "cycle", "--schedule-period", "4")
    assert harness["fit"][0]["schedule_params"]["schedule_preset"] == "cycle"
    assert harness["fit"][0]["schedule_params"]["schedule_period"] == 4


def test_displayed_weights_come_from_the_schedule_that_drove_the_fit(tmp_path, harness):
    """The overlay and the schedule strip must show the weights the optimizer
    actually saw - otherwise a cycle run would render a vanilla strip."""
    _run(tmp_path, "--schedule-preset", "cycle", "--schedule-period", "4")
    S = harness["fit"][0]["schedule"]
    W = harness["render"][0]["W"]
    assert np.array_equal(W, np.vstack([S[0], S]))


def test_displayed_weights_keep_the_init_row_convention(tmp_path, harness):
    """W is indexed by snapshot index, and frame 0 is the initialization, so it
    carries one more row than there are iterations."""
    _run(tmp_path, "--schedule-preset", "cycle", "--schedule-period", "4")
    assert len(harness["render"][0]["W"]) == sum(NUM_ITERS) + 1


def test_vanilla_run_still_displays_pacmaps_own_schedule(tmp_path, harness):
    _run(tmp_path)
    assert np.array_equal(harness["render"][0]["W"], cli.weight_schedule(NUM_ITERS))


def test_cycle_run_hints_at_fixed_camera(tmp_path, harness, capsys):
    """The default camera only ever zooms out, so a breathing embedding reads
    as shrinking on the inhale."""
    _run(tmp_path, "--schedule-preset", "cycle")
    assert "--fixed-camera" in capsys.readouterr().out


def test_no_camera_hint_when_fixed_camera_is_already_set(tmp_path, harness, capsys):
    _run(tmp_path, "--schedule-preset", "cycle", "--fixed-camera")
    assert "--fixed-camera" not in capsys.readouterr().out


def test_no_camera_hint_for_a_vanilla_run(tmp_path, harness, capsys):
    _run(tmp_path)
    assert "--fixed-camera" not in capsys.readouterr().out


def test_breathe_preset_reaches_the_fit_with_its_fp_knobs(tmp_path, harness):
    _run(tmp_path, "--schedule-preset", "breathe", "--schedule-period", "4",
         "--schedule-fp-min", "0.2", "--schedule-fp-max", "5")
    schedule = harness["fit"][0]["schedule"]
    # Unset knobs come from breathe's own defaults (w_MN held at 3.0, phase 0),
    # not from DEFAULT_CONFIG - that's what makes a preset a shape rather than
    # just a name.
    expected = cli.build_schedule(
        "breathe", NUM_ITERS, period=4, mn_min=3.0, mn_max=3.0,
        fp_min=0.2, fp_max=5.0, fp_phase=0.0,
    )
    assert np.array_equal(schedule, expected)


def test_fp_knobs_reach_the_cache_key(tmp_path, harness):
    _run(tmp_path, "--schedule-preset", "cycle", "--schedule-fp-max", "5")
    assert harness["fit"][0]["schedule_params"]["schedule_fp_max"] == 5.0
