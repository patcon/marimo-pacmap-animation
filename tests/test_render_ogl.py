"""The ogl backend: exports a .pcmp data file rather than rasterizing pixels.

Unlike the fastplotlib backend this needs no GPU and no optional dependency,
so all of it runs in CI.
"""
import numpy as np
import pytest

from _loader import cli
from _synthetic import synthetic_render_inputs


def _export(tmp_path, name="out.pcmp", **overrides):
    inputs = synthetic_render_inputs(**overrides.pop("inputs", {}))
    out = tmp_path / name
    cli.render_animation(renderer="ogl", out_path=str(out), n_lines=5, **{**inputs, **overrides})
    return cli.pcmp.read_pcmp(out)


def test_exports_every_iteration_at_step_1(tmp_path):
    header, arrays = _export(tmp_path, step=1)
    total = sum((2, 2, 2))
    assert header["frames"] == total + 1
    assert header["iters"] == list(range(total + 1))
    assert arrays["positions"].shape == (total + 1, 40, 2)


def test_step_thins_frames_the_same_way_matplotlib_does(tmp_path):
    """--step means one thing across backends: the frame list here must match
    the one _render_animation_mpl builds for the same arguments."""
    header, arrays = _export(tmp_path, step=3)
    total = sum((2, 2, 2))
    expected = list(range(0, total + 1, 3))
    assert header["iters"] == expected
    assert header["frames"] == len(expected)
    assert arrays["positions"].shape[0] == len(expected)


def test_start_and_end_narrow_the_exported_range(tmp_path):
    header, _ = _export(tmp_path, step=1, start=2, end=4)
    assert header["iters"] == [2, 3, 4]
    assert header["frames"] == 3


def test_positions_round_trip_equal_to_the_trace(tmp_path):
    inputs = synthetic_render_inputs()
    out = tmp_path / "out.pcmp"
    cli.render_animation(renderer="ogl", out_path=str(out), n_lines=5, step=1, **inputs)

    _, arrays = cli.pcmp.read_pcmp(out)
    np.testing.assert_array_equal(arrays["positions"], inputs["trace"])


def test_per_frame_metadata_is_aligned_with_the_exported_frames(tmp_path):
    """weights/center/radius are indexed by exported frame, not by trace
    index, so the player can read row f without a lookup table."""
    inputs = synthetic_render_inputs()
    out = tmp_path / "out.pcmp"
    cli.render_animation(renderer="ogl", out_path=str(out), n_lines=5, step=3, **inputs)

    header, _ = cli.pcmp.read_pcmp(out)
    iters = header["iters"]
    assert len(header["weights"]) == len(iters)
    assert len(header["center"]) == len(iters)
    assert len(header["radius"]) == len(iters)
    np.testing.assert_allclose(header["weights"][1], inputs["W"][iters[1]], rtol=1e-6)
    np.testing.assert_allclose(header["radius"][1], inputs["r_s"][iters[1]], rtol=1e-6)


def test_colors_are_rgb_floats_in_unit_range(tmp_path):
    _, arrays = _export(tmp_path, step=1, cmap="tab10")
    colors = arrays["colors"]
    assert colors.shape == (40, 3)
    assert colors.min() >= 0.0 and colors.max() <= 1.0


def test_colors_differ_between_categorical_and_continuous_colormaps(tmp_path):
    _, categorical = _export(tmp_path, name="a.pcmp", step=1, cmap="tab10")
    _, continuous = _export(tmp_path, name="b.pcmp", step=1, cmap="viridis")
    assert not np.allclose(categorical["colors"], continuous["colors"])


def test_colors_match_matplotlibs_own_normalization(tmp_path):
    """Baked here rather than in JS so an ogl export and the mp4 of the same
    run are colored identically."""
    import matplotlib

    inputs = synthetic_render_inputs()
    _, arrays = _export(tmp_path, step=1, cmap="viridis", inputs={})

    y = inputs["y"]
    norm = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
    expected = matplotlib.colormaps["viridis"](norm(y))[:, :3]
    np.testing.assert_allclose(arrays["colors"], expected, atol=1e-6)


def test_3d_trace_records_dims_3(tmp_path):
    header, arrays = _export(tmp_path, step=1, inputs={"n_components": 3})
    assert header["dims"] == 3
    assert arrays["positions"].shape[2] == 3


def test_header_records_the_phase_boundaries(tmp_path):
    header, _ = _export(tmp_path, step=1)
    assert header["num_iters"] == [2, 2, 2]


def test_render_frame_exports_a_single_frame(tmp_path):
    inputs = synthetic_render_inputs()
    out = tmp_path / "one.pcmp"
    cli.render_frame(renderer="ogl", out_path=str(out), frame=4, n_lines=5, **inputs)

    header, arrays = cli.pcmp.read_pcmp(out)
    assert header["frames"] == 1
    assert header["iters"] == [4]
    assert arrays["positions"].shape == (1, 40, 2)
    np.testing.assert_array_equal(arrays["positions"][0], inputs["trace"][4])


def test_edge_arguments_are_accepted_and_ignored(tmp_path):
    """Signature parity with the pixel backends is the contract
    render_animation()'s **kwargs pass-through depends on; edges themselves
    are out of scope for this backend."""
    inputs = synthetic_render_inputs()
    out = tmp_path / "out.pcmp"
    cli.render_animation(
        renderer="ogl", out_path=str(out), n_lines=5, step=1,
        edge_style_preset="v2", edge_gamma=0.5, line_alpha=0.3,
        overlay_style_preset="v1", point_size=9, point_alpha=0.4,
        title_prefix="localmap ", fps=30, rotate=True, n_components=2,
        **inputs,
    )
    header, arrays = cli.pcmp.read_pcmp(out)
    assert "edges" not in arrays
    assert header["frames"] == sum((2, 2, 2)) + 1
