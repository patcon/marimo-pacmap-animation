"""Tests for the fastplotlib rendering backend.

These construct real offscreen WGPU canvases, so they are skipped wholesale
when fastplotlib isn't installed (the default env, and CI - see plan Task 8)
or no GPU adapter is available. Run locally with:

    uv run --extra fastplotlib pytest tests/test_render_fpl.py
"""

import importlib.util

import numpy as np
import pytest

from _loader import cli


def _fpl_offscreen_available():
    if importlib.util.find_spec("fastplotlib") is None:
        return False
    try:
        import wgpu

        return wgpu.gpu.request_adapter_sync() is not None
    except Exception:
        return False


requires_fpl = pytest.mark.skipif(
    not _fpl_offscreen_available(),
    reason="fastplotlib or a WGPU adapter is unavailable",
)


def synthetic_render_inputs(n_points=40, num_iters=(2, 2, 2), n_components=2, seed=0):
    """Small, fit-free stand-ins for everything run_algorithm() computes
    before rendering, shaped exactly like the real pipeline's outputs."""
    rs = np.random.RandomState(seed)
    total = sum(num_iters)
    trace = (rs.rand(total + 1, n_points, n_components) * 10).astype(np.float32)
    y = rs.randint(0, 10, size=n_points)
    W = cli.weight_schedule(num_iters)
    pair_neighbors = rs.randint(0, n_points, size=(60, 2))
    pair_MN = rs.randint(0, n_points, size=(30, 2))
    pair_FP_history = [(0, rs.randint(0, n_points, size=(80, 2)))]
    center, r_s = cli.camera_path(trace)
    return dict(
        trace=trace, y=y, W=W, pair_neighbors=pair_neighbors, pair_MN=pair_MN,
        pair_FP_history=pair_FP_history, num_iters=num_iters,
        center=center, r_s=r_s, rs=rs,
    )


@requires_fpl
def test_render_frame_fpl_writes_nonempty_png(tmp_path):
    inputs = synthetic_render_inputs()
    out = tmp_path / "frame.png"
    result = cli.render.render_frame(
        renderer="fastplotlib", out_path=str(out), frame=3, n_lines=5, **inputs,
    )
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


@requires_fpl
def test_render_frame_fpl_draws_points_on_dark_background(tmp_path):
    # The png should be dominated by the dark background but not uniform -
    # i.e. the scatter actually drew something.
    from PIL import Image

    inputs = synthetic_render_inputs()
    out = tmp_path / "frame.png"
    cli.render.render_frame(
        renderer="fastplotlib", out_path=str(out), frame=0, n_lines=5, **inputs,
    )
    img = np.asarray(Image.open(out).convert("RGB"))
    assert img.std() > 0  # not a blank canvas
    # background #0d0d10 -> most pixels very dark
    assert (img.mean(axis=2) < 40).mean() > 0.5


# --- Task 4: edge layers ---

def test_edge_segments_interleaves_nan_breaks():
    # (k, 2) pairs over an (n, 2) embedding -> (3k, 2) vertex buffer laid out
    # [start, end, nan] per edge, so one graphic draws k disjoint segments.
    from pacmap_cli import render_fpl

    Y = np.arange(10, dtype=np.float32).reshape(5, 2)
    P = np.array([[0, 2], [4, 1]])
    buf = render_fpl.edge_segments(Y, P)
    assert buf.shape == (6, 2)
    assert buf.dtype == np.float32
    np.testing.assert_array_equal(buf[0], Y[0])
    np.testing.assert_array_equal(buf[1], Y[2])
    assert np.isnan(buf[2]).all()
    np.testing.assert_array_equal(buf[3], Y[4])
    np.testing.assert_array_equal(buf[4], Y[1])
    assert np.isnan(buf[5]).all()


def test_edge_vertex_alphas_scalar_broadcasts_and_array_repeats():
    from pacmap_cli import render_fpl

    scalar = render_fpl.edge_vertex_alphas(0.5, n_edges=3, line_alpha=1.0)
    np.testing.assert_allclose(scalar, np.full(9, 0.5))
    per_edge = render_fpl.edge_vertex_alphas(np.array([0.2, 0.4]), n_edges=2, line_alpha=0.5)
    np.testing.assert_allclose(per_edge, np.repeat([0.1, 0.2], 3))


def test_edge_vertex_alphas_clips_to_unit_range():
    from pacmap_cli import render_fpl

    clipped = render_fpl.edge_vertex_alphas(np.array([-1.0, 3.0]), n_edges=2, line_alpha=1.0)
    np.testing.assert_allclose(clipped, np.repeat([0.0, 1.0], 3))


@requires_fpl
@pytest.mark.parametrize("preset", ["v1", "v2", "v3"])
def test_render_frame_fpl_with_edges_all_presets(tmp_path, preset):
    inputs = synthetic_render_inputs()
    out = tmp_path / f"frame_{preset}.png"
    cli.render.render_frame(
        renderer="fastplotlib", out_path=str(out), frame=3, n_lines=20,
        edge_style_preset=preset, **inputs,
    )
    assert out.exists() and out.stat().st_size > 0


@requires_fpl
def test_render_frame_fpl_edges_add_pixels_beyond_scatter(tmp_path):
    # Same frame with and without lines: the edge layers must actually draw.
    from PIL import Image

    inputs = synthetic_render_inputs()
    lit = {}
    for n_lines, name in [(50, "with"), (1, "without")]:
        out = tmp_path / f"{name}.png"
        cli.render.render_frame(
            renderer="fastplotlib", out_path=str(out), frame=3, n_lines=n_lines,
            edge_style_preset="v2", **inputs,
        )
        img = np.asarray(Image.open(out).convert("RGB"))
        lit[name] = (img.mean(axis=2) > 40).sum()
    assert lit["with"] > lit["without"]
