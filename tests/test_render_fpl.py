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
