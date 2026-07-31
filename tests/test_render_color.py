"""The colormap the renderers color points with.

Which map to use is decided once in `datasets.py` (categorical vs continuous)
and threaded through as a single `cmap` string, so the renderers themselves
stay scheme-agnostic. These tests pin that the string actually reaches the
scatter artist rather than being dropped on the way through.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from _loader import cli


def _synthetic_inputs(n_points=20, n_components=2, seed=0):
    rs = np.random.RandomState(seed)
    num_iters = (1, 1, 1)
    trace = rs.normal(size=(sum(num_iters) + 1, n_points, n_components)).astype(np.float32)
    y = rs.randint(0, 10, size=n_points)
    W = cli.weight_schedule(num_iters)
    pair_neighbors = rs.randint(0, n_points, size=(30, 2))
    pair_MN = rs.randint(0, n_points, size=(20, 2))
    pair_FP_history = [(0, rs.randint(0, n_points, size=(20, 2)))]
    center, r_s = cli.camera_path(trace)
    return (trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters,
            center, r_s, np.random.RandomState(1))


def test_build_renderer_defaults_to_the_categorical_colormap():
    fig, _update, _total, _BG = cli.render._build_renderer(*_synthetic_inputs(), n_lines=10)
    scat = fig.axes[0].collections[-1]
    assert scat.get_cmap().name == cli.datasets.CATEGORICAL_CMAP


def test_build_renderer_uses_the_requested_colormap():
    fig, _update, _total, _BG = cli.render._build_renderer(
        *_synthetic_inputs(), n_lines=10, cmap=cli.datasets.CONTINUOUS_CMAP)
    scat = fig.axes[0].collections[-1]
    assert scat.get_cmap().name == cli.datasets.CONTINUOUS_CMAP


def test_build_renderer_3d_uses_the_requested_colormap():
    fig, _update, _total, _BG = cli.render._build_renderer_3d(
        *_synthetic_inputs(n_components=3), n_lines=10, cmap=cli.datasets.CONTINUOUS_CMAP)
    scat = fig.axes[0].collections[-1]
    assert scat.get_cmap().name == cli.datasets.CONTINUOUS_CMAP


def test_render_frame_forwards_cmap_to_the_builder(tmp_path):
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs = _synthetic_inputs()
    seen = {}
    real = cli.render._build_renderer

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    cli.render._build_renderer = spy
    try:
        cli.render.render_frame(
            trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
            out_path=str(tmp_path / "f.png"), frame=0, n_lines=5,
            cmap=cli.datasets.CONTINUOUS_CMAP,
        )
    finally:
        cli.render._build_renderer = real

    assert seen["cmap"] == cli.datasets.CONTINUOUS_CMAP


def test_every_backend_accepts_a_cmap():
    """Structural, so it runs without a GPU: `cmap` is part of the renderer
    contract, not a matplotlib-only extra."""
    import inspect

    from pacmap_cli import render_fpl

    for fn in (cli.render._render_animation_mpl, cli.render._render_frame_mpl,
               render_fpl.render_animation_fpl, render_fpl.render_frame_fpl):
        assert "cmap" in inspect.signature(fn).parameters, fn.__name__
