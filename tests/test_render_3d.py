"""Unit tests for the 3D renderer's edge artists, built on synthetic (not
real pacmap-fitted) data so they run fast and in isolation from the fit
pipeline."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from mpl_toolkits.mplot3d.art3d import Line3DCollection

from _loader import cli


def _synthetic_inputs(n_points=20, seed=0):
    rs = np.random.RandomState(seed)
    num_iters = (1, 1, 1)
    n_frames = sum(num_iters) + 1  # matches fit_trace()'s intermediate_snapshots convention
    trace = rs.normal(size=(n_frames, n_points, 3)).astype(np.float32)
    y = rs.randint(0, 10, size=n_points)
    W = cli.weight_schedule(num_iters)
    pair_neighbors = rs.randint(0, n_points, size=(30, 2))
    pair_MN = rs.randint(0, n_points, size=(20, 2))
    pair_FP = rs.randint(0, n_points, size=(20, 2))
    pair_FP_history = [(0, pair_FP)]
    center, r_s = cli.camera_path(trace)
    draw_rs = np.random.RandomState(1)
    return trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, draw_rs


def test_build_renderer_3d_draws_edges_as_line3dcollections():
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, draw_rs = _synthetic_inputs()

    fig, update, total, BG = cli.render._build_renderer_3d(
        trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, draw_rs,
        n_lines=10,
    )
    update(0)

    ax = fig.axes[0]
    line3d_collections = [c for c in ax.collections if isinstance(c, Line3DCollection)]
    assert len(line3d_collections) == 3
    for lc in line3d_collections:
        segs = lc._segments3d  # get_segments() returns the 2D-projected form, only valid post-draw
        assert len(segs) == 10
        assert np.asarray(segs).shape[1:] == (2, 3)  # each segment: 2 endpoints, 3 coords


def test_build_renderer_3d_edges_track_frame_updates():
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, draw_rs = _synthetic_inputs()

    fig, update, total, BG = cli.render._build_renderer_3d(
        trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, draw_rs,
        n_lines=10,
    )
    update(0)
    ax = fig.axes[0]
    lc = next(c for c in ax.collections if isinstance(c, Line3DCollection))
    segs_f0 = np.asarray(lc._segments3d)

    update(total)
    segs_fT = np.asarray(lc._segments3d)
    assert not np.allclose(segs_f0, segs_fT)


def test_build_renderer_3d_v3_edge_preset_works():
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, draw_rs = _synthetic_inputs()

    fig, update, total, BG = cli.render._build_renderer_3d(
        trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, draw_rs,
        n_lines=10, edge_style_preset="v3",
    )
    update(1)  # should not raise
