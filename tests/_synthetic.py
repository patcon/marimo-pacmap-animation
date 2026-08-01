"""Fit-free stand-ins for the pipeline's intermediate arrays.

Shared by every renderer-backend test so they all exercise the same shapes the
real run_algorithm() produces, without paying for a pacmap fit.
"""
import numpy as np

from _loader import cli


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
