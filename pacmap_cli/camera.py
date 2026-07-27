"""Replaying the pair-weight schedule and computing the camera path."""

import numpy as np


def weight_schedule(num_iters):
    """w_MN / w_NB / w_FP at every snapshot index, including the init frame."""
    from pacmap.pacmap import find_weight

    total = sum(num_iters)
    W = np.array([find_weight(1000.0, i, num_iters=num_iters) for i in range(total)])
    W = np.vstack([W[0], W])  # prepend so index == snapshot index
    return W


def camera_path(trace, y=None, focus_label=None, smooth_window=15, headroom=1.15, fixed=False, zoom=1.0):
    """Per-frame camera (center, radius). Smoothed, monotonic zoom-out by
    default so early iterations stay legible; `fixed=True` instead locks a
    single radius (sized to the trace's largest extent) for the whole
    animation, so you can see the true scale of the movement even though
    early frames start as a tiny dot.

    If `focus_label` is set, the camera instead tracks just the points
    where `y == focus_label`: `center` is that subset's per-frame centroid
    (smoothed, not monotonic - the cluster can legitimately drift back) and
    `radius` is sized to its own extent from that centroid. When
    `focus_label` is None, `center` is all zeros, matching the original
    origin-centered behavior.

    `zoom` divides the final radius (applied after `fixed`/monotonic
    zoom-out/`focus_label` all compute it) - `zoom=2.0` frames half the
    extent, i.e. sees closer/finer detail at the cost of cutting off edges."""
    pts = trace if focus_label is None else trace[:, y == focus_label, :]
    center = pts.mean(axis=1)  # (T, 2)
    r = np.percentile(np.abs(pts - center[:, None, :]).reshape(len(pts), -1), 99.5, axis=1)
    k = smooth_window
    if fixed:
        r_out = np.full(len(trace), r.max() * headroom)
    else:
        r_s = np.convolve(np.r_[np.full(k, r[0]), r], np.ones(k) / k, mode="valid")
        r_out = np.maximum.accumulate(r_s) * headroom
    r_out = r_out / zoom
    if focus_label is None:
        return np.zeros((len(trace), 2)), r_out
    smooth = lambda col: np.convolve(np.r_[np.full(k, col[0]), col], np.ones(k) / k, mode="valid")
    center_s = np.stack([smooth(center[:, 0]), smooth(center[:, 1])], axis=1)
    return center_s, r_out
