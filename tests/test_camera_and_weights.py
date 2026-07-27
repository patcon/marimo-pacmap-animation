import numpy as np

from _loader import cli


def _linear_trace(n_frames=20, n_points=30, seed=0, n_components=2):
    """A trace whose extent grows linearly, like a real PaCMAP run expanding
    over iterations."""
    rs = np.random.RandomState(seed)
    base = rs.normal(size=(n_points, n_components))
    scale = np.linspace(1.0, 30.0, n_frames)
    return np.stack([base * s for s in scale])


def test_weight_schedule_mn_collapses_from_1000_to_0():
    num_iters = (100, 100, 250)
    W = cli.weight_schedule(num_iters)
    assert W[0, 0] == 1000.0
    assert W[-1, 0] == 0.0


def test_weight_schedule_length_includes_init_frame():
    num_iters = (10, 10, 10)
    W = cli.weight_schedule(num_iters)
    assert len(W) == sum(num_iters) + 1


def test_camera_path_default_centers_on_origin():
    trace = _linear_trace()
    center, r = cli.camera_path(trace)
    assert np.allclose(center, 0.0)


def test_camera_path_radius_length_matches_trace_length():
    trace = _linear_trace()
    _, r = cli.camera_path(trace)
    assert r.shape == (len(trace),)


def test_camera_path_focus_label_center_length_matches_trace_length():
    trace = _linear_trace()
    y = np.zeros(trace.shape[1], dtype=int)
    y[:15] = 1
    center, _ = cli.camera_path(trace, y=y, focus_label=1)
    assert center.shape == (len(trace), 2)


def test_camera_path_radius_is_monotonic_by_default():
    trace = _linear_trace()
    _, r = cli.camera_path(trace)
    assert np.all(np.diff(r) >= 0)


def test_camera_path_fixed_radius_is_constant():
    trace = _linear_trace()
    _, r = cli.camera_path(trace, fixed=True)
    assert np.allclose(r, r[0])


def test_camera_path_fixed_radius_covers_largest_extent():
    trace = _linear_trace()
    _, r_zoom = cli.camera_path(trace, fixed=False)
    _, r_fixed = cli.camera_path(trace, fixed=True)
    assert r_fixed[0] >= r_zoom.max()


def test_camera_path_focus_label_tracks_only_that_labels_points():
    n_frames, n_points = 10, 40
    rs = np.random.RandomState(1)
    y = np.array([0] * 20 + [1] * 20)
    # label 1's points drift far away over time; label 0 stays put.
    base = rs.normal(size=(n_points, 2))
    trace = np.stack([base.copy() for _ in range(n_frames)])
    drift = np.linspace(0, 100, n_frames)
    for f in range(n_frames):
        trace[f, y == 1] += drift[f]

    center, _ = cli.camera_path(trace, y=y, focus_label=1, smooth_window=1)
    # centroid should end up near the drifted cluster, not at the origin.
    assert center[-1, 0] > 50


def test_camera_path_focus_label_center_is_nonzero():
    trace = _linear_trace()
    y = np.zeros(trace.shape[1], dtype=int)
    y[:15] = 1
    center, _ = cli.camera_path(trace, y=y, focus_label=1)
    assert not np.allclose(center, 0.0)


def test_camera_path_zoom_default_is_unchanged():
    trace = _linear_trace()
    _, r_default = cli.camera_path(trace)
    _, r_zoom1 = cli.camera_path(trace, zoom=1.0)
    assert np.allclose(r_default, r_zoom1)


def test_camera_path_zoom_2x_halves_radius():
    trace = _linear_trace()
    _, r = cli.camera_path(trace)
    _, r_zoomed = cli.camera_path(trace, zoom=2.0)
    assert np.allclose(r_zoomed, r / 2.0)


def test_camera_path_zoom_applies_when_fixed():
    trace = _linear_trace()
    _, r = cli.camera_path(trace, fixed=True)
    _, r_zoomed = cli.camera_path(trace, fixed=True, zoom=2.0)
    assert np.allclose(r_zoomed, r / 2.0)


def test_camera_path_zoom_applies_when_focus_label_set():
    trace = _linear_trace()
    y = np.zeros(trace.shape[1], dtype=int)
    y[:15] = 1
    _, r = cli.camera_path(trace, y=y, focus_label=1)
    _, r_zoomed = cli.camera_path(trace, y=y, focus_label=1, zoom=2.0)
    assert np.allclose(r_zoomed, r / 2.0)


def test_camera_path_3d_center_and_radius_shapes():
    trace = _linear_trace(n_components=3)
    center, r = cli.camera_path(trace)
    assert center.shape == (len(trace), 3)
    assert r.shape == (len(trace),)


def test_camera_path_3d_radius_is_monotonic_by_default():
    trace = _linear_trace(n_components=3)
    _, r = cli.camera_path(trace)
    assert np.all(np.diff(r) >= 0)


def test_camera_path_3d_fixed_radius_covers_largest_extent():
    trace = _linear_trace(n_components=3)
    _, r_zoom = cli.camera_path(trace, fixed=False)
    _, r_fixed = cli.camera_path(trace, fixed=True)
    assert r_fixed[0] >= r_zoom.max()


def test_camera_path_3d_focus_label_center_has_3_columns():
    trace = _linear_trace(n_components=3)
    y = np.zeros(trace.shape[1], dtype=int)
    y[:15] = 1
    center, _ = cli.camera_path(trace, y=y, focus_label=1)
    assert center.shape == (len(trace), 3)
    assert not np.allclose(center, 0.0)
