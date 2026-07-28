"""Tests for the fastplotlib rendering backend.

These construct real offscreen WGPU canvases, so they are skipped wholesale
when fastplotlib isn't installed (the default env, and CI - see plan Task 8)
or no GPU adapter is available. Run locally with:

    uv run --extra fastplotlib pytest tests/test_render_fpl.py
"""

import importlib.util
from unittest import mock

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


# --- Task 5: overlay text, legend, weight strip ---

def _regions(png_path):
    """Split a rendered png into (top strip, main body, bottom strip) arrays."""
    from PIL import Image

    img = np.asarray(Image.open(png_path).convert("RGB"))
    h = img.shape[0]
    # Top strip covers the figure margin (~3%) plus the main subplot's 60px
    # top dock, where the overlay title text renders.
    return img[: int(h * 0.13)], img[int(h * 0.13): int(h * 0.84)], img[int(h * 0.84):]


@requires_fpl
def test_render_frame_fpl_overlay_text_present_and_updates(tmp_path):
    # Render the SAME frame with the two overlay presets: the scatter and
    # edges are identical, so any difference in the top strip must be the
    # overlay text itself.
    inputs = synthetic_render_inputs()
    tops = {}
    for preset in ("v1", "v2"):
        out = tmp_path / f"{preset}.png"
        cli.render.render_frame(
            renderer="fastplotlib", out_path=str(out), frame=5, n_lines=5,
            overlay_style_preset=preset, **inputs,
        )
        top, _, _ = _regions(out)
        tops[preset] = top
    # white-ish text pixels present (r~g~b, bright), not just colored points
    rgb = tops["v2"].astype(int)
    whiteish = (rgb.min(axis=2) > 150) & (rgb.max(axis=2) - rgb.min(axis=2) < 40)
    assert whiteish.sum() > 20
    assert not np.array_equal(tops["v1"], tops["v2"])


@requires_fpl
def test_render_frame_fpl_weight_strip_present_and_cursor_moves(tmp_path):
    inputs = synthetic_render_inputs()
    bottoms = []
    for frame in (0, 5):
        out = tmp_path / f"f{frame}.png"
        cli.render.render_frame(
            renderer="fastplotlib", out_path=str(out), frame=frame, n_lines=5, **inputs,
        )
        _, _, bottom = _regions(out)
        bottoms.append(bottom)
    # weight curves: colored (non-grey) pixels present in the strip
    rgb = bottoms[0].astype(int)
    colored = (np.abs(rgb[..., 0] - rgb[..., 2]) > 30).sum()
    assert colored > 50
    # the current-frame cursor moved between renders
    assert not np.array_equal(bottoms[0], bottoms[1])


# --- Task 6: animation loop -> mp4 ---

@requires_fpl
def test_render_animation_fpl_writes_playable_mp4(tmp_path):
    import imageio_ffmpeg

    inputs = synthetic_render_inputs()
    out = tmp_path / "anim.mp4"
    result = cli.render.render_animation(
        renderer="fastplotlib", out_path=str(out), n_lines=5, step=1, fps=5, **inputs,
    )
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0
    # full range at step 1 -> frames 0..6 inclusive = 7 frames
    reader = imageio_ffmpeg.read_frames(str(out))
    meta = reader.__next__()
    n = sum(1 for _ in reader)
    assert meta["fps"] == 5
    assert n == 7


@requires_fpl
def test_render_animation_fpl_honors_start_end_and_step(tmp_path):
    import imageio_ffmpeg

    inputs = synthetic_render_inputs()
    out = tmp_path / "anim_range.mp4"
    cli.render.render_animation(
        renderer="fastplotlib", out_path=str(out), n_lines=5, step=2, fps=5,
        start=1, end=5, **inputs,
    )
    reader = imageio_ffmpeg.read_frames(str(out))
    reader.__next__()
    # frames 1, 3, 5
    assert sum(1 for _ in reader) == 3


@requires_fpl
def test_main_end_to_end_with_fastplotlib_renderer(tmp_path, monkeypatch):
    def fake_load_mnist(n=None, seed=0):
        rs = np.random.RandomState(seed)
        n_points = 60 if n is None else int(n)
        X = rs.rand(n_points, 784).astype(np.float32)
        y = rs.randint(0, 10, size=n_points)
        return X, y, rs

    monkeypatch.setattr(cli.orchestrate, "load_mnist", fake_load_mnist)
    out_dir = tmp_path / "run"
    cli.main([
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--renderer", "fastplotlib",
        "--output-dir", str(out_dir),
    ])
    out_file = out_dir / "pacmap_mnist_fpl.mp4"
    assert out_file.exists() and out_file.stat().st_size > 0


# --- Task 7: 3D support ---

def test_edge_segments_supports_3_column_embeddings():
    from pacmap_cli import render_fpl

    Y = np.arange(15, dtype=np.float32).reshape(5, 3)
    P = np.array([[0, 2], [4, 1]])
    buf = render_fpl.edge_segments(Y, P)
    assert buf.shape == (6, 3)
    np.testing.assert_array_equal(buf[0], Y[0])
    np.testing.assert_array_equal(buf[1], Y[2])
    assert np.isnan(buf[2]).all()


@requires_fpl
def test_render_frame_fpl_3d_writes_nonempty_png(tmp_path):
    from PIL import Image

    inputs = synthetic_render_inputs(n_components=3)
    out = tmp_path / "frame3d.png"
    cli.render.render_frame(
        renderer="fastplotlib", out_path=str(out), frame=3, n_lines=5,
        n_components=3, **inputs,
    )
    assert out.exists() and out.stat().st_size > 0
    img = np.asarray(Image.open(out).convert("RGB"))
    assert img.std() > 0  # points actually drew
    assert (img.mean(axis=2) < 40).mean() > 0.5  # on the dark background


@requires_fpl
def test_render_animation_fpl_3d_writes_mp4(tmp_path):
    import imageio_ffmpeg

    inputs = synthetic_render_inputs(n_components=3)
    out = tmp_path / "anim3d.mp4"
    cli.render.render_animation(
        renderer="fastplotlib", out_path=str(out), n_lines=5, step=1, fps=5,
        n_components=3, rotate=True, **inputs,
    )
    reader = imageio_ffmpeg.read_frames(str(out))
    reader.__next__()
    assert sum(1 for _ in reader) == 7


@requires_fpl
def test_render_frame_fpl_3d_rotate_changes_view_angle():
    # Pixel comparison is unreliable (GPU renders are not bit-deterministic),
    # so assert on the camera state itself: at a non-zero frame the rotating
    # camera must sit at a different position than the fixed one.
    from pacmap_cli import render_fpl

    inputs = synthetic_render_inputs(n_components=3)
    positions = {}
    for rotate in (False, True):
        fig, update, total, BG = render_fpl._build_renderer_fpl(
            n_lines=5, rotate=rotate, **inputs,
        )
        update(3)
        positions[rotate] = fig[0].camera.get_state()["position"].copy()
    assert not np.allclose(positions[False], positions[True])


@requires_fpl
@pytest.mark.parametrize("n_components", [2, 3])
def test_render_fpl_edges_never_write_depth(n_components):
    # Edges are semi-transparent. If they wrote depth, anything drawn after
    # them and sitting behind them in z would be depth-culled instead of
    # alpha-blended: in 3D the edges would "erase" the points behind them and
    # blend with the background - visible as opaque dark streaks cutting
    # through clusters. True under either OPAQUE_POINTS setting.
    from pacmap_cli import render_fpl

    inputs = synthetic_render_inputs(n_components=n_components)
    fig, update, total, BG = render_fpl._build_renderer_fpl(n_lines=5, **inputs)
    main_graphics = [g for g in fig[0].graphics if hasattr(g, "colors")]
    assert len(main_graphics) == 2  # one merged edge buffer + scatter
    edges, = [g for g in main_graphics if type(g).__name__ == "LineGraphic"]
    assert edges.world_object.material.depth_write is False


@requires_fpl
@pytest.mark.parametrize("n_components", [2, 3])
def test_render_fpl_point_depth_write_follows_opaque_points_toggle(n_components):
    # OPAQUE_POINTS is what lets an edge in front of a cluster draw in front of
    # it: the points write depth (and pygfx draws opaque before blended), so
    # edge fragments behind a point are discarded rather than composited under
    # the whole scatter. Off, the points must not write depth either, so a
    # dense low-point-alpha cloud can still accumulate opacity.
    from pacmap_cli import render_fpl

    inputs = synthetic_render_inputs(n_components=n_components)
    for opaque in (True, False):
        with mock.patch.object(render_fpl, "OPAQUE_POINTS", opaque):
            fig, update, total, BG = render_fpl._build_renderer_fpl(n_lines=5, **inputs)
        scat, = [g for g in fig[0].graphics if type(g).__name__ == "ScatterGraphic"]
        assert scat.world_object.material.depth_write is opaque


@requires_fpl
def test_render_fpl_3d_sorts_points_back_to_front():
    # Points don't write depth, so they blend in buffer order. In 3D that
    # order must be re-sorted per frame along the view axis, otherwise a
    # point passing behind another draws on top of it.
    from pacmap_cli import render_fpl

    inputs = synthetic_render_inputs(n_components=3)
    fig, update, total, BG = render_fpl._build_renderer_fpl(
        n_lines=5, rotate=True, **inputs,
    )
    trace = inputs["trace"]
    scat, = [g for g in fig[0].graphics if type(g).__name__ == "ScatterGraphic"]
    for f in (1, 4):
        update(f)
        drawn = np.asarray(scat.data.value)[:, :3]
        # Same point set as the trace frame, just permuted...
        assert np.allclose(np.sort(drawn, axis=0), np.sort(trace[f], axis=0))
        # ...into ascending depth (farthest from the camera drawn first).
        depth = drawn @ render_fpl_view_direction(f, total)
        assert np.all(np.diff(depth) >= 0)


@requires_fpl
def test_render_fpl_3d_sorts_edges_back_to_front():
    # Same problem as the points: edge segments must composite back-to-front
    # by midpoint depth. All three pair types share one buffer, so the sort
    # orders them against each other and not just within a type - which is
    # the whole reason they were merged.
    from pacmap_cli import render_fpl

    inputs = synthetic_render_inputs(n_components=3)
    fig, update, total, BG = render_fpl._build_renderer_fpl(
        n_lines=5, rotate=True, **inputs,
    )
    line, = [g for g in fig[0].graphics if type(g).__name__ == "LineGraphic"]
    for f in (1, 4):
        update(f)
        verts = np.asarray(line.data.value)[:, :3]
        mid = (verts[0::3] + verts[1::3]) / 2  # [start, end, nan] per edge
        depth = mid @ render_fpl_view_direction(f, total)
        assert np.all(np.diff(depth) >= -1e-5)
        # Colors travel with their edges, so the buffer is no longer three
        # contiguous per-type blocks.
        rgb = np.asarray(line.colors.value)[0::3, :3]
        assert len(np.unique(rgb, axis=0)) == 3
        blocks = 1 + np.sum(np.any(np.diff(rgb, axis=0) != 0, axis=1))
        assert blocks > 3


@requires_fpl
@pytest.mark.parametrize("flag", ["DEPTH_SORT_POINTS", "DEPTH_SORT_EDGES"])
def test_render_fpl_depth_sort_flags_disable_sorting(monkeypatch, flag):
    # The toggles must actually reach the update path, so a render can be
    # compared against the raw buffer-order behaviour.
    from pacmap_cli import render_fpl

    monkeypatch.setattr(render_fpl, flag, False)
    inputs = synthetic_render_inputs(n_components=3)
    fig, update, total, BG = render_fpl._build_renderer_fpl(
        n_lines=5, rotate=True, **inputs,
    )
    update(4)
    if flag == "DEPTH_SORT_POINTS":
        scat, = [g for g in fig[0].graphics if type(g).__name__ == "ScatterGraphic"]
        drawn = np.asarray(scat.data.value)[:, :3]
        assert np.allclose(drawn, inputs["trace"][4])  # untouched trace order
    else:
        line = [g for g in fig[0].graphics if type(g).__name__ == "LineGraphic"][0]
        verts = np.asarray(line.data.value)[:, :3]
        mid = (verts[0::3] + verts[1::3]) / 2
        depth = mid @ render_fpl_view_direction(4, total)
        assert not np.all(np.diff(depth) >= -1e-5)


def render_fpl_view_direction(f, total, elev=20, azim0=-60):
    """Mirror of the backend's camera direction, for the depth-sort test."""
    azim, elev = np.radians(azim0 + 360 * f / total), np.radians(elev)
    return np.array([
        np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev),
    ])


def test_run_algorithm_no_longer_rejects_fastplotlib_3d(monkeypatch):
    # The Task 1 guard must be gone: with a monkeypatched fit and renderer,
    # a fastplotlib+3D run reaches rendering rather than raising.
    calls = []

    def fake_fit_trace(*args, **kwargs):
        rs = np.random.RandomState(0)
        trace = rs.rand(7, 10, 3).astype(np.float32)
        pairs = rs.randint(0, 10, size=(5, 2))
        return trace, pairs, pairs.copy(), [(0, pairs.copy())]

    monkeypatch.setattr(cli.orchestrate, "fit_trace", fake_fit_trace)
    monkeypatch.setitem(
        cli.render.RENDERERS, "fastplotlib",
        lambda: {"animation": lambda **kw: calls.append(kw) or kw["out_path"],
                 "frame": lambda **kw: calls.append(kw) or kw["out_path"]},
    )
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg.update(renderer="fastplotlib", n_components=3, num_iters=(2, 2, 2), n_lines=5)
    cli.run_algorithm(
        X=np.zeros((10, 4)), y=np.zeros(10, dtype=int),
        rs=np.random.RandomState(0), algorithm="pacmap", cfg=cfg,
        iter_out_paths=[(None, "out.mp4")],
    )
    assert len(calls) == 1
