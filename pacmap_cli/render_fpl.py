"""fastplotlib rendering backend: offscreen GPU rendering to png/mp4.

fastplotlib (plus imageio-ffmpeg for the mp4 writer) is an optional
dependency - see the `fastplotlib` extra in pyproject.toml - so it is
imported lazily inside the render functions, never at module import time.
Selecting `--renderer fastplotlib` without it installed exits with an
actionable message instead of a traceback.

Unlike the matplotlib backend there is no FuncAnimation here: the figure is
built on an offscreen canvas, `update(f)` mutates the graphics' GPU buffers,
and each frame is drawn and snapshotted to a numpy array (`fig.export_numpy()`
after `fig.canvas.draw()` - snapshotting without a draw first crashes in
pygfx). Stills are written with PIL (already present via matplotlib's pillow
dependency); animations stream the arrays into imageio-ffmpeg (plan Task 6).
"""

import time

import numpy as np

from .datasets import CATEGORICAL_CMAP
from .fp_history import checkpoint_index_for_frame
from .overlay import compute_overlay_text
from .pairs import compute_edge_alphas, count_drawn, subsample_pairs, subsample_pairs_indices

_INSTALL_HINT = (
    "--renderer fastplotlib requires the optional fastplotlib dependencies.\n"
    "Run via the project env with the extra enabled (extras don't apply to\n"
    "PEP 723 script runs, so invoke through `python`):\n"
    "    uv run --extra fastplotlib python pacmap_animation_mnist.cli.py --renderer fastplotlib ...\n"
    "or, when running the script standalone:\n"
    "    uv run --with fastplotlib==0.6.1 --with imageio-ffmpeg==0.6.0 pacmap_animation_mnist.cli.py ..."
)


def _import_fastplotlib():
    # Must be set before fastplotlib/rendercanvas import: without it,
    # rendercanvas's auto backend selection looks for a GUI toolkit (glfw/qt)
    # and raises ImportError in headless use even when we only ever ask for
    # offscreen canvases.
    import os

    os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")
    try:
        import fastplotlib as fpl
    except ImportError as exc:
        raise SystemExit(_INSTALL_HINT) from exc
    return fpl


# Depth-sorting toggles (code-level only - deliberately not CLI/config flags).
# Points and edges both render with `depth_write = False` so semi-transparent
# geometry alpha-blends instead of depth-culling; the GPU then composites them
# in buffer order, which is index order and unrelated to what is actually in
# front. In 3D each frame therefore re-sorts the buffers back-to-front along
# the view axis (painter's algorithm). Flip either flag off to get the raw
# buffer-order behaviour back - useful for comparing renders or for ruling the
# sort out as the cause of a rendering oddity. Both are no-ops in 2D, where
# every vertex is coplanar and buffer order is already the right answer.
DEPTH_SORT_POINTS = True
DEPTH_SORT_EDGES = True

# Sorting each buffer only orders geometry *within* that buffer. Points and
# edges are separate draw calls, so with neither writing depth the GPU
# composites them in graphic order and the scatter always wins - an edge
# passing in front of a cluster still draws under it. Turning this on makes
# the points opaque (alpha_mode "auto", which writes depth) and the edges
# explicitly blended; pygfx draws opaque objects before blended ones, so the
# points land in the depth buffer first and each edge fragment is then either
# discarded (it is behind a point) or blended over the point's actual color
# (it is in front). Note this is NOT the pre-870d641 arrangement that produced
# dark streaks - that was the mirror image, with the *edges* writing depth and
# drawing first, so a near-invisible edge could cull whole clusters behind it
# and composite against the background. Here edges never occlude anything.
# The cost lands on the points instead: a point behind another point is now
# depth-culled, so a dense low --point-alpha cloud stops accumulating opacity
# the way matplotlib's does. Flip off to get that accumulation back at the
# price of edges never appearing in front. No-op in 2D, where everything is
# coplanar and there is no "in front" to get wrong.
OPAQUE_POINTS = True


def edge_segments(Y, P):
    """Vertex buffer drawing the (k, 2) index pairs `P` over embedding `Y` as
    k disjoint segments in a single line graphic: laid out [start, end, nan]
    per edge, (3k, 2) float32 - pygfx treats nan vertices as breaks, which is
    far cheaper than one graphic per edge."""
    buf = np.full((3 * len(P), Y.shape[1]), np.nan, dtype=np.float32)
    buf[0::3] = Y[P[:, 0]]
    buf[1::3] = Y[P[:, 1]]
    return buf


def edge_vertex_alphas(alpha, n_edges, line_alpha):
    """Expand a scalar or per-edge alpha into the (3 * n_edges,) per-vertex
    alpha column for an edge_segments() buffer, scaled by line_alpha and
    clipped to [0, 1] - the fastplotlib analogue of the matplotlib backend's
    apply_alpha()."""
    alpha = np.clip(np.asarray(alpha, dtype=np.float32) * line_alpha, 0.0, 1.0)
    if alpha.ndim == 0:
        return np.full(3 * n_edges, float(alpha), dtype=np.float32)
    return np.repeat(alpha, 3)


def _build_renderer_fpl(
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
    n_lines=150, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
    line_alpha=1.0,
    cmap=CATEGORICAL_CMAP,
    rotate=False,
):
    """fastplotlib counterpart to render.py's `_build_renderer()` (and, for
    3-column traces, `_build_renderer_3d()`): same contract - returns
    `(fig, update, total, BG)` where `update(f)` mutates the graphics to
    show trace index `f` - but the figure is an offscreen fastplotlib Figure
    and updates write straight into GPU buffers. Unlike matplotlib, no
    separate 3D artist set is needed: scatter/line buffers are (n, 3)
    natively, so 2D vs 3D differs only in camera handling. In 3D the camera
    is orthographic at matplotlib's default viewpoint (elev=20, azim=-60,
    z-up); `rotate=True` sweeps the azimuth through one revolution over the
    frame range, matching the matplotlib backend (including single-frame
    renders showing that frame's angle).

    Remaining gap: the overlay/legend/weight strip (plan Task 5, deferred).
    """
    fpl = _import_fastplotlib()

    total = sum(num_iters)
    PN = subsample_pairs(pair_neighbors, n_lines, rs)
    PM = subsample_pairs(pair_MN, n_lines, rs)
    fp_idx = subsample_pairs_indices(pair_FP_history[0][1], n_lines, rs)
    checkpoint_frames = np.array([f for f, _arr in pair_FP_history])
    checkpoint_PF = [arr[fp_idx] for _f, arr in pair_FP_history]
    counts = count_drawn(len(trace[0]), PN, PM, checkpoint_PF[0])
    print(
        f"Drawing {counts['nodes']} nodes and {counts['edges_total']} edges (fastplotlib) "
        f"(neighbour={counts['edges_neighbor']}, mid-near={counts['edges_midnear']}, "
        f"further={counts['edges_further']})"
    )
    BG = "#0d0d10"
    NB_COLOR, MN_COLOR, FP_COLOR = "#4da6ff", "#ffa53d", "#ff4d4d"

    # Same logical size as the matplotlib figure (7x8in at 110dpi), with the
    # same two-region layout: main plot on top, weight-schedule strip along
    # the bottom (fractional rects mirror the matplotlib add_axes boxes).
    # The exported array may come back at an integer multiple (the canvas's
    # pixel ratio); consumers must read the actual exported shape rather
    # than assume this size.
    fig = fpl.Figure(
        rects=[(0.02, 0.03, 0.96, 0.83), (0.09, 0.86, 0.82, 0.12)],
        size=(770, 880), canvas="offscreen",
    )
    # pygfx supersamples at pixel_ratio 2 by default, which quadruples the
    # pixels snapshotted and encoded per frame; ratio 1 matches matplotlib's
    # 770x880 output exactly (materials still shader-antialias) and roughly
    # halves the per-frame cost.
    fig.renderer.pixel_ratio = 1
    sub, subw = fig[0], fig[1]
    for s in (sub, subw):
        s.background_color = (BG,)  # tuple: the setter iterates (gradient corners)
        s.axes.visible = False
        s.title = ""
        # The Frame reserves hardcoded strips for the subplot title (top) and
        # resize handle (bottom) that render in the frame plane's own color;
        # recolor the plane to BG and hide the handle so they disappear.
        s.frame.plane.material.color = BG
        s.frame.resize_handle.visible = False

    dim = trace.shape[2]
    Y0 = trace[0]
    from pygfx.utils import Color

    # All three pair types share one line graphic, concatenated in what used
    # to be their layer order (further, then mid-near, then neighbour) so the
    # unsorted 2D render still composites exactly as separate layers did -
    # and matplotlib's zorder does. One buffer is what makes depth sorting
    # work *across* pair types in 3D: with three graphics, layer order beat
    # depth, so a further-pair edge in front of a neighbour edge still drew
    # under it. The cost is a single shared `thickness` (matplotlib varies it,
    # 0.5 for further vs 0.7 for the rest); the thinner value wins, since
    # merged rendering already gives the dense further layer more presence.
    # Row counts are fixed - every far-pair checkpoint is subsampled through
    # the same fp_idx - so the per-type slices below hold for every frame.
    checkpoint_P_all = [np.vstack([PF, PM, PN]) for PF in checkpoint_PF]
    n_edges = len(checkpoint_P_all[0])
    edge_spans = []  # (vertex_slice, edge_count) per type, in buffer order
    at = 0
    for P, color in ((checkpoint_PF[0], FP_COLOR), (PM, MN_COLOR), (PN, NB_COLOR)):
        edge_spans.append((slice(3 * at, 3 * (at + len(P))), len(P), color))
        at += len(P)

    edges = sub.add_line(
        edge_segments(Y0, checkpoint_P_all[0]), thickness=1.0, colors="w",
    )
    # Semi-transparent overlays must not write depth: if the edges wrote depth,
    # anything drawn after them and sitting behind them in z would be
    # depth-culled instead of alpha-blended - the edge then blends with the
    # background instead of the points, showing up as opaque dark streaks
    # cutting through clusters in 3D. Under OPAQUE_POINTS the mode is set
    # explicitly rather than via depth_write so pygfx also classifies the edges
    # as transparent and draws them after the (opaque) scatter.
    if OPAQUE_POINTS:
        edges.world_object.material.alpha_mode = "blend"  # implies depth_write False
    else:
        edges.world_object.material.depth_write = False
    # RGB is static per type; only the alpha column and the row order change
    # per frame.
    edge_rgba = np.zeros((3 * n_edges, 4), dtype=np.float32)
    for span, _n, color in edge_spans:
        edge_rgba[span, :3] = np.asarray(Color(color))[:3]

    scat = sub.add_scatter(
        np.ascontiguousarray(trace[0], dtype=np.float32),
        cmap=cmap, cmap_transform=y, sizes=point_size,
    )
    scat.colors[:, -1] = point_alpha
    # With OPAQUE_POINTS the points keep the default depth-writing mode, which
    # is what lets edges in front of them draw in front (see the toggle's note).
    # Without it they must not write depth either, or semi-transparent points
    # depth-cull each other and a dense low-point-alpha cloud can't accumulate
    # opacity the way matplotlib's does. Either way the points blend in buffer
    # order, so in 3D `update()` sorts them back-to-front per frame; in 2D
    # everything is coplanar and buffer order is the right answer anyway (it's
    # what matplotlib's single scatter does too).
    if not OPAQUE_POINTS:
        scat.world_object.material.depth_write = False
    point_rgba = np.array(scat.colors.value, dtype=np.float32)

    from fastplotlib.graphics import TextGraphic

    # Overlay text lives in the main subplot's top dock and the pair-type
    # legend in its bottom dock: docks are separate plot areas with their own
    # static cameras, so the text stays pinned regardless of the main
    # camera's per-frame framing. Both dock cameras are fixed to a unit rect
    # after fig.show() (which can rescale cameras) so offsets are fractions.
    dock_top, dock_bot = sub.docks["top"], sub.docks["bottom"]
    dock_top.size = 72  # four v2 overlay lines at font_size 12 need ~68px
    dock_bot.size = 24
    # center=False everywhere text is added: centering calls
    # camera.show_object() on the graphic, and screen-space text has no
    # bounding sphere for it to frame (pygfx raises ValueError).
    title = TextGraphic("", font_size=12, face_color="w", anchor="top-left", offset=(0.005, 0.92, 0))
    dock_top.add_graphic(title, center=False)
    for x, label, color in ((0.005, "neighbour", NB_COLOR), (0.12, "mid-near", MN_COLOR), (0.23, "further", FP_COLOR)):
        dock_bot.add_graphic(TextGraphic(label, font_size=11, face_color=color, anchor="middle-left", offset=(x, 0.5, 0)), center=False)

    # Weight-schedule strip: the three log-weight curves, phase boundaries,
    # a moving current-frame cursor, and the axis label.
    it = np.arange(total + 1, dtype=np.float32)
    Wlog = np.log10(W + 1).astype(np.float32)
    for j, c in enumerate((MN_COLOR, NB_COLOR, FP_COLOR)):
        subw.add_line(np.column_stack([it, Wlog[:, j]]), thickness=1.4, colors=c)
    ylo, yhi = float(Wlog.min()), float(Wlog.max())
    yrange = (yhi - ylo) or 1.0
    for b in (num_iters[0], num_iters[0] + num_iters[1]):
        subw.add_line(np.array([[b, ylo], [b, yhi]], dtype=np.float32), thickness=0.8, colors="#555555")
    vline = subw.add_line(np.array([[0, ylo], [0, yhi]], dtype=np.float32), thickness=1.2, colors="w")

    fig.show()  # required once to initialize the render pipeline

    subw.add_graphic(TextGraphic(
        "iteration  (log weight)", font_size=10, face_color="#888888",
        anchor="top-center", offset=(total / 2, ylo - 0.25 * yrange, 0),
    ), center=False)

    # Static cameras, set after show() so nothing rescales them: unit rect
    # for the text docks, data-extent rect (with label room below) for the
    # weight strip.
    unit_state = {
        "position": np.array([0.5, 0.5, 0.0]), "width": 1.0, "height": 1.0,
        "zoom": 1.0, "maintain_aspect": False, "fov": 0.0,
    }
    dock_top.camera.set_state(unit_state)
    dock_bot.camera.set_state(unit_state)
    y_view_lo, y_view_hi = ylo - 0.6 * yrange, yhi + 0.1 * yrange
    subw.camera.set_state({
        "position": np.array([total / 2, (y_view_lo + y_view_hi) / 2, 0.0]),
        "width": total * 1.02, "height": y_view_hi - y_view_lo,
        "zoom": 1.0, "maintain_aspect": False, "fov": 0.0,
    })

    def apply_edges(Y, P_all, alphas, direction):
        """Write one frame of every pair type into the shared edge buffer.
        `alphas` is (further, mid-near, neighbour) - each a scalar or a
        per-edge array - in the same order as `edge_spans`."""
        segs = edge_segments(Y, P_all)
        rgba = edge_rgba
        for (span, n, _color), alpha in zip(edge_spans, alphas):
            rgba[span, 3] = edge_vertex_alphas(alpha, n, line_alpha)
        if direction is not None and DEPTH_SORT_EDGES:
            # Sort every edge back-to-front by midpoint depth, in blocks of
            # three vertices ([start, end, nan] per edge), colors carried
            # along. Sorting the merged buffer is what orders edges across
            # pair types and not just within one.
            order = np.argsort((Y[P_all[:, 0]] + Y[P_all[:, 1]]) @ direction)
            segs = segs.reshape(n_edges, 3, -1)[order].reshape(3 * n_edges, -1)
            rgba = rgba.reshape(n_edges, 3, 4)[order].reshape(3 * n_edges, 4)
        edges.data[:, :dim] = segs
        edges.colors[:] = rgba

    def view_direction(f):
        """Unit vector from the frame's camera position toward the scene: the
        3D camera sits at `center[f] + direction * 3L` looking back down it,
        so a point's dot product with it grows as the point gets closer."""
        azim = np.radians(-60 + (360 * f / total if rotate else 0))
        elev = np.radians(20)
        return np.array([
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ])

    def set_camera(f):
        L = float(r_s[f])
        if dim == 2:
            cx, cy = (float(c) for c in center[f])
            sub.camera.set_state({
                "position": np.array([cx, cy, 0.0]),
                "width": 2 * L, "height": 2 * L,
                "zoom": 1.0, "maintain_aspect": True, "fov": 0.0,
            })
            return
        # 3D: orthographic camera on a sphere around the frame's center,
        # matching matplotlib's elev=20/azim=-60 default (z-up). Azimuth
        # sweeps one revolution over the frame range when rotate is set.
        c = center[f].astype(np.float64)
        direction = view_direction(f)
        sub.camera.set_state({
            "position": c + direction * 3 * L,
            "width": 2 * L, "height": 2 * L,
            "zoom": 1.0, "maintain_aspect": True, "fov": 0.0,
            "reference_up": np.array([0.0, 0.0, 1.0]),
            # ortho frustum depth: scene spans +-sqrt(3)*L around a center
            # 3L away, so (L, 6L) covers it with margin on both sides
            "depth_range": (L, 6 * L),
        })
        sub.camera.look_at(c)

    def update(f):
        Y = trace[f]
        w_MN, w_NB, w_FP = W[f]
        ck = checkpoint_index_for_frame(f, checkpoint_frames)
        PF, P_all = checkpoint_PF[ck], checkpoint_P_all[ck]
        if edge_style_preset == "v3":
            a_nb, a_mn, a_fp = compute_edge_alphas(
                w_NB, w_MN, w_FP, preset=edge_style_preset, gamma=edge_gamma, Y=Y, pairs=(PN, PM, PF))
        else:
            a_nb, a_mn, a_fp = compute_edge_alphas(w_NB, w_MN, w_FP, preset=edge_style_preset, gamma=edge_gamma)
        direction = view_direction(f) if dim == 3 else None
        if direction is not None and DEPTH_SORT_POINTS:
            # Reorder the point buffer back-to-front so a point passing behind
            # another is drawn under it instead of over it. The colors have to
            # be permuted in lockstep, since they're per-index too.
            order = np.argsort(Y @ direction)
            scat.data[:, :dim] = Y[order]
            scat.colors[:] = point_rgba[order]
        else:
            scat.data[:, :dim] = Y
        apply_edges(Y, P_all, (a_fp, a_mn, a_nb), direction)
        set_camera(f)
        ph = 1 if f <= num_iters[0] else (2 if f <= num_iters[0] + num_iters[1] else 3)
        title.text = compute_overlay_text(f, total, ph, w_MN, w_NB, w_FP, title_prefix, preset=overlay_style_preset)
        vline.data[:, 0] = f
        return ()

    return fig, update, total, BG


def _export_frame(fig):
    """Draw the offscreen canvas and return the frame as an RGBA uint8 array."""
    fig.canvas.draw()
    return fig.export_numpy()


def render_frame_fpl(
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
    out_path, frame, n_lines=150, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
    line_alpha=1.0,
    cmap=CATEGORICAL_CMAP,
    n_components=2, rotate=False,
):
    """Render a single trace index `frame` as a png via fastplotlib."""
    from PIL import Image

    fig, update, total, BG = _build_renderer_fpl(
        trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
        n_lines=n_lines, title_prefix=title_prefix,
        point_size=point_size, point_alpha=point_alpha,
        edge_style_preset=edge_style_preset, edge_gamma=edge_gamma,
        overlay_style_preset=overlay_style_preset, line_alpha=line_alpha,
        cmap=cmap, rotate=rotate,
    )
    print(f"Rendering iteration {frame} of {total} to {out_path}...")
    t0 = time.time()
    update(frame)
    Image.fromarray(_export_frame(fig)).save(out_path)
    print("rendered %s in %.0fs" % (out_path, time.time() - t0))
    return out_path


def render_animation_fpl(
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
    out_path, n_lines=150, step=1, fps=25, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
    line_alpha=1.0,
    cmap=CATEGORICAL_CMAP,
    start=None, end=None,
    n_components=2, rotate=False,
):
    """Render trace indices `start`..`end` inclusive (default: the whole
    trace) as an mp4, stepping by `step` - the fastplotlib counterpart to
    render.py's `_render_animation_mpl()`. No FuncAnimation: each frame is
    update() -> offscreen draw -> numpy snapshot, streamed straight into an
    imageio-ffmpeg writer process."""
    import imageio_ffmpeg

    fig, update, total, BG = _build_renderer_fpl(
        trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
        n_lines=n_lines, title_prefix=title_prefix,
        point_size=point_size, point_alpha=point_alpha,
        edge_style_preset=edge_style_preset, edge_gamma=edge_gamma,
        overlay_style_preset=overlay_style_preset, line_alpha=line_alpha,
        cmap=cmap, rotate=rotate,
    )
    start = 0 if start is None else start
    end = total if end is None else end
    frames = list(range(start, end + 1, step))

    n_frames = len(frames)
    print(f"Rendering {n_frames} frames (iterations {start}-{end} of {len(trace)} captured, step={step}) to {out_path}...")
    t0 = time.time()
    report_every = max(1, n_frames // 20)  # ~20 progress lines regardless of frame count

    # First frame decides the pixel size (the offscreen canvas renders at
    # its own pixel ratio, so this can be a multiple of the logical figure
    # size); macro_block_size=2 keeps ffmpeg happy (h264 needs even dims)
    # without silently rescaling to a multiple of 16.
    update(frames[0])
    frame0 = _export_frame(fig)
    h, w = frame0.shape[:2]
    writer = imageio_ffmpeg.write_frames(out_path, (w, h), fps=fps, macro_block_size=2)
    writer.send(None)  # seed the generator

    try:
        for i, f in enumerate(frames):
            if i == 0:
                rgb = frame0[:, :, :3]
            else:
                update(f)
                rgb = _export_frame(fig)[:, :, :3]
            writer.send(np.ascontiguousarray(rgb))
            if i % report_every == 0 or i == n_frames - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_frames - i - 1) / rate if rate > 0 else float("nan")
                print(f"  frame {i + 1}/{n_frames}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    finally:
        writer.close()
    print("rendered %s in %.0fs" % (out_path, time.time() - t0))
    return out_path
