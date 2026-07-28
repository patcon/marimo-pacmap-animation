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

from .fp_history import checkpoint_index_for_frame
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
):
    """fastplotlib counterpart to render.py's `_build_renderer()`: same
    contract - returns `(fig, update, total, BG)` where `update(f)` mutates
    the graphics to show trace index `f` - but the figure is an offscreen
    fastplotlib Figure and updates write straight into GPU buffers.

    Task 3 scope: scatter + camera only; edges (Task 4) and the overlay/
    weight strip (Task 5) land next.
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

    # Same logical size as the matplotlib figure (7x8in at 110dpi). The
    # exported array may come back at an integer multiple (the canvas's
    # pixel ratio); consumers must read the actual exported shape rather
    # than assume this size.
    fig = fpl.Figure(size=(770, 880), canvas="offscreen")
    sub = fig[0, 0]
    sub.background_color = (BG,)  # tuple: the setter iterates (gradient corners)
    sub.axes.visible = False
    sub.title = ""
    # The Frame reserves hardcoded strips for the subplot title (top) and
    # resize handle (bottom) that render in the frame plane's own color;
    # recolor the plane to BG and hide the handle so they disappear.
    sub.frame.plane.material.color = BG
    sub.frame.resize_handle.visible = False

    # Edge layers added before the scatter so points draw on top (render
    # order follows add order), and back-to-front within themselves to match
    # the matplotlib zorder: further < mid-near < neighbour.
    Y0 = trace[0][:, :2]
    from pygfx.utils import Color

    def add_edges(P, color, thickness):
        line = sub.add_line(edge_segments(Y0, P), thickness=thickness, colors=color)
        base = np.tile(np.asarray(Color(color)), (3 * len(P), 1)).astype(np.float32)
        return line, base

    lc_fp, fp_rgba = add_edges(checkpoint_PF[0], FP_COLOR, 1.0)
    lc_mn, mn_rgba = add_edges(PM, MN_COLOR, 1.2)
    lc_nb, nb_rgba = add_edges(PN, NB_COLOR, 1.2)

    scat = sub.add_scatter(
        np.ascontiguousarray(trace[0][:, :2], dtype=np.float32),
        cmap="tab10", cmap_transform=y, sizes=point_size,
    )
    scat.colors[:, -1] = point_alpha

    fig.show()  # required once to initialize the render pipeline

    def apply_edges(line, base_rgba, Y, P, alpha):
        line.data[:, :2] = edge_segments(Y, P)
        base_rgba[:, 3] = edge_vertex_alphas(alpha, len(P), line_alpha)
        line.colors[:] = base_rgba

    def update(f):
        Y = trace[f][:, :2]
        w_MN, w_NB, w_FP = W[f]
        PF = checkpoint_PF[checkpoint_index_for_frame(f, checkpoint_frames)]
        if edge_style_preset == "v3":
            a_nb, a_mn, a_fp = compute_edge_alphas(
                w_NB, w_MN, w_FP, preset=edge_style_preset, gamma=edge_gamma, Y=Y, pairs=(PN, PM, PF))
        else:
            a_nb, a_mn, a_fp = compute_edge_alphas(w_NB, w_MN, w_FP, preset=edge_style_preset, gamma=edge_gamma)
        scat.data[:, :2] = Y
        apply_edges(lc_nb, nb_rgba, Y, PN, a_nb)
        apply_edges(lc_mn, mn_rgba, Y, PM, a_mn)
        apply_edges(lc_fp, fp_rgba, Y, PF, a_fp)
        L = float(r_s[f])
        cx, cy = (float(c) for c in center[f][:2])
        sub.camera.set_state({
            "position": np.array([cx, cy, 0.0]),
            "width": 2 * L, "height": 2 * L,
            "zoom": 1.0, "maintain_aspect": True, "fov": 0.0,
        })
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
    )
    print(f"Rendering iteration {frame} of {total} to {out_path}...")
    t0 = time.time()
    update(frame)
    Image.fromarray(_export_frame(fig)).save(out_path)
    print("rendered %s in %.0fs" % (out_path, time.time() - t0))
    return out_path


def render_animation_fpl(*args, **kwargs):
    """Render an iteration range as an mp4. Not implemented yet - lands in
    plan Task 6 (offscreen frame loop -> imageio-ffmpeg)."""
    _import_fastplotlib()
    raise NotImplementedError("fastplotlib animation rendering is not implemented yet (plan Task 6)")
