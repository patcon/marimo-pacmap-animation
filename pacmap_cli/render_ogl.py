"""The `ogl` renderer backend: export the trace instead of rasterizing it.

Every other backend bakes frames into pixels. This one writes a single .pcmp
file (see pcmp.py for the byte layout) holding the embedding positions, baked
per-point colors and the per-frame metadata the player needs, and hands the
drawing to the static WebGL app in app/. That buys free orbit/zoom and a
scrubbable timeline without a re-render per viewpoint.

It sits in the RENDERERS registry because that is where the trace, the weight
schedule and camera_path()'s output already converge -- so it needs no new
plumbing through main()/run_algorithm(). The one place the "data, not pixels"
mismatch shows is the output extension, handled by RENDERER_OUTPUT_EXT in
render.py.

Two deliberate non-features:

- **No edges.** The pair_* arguments are accepted and ignored. Points first;
  edges are additive later since the arguments already arrive.
- **No overlay/style arguments are honored** (edge_style_preset, line_alpha,
  point_size, ...). Those are viewer-side concerns now, adjustable live in the
  page rather than frozen at export time. They are still accepted, because
  signature parity across backends is what render_animation()'s **kwargs
  pass-through relies on.

Colors *are* baked here rather than in JavaScript, so that an ogl export and
the mp4 of the same run agree: matplotlib's own Normalize over the data range
is what ax.scatter(c=y, cmap=...) applies, and reimplementing that in JS for
both the categorical (tab10) and continuous (viridis) schemes would be a
second source of truth.
"""
import os
import time
from pathlib import Path

import numpy as np

from .datasets import CATEGORICAL_CMAP
from .pcmp import quantize_positions, write_pcmp


def render_animation_ogl(
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
    """Export trace indices `start`..`end` inclusive (default: the whole
    trace), stepped by `step`, as a .pcmp."""
    total = len(trace) - 1
    start = 0 if start is None else start
    end = total if end is None else end
    # The same frame list _render_animation_mpl builds, so --step means one
    # thing regardless of backend.
    frames = list(range(start, end + 1, step))
    return _write(trace, y, W, num_iters, center, r_s, frames, out_path, cmap, title_prefix, fps)


def render_frame_ogl(
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
    out_path, frame, n_lines=150, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
    line_alpha=1.0,
    cmap=CATEGORICAL_CMAP,
    n_components=2, rotate=False,
):
    """Export a single trace index as a one-frame .pcmp. Degenerate but
    consistent -- the player shows a static scatter and the scrubber has one
    position."""
    return _write(trace, y, W, num_iters, center, r_s, [frame], out_path, cmap, title_prefix, fps=25)


def _write(trace, y, W, num_iters, center, r_s, frames, out_path, cmap, title_prefix, fps):
    t0 = time.time()
    idx = np.asarray(frames)
    positions = np.ascontiguousarray(trace[idx], dtype=np.float32)
    n_frames, n_points, dims = positions.shape
    # Halves the file, which is almost entirely these coordinates, and costs
    # the player nothing: a normalized uint16 vertex attribute is expanded to
    # float by the GPU during the fetch. See quantize_positions for why the
    # range is per frame rather than global.
    quantized, pos_min, pos_extent = quantize_positions(positions)

    header = {
        # Enough to label the view; the dataset/algorithm identity itself is
        # not among the renderer arguments, so it is taken from the filename
        # main() already encoded it into.
        "label": Path(out_path).stem,
        "title_prefix": title_prefix,
        "frames": n_frames,
        "points": n_points,
        "dims": dims,
        "iters": [int(i) for i in idx],
        "num_iters": [int(n) for n in num_iters],
        "fps": int(fps),
        "cmap": cmap,
        # Per exported frame, not per trace index, so the player reads row f
        # directly rather than mapping through `iters`.
        "weights": np.asarray(W)[idx].tolist(),
        "center": np.asarray(center)[idx].tolist(),
        "radius": np.asarray(r_s)[idx].tolist(),
        # The per-frame decode for `positions`: p = pos_min + q/QUANT_MAX * pos_extent.
        "pos_min": pos_min.tolist(),
        "pos_extent": pos_extent.tolist(),
    }
    arrays = {"positions": quantized, "colors": _bake_colors(y, cmap)}

    print(f"Exporting {n_frames} frames x {n_points} points ({dims}D) to {out_path}...")
    write_pcmp(out_path, header, arrays)
    size_mb = os.path.getsize(out_path) / 1e6
    print("exported %s (%.1f MB) in %.1fs" % (out_path, size_mb, time.time() - t0))
    return out_path


def _bake_colors(y, cmap):
    """Resolve `y` to per-point RGB through `cmap`, matching what matplotlib's
    scatter would do with the same colormap name and data."""
    import matplotlib

    y = np.asarray(y)
    norm = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
    rgba = matplotlib.colormaps[cmap](norm(y))
    return np.ascontiguousarray(rgba[:, :3], dtype=np.float32)
