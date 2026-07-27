"""Building the figure/artists and rendering frames to mp4 or png."""

import time

import numpy as np

from .fp_history import checkpoint_index_for_frame
from .overlay import compute_overlay_text
from .pairs import compute_edge_alphas, count_drawn, subsample_pairs, subsample_pairs_indices


def _build_renderer(
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
    n_lines=150, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
    line_alpha=1.0,
):
    """Build the figure/artists and return `(fig, update, total, BG)`, where
    `update(f)` mutates all artists in place to show trace index `f`. Shared
    by `render_animation` (many frames -> mp4) and `render_frame` (one frame
    -> png)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    total = sum(num_iters)
    PN = subsample_pairs(pair_neighbors, n_lines, rs)
    PM = subsample_pairs(pair_MN, n_lines, rs)
    # Sample far-pair row indices once (against any checkpoint - they all
    # share the same shape) so the same source point's edge is tracked by
    # index as its target endpoint changes across LocalMAP's resampled
    # snapshots, instead of independently re-subsampling fresh rows at
    # every checkpoint.
    fp_idx = subsample_pairs_indices(pair_FP_history[0][1], n_lines, rs)
    checkpoint_frames = np.array([f for f, _arr in pair_FP_history])
    checkpoint_PF = [arr[fp_idx] for _f, arr in pair_FP_history]
    counts = count_drawn(len(trace[0]), PN, PM, checkpoint_PF[0])
    print(
        f"Drawing {counts['nodes']} nodes and {counts['edges_total']} edges "
        f"(neighbour={counts['edges_neighbor']}, mid-near={counts['edges_midnear']}, "
        f"further={counts['edges_further']})"
    )
    BG = "#0d0d10"
    NB_COLOR, MN_COLOR, FP_COLOR = "#4da6ff", "#ffa53d", "#ff4d4d"

    fig = plt.figure(figsize=(7, 8), dpi=110)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.83])
    ax.set_facecolor(BG)
    axw = fig.add_axes([0.09, 0.05, 0.82, 0.07])
    axw.set_facecolor(BG)
    for a in (ax, axw):
        for s in a.spines.values():
            s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    lc_fp = LineCollection([], colors=FP_COLOR, linewidths=0.5, zorder=1)
    lc_mn = LineCollection([], colors=MN_COLOR, linewidths=0.7, zorder=2)
    lc_nb = LineCollection([], colors=NB_COLOR, linewidths=0.7, zorder=3)
    for lc in (lc_fp, lc_mn, lc_nb):
        ax.add_collection(lc)
    scat = ax.scatter(trace[0][:, 0], trace[0][:, 1], c=y, cmap="tab10",
                       s=point_size, alpha=point_alpha, linewidths=0, zorder=4)
    title = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="w", fontsize=11, va="top", family="monospace")
    ax.text(0.02, 0.03, "neighbour", transform=ax.transAxes, color=NB_COLOR, fontsize=9)
    ax.text(0.16, 0.03, "mid-near", transform=ax.transAxes, color=MN_COLOR, fontsize=9)
    ax.text(0.29, 0.03, "further", transform=ax.transAxes, color=FP_COLOR, fontsize=9)
    it = np.arange(total + 1)
    for j, c in enumerate(("#ffa53d", "#4da6ff", "#ff4d4d")):
        axw.plot(it, np.log10(W[:, j] + 1), color=c, lw=1.4)
    for b in (num_iters[0], num_iters[0] + num_iters[1]):
        axw.axvline(b, color="#555", lw=0.8, ls=":")
    vline = axw.axvline(0, color="w", lw=1.2)
    axw.set_xlim(0, total)
    axw.set_yticks([])
    axw.tick_params(colors="#888", labelsize=7)
    axw.set_xlabel("iteration  (log weight)", color="#888", fontsize=8)

    def seg(Y, p):
        return np.stack([Y[p[:, 0]], Y[p[:, 1]]], axis=1)

    def apply_alpha(lc, base_color, alpha):
        """Apply a scalar or per-edge alpha to a LineCollection. Arrays are
        baked into per-segment RGBA (via set_color) rather than passed to
        set_alpha, since array support there is matplotlib-version-dependent."""
        alpha = np.clip(np.asarray(alpha) * line_alpha, 0.0, 1.0)
        if alpha.ndim == 0:
            lc.set_alpha(float(alpha))
        else:
            rgba = np.tile(to_rgba(base_color), (len(alpha), 1))
            rgba[:, 3] = alpha
            lc.set_alpha(None)
            lc.set_color(rgba)

    def update(f):
        Y = trace[f]
        w_MN, w_NB, w_FP = W[f]
        PF = checkpoint_PF[checkpoint_index_for_frame(f, checkpoint_frames)]
        if edge_style_preset == "v3":
            a_nb, a_mn, a_fp = compute_edge_alphas(
                w_NB, w_MN, w_FP, preset=edge_style_preset, gamma=edge_gamma, Y=Y, pairs=(PN, PM, PF))
        else:
            a_nb, a_mn, a_fp = compute_edge_alphas(w_NB, w_MN, w_FP, preset=edge_style_preset, gamma=edge_gamma)
        scat.set_offsets(Y)
        lc_nb.set_segments(seg(Y, PN)); apply_alpha(lc_nb, NB_COLOR, a_nb)
        lc_mn.set_segments(seg(Y, PM)); apply_alpha(lc_mn, MN_COLOR, a_mn)
        lc_fp.set_segments(seg(Y, PF)); apply_alpha(lc_fp, FP_COLOR, a_fp)
        L = r_s[f]; cx, cy = center[f]
        ax.set_xlim(cx - L, cx + L); ax.set_ylim(cy - L, cy + L)
        ph = 1 if f <= num_iters[0] else (2 if f <= num_iters[0] + num_iters[1] else 3)
        title.set_text(compute_overlay_text(f, total, ph, w_MN, w_NB, w_FP, title_prefix, preset=overlay_style_preset))
        vline.set_xdata([f, f])
        return ()

    return fig, update, total, BG


def render_animation(
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
    out_path, n_lines=150, step=3, fps=25, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
    line_alpha=1.0,
    start=None, end=None,
):
    """Render trace indices `start`..`end` inclusive (default: the whole
    trace) as an mp4, stepping by `step`."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, update, total, BG = _build_renderer(
        trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
        n_lines=n_lines, title_prefix=title_prefix,
        point_size=point_size, point_alpha=point_alpha,
        edge_style_preset=edge_style_preset, edge_gamma=edge_gamma,
        overlay_style_preset=overlay_style_preset, line_alpha=line_alpha,
    )
    start = 0 if start is None else start
    end = total if end is None else end
    frames = list(range(start, end + 1, step))

    n_frames = len(frames)
    print(f"Rendering {n_frames} frames (iterations {start}-{end} of {len(trace)} captured, step={step}) to {out_path}...")
    t0 = time.time()
    report_every = max(1, n_frames // 20)  # ~20 progress lines regardless of frame count

    def progress(current, total):
        if current % report_every != 0 and current != total - 1:
            return
        elapsed = time.time() - t0
        rate = (current + 1) / elapsed if elapsed > 0 else 0
        eta = (total - current - 1) / rate if rate > 0 else float("nan")
        print(f"  frame {current + 1}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)
    anim.save(out_path, writer="ffmpeg", fps=fps, savefig_kwargs={"facecolor": BG},
              progress_callback=progress)
    plt.close(fig)
    print("rendered %s in %.0fs" % (out_path, time.time() - t0))
    return out_path


def render_frame(
    trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
    out_path, frame, n_lines=150, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
    line_alpha=1.0,
):
    """Render a single trace index `frame` as a png."""
    import matplotlib.pyplot as plt

    fig, update, total, BG = _build_renderer(
        trace, y, W, pair_neighbors, pair_MN, pair_FP_history, num_iters, center, r_s, rs,
        n_lines=n_lines, title_prefix=title_prefix,
        point_size=point_size, point_alpha=point_alpha,
        edge_style_preset=edge_style_preset, edge_gamma=edge_gamma,
        overlay_style_preset=overlay_style_preset, line_alpha=line_alpha,
    )
    print(f"Rendering iteration {frame} of {total} to {out_path}...")
    t0 = time.time()
    update(frame)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    print("rendered %s in %.0fs" % (out_path, time.time() - t0))
    return out_path
