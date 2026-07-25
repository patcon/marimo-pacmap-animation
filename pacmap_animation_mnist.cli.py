# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pacmap",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
"""Render a PaCMAP / LocalMAP optimization animation on MNIST from the CLI.

    uv run pacmap_animation_mnist.cli.py --algorithm both --n 5000 --output-dir out/

Functions here are factored so they can also be imported directly into a
marimo notebook.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist(n=None, seed=0):
    """Load MNIST, optionally subsampled to `n` points. n=None loads all ~70_000."""
    try:
        from tensorflow.keras.datasets import mnist
        (Xtr, ytr), _ = mnist.load_data()
        Xfull = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
        yfull = ytr.astype(int)
    except Exception as e:
        print("keras unavailable (%s), falling back to openml" % type(e).__name__)
        from sklearn.datasets import fetch_openml
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        Xfull = d.data.astype(np.float32) / 255.0
        yfull = d.target.astype(int)

    rs = np.random.RandomState(seed)
    sel = np.arange(len(Xfull)) if n is None else rs.choice(len(Xfull), n, replace=False)

    X, y = np.ascontiguousarray(Xfull[sel]), yfull[sel]
    print(X.shape, X.dtype, np.bincount(y))
    return X, y, rs


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_trace(X, algorithm, n_neighbors, mn_ratio, fp_ratio, num_iters, seed=42):
    """Run PaCMAP or LocalMAP, capturing the embedding at every iteration."""
    import pacmap

    reducer_cls = {"pacmap": pacmap.PaCMAP, "localmap": pacmap.LocalMAP}[algorithm]
    total = sum(num_iters)

    print(f"Running {algorithm}...")
    t0 = time.time()
    reducer = reducer_cls(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=mn_ratio,
        FP_ratio=fp_ratio,
        num_iters=num_iters,
        intermediate=True,
        intermediate_snapshots=list(range(total + 1)),
        random_state=seed,
        verbose=False,
    )
    trace = reducer.fit_transform(X)  # (total+1, N, 2) float32
    print("%s fit %.1fs" % (algorithm, time.time() - t0), trace.shape, trace.nbytes / 1e6, "MB")

    pair_neighbors = reducer.pair_neighbors
    pair_MN = reducer.pair_MN
    pair_FP = reducer.pair_FP
    print(f"{algorithm} pairs:", pair_neighbors.shape, pair_MN.shape, pair_FP.shape)
    return trace, pair_neighbors, pair_MN, pair_FP


# ---------------------------------------------------------------------------
# Replay internals / camera
# ---------------------------------------------------------------------------

def weight_schedule(num_iters):
    """w_MN / w_NB / w_FP at every snapshot index, including the init frame."""
    from pacmap.pacmap import find_weight

    total = sum(num_iters)
    W = np.array([find_weight(1000.0, i, num_iters=num_iters) for i in range(total)])
    W = np.vstack([W[0], W])  # prepend so index == snapshot index
    return W


def camera_path(trace, smooth_window=15, headroom=1.15, fixed=False):
    """Per-frame camera radius. Smoothed, monotonic zoom-out by default so
    early iterations stay legible; `fixed=True` instead locks a single radius
    (sized to the trace's largest extent) for the whole animation, so you can
    see the true scale of the movement even though early frames start as a
    tiny dot."""
    r = np.percentile(np.abs(trace).reshape(len(trace), -1), 99.5, axis=1)
    if fixed:
        return np.full(len(trace), r.max() * headroom)
    k = smooth_window
    r_s = np.convolve(np.r_[np.full(k, r[0]), r], np.ones(k) / k, mode="valid")
    r_s = np.maximum.accumulate(r_s) * headroom
    return r_s


def subsample_pairs(pairs, m, rs):
    return pairs[rs.choice(len(pairs), min(m, len(pairs)), replace=False)]


# Per-type opacity ceilings used by the "v2" edge-style preset (see
# compute_edge_alphas). Tuned so each pair type can reach a comparable peak
# visibility instead of mid-near's raw weight (up to 1000) drowning out
# neighbour (2-3) and further (always 1).
EDGE_ALPHA_MAX_V2 = {"nb": 0.5, "mn": 0.6, "fp": 0.45}


def compute_edge_alphas(w_NB, w_MN, w_FP, preset="v1", gamma=0.2):
    """Per-frame LineCollection alpha for (neighbour, mid-near, further).

    "v1" is the original mapping: each type's alpha is a fixed function of
    its own raw weight, independent of the others. Because w_MN ranges from
    1000 down to 0 while w_NB and w_FP barely move (2-3 and 1 respectively),
    mid-near dominates visually for most of phase 1 and further pairs (fixed
    weight of 1) never rise above alpha 0.05 - effectively invisible.

    "v2" normalizes the three weights against their per-frame max (the
    "combined force" for that frame) and applies a gamma < 1 to compress the
    resulting ratio, so a type that is orders of magnitude weaker than the
    frame's dominant force still gets a visible (if faint) share instead of
    being crushed to ~0. Each type is then scaled by its own opacity ceiling
    so mid-near can still read as the strongest without hiding the others.
    """
    if preset == "v1":
        a_nb = 0.10 * w_NB / 3
        a_mn = 0.55 * w_MN / (w_MN + 3)
        a_fp = 0.05 * w_FP
        return a_nb, a_mn, a_fp

    w_max = max(w_NB, w_MN, w_FP, 1e-9)
    r_nb = (w_NB / w_max) ** gamma
    r_mn = (w_MN / w_max) ** gamma
    r_fp = (w_FP / w_max) ** gamma
    a_nb = EDGE_ALPHA_MAX_V2["nb"] * r_nb
    a_mn = EDGE_ALPHA_MAX_V2["mn"] * r_mn
    a_fp = EDGE_ALPHA_MAX_V2["fp"] * r_fp
    return a_nb, a_mn, a_fp


def compute_overlay_text(f, total, ph, w_MN, w_NB, w_FP, title_prefix="", preset="v1"):
    """Title-block text for a frame. "v1" is the original single-line format,
    with w_MN shown as a float (it's fractional during phase 1's ramp) and no
    w_FP. "v2" rounds w_MN to an integer for readability and stacks all three
    weights as separate, label-aligned lines so their digits fall in the same
    column frame to frame - easier to see them move together at a glance."""
    if preset == "v1":
        return f"{title_prefix}iter %3d/%d   phase %d   w_MN=%7.1f  w_NB=%.0f" % (f, total, ph, w_MN, w_NB)
    return (
        "%siter %3d/%d   phase %d\n"
        "w_MN %4d\n"
        "w_NB %4d\n"
        "w_FP %4d"
    ) % (title_prefix, f, total, ph, round(w_MN), round(w_NB), round(w_FP))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_animation(
    trace, y, W, pair_neighbors, pair_MN, pair_FP, num_iters, r_s, rs,
    out_path, n_lines=150, step=3, fps=25, title_prefix="",
    point_size=5, point_alpha=1.0,
    edge_style_preset="v1", edge_gamma=0.2,
    overlay_style_preset="v1",
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    total = sum(num_iters)
    PN = subsample_pairs(pair_neighbors, n_lines, rs)
    PM = subsample_pairs(pair_MN, n_lines, rs)
    PF = subsample_pairs(pair_FP, n_lines, rs)
    frames = list(range(0, len(trace), step))
    BG = "#0d0d10"

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
    lc_fp = LineCollection([], colors="#ff4d4d", linewidths=0.5, zorder=1)
    lc_mn = LineCollection([], colors="#ffa53d", linewidths=0.7, zorder=2)
    lc_nb = LineCollection([], colors="#4da6ff", linewidths=0.7, zorder=3)
    for lc in (lc_fp, lc_mn, lc_nb):
        ax.add_collection(lc)
    scat = ax.scatter(trace[0][:, 0], trace[0][:, 1], c=y, cmap="tab10",
                       s=point_size, alpha=point_alpha, linewidths=0, zorder=4)
    title = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="w", fontsize=11, va="top", family="monospace")
    ax.text(0.02, 0.03, "neighbour", transform=ax.transAxes, color="#4da6ff", fontsize=9)
    ax.text(0.16, 0.03, "mid-near", transform=ax.transAxes, color="#ffa53d", fontsize=9)
    ax.text(0.29, 0.03, "further", transform=ax.transAxes, color="#ff4d4d", fontsize=9)
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

    def update(f):
        Y = trace[f]
        w_MN, w_NB, w_FP = W[f]
        a_nb, a_mn, a_fp = compute_edge_alphas(w_NB, w_MN, w_FP, preset=edge_style_preset, gamma=edge_gamma)
        scat.set_offsets(Y)
        lc_nb.set_segments(seg(Y, PN)); lc_nb.set_alpha(a_nb)
        lc_mn.set_segments(seg(Y, PM)); lc_mn.set_alpha(a_mn)
        lc_fp.set_segments(seg(Y, PF)); lc_fp.set_alpha(a_fp)
        L = r_s[f]; ax.set_xlim(-L, L); ax.set_ylim(-L, L)
        ph = 1 if f <= num_iters[0] else (2 if f <= num_iters[0] + num_iters[1] else 3)
        title.set_text(compute_overlay_text(f, total, ph, w_MN, w_NB, w_FP, title_prefix, preset=overlay_style_preset))
        vline.set_xdata([f, f])
        return ()

    n_frames = len(frames)
    print(f"Rendering {n_frames} frames ({len(trace)} captured iterations, step={step}) to {out_path}...")
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def unique_path(path):
    """Return `path` if it doesn't exist. Otherwise ask whether to overwrite it;
    if not, return `path` with an incrementing number appended before the
    suffix (e.g. foo.mp4 -> foo_1.mp4 -> foo_2.mp4) instead."""
    path = Path(path)
    if not path.exists():
        return path
    reply = input(f"{path} already exists. Overwrite? [y/N] ").strip().lower()
    if reply == "y":
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def run_algorithm(X, y, rs, algorithm, cfg, out_path):
    trace, pair_neighbors, pair_MN, pair_FP = fit_trace(
        X,
        algorithm,
        n_neighbors=cfg["n_neighbors"],
        mn_ratio=cfg["mn_ratio"],
        fp_ratio=cfg["fp_ratio"],
        num_iters=cfg["num_iters"],
        seed=cfg["seed"],
    )
    W = weight_schedule(cfg["num_iters"])
    r_s = camera_path(trace, fixed=cfg["fixed_camera"])
    return render_animation(
        trace, y, W, pair_neighbors, pair_MN, pair_FP, cfg["num_iters"], r_s, rs,
        out_path=str(out_path),
        n_lines=cfg["n_lines"],
        step=cfg["step"],
        fps=cfg["fps"],
        title_prefix=f"{algorithm} " if algorithm == "localmap" else "",
        point_size=cfg["point_size"],
        point_alpha=cfg["point_alpha"],
        edge_style_preset=cfg["edge_style_preset"],
        edge_gamma=cfg["edge_gamma"],
        overlay_style_preset=cfg["overlay_style_preset"],
    )


DEFAULT_CONFIG = {
    "n": 5000,
    "algorithm": "both",       # "pacmap", "localmap", or "both"
    "n_neighbors": 10,
    "mn_ratio": 0.5,
    "fp_ratio": 2.0,
    "num_iters": [100, 100, 250],
    "seed": 42,
    "n_lines": 150,
    "step": 3,
    "fps": 25,
    "point_size": 5,
    "point_alpha": 1.0,
    "edge_style_preset": "v1",  # "v1" (raw per-type weight) or "v2" (normalized/gamma-compressed)
    "edge_gamma": 0.2,         # v2 only: compression exponent for weight ratios
    "overlay_style_preset": "v1",  # "v1" (single-line, w_MN as float) or "v2" (w_MN/NB/FP stacked, integer, aligned) - config-file only, not a CLI flag
    "fixed_camera": False,     # True -> lock a single radius instead of zooming out
    "output_dir": "",          # "" -> outputs/; see resolve_output_dir()
}


def resolve_output_dir(output_dir):
    """"" -> outputs/. An absolute path, or one starting with "./" or "../",
    is used as-is. Any other relative path is nested under outputs/, e.g.
    "myrun" -> outputs/myrun."""
    if not output_dir:
        return Path("outputs")
    p = Path(output_dir)
    if p.is_absolute() or str(output_dir).startswith(("./", "../")):
        return p
    return Path("outputs") / p


# Params worth encoding in a --tag-output slug, in display order. "algorithm"
# and "output_dir" are deliberately excluded: they don't affect the fit/render
# in a way you'd want to distinguish runs by in a shared comparison folder.
TAG_PARAMS = [
    ("n", "n"),
    ("n_neighbors", "nn"),
    ("mn_ratio", "mnr"),
    ("fp_ratio", "fpr"),
    ("num_iters", "iters"),
    ("seed", "seed"),
    ("n_lines", "nlines"),
    ("step", "step"),
    ("fps", "fps"),
    ("point_size", "psize"),
    ("point_alpha", "palpha"),
    ("edge_style_preset", "edge"),
    ("edge_gamma", "gamma"),
    ("fixed_camera", "camfixed"),
]


def param_tag(cfg):
    """Slug of the params in `cfg` that differ from DEFAULT_CONFIG, e.g.
    "nn5_mnr0.8". Falls back to "default" if nothing differs."""
    parts = []
    for key, abbr in TAG_PARAMS:
        val, default = cfg[key], DEFAULT_CONFIG[key]
        if key == "num_iters":
            val, default = tuple(val), tuple(default)
        if val == default:
            continue
        if key == "n":
            val_str = "all" if val is None else str(val)
        elif key == "num_iters":
            val_str = "-".join(map(str, val))
        else:
            val_str = str(val)
        parts.append(f"{abbr}{val_str}")
    return "_".join(parts) if parts else "default"


def load_config(config_path):
    cfg = dict(DEFAULT_CONFIG)
    if config_path:
        with open(config_path) as f:
            cfg.update(json.load(f))
    return cfg


def parse_args(argv=None):
    d = DEFAULT_CONFIG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None,
                    help="JSON config file; CLI flags override its values. Unset fields fall back to the built-in defaults shown below")
    p.add_argument("--n", type=str, default=None,
                    help=f"subsample size, or 'all' for the full ~70,000 points (default: {d['n']})")
    p.add_argument("--algorithm", choices=["pacmap", "localmap", "both"], default=None,
                    help=f"(default: {d['algorithm']})")
    p.add_argument("--n-neighbors", type=int, default=None,
                    help=f"(default: {d['n_neighbors']})")
    p.add_argument("--mn-ratio", type=float, default=None,
                    help=f"(default: {d['mn_ratio']})")
    p.add_argument("--fp-ratio", type=float, default=None,
                    help=f"(default: {d['fp_ratio']})")
    p.add_argument("--num-iters", type=str, default=None,
                    help=f"comma-separated PaCMAP phase lengths, e.g. 100,100,250 (default: {','.join(map(str, d['num_iters']))})")
    p.add_argument("--seed", type=int, default=None,
                    help=f"(default: {d['seed']})")
    p.add_argument("--n-lines", type=int, default=None,
                    help=f"pairs drawn per pair-type per frame (default: {d['n_lines']})")
    p.add_argument("--step", type=int, default=None,
                    help=f"render every Nth captured iteration (default: {d['step']})")
    p.add_argument("--fps", type=int, default=None,
                    help=f"(default: {d['fps']})")
    p.add_argument("--point-size", type=float, default=None,
                    help=f"scatter marker size; lower to see density through overlap (default: {d['point_size']})")
    p.add_argument("--point-alpha", type=float, default=None,
                    help=f"scatter marker opacity 0-1; lower so overlapping points blend into visibly denser regions (default: {d['point_alpha']})")
    p.add_argument("--edge-style-preset", choices=["v1", "v2"], default=None,
                    help="'v1': each pair type's line alpha is a fixed function of its own "
                         "raw weight (default) - mid-near's weight of up to 1000 dominates "
                         "and further pairs (fixed weight 1) are nearly invisible. "
                         "'v2': normalizes each frame's weights against their max and applies "
                         "--edge-gamma compression, so weaker pair types stay visible "
                         f"(default: {d['edge_style_preset']})")
    p.add_argument("--edge-gamma", type=float, default=None,
                    help="v2 edge-style-preset only: exponent compressing per-frame weight "
                         f"ratios before mapping to alpha; lower = weaker types more visible (default: {d['edge_gamma']})")
    p.add_argument("--fixed-camera", action="store_true", default=None,
                    help="lock a single camera radius sized to the trace's largest extent "
                         "instead of the default smoothed zoom-out, so you can see the true "
                         "scale of movement (early frames start as a tiny dot)")
    p.add_argument("--output-dir", type=str, default=None,
                    help="output directory (default: outputs/). An absolute path, or one "
                         "starting with ./ or ../, is used as-is; any other relative path "
                         "is nested under outputs/, e.g. 'myrun' -> outputs/myrun")
    p.add_argument("--tag-output", action="store_true",
                    help="nest the render under a subdirectory named after the "
                         "non-default params (e.g. outputs/nn5_mnr0.8/), so runs "
                         "with different settings are easy to tell apart at a glance")
    return p.parse_args(argv)


def build_config(args):
    cfg = load_config(args.config)
    if args.n is not None:
        cfg["n"] = None if args.n.strip().lower() == "all" else int(args.n)
    overrides = {
        "algorithm": args.algorithm,
        "n_neighbors": args.n_neighbors,
        "mn_ratio": args.mn_ratio,
        "fp_ratio": args.fp_ratio,
        "num_iters": [int(x) for x in args.num_iters.split(",")] if args.num_iters else None,
        "seed": args.seed,
        "n_lines": args.n_lines,
        "step": args.step,
        "fps": args.fps,
        "point_size": args.point_size,
        "point_alpha": args.point_alpha,
        "edge_style_preset": args.edge_style_preset,
        "edge_gamma": args.edge_gamma,
        "fixed_camera": args.fixed_camera,
        "output_dir": args.output_dir,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["num_iters"] = tuple(cfg["num_iters"])
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    output_dir = resolve_output_dir(cfg["output_dir"])
    if args.tag_output:
        output_dir = output_dir / param_tag(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["pacmap", "localmap"] if cfg["algorithm"] == "both" else [cfg["algorithm"]]
    # Resolve (and confirm any overwrite of) output filenames before running
    # any computation, so approval doesn't happen after a long fit/render.
    out_paths = {a: unique_path(output_dir / f"{a}_mnist.mp4") for a in algorithms}

    X, y, rs = load_mnist(n=cfg["n"], seed=cfg["seed"])

    for algorithm in algorithms:
        run_algorithm(X, y, rs, algorithm, cfg, out_paths[algorithm])


if __name__ == "__main__":
    main()
