"""CLI argument parsing and config-file/default merging."""

import argparse
import json


DEFAULT_CONFIG = {
    "n": 5000,
    "algorithm": "both",       # "pacmap", "localmap", or "both"
    "n_components": 2,        # 2 or 3; 3 renders via the 3D renderer
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
    "edge_style_preset": "v1",  # "v1" (raw per-type weight), "v2" (normalized/gamma-compressed), or "v3" (per-edge, distance-aware force)
    "edge_gamma": 0.2,         # v2 only: compression exponent for weight ratios
    "overlay_style_preset": "v2",  # "v1" (single-line, w_MN as float) or "v2" (w_MN/NB/FP stacked, integer, aligned) - config-file only, not a CLI flag
    "line_alpha": 1.0,       # multiplier on all edge line alphas; turn down when n_lines is high so overlapping lines don't wash out
    "fixed_camera": False,     # True -> lock a single radius instead of zooming out
    "zoom": 1.0,               # >1 frames a smaller radius (closer/finer detail, cuts off edges); works with fixed_camera and focus_label alike
    "focus_label": None,       # int -> camera tracks just that MNIST digit's cluster; "__prompt__" -> resolved interactively in main()
    "iter": None,              # None -> full-range video (default); otherwise a list of items (int -> single-iteration png, (start, end) tuple -> range video), one output rendered per item
    "output_dir": "",          # "" -> outputs/; see resolve_output_dir()
}


def load_config(config_path):
    cfg = dict(DEFAULT_CONFIG)
    if config_path:
        with open(config_path) as f:
            cfg.update(json.load(f))
    return cfg


def parse_count_arg(value):
    """Parse a --n/--n-lines CLI value: an absolute integer, or a fraction in
    (0, 1] treated as a proportion of the relevant total at the point of use
    (see resolve_proportion()); 1 or 1.0 means "all" (100%)."""
    v = float(value)
    return v if 0 < v <= 1 else int(v)


def parse_iter_arg(value):
    """Parse a single --iter token: "N" -> int N (single iteration -> png), or
    "A-B" -> (int(A), int(B)) tuple (iteration range -> mp4)."""
    if "-" in value:
        start, end = value.split("-", 1)
        return (int(start), int(end))
    return int(value)


def parse_iter_list(value):
    """Parse a (possibly comma-separated) --iter CLI value into a list of
    items, each produced by parse_iter_arg(), e.g. "50,150,250-400" ->
    [50, 150, (250, 400)] - one png/mp4 fragment rendered per item."""
    return [parse_iter_arg(token) for token in value.split(",")]


def parse_args(argv=None):
    d = DEFAULT_CONFIG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None,
                    help="JSON config file; CLI flags override its values. Unset fields fall back to the built-in defaults shown below")
    p.add_argument("--n", type=str, default=None,
                    help=f"subsample size, a fraction in (0, 1) for a proportion of the full "
                         f"~70,000 points, or 'all' for all of them (default: {d['n']})")
    p.add_argument("--algorithm", choices=["pacmap", "localmap", "both"], default=None,
                    help=f"(default: {d['algorithm']})")
    p.add_argument("--n-components", type=int, choices=[2, 3], default=None,
                    help=f"embedding dimensionality; 3 renders via the 3D renderer (default: {d['n_components']})")
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
    p.add_argument("--n-lines", type=str, default=None,
                    help=f"pairs drawn per pair-type per frame, or a fraction in (0, 1) for a "
                         f"proportion of that pair type's available pool (default: {d['n_lines']})")
    p.add_argument("--step", type=int, default=None,
                    help=f"render every Nth captured iteration (default: {d['step']})")
    p.add_argument("--fps", type=int, default=None,
                    help=f"(default: {d['fps']})")
    p.add_argument("--point-size", type=float, default=None,
                    help=f"scatter marker size; lower to see density through overlap (default: {d['point_size']})")
    p.add_argument("--point-alpha", type=float, default=None,
                    help=f"scatter marker opacity 0-1; lower so overlapping points blend into visibly denser regions (default: {d['point_alpha']})")
    p.add_argument("--edge-style-preset", choices=["v1", "v2", "v3"], default=None,
                    help="'v1': each pair type's line alpha is a fixed function of its own "
                         "raw weight (default) - mid-near's weight of up to 1000 dominates "
                         "and further pairs (fixed weight 1) are nearly invisible. "
                         "'v2': normalizes each frame's weights against their max and applies "
                         "--edge-gamma compression, so weaker pair types stay visible. "
                         "'v3': shades each individual drawn edge by its actual instantaneous "
                         "PaCMAP gradient magnitude (weight and current low-dim distance), so "
                         "e.g. a further pair already pushed apart fades even while its type's "
                         f"weight stays high (default: {d['edge_style_preset']})")
    p.add_argument("--edge-gamma", type=float, default=None,
                    help="v2/v3 edge-style-preset only: exponent compressing per-frame (v2) or "
                         "per-edge (v3) force ratios before mapping to alpha; lower = weaker "
                         f"types/edges more visible (default: {d['edge_gamma']})")
    p.add_argument("--line-alpha", type=float, default=None,
                    help="multiplier applied to every edge line's alpha; turn down when "
                         f"--n-lines is high so overlapping lines don't wash out (default: {d['line_alpha']})")
    p.add_argument("--fixed-camera", action="store_true", default=None,
                    help="lock a single camera radius sized to the trace's largest extent "
                         "instead of the default smoothed zoom-out, so you can see the true "
                         "scale of movement (early frames start as a tiny dot)")
    p.add_argument("--zoom", type=float, default=None,
                    help="camera zoom multiplier; >1 frames a smaller radius to see finer "
                         "detail at the cost of cutting off the edges of the embedding. "
                         f"Works alongside --fixed-camera and --focus-label alike (default: {d['zoom']})")
    p.add_argument("--focus-label", type=str, nargs="?", const="__prompt__", default=None,
                    help="camera tracks just this MNIST digit's cluster instead of the whole "
                         "embedding. Pass a digit (e.g. --focus-label 3), or pass the flag with "
                         "no value to be prompted for one interactively after MNIST loads "
                         f"(default: {d['focus_label']})")
    p.add_argument("--iter", type=str, default=None,
                    help="render only this iteration or range of iterations, indexed directly "
                         "against pacmap's iteration count (0-sum(--num-iters)) rather than the "
                         "--step-derived animation frame. A single value (e.g. --iter 150) "
                         "renders one still frame as a png; a range (e.g. --iter 50-300) renders "
                         "an mp4 spanning just those iterations, still subsampled by --step. "
                         "Comma-separate multiple values/ranges (e.g. --iter 50,150,250-400) to "
                         "render several outputs - one png/mp4 fragment per item, sharing a "
                         "single fit/trace - in one invocation "
                         "(default: full 0-total range as an mp4)")
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
        cfg["n"] = None if args.n.strip().lower() == "all" else parse_count_arg(args.n)
    overrides = {
        "algorithm": args.algorithm,
        "n_components": args.n_components,
        "n_neighbors": args.n_neighbors,
        "mn_ratio": args.mn_ratio,
        "fp_ratio": args.fp_ratio,
        "num_iters": [int(x) for x in args.num_iters.split(",")] if args.num_iters else None,
        "seed": args.seed,
        "n_lines": parse_count_arg(args.n_lines) if args.n_lines is not None else None,
        "step": args.step,
        "fps": args.fps,
        "point_size": args.point_size,
        "point_alpha": args.point_alpha,
        "edge_style_preset": args.edge_style_preset,
        "edge_gamma": args.edge_gamma,
        "line_alpha": args.line_alpha,
        "fixed_camera": args.fixed_camera,
        "zoom": args.zoom,
        "focus_label": args.focus_label,
        "iter": parse_iter_list(args.iter) if args.iter is not None else None,
        "output_dir": args.output_dir,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["num_iters"] = tuple(cfg["num_iters"])
    return cfg


def resolve_focus_label(focus_label, y):
    """None passes through. "__prompt__" prompts interactively for one of
    the labels actually present in `y`, reprompting on invalid input.
    Otherwise parses `focus_label` as an int."""
    if focus_label is None:
        return None
    if focus_label != "__prompt__":
        return int(focus_label)
    labels = sorted(set(y.tolist()))
    while True:
        reply = input(f"Focus on which label? {labels}: ").strip()
        try:
            choice = int(reply)
        except ValueError:
            print(f"Not a number: {reply!r}")
            continue
        if choice in labels:
            return choice
        print(f"{choice} not in {labels}")
