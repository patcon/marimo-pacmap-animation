"""Top-level orchestration: run one algorithm end to end, and the CLI entry point."""

from .camera import camera_path, weight_schedule
from .config import build_config, parse_args, resolve_focus_label
from .data import load_mnist
from .fit import fit_trace
from .paths import param_tag, resolve_output_dir, unique_path
from .render import render_animation, render_frame


def run_algorithm(X, y, rs, algorithm, cfg, iter_out_paths):
    """Fit once, then render one output per (iter_item, out_path) pair in
    iter_out_paths - iter_item is None (full range), an int (single-iteration
    png), or a (start, end) tuple (range mp4)."""
    trace, pair_neighbors, pair_MN, pair_FP_history = fit_trace(
        X,
        algorithm,
        n_neighbors=cfg["n_neighbors"],
        mn_ratio=cfg["mn_ratio"],
        fp_ratio=cfg["fp_ratio"],
        num_iters=cfg["num_iters"],
        seed=cfg["seed"],
    )
    W = weight_schedule(cfg["num_iters"])
    center, r_s = camera_path(trace, y=y, focus_label=cfg["focus_label"], fixed=cfg["fixed_camera"], zoom=cfg["zoom"])

    total = sum(cfg["num_iters"])
    common = dict(
        trace=trace, y=y, W=W, pair_neighbors=pair_neighbors, pair_MN=pair_MN, pair_FP_history=pair_FP_history,
        num_iters=cfg["num_iters"], center=center, r_s=r_s, rs=rs,
        n_lines=cfg["n_lines"],
        title_prefix=f"{algorithm} " if algorithm == "localmap" else "",
        point_size=cfg["point_size"],
        point_alpha=cfg["point_alpha"],
        edge_style_preset=cfg["edge_style_preset"],
        edge_gamma=cfg["edge_gamma"],
        overlay_style_preset=cfg["overlay_style_preset"],
        line_alpha=cfg["line_alpha"],
    )

    results = []
    for iter_item, out_path in iter_out_paths:
        if iter_item is not None:
            bounds = iter_item if isinstance(iter_item, tuple) else (iter_item, iter_item)
            if not (0 <= bounds[0] <= bounds[1] <= total):
                raise ValueError(f"--iter {iter_item} out of range: iterations run 0-{total}")
        if isinstance(iter_item, int):
            results.append(render_frame(**common, out_path=str(out_path), frame=iter_item))
        else:
            start, end = iter_item if isinstance(iter_item, tuple) else (None, None)
            results.append(render_animation(
                **common, out_path=str(out_path), step=cfg["step"], fps=cfg["fps"], start=start, end=end,
            ))
    return results


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    # Loaded before output-path resolution (cheap) so --focus-label can be
    # prompted against the real label set; the expensive step (fit_trace)
    # still stays gated behind the overwrite-confirmation below.
    X, y, rs = load_mnist(n=cfg["n"], seed=cfg["seed"])
    cfg["focus_label"] = resolve_focus_label(cfg["focus_label"], y)

    output_dir = resolve_output_dir(cfg["output_dir"])
    if args.tag_output:
        output_dir = output_dir / param_tag(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["pacmap", "localmap"] if cfg["algorithm"] == "both" else [cfg["algorithm"]]
    # cfg["iter"] is None (full range -> one mp4) or a list of items, each an
    # int (single-iteration png) or (start, end) tuple (range mp4); one output
    # is rendered per item. The suffix makes different --iter renders in the
    # same directory distinguishable at a glance.
    iter_items = cfg["iter"] if cfg["iter"] is not None else [None]

    def _suffix_ext(iter_item):
        if iter_item is None:
            return "", "mp4"
        if isinstance(iter_item, tuple):
            return f"_iter{iter_item[0]}-{iter_item[1]}", "mp4"
        return f"_iter{iter_item}", "png"

    # Resolve (and confirm any overwrite of) output filenames before running
    # any computation, so approval doesn't happen after a long fit/render.
    out_paths = {
        a: [
            (iter_item, unique_path(output_dir / f"{a}_mnist{suffix}.{ext}"))
            for iter_item in iter_items
            for suffix, ext in [_suffix_ext(iter_item)]
        ]
        for a in algorithms
    }

    for algorithm in algorithms:
        run_algorithm(X, y, rs, algorithm, cfg, out_paths[algorithm])
