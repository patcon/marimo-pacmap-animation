"""Top-level orchestration: run one algorithm end to end, and the CLI entry point."""

from .camera import camera_path, weight_schedule
from .config import build_config, parse_args, resolve_focus_label
from .data import load_mnist
from .fit import fit_trace
from .paths import param_tag, resolve_output_dir, unique_path
from .render import render_animation, render_frame


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
    center, r_s = camera_path(trace, y=y, focus_label=cfg["focus_label"], fixed=cfg["fixed_camera"])

    total = sum(cfg["num_iters"])
    iter_cfg = cfg["iter"]
    if iter_cfg is not None:
        bounds = iter_cfg if isinstance(iter_cfg, tuple) else (iter_cfg, iter_cfg)
        if not (0 <= bounds[0] <= bounds[1] <= total):
            raise ValueError(f"--iter {iter_cfg} out of range: iterations run 0-{total}")

    common = dict(
        trace=trace, y=y, W=W, pair_neighbors=pair_neighbors, pair_MN=pair_MN, pair_FP=pair_FP,
        num_iters=cfg["num_iters"], center=center, r_s=r_s, rs=rs,
        out_path=str(out_path),
        n_lines=cfg["n_lines"],
        title_prefix=f"{algorithm} " if algorithm == "localmap" else "",
        point_size=cfg["point_size"],
        point_alpha=cfg["point_alpha"],
        edge_style_preset=cfg["edge_style_preset"],
        edge_gamma=cfg["edge_gamma"],
        overlay_style_preset=cfg["overlay_style_preset"],
        line_alpha=cfg["line_alpha"],
    )
    if isinstance(iter_cfg, int):
        return render_frame(**common, frame=iter_cfg)
    start, end = iter_cfg if isinstance(iter_cfg, tuple) else (None, None)
    return render_animation(**common, step=cfg["step"], fps=cfg["fps"], start=start, end=end)


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
    # A single --iter value renders one still frame (png); None or a range
    # renders the usual mp4. The suffix makes different --iter renders in
    # the same directory distinguishable at a glance.
    iter_cfg = cfg["iter"]
    if iter_cfg is None:
        suffix, ext = "", "mp4"
    elif isinstance(iter_cfg, tuple):
        suffix, ext = f"_iter{iter_cfg[0]}-{iter_cfg[1]}", "mp4"
    else:
        suffix, ext = f"_iter{iter_cfg}", "png"
    # Resolve (and confirm any overwrite of) output filenames before running
    # any computation, so approval doesn't happen after a long fit/render.
    out_paths = {a: unique_path(output_dir / f"{a}_mnist{suffix}.{ext}") for a in algorithms}

    for algorithm in algorithms:
        run_algorithm(X, y, rs, algorithm, cfg, out_paths[algorithm])
