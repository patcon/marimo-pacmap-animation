"""Output path resolution: where renders land and how they're named."""

from pathlib import Path

from .config import DEFAULT_CONFIG


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
    ("low_dist_thres", "ldt"),
    ("num_iters", "iters"),
    ("schedule_preset", "sched"),
    ("schedule_period", "period"),
    ("schedule_mn_min", "mnmin"),
    ("schedule_mn_max", "mnmax"),
    ("schedule_fp_min", "fpmin"),
    ("schedule_fp_max", "fpmax"),
    ("schedule_fp_phase", "fpphase"),
    ("seed", "seed"),
    ("n_lines", "nlines"),
    ("step", "step"),
    ("fps", "fps"),
    ("point_size", "psize"),
    ("point_alpha", "palpha"),
    ("edge_style_preset", "edge"),
    ("edge_gamma", "gamma"),
    ("line_alpha", "linealpha"),
    ("fixed_camera", "camfixed"),
    ("focus_label", "focus"),
]


def param_tag(cfg):
    """Slug of the params in `cfg` that differ from DEFAULT_CONFIG, e.g.
    "nn5_mnr0.8". Falls back to "default" if nothing differs."""
    parts = []
    # The cycle knobs mean nothing under the vanilla preset (which doesn't
    # patch the fit at all), so they shouldn't split a comparison folder in
    # two - the same rule the cache key applies to them.
    skip = set() if cfg["schedule_preset"] != "vanilla" else {
        key for key, _ in TAG_PARAMS if key.startswith("schedule_") and key != "schedule_preset"
    }
    for key, abbr in TAG_PARAMS:
        if key in skip:
            continue
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
