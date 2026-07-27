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

Thin entry point: the implementation lives in the `pacmap_cli` package
(sitting next to this file), split into modules mirroring the pipeline
stages (data -> fit -> camera -> pairs/render -> config/paths ->
orchestrate). This file re-exports everything so it can still be imported
directly (e.g. from a notebook) exactly as before the split.
"""

from pacmap_cli import camera, config, data, fit, fp_history, orchestrate, overlay, pairs, paths, render
from pacmap_cli.camera import camera_path, weight_schedule
from pacmap_cli.config import (
    DEFAULT_CONFIG,
    build_config,
    load_config,
    parse_args,
    parse_count_arg,
    parse_iter_arg,
    resolve_focus_label,
)
from pacmap_cli.data import load_mnist, resolve_proportion
from pacmap_cli.fit import fit_trace
from pacmap_cli.fp_history import capture_fp_history, checkpoint_index_for_frame, fp_resample_iterations
from pacmap_cli.orchestrate import main, run_algorithm
from pacmap_cli.overlay import compute_overlay_text
from pacmap_cli.pairs import (
    EDGE_ALPHA_MAX_V2,
    compute_edge_alphas,
    count_drawn,
    pacmap_force,
    pair_dist,
    subsample_pairs,
    subsample_pairs_indices,
)
from pacmap_cli.paths import TAG_PARAMS, param_tag, resolve_output_dir, unique_path
from pacmap_cli.render import render_animation, render_frame


if __name__ == "__main__":
    main()
