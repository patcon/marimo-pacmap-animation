# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small, self-contained project that animates PaCMAP and LocalMAP (dimensionality reduction algorithms) optimizing a 2D embedding of MNIST, frame by frame, and renders the result as MP4 video. It captures the embedding at every optimization iteration via the `pacmap` library's public `intermediate_snapshots` kwarg — no forking or monkey-patching of the library.

## Files

- `pacmap_animation_mnist.marimo.py` — the marimo notebook (interactive, cell-based). This is the primary/reference implementation.
- `pacmap_animation_mnist.cli.py` — headless CLI script that mirrors the notebook's logic, factored into importable functions (`load_mnist`, `fit_trace`, `weight_schedule`, `camera_path`, `render_animation`, `run_algorithm`).
- `pacmap_animation_mnist.ipynb` — the original Jupyter notebook the marimo version was converted from. Treat as historical reference only; do not edit as the source of truth.

The marimo notebook and CLI script independently duplicate the same pipeline (fit → replay weight schedule → compute camera path → render). When changing core logic (weight schedule, camera smoothing, rendering), check whether the equivalent change is needed in both files.

## Running

Both scripts declare dependencies inline (PEP 723) and run via `uv`/`uvx` without a separate install step:

```bash
# Interactive notebook
uvx marimo edit --sandbox pacmap_animation_mnist.marimo.py

# Headless CLI render
uv run pacmap_animation_mnist.cli.py --algorithm both --n 5000
```

CLI flags: `--n` (subsample size, or `all` for the full ~70,000 points), `--algorithm {pacmap,localmap,both}`, `--n-neighbors`, `--mn-ratio`, `--fp-ratio`, `--num-iters` (comma-separated PaCMAP phase lengths), `--seed`, `--n-lines`, `--step`, `--fps`, `--point-size`, `--point-alpha`, `--fixed-camera`, `--output-dir`, `--tag-output`, and `--config path/to/config.json` for file-based overrides (CLI flags win over the config file; unset config fields fall back to `DEFAULT_CONFIG` in `pacmap_animation_mnist.cli.py`).

Renders default to `outputs/` (gitignored) via `resolve_output_dir()` in the CLI script: a relative `--output-dir` value with no `./`/`../` prefix is nested under `outputs/` (e.g. `myrun` → `outputs/myrun`); an absolute path or one starting with `./`/`../` is used as-is. `--tag-output` additionally nests the render under a subdirectory named for whichever params differ from `DEFAULT_CONFIG` (via `param_tag()`/`TAG_PARAMS`), e.g. `outputs/nn5_mnr0.8/`, so a directory listing doubles as a way to compare runs by parameter; `algorithm` and `output_dir` are deliberately excluded from the tag so a `both` run and separate `pacmap`/`localmap` runs with the same other params land in the same folder. `unique_path()` asks before overwriting an existing render; declining appends `_1`, `_2`, etc. instead. `main()` resolves and confirms all output filenames up front, before `load_mnist`/`fit_trace` run, so approval happens before any expensive computation rather than after. This output-path logic lives only in the CLI script, not the marimo notebook.

MNIST is loaded via `keras` if available, falling back to `sklearn.datasets.fetch_openml` (pinned to the `liac-arff` parser to avoid a hidden pandas dependency).

There is no test suite, linter, or build step configured in this repo.

## Architecture / key mechanics

1. **Fitting with full history** — `intermediate_snapshots` must start at 0 (otherwise the internal `itr_ind` counter never binds and a `NameError` results) and have length `sum(num_iters) + 1`. Frame 0 is the initialization; frame *k* is the state after *k* Adam steps. `num_iters` is a 3-tuple of PaCMAP's three optimization phases (neighbor / mid-near / further pairs).

2. **Replaying internals offline** — the pair-weight schedule (`w_MN`, `w_NB`, `w_FP`) and gradients are pure functions of the state already captured, computed via `pacmap.pacmap.find_weight` / `pacmap.pacmap.pacmap_grad` after the fact — nothing needs to be intercepted during the fit. `w_MN` collapsing from 1000 → 3 → 0 across the three phases is the core PaCMAP mechanism: pull global structure into place first, then let the local neighbor term refine it.

3. **Camera** — the embedding expands ~30x over the run. A smoothed, monotonic zoom-out (rolling mean of the 99.5th percentile radius, then `np.maximum.accumulate`) keeps early iterations legible without introducing per-frame jitter. `--fixed-camera` (CLI only) instead locks a single radius sized to the trace's largest extent, trading early-frame legibility (starts as a tiny dot) for an honest view of how much the embedding actually moves. See `camera_path()` in the CLI script.

4. **LocalMAP caveat** — LocalMAP resamples its far-pair graph every 10 iterations after iteration 200, and only the final set survives on the fitted object. Positions are captured fine across all frames, but drawing its *evolving* far-pair graph would require a small monkey-patch of `pacmap.pacmap.localmap` to append `pair_FP.copy()` inside the resampling branch — not currently implemented.

5. **Rendering** — `matplotlib.animation.FuncAnimation` draws a subsampled set of pairs (colored by type: neighbour/mid-near/further) with opacity driven by the live weight, plus a scatter of points colored by MNIST digit label and a small weight-schedule strip synced to the main frame. Output is written via the `ffmpeg` writer, so `ffmpeg` must be available on `PATH`. Scatter marker size/opacity are `--point-size`/`--point-alpha` (CLI only); lowering `--point-alpha` lets overlapping points blend into visibly denser regions instead of fully occluding each other.

## Skills

This repo uses marimo-team skills (`add-molab-badge`, `jupyter-to-marimo`), installed under `.agents/skills/` and tracked in `skills-lock.json`.
