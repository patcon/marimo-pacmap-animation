# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small, self-contained project that animates PaCMAP and LocalMAP (dimensionality reduction algorithms) optimizing a 2D embedding of MNIST, frame by frame, and renders the result as MP4 video. It captures the embedding at every optimization iteration via the `pacmap` library's public `intermediate_snapshots` kwarg — no forking or monkey-patching of the library.

## Files

- `pacmap_animation_mnist.marimo.py` — the marimo notebook (interactive, cell-based). This is the primary/reference implementation.
- `pacmap_animation_mnist.cli.py` — headless CLI script that mirrors the notebook's logic. Thin entry point only (PEP 723 header + `main()` call): the actual implementation lives in the `pacmap_cli/` package next to it, re-exported here so `import pacmap_animation_mnist.cli as cli; cli.fit_trace(...)`-style access still works unchanged.
- `pacmap_cli/` — the CLI's implementation, split by pipeline stage: `data.py` (`load_mnist`, `resolve_proportion`), `fit.py` (`fit_trace`), `camera.py` (`weight_schedule`, `camera_path`), `pairs.py` (pair subsampling, `compute_edge_alphas`, `pacmap_force`), `overlay.py` (`compute_overlay_text`), `render.py` (`_build_renderer`, `render_animation`, `render_frame`), `paths.py` (`resolve_output_dir`, `param_tag`, `unique_path`), `config.py` (`DEFAULT_CONFIG`, `parse_args`, `build_config`), `orchestrate.py` (`run_algorithm`, `main`).
- `pacmap_animation_mnist.ipynb` — the original Jupyter notebook the marimo version was converted from. Treat as historical reference only; do not edit as the source of truth.
- `tests/` — pytest suite covering `pacmap_cli`'s pure functions plus an end-to-end `main()` smoke test. Run with `uv run pytest`. `tests/_loader.py` loads the entry shim via `importlib` (its filename isn't a valid dotted module path) and adds the repo root to `sys.path` so the shim's own `from pacmap_cli import ...` resolves. `pyproject.toml` pins dependency versions to a known-working set rather than letting a fresh resolution pick incompatible ones (see git history for why).

The marimo notebook and CLI package independently duplicate the same pipeline (fit → replay weight schedule → compute camera path → render). When changing core logic (weight schedule, camera smoothing, rendering), check whether the equivalent change is needed in both.

## Running

Both scripts declare dependencies inline (PEP 723) and run via `uv`/`uvx` without a separate install step:

```bash
# Interactive notebook
uvx marimo edit --sandbox pacmap_animation_mnist.marimo.py

# Headless CLI render
uv run pacmap_animation_mnist.cli.py --algorithm both --n 5000
```

CLI flags: `--n` (subsample size, a fraction in (0, 1] for a proportion of the full ~70,000 points — 1 or 1.0 means all of them, same as passing `all` — or an absolute count), `--algorithm {pacmap,localmap,both}`, `--n-neighbors`, `--mn-ratio`, `--fp-ratio`, `--num-iters` (comma-separated PaCMAP phase lengths), `--seed`, `--n-lines` (also accepts a fraction in (0, 1], as a proportion of each pair type's available pool; 1 or 1.0 means the full pool), `--step`, `--fps`, `--point-size`, `--point-alpha`, `--edge-style-preset {v1,v2}`, `--edge-gamma`, `--line-alpha`, `--fixed-camera`, `--focus-label` (camera tracks one MNIST digit's cluster instead of the whole embedding; pass a digit, or pass with no value to be prompted interactively after MNIST loads), `--iter` (render only one iteration, as a png, or a range of iterations, as an mp4, indexed directly against pacmap's own iteration count rather than the `--step`-derived animation frame — see `render_frame()`/`render_animation()`'s `start`/`end` args; comma-separate multiple values/ranges, e.g. `--iter 50,150,250-400`, to render several outputs from a single fit/trace in one invocation), `--output-dir`, `--tag-output`, and `--config path/to/config.json` for file-based overrides (CLI flags win over the config file; unset config fields fall back to `DEFAULT_CONFIG` in `pacmap_cli/config.py`).

Renders default to `outputs/` (gitignored) via `resolve_output_dir()` in the CLI script: a relative `--output-dir` value with no `./`/`../` prefix is nested under `outputs/` (e.g. `myrun` → `outputs/myrun`); an absolute path or one starting with `./`/`../` is used as-is. `--tag-output` additionally nests the render under a subdirectory named for whichever params differ from `DEFAULT_CONFIG` (via `param_tag()`/`TAG_PARAMS`), e.g. `outputs/nn5_mnr0.8/`, so a directory listing doubles as a way to compare runs by parameter; `algorithm` and `output_dir` are deliberately excluded from the tag so a `both` run and separate `pacmap`/`localmap` runs with the same other params land in the same folder. `--iter` is deliberately excluded from `--tag-output`'s slug too, since `main()` already bakes it into the output filename directly — `{algorithm}_mnist.mp4` by default, `{algorithm}_mnist_iter150.png` for a single iteration, `{algorithm}_mnist_iter50-300.mp4` for a range — so a tag would be redundant. `unique_path()` asks before overwriting an existing render; declining appends `_1`, `_2`, etc. instead. `main()` resolves and confirms all output filenames up front, before `load_mnist`/`fit_trace` run, so approval happens before any expensive computation rather than after. This output-path logic lives only in the CLI script, not the marimo notebook.

MNIST is loaded via `keras` if available, falling back to `sklearn.datasets.fetch_openml` (pinned to the `liac-arff` parser to avoid a hidden pandas dependency).

There is no test suite, linter, or build step configured in this repo.

## Architecture / key mechanics

1. **Fitting with full history** — `intermediate_snapshots` must start at 0 (otherwise the internal `itr_ind` counter never binds and a `NameError` results) and have length `sum(num_iters) + 1`. Frame 0 is the initialization; frame *k* is the state after *k* Adam steps. `num_iters` is a 3-tuple of PaCMAP's three optimization phases (neighbor / mid-near / further pairs).

2. **Replaying internals offline** — the pair-weight schedule (`w_MN`, `w_NB`, `w_FP`) and gradients are pure functions of the state already captured, computed via `pacmap.pacmap.find_weight` / `pacmap.pacmap.pacmap_grad` after the fact — nothing needs to be intercepted during the fit. `w_MN` collapsing from 1000 → 3 → 0 across the three phases is the core PaCMAP mechanism: pull global structure into place first, then let the local neighbor term refine it.

3. **Camera** — the embedding expands ~30x over the run. A smoothed, monotonic zoom-out (rolling mean of the 99.5th percentile radius, then `np.maximum.accumulate`) keeps early iterations legible without introducing per-frame jitter. `--fixed-camera` (CLI only) instead locks a single radius sized to the trace's largest extent, trading early-frame legibility (starts as a tiny dot) for an honest view of how much the embedding actually moves. `--focus-label` (CLI only) instead scopes the camera to a single MNIST digit's cluster: both the radius and a smoothed per-frame centroid are computed from just that label's points, so the camera follows the cluster instead of framing the whole embedding from the origin. See `camera_path()` in the CLI script.

4. **LocalMAP caveat** — LocalMAP resamples its far-pair graph every 10 iterations after iteration 200, and only the final set survives on the fitted object. Positions are captured fine across all frames, but drawing its *evolving* far-pair graph would require a small monkey-patch of `pacmap.pacmap.localmap` to append `pair_FP.copy()` inside the resampling branch — not currently implemented.

5. **Rendering** — `matplotlib.animation.FuncAnimation` draws a subsampled set of pairs (colored by type: neighbour/mid-near/further) with opacity driven by the live weight, plus a scatter of points colored by MNIST digit label and a small weight-schedule strip synced to the main frame. Output is written via the `ffmpeg` writer, so `ffmpeg` must be available on `PATH`. Scatter marker size/opacity are `--point-size`/`--point-alpha` (CLI only); lowering `--point-alpha` lets overlapping points blend into visibly denser regions instead of fully occluding each other.

   Figure/artist setup and the per-frame `update(f)` closure are factored into a shared `_build_renderer()` helper (CLI only), used by both `render_animation()` (many frames → mp4 via `FuncAnimation`, the default) and `render_frame()` (`update()` called once, then `fig.savefig()` → png). Which one runs is controlled by `--iter` (CLI only): unset renders the full `0`-`sum(num_iters)` range as before; a single value (e.g. `--iter 150`) calls `render_frame()`; a range (e.g. `--iter 50-300`) calls `render_animation()` with `start`/`end` bounding `frames` instead of the default `0`-`total`, still stepped by `--step`. `--iter` indexes directly into `trace` by pacmap iteration number — safe because `intermediate_snapshots` already captures every iteration 1:1, so no rounding is needed. Out-of-range values raise a `ValueError` in `run_algorithm()` before rendering starts. A comma-separated `--iter` (e.g. `--iter 50,150,250-400`) is parsed by `parse_iter_list()` in `pacmap_cli/config.py` into a list of items (`cfg["iter"]`); `fit_trace()` still runs once per algorithm in `run_algorithm()` (`pacmap_cli/orchestrate.py`), which then loops over the items calling `render_frame()`/`render_animation()` once per item against the shared trace, so multiple pngs/mp4s come from one fit rather than refitting per item.

   Edge line opacity per pair type is computed by `compute_edge_alphas()` (CLI only), selected via `--edge-style-preset`. `"v1"` (default) maps each type's alpha from its own raw weight independently — since `w_MN` ranges up to 1000 while `w_NB` (2-3) and `w_FP` (fixed at 1) barely move, mid-near visually dominates phase 1 and further pairs stay near-invisible (alpha capped at 0.05). `"v2"` normalizes the three weights against their per-frame max and raises the ratio to `--edge-gamma` (default 0.2, i.e. strong compression) before scaling by a per-type ceiling (`EDGE_ALPHA_MAX_V2`), so neighbour/further pairs stay visibly present even while mid-near dominates the fit. `--line-alpha` (default 1.0, CLI only) multiplies every line's computed alpha afterward — turn it down when `--n-lines` draws a lot of pairs so overlapping lines don't wash out into a solid wash of color.

   `--n` and `--n-lines` (CLI only) both accept either an absolute count or a fraction in (0, 1], resolved via `resolve_proportion()` as a proportion of the relevant total at the point of use: the full dataset size for `--n` (in `load_mnist()`), or a pair type's available pool for `--n-lines` (in `subsample_pairs()`, called separately per pair type, so e.g. `--n-lines 0.1` draws 10% of neighbour pairs, 10% of mid-near pairs, and 10% of further pairs — each against its own pool size, not a shared one). `1`/`1.0` is inclusive of the upper bound, so it resolves to the full total rather than a literal count of `1`.

   The top-left overlay text is computed by `compute_overlay_text()`, selected via `cfg["overlay_style_preset"]` — a `DEFAULT_CONFIG` field settable only through a `--config` JSON file, not a CLI flag. `"v1"` is the original single line with `w_MN` shown as a float and no `w_FP`. `"v2"` (default) rounds all three weights to integers and stacks them as separate label-aligned lines (`w_MN`/`w_NB`/`w_FP`), so their digits land in the same column frame to frame, making it easier to see the three move together.

## Skills

This repo uses marimo-team skills (`add-molab-badge`, `jupyter-to-marimo`), installed under `.agents/skills/` and tracked in `skills-lock.json`.
