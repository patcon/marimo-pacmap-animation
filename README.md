# PaCMAP / LocalMAP MNIST Animation

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/patcon/pacmap-animation/blob/main/pacmap_animation_mnist.marimo.py)

A [marimo](https://marimo.io) notebook that animates [PaCMAP](https://github.com/YingfanWang/PaCMAP) and LocalMAP optimizing a 2D embedding of MNIST, frame by frame.

It captures the embedding at **every** iteration via the public `intermediate_snapshots` kwarg (no fork or monkey-patch needed) and renders a video showing:

- the three PaCMAP optimization phases (neighbor / mid-near / further pairs)
- the pair-weight schedule that drives them, with `w_MN` collapsing from 1000 → 3 → 0 as global structure gets pulled into place and then refined locally
- a smoothed, monotonic camera zoom to keep the ~30x expansion of the embedding legible
- a side-by-side comparison of PaCMAP vs. LocalMAP from the same initialization

## Running

The notebook declares its dependencies inline (PEP 723), so it can be run directly with [uv](https://github.com/astral-sh/uv):

```bash
uvx marimo edit --sandbox pacmap_animation_mnist.marimo.py
```

MNIST is loaded via `keras` if available, falling back to `sklearn.datasets.fetch_openml`.

There's also a plain CLI script for rendering a video without the notebook UI:

```bash
uv run pacmap_animation_mnist.cli.py --algorithm both --n 5000
```

Renders land in `outputs/<dataset>/` by default (gitignored) — e.g.
`outputs/mnist/pacmap.mp4` — and never overwrite an existing file: a repeat run
gets `pacmap_1.mp4`, `pacmap_2.mp4`, etc. Pass `--output-dir myrun` to nest under
`outputs/myrun/<dataset>/` instead, or `--output-dir ./out` (or an absolute path)
to use that directory as-is.

Add `--tag-output` to nest the render under a further subdirectory named after
whichever params you changed from the defaults (e.g.
`outputs/mnist/nn5_mnr0.8/`), so a directory listing doubles as a quick way to
compare runs by parameter. `--algorithm` isn't part of the tag, so a `both` run
and separate `pacmap`/`localmap` runs with the same other params land in the same
tagged folder.

Flags: `--dataset`, `--color`, `--n`, `--algorithm {pacmap,localmap,both}`,
`--n-neighbors`, `--mn-ratio`, `--fp-ratio`, `--num-iters`, `--seed`, `--n-lines`,
`--step`, `--fps`, `--point-size`, `--point-alpha`, `--output-dir`,
`--tag-output`, and `--config path/to/config.json` for file-based overrides (CLI
flags win over the config file). Its functions are factored so they can be
imported directly into a notebook rather than only invoked from the CLI. Run
`uv run pacmap_animation_mnist.cli.py --help` for the full list — the CLI has a
good deal more (3D embeddings, GPU rendering, single-frame stills, pair-weight
schedules) than this summary.

Lower `--point-alpha` (e.g. `0.4`) and/or `--point-size` if you want the scatter
to better reflect point density — overlapping points blend into visibly denser
regions instead of fully occluding each other.

### Other datasets

`--dataset polis:<source>` embeds a [Polis](https://pol.is) conversation instead
of MNIST: points are participants, laid out from their vote matrix, so the
animation shows opinion groups separating. `<source>` is a conversation or report
id, a `hf:user/dataset` slug, a pol.is URL, or a local CSV export directory.
Color them by cluster (`--color polis:group-id`, the default) or by how much
each participant voted (`--color polis:n-votes`).

This path is backed by the vendored
[valency-anndata](https://github.com/patcon/valency-anndata) submodule, which
owns all the Polis-specific work — including the vote-matrix imputation, which is
its behaviour rather than anything decided here. Its dependencies conflict with
this repo's pins, so they live behind an optional extra, and extras don't apply
to PEP 723 script runs:

```bash
git submodule update --init
uv run --extra polis python pacmap_animation_mnist.cli.py \
    --dataset polis:35bmpjr8um --algorithm pacmap --fixed-camera
```

Only the CLI supports this; the marimo notebook still loads MNIST only.

## Files

- `pacmap_animation_mnist.marimo.py` — the marimo notebook
- `pacmap_animation_mnist.cli.py` — CLI script / importable functions for headless rendering
- `pacmap_cli/` — the CLI's implementation, split by pipeline stage
- `vendor/valency-anndata` — git submodule backing the `polis` dataset
- `pacmap_animation_mnist.ipynb` — the original Jupyter notebook this was converted from
