# PaCMAP / LocalMAP MNIST Animation

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/patcon/marimo-pacmap-animation/blob/main/pacmap_animation_mnist.marimo.py)

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

Renders land in `outputs/` by default (gitignored) and never overwrite an existing
file — a repeat run gets `pacmap_mnist_1.mp4`, `pacmap_mnist_2.mp4`, etc. Pass
`--output-dir myrun` to nest under `outputs/myrun` instead, or `--output-dir ./out`
(or an absolute path) to use that directory as-is.

Add `--tag-output` to nest the render under a subdirectory named after whichever
params you changed from the defaults (e.g. `outputs/nn5_mnr0.8/`), so a directory
listing doubles as a quick way to compare runs by parameter. `--algorithm` isn't
part of the tag, so a `both` run and separate `pacmap`/`localmap` runs with the
same other params land in the same tagged folder.

Flags: `--n`, `--algorithm {pacmap,localmap,both}`, `--n-neighbors`, `--mn-ratio`,
`--fp-ratio`, `--num-iters`, `--seed`, `--n-lines`, `--step`, `--fps`, `--output-dir`,
`--tag-output`, and `--config path/to/config.json` for file-based overrides (CLI
flags win over the config file). Its functions are factored so they can be
imported directly into a notebook rather than only invoked from the CLI.

## Files

- `pacmap_animation_mnist.marimo.py` — the marimo notebook
- `pacmap_animation_mnist.cli.py` — CLI script / importable functions for headless rendering
- `pacmap_animation_mnist.ipynb` — the original Jupyter notebook this was converted from
