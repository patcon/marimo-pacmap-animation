"""Which data gets embedded, and what colors the points.

The single seam between "load some data" and the rest of the pipeline. Built as
a registry of lazy factories, mirroring `render.py`'s `RENDERERS`: a dataset
whose loader needs heavy or optional dependencies (polis, via the vendored
valency-anndata submodule) must not cost anything on an `mnist` run, so the
factory imports its module only when that dataset is actually selected.

A loader has the uniform signature `loader(source, n, seed, color)` and returns
`(X, y, rs)`, matching what `data.load_mnist` already returned. `load_dataset`
wraps that with a `meta` dict carrying the two things the rest of the CLI needs
to know about the choice: the `slug` (an output *directory* level, see
`orchestrate.main`) and the `cmap` the renderers color points with.

Color is data, not a rendering branch: each scheme declares whether it is
categorical or continuous, and that collapses to a single colormap name chosen
here. Renderers stay preset-agnostic - they take `y` and a `cmap` string, and
both matplotlib and fastplotlib normalize a continuous map over the data range
themselves.
"""


# Discrete colors for a small set of labels (MNIST digits, Polis groups) vs a
# continuous ramp for a magnitude (vote counts). tab10 is what the renderers
# hardcoded before datasets existed, so categorical output is unchanged.
CATEGORICAL_CMAP = "tab10"
CONTINUOUS_CMAP = "viridis"

# Color schemes, keyed by the value `--color` takes. Namespaced by dataset
# (`polis:n-votes`) where a scheme only means something for one source; plain
# (`label`) where it is the generic "whatever this dataset calls its class".
COLOR_SCHEMES = {
    "label": {"dataset": "mnist", "kind": "categorical"},
    "polis:group-id": {"dataset": "polis", "kind": "categorical"},
    "polis:n-votes": {"dataset": "polis", "kind": "continuous"},
}

# Which scheme a dataset uses when `--color` isn't given.
DEFAULT_COLORS = {
    "mnist": "label",
    "polis": "polis:group-id",
}

# Datasets needing a source to say *which* data (a Polis conversation id, report
# id, HuggingFace slug or local export dir); mnist is just itself.
SOURCE_REQUIRED = {"polis"}


def _loader_mnist():
    from .data import load_mnist

    def load(source=None, n=None, seed=0, color="label"):
        # source/color are fixed for mnist (there is one dataset and one
        # scheme); accepted anyway to keep every loader's signature uniform.
        return load_mnist(n=n, seed=seed)

    return load


def _loader_polis():
    from . import datasets_polis

    return datasets_polis.load_polis


# Dataset registry: name -> zero-arg factory returning the loader. Adding a
# dataset = write its loader module, add an entry here, and give it an entry in
# DEFAULT_COLORS plus one or more COLOR_SCHEMES; `--dataset`'s help and the
# error messages below are all derived from these tables.
DATASETS = {
    "mnist": _loader_mnist,
    "polis": _loader_polis,
}


def parse_dataset_spec(spec):
    """Split a `--dataset` value into `(name, source)`.

    `"mnist"` -> `("mnist", None)`; `"polis:35bmpjr8um"` -> `("polis",
    "35bmpjr8um")`. Only the leading name is split off, since a source can
    itself contain colons (`polis:hf:user/dataset`, or a full URL).
    """
    name, sep, source = spec.partition(":")
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}: expected one of {sorted(DATASETS)}")
    source = source if sep else None
    if name in SOURCE_REQUIRED and not source:
        raise ValueError(
            f"dataset {name!r} requires a source, e.g. --dataset {name}:<conversation-id> "
            f"or --dataset {name}:./path/to/export")
    if name not in SOURCE_REQUIRED and source:
        raise ValueError(f"dataset {name!r} takes no source, but got {source!r}")
    return name, source


def get_loader(name):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}: expected one of {sorted(DATASETS)}")
    return DATASETS[name]()


def schemes_for(dataset):
    """The `--color` values valid for `dataset`, in declaration order."""
    return [name for name, spec in COLOR_SCHEMES.items() if spec["dataset"] == dataset]


def resolve_color(dataset, color):
    """The color scheme this run uses: the dataset's default when `color` is
    None, otherwise `color` validated against that dataset."""
    if color is None:
        return DEFAULT_COLORS[dataset]
    if color not in COLOR_SCHEMES:
        raise ValueError(
            f"unknown color scheme {color!r}: expected one of {sorted(COLOR_SCHEMES)}")
    owner = COLOR_SCHEMES[color]["dataset"]
    if owner != dataset:
        raise ValueError(
            f"color scheme {color!r} belongs to dataset {owner!r}, not {dataset!r}; "
            f"{dataset!r} supports {schemes_for(dataset)}")
    return color


def is_continuous(color):
    """Whether `color` is a magnitude (continuous ramp) rather than a class.

    Callers use this for more than the colormap: `--focus-label` compares `y`
    for equality, which is meaningless against a continuous scheme.
    """
    return COLOR_SCHEMES[color]["kind"] == "continuous"


def cmap_for(color):
    return CONTINUOUS_CMAP if is_continuous(color) else CATEGORICAL_CMAP


def _sanitize(text):
    """`text` reduced to characters safe in a directory name."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in text.lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-.")


def dataset_slug(name, source):
    """Filesystem-safe identifier for this dataset choice, used as an output
    *directory* level (`outputs/mnist/`, `outputs/polis-35bmpjr8um/`).

    For a source with path segments - a local export dir or a HuggingFace slug
    - only the last one is kept: the whole path can't go in a directory name,
    and the leaf is the part that names the conversation.
    """
    if not source:
        return name
    leaf = [part for part in source.replace("\\", "/").split("/") if part not in ("", ".", "..")]
    return f"{name}-{_sanitize(leaf[-1] if leaf else source)}"


def color_marker(dataset, color):
    """Filename marker for a non-default color scheme (`"_colorn-votes"`), or
    `""` for the dataset's own default.

    Color is a *rendering* choice, so it belongs in the filename rather than
    the directory (unlike the dataset itself) - two colorings of one fit land
    side by side in the same directory. The dataset namespace is dropped,
    since the directory already carries it.
    """
    if color == DEFAULT_COLORS[dataset]:
        return ""
    return "_color" + _sanitize(color.partition(":")[2] or color)


def dataset_meta(spec, color=None):
    """Everything the CLI needs to know about a dataset choice *without*
    loading it: `dataset`, `source`, `slug`, `color`, `cmap`.

    Pure and cheap, so output paths and flag validation are resolved before
    the expensive load, not after it.
    """
    name, source = parse_dataset_spec(spec)
    color = resolve_color(name, color)
    return {
        "dataset": name,
        "source": source,
        "slug": dataset_slug(name, source),
        "color": color,
        "cmap": cmap_for(color),
    }


def load_dataset(spec, n=None, seed=0, color=None):
    """Load the dataset named by `spec`, returning `(X, y, rs, meta)`.

    `X` is the data to embed, `y` the per-point color values, `rs` a seeded
    RandomState the render reuses for pair subsampling, and `meta` is
    `dataset_meta(spec, color)`.
    """
    meta = dataset_meta(spec, color)
    X, y, rs = get_loader(meta["dataset"])(meta["source"], n=n, seed=seed, color=meta["color"])
    return X, y, rs, meta
