"""Loading Polis conversation data as a participants x statements matrix.

Backed by the vendored `valency-anndata` submodule (see `vendor/`), which owns
every Polis-specific concern: source parsing (conversation id, report id,
HuggingFace slug, local CSV export directory), fetching and caching, rebuilding
the vote matrix from raw vote events, and the imputation/clustering pipeline.
This module is only an adapter down to the `(X, y, rs)` tuple the rest of the
CLI consumes.

What was verified about that package's API (v0.4.0, submodule pinned by SHA),
since none of it is guessable from the outside:

- `val.datasets` exposes both a generic `polis.load(source)` and one function
  per curated reference conversation (`japanchoice(topic)`, `vtaiwan(topic)`,
  `aufstehen()`, ...), listed in its `__all__`. Which one a `--dataset
  polis:<source>` value reaches is decided by `_load_source()` below.
- `val.datasets.polis.load(source)` returns an AnnData of shape
  (participants, statements) with votes in {-1, 0, +1} and NaN for unvoted -
  ~70% NaN on a real conversation, so it is not directly fittable.
- Its module-scope imports pull in anndata, scanpy, googletrans, polis-client
  and huggingface-hub. There is no lighter slice to import (even
  `preprocessing/__init__.py` imports scanpy at line 1), which is why the whole
  dependency set lives in the `polis` extra rather than being trimmed.
- `val.tl.recipe_polis(adata)` runs the Small et al. / red-dwarf pipeline:
  zero-mask meta+moderated statements, impute statement-wise means, PCA,
  sparsity-aware scaling, then k-means for 2<=k<=5 picking k by silhouette.
  It leaves the dense matrix in `layers["X_masked_imputed_mean"]` (that
  statement-wise mean imputation is deliberately the package's own, not
  reimplemented here) and the labels in `obs["kmeans_polis"]`.
- It only clusters participants above a 7-vote threshold; the rest come back
  NaN. They are kept and labelled -1 rather than dropped, so the animation
  still shows every participant.
- It raises if `var["is_meta"]` is entirely missing, which happens for minimal
  exports whose comments.csv has no `is-meta` column. Its own error message
  says to force the values False, which is what `_ensure_is_meta` does.
"""

import warnings

import numpy as np

from .data import subsample_indices

# Keys valency-anndata writes, named here so a package change surfaces as one
# failure with a clear cause rather than a KeyError deep in this module.
DENSE_LAYER = "X_masked_imputed_mean"
RAW_LAYER = "raw_sparse"
GROUP_KEY = "kmeans_polis"

# Label for participants Polis's pipeline left unclustered (below its vote
# threshold). Negative so it can't collide with a real group id.
UNCLUSTERED = -1

_MISSING_MSG = (
    "The polis dataset needs the vendored valency-anndata submodule and its\n"
    "dependencies, which are an optional extra:\n\n"
    "    git submodule update --init\n"
    "    uv run --extra polis python pacmap_animation_mnist.cli.py --dataset polis:<source> ...\n\n"
    "(extras don't apply to PEP 723 script runs, hence the `python`)."
)


def _raise_missing():
    raise SystemExit(_MISSING_MSG)


def _import_valency():
    try:
        import valency_anndata as val
    except ImportError:
        _raise_missing()
    return val


# Names in `val.datasets.__all__` that aren't conversations, so a source
# starting with one is a plain source rather than a reference dataset.
_NOT_REFERENCE_DATASETS = {"load", "translate_statements"}


def _reference_loaders(val):
    """The named reference conversations the package ships, `{name: callable}`.

    Read off `val.datasets` rather than duplicated here, so a dataset added
    upstream is usable through `--dataset polis:<name>` with no change on this
    side (and a submodule bump is the only thing that gates it).
    """
    return {
        name: getattr(val.datasets, name)
        for name in getattr(val.datasets, "__all__", [])
        if name not in _NOT_REFERENCE_DATASETS and callable(getattr(val.datasets, name, None))
    }


def _load_source(val, source):
    """Dispatch one `--dataset polis:<source>` value to the right entry point.

    Two forms, distinguished by whether the leading segment names one of the
    package's reference conversations:

    - `japanchoice:2025_foreign_affairs_security` -> `val.datasets.japanchoice(
      "2025_foreign_affairs_security")`, and `aufstehen` -> `val.datasets
      .aufstehen()` for the ones that take no variant;
    - anything else -> `val.datasets.polis.load(source)` verbatim: a
      conversation/report id, a pol.is URL, an `hf:user/dataset` slug (whose
      colon must survive the split), or a local export directory.
    """
    name, _sep, variant = source.partition(":")
    loader = _reference_loaders(val).get(name)
    if loader is None:
        return val.datasets.polis.load(source)
    try:
        return loader(variant) if variant else loader()
    except TypeError as e:
        # The variant-taking loaders declare it positionally, so omitting it
        # is a TypeError rather than something the package explains.
        raise ValueError(
            f"the {name!r} dataset needs a variant, e.g. "
            f"--dataset polis:{name}:<variant> (see {name}'s docstring for the list)") from e


def _ensure_is_meta(adata):
    """`recipe_polis` refuses to build its zero-mask when no statement carries
    is-meta data. A minimal CSV export simply has no such column, so treat
    "unknown" as "not meta" - the override the package's own error suggests."""
    is_meta = adata.var["is_meta"]
    if is_meta.isna().all():
        warnings.warn(
            "Polis export has no is-meta statement data; treating all statements as "
            "non-meta so the zero-mask can be built.", stacklevel=2)
        adata.var["is_meta"] = False


def _group_labels(adata):
    """Cluster labels as ints, with unclustered participants at UNCLUSTERED."""
    raw = adata.obs[GROUP_KEY].astype("float64").to_numpy()
    return np.where(np.isnan(raw), UNCLUSTERED, raw).astype(np.int64)


def _vote_counts(adata):
    """How many statements each participant actually voted on."""
    raw = np.asarray(adata.layers[RAW_LAYER], dtype=float)
    return (~np.isnan(raw)).sum(axis=1).astype(np.int64)


def load_polis(source, n=None, seed=0, color="polis:group-id"):
    """Load a Polis conversation, optionally subsampled to `n` participants (or
    a proportion of them if `n` is a float in (0, 1]). `n=None` keeps all.

    `source` is either one of the package's named reference conversations
    (`japanchoice:2025_foreign_affairs_security`, `aufstehen`) or anything its
    generic loader accepts: a conversation id, a report id, a `hf:user/dataset`
    slug, a pol.is URL, or a local CSV export directory. See `_load_source()`.
    Returns `(X, y, rs)` like `load_mnist`, with `y` holding whichever color
    scheme was asked for.
    """
    val = _import_valency()

    adata = _load_source(val, source)
    _ensure_is_meta(adata)
    val.tl.recipe_polis(adata)

    Xfull = np.ascontiguousarray(adata.layers[DENSE_LAYER], dtype=np.float32)
    yfull = _vote_counts(adata) if color == "polis:n-votes" else _group_labels(adata)

    rs = np.random.RandomState(seed)
    sel = subsample_indices(len(Xfull), n, rs)

    X, y = np.ascontiguousarray(Xfull[sel]), yfull[sel]
    print(f"{X.shape} {X.dtype} participants x statements, color={color} "
          f"range=({y.min()}, {y.max()})")
    return X, y, rs
