"""The dataset registry and the colour-scheme table.

Pure tests: no dataset is actually loaded here (that needs MNIST on disk or a
Polis export), so `load_dataset`'s dispatch is exercised against a fake loader
registered into the registry. The real loaders are covered by the end-to-end
smoke tests.
"""

import numpy as np
import pytest

from _loader import cli

datasets = cli.datasets


# --- parse_dataset_spec ---

def test_parse_dataset_spec_bare_name_has_no_source():
    assert datasets.parse_dataset_spec("mnist") == ("mnist", None)


def test_parse_dataset_spec_splits_name_from_source():
    assert datasets.parse_dataset_spec("polis:35bmpjr8um") == ("polis", "35bmpjr8um")


def test_parse_dataset_spec_splits_on_first_colon_only():
    # A source can itself contain colons (valency-anndata accepts `hf:user/ds`
    # and full URLs), so only the leading dataset name is split off.
    assert datasets.parse_dataset_spec("polis:hf:patcon/polis-aufstehen-2018") == (
        "polis", "hf:patcon/polis-aufstehen-2018")


def test_parse_dataset_spec_keeps_relative_path_source_intact():
    assert datasets.parse_dataset_spec("polis:./exports/convo") == ("polis", "./exports/convo")


def test_parse_dataset_spec_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.parse_dataset_spec("imagenet")


def test_parse_dataset_spec_rejects_polis_without_a_source():
    # Unlike mnist, polis has nothing to load without being told where from.
    with pytest.raises(ValueError, match="requires a source"):
        datasets.parse_dataset_spec("polis")


def test_parse_dataset_spec_rejects_source_on_mnist():
    with pytest.raises(ValueError, match="takes no source"):
        datasets.parse_dataset_spec("mnist:somewhere")


# --- the registry ---

def test_datasets_registry_has_mnist_and_polis():
    assert set(datasets.DATASETS) == {"mnist", "polis"}


def test_get_loader_returns_a_callable():
    assert callable(datasets.get_loader("mnist"))


def test_get_loader_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.get_loader("imagenet")


def test_mnist_loader_does_not_import_the_polis_module():
    # The factory indirection exists so an mnist run never touches the
    # valency-anndata submodule or its heavy optional deps.
    import sys
    sys.modules.pop("pacmap_cli.datasets_polis", None)
    datasets.get_loader("mnist")
    assert "pacmap_cli.datasets_polis" not in sys.modules


# --- colour schemes ---

def test_resolve_color_defaults_to_label_for_mnist():
    assert datasets.resolve_color("mnist", None) == "label"


def test_resolve_color_defaults_to_group_id_for_polis():
    assert datasets.resolve_color("polis", None) == "polis:group-id"


def test_resolve_color_passes_through_a_supported_scheme():
    assert datasets.resolve_color("polis", "polis:n-votes") == "polis:n-votes"


def test_resolve_color_rejects_a_scheme_from_another_dataset():
    with pytest.raises(ValueError) as e:
        datasets.resolve_color("mnist", "polis:n-votes")
    # The message should say what this dataset *does* support.
    assert "label" in str(e.value)


def test_resolve_color_rejects_an_unknown_scheme():
    with pytest.raises(ValueError, match="unknown colour scheme|unknown color scheme"):
        datasets.resolve_color("mnist", "chartreuse")


def test_categorical_schemes_use_a_discrete_colormap():
    assert datasets.cmap_for("label") == datasets.CATEGORICAL_CMAP
    assert datasets.cmap_for("polis:group-id") == datasets.CATEGORICAL_CMAP


def test_continuous_scheme_uses_a_continuous_colormap():
    assert datasets.cmap_for("polis:n-votes") == datasets.CONTINUOUS_CMAP


def test_categorical_and_continuous_colormaps_differ():
    assert datasets.CATEGORICAL_CMAP != datasets.CONTINUOUS_CMAP


def test_is_continuous_distinguishes_the_two_kinds():
    assert datasets.is_continuous("polis:n-votes")
    assert not datasets.is_continuous("polis:group-id")
    assert not datasets.is_continuous("label")


# --- slugs (these become an output directory level) ---

def test_slug_for_mnist_is_just_the_name():
    assert datasets.dataset_slug("mnist", None) == "mnist"


def test_slug_for_polis_includes_the_conversation_id():
    assert datasets.dataset_slug("polis", "35bmpjr8um") == "polis-35bmpjr8um"


def test_slug_for_a_path_source_uses_its_basename():
    # The whole path can't go in a directory name, and the leaf is the part
    # that identifies the conversation.
    assert datasets.dataset_slug("polis", "./exports/convo-abc/") == "polis-convo-abc"


def test_slug_strips_characters_that_are_unsafe_in_a_filename():
    slug = datasets.dataset_slug("polis", "hf:patcon/polis-aufstehen-2018")
    assert "/" not in slug and ":" not in slug
    assert slug.startswith("polis-")


# --- load_dataset dispatch ---

@pytest.fixture
def fake_polis(monkeypatch):
    """Register a loader that returns a tiny vote-matrix-shaped result."""
    calls = {}

    def loader(source, n=None, seed=0, color="polis:group-id"):
        calls["source"] = source
        calls["n"] = n
        calls["seed"] = seed
        calls["color"] = color
        rs = np.random.RandomState(seed)
        X = rs.uniform(-1, 1, size=(12, 5)).astype(np.float32)
        y = rs.randint(0, 3, size=12)
        return X, y, rs

    monkeypatch.setitem(datasets.DATASETS, "polis", lambda: loader)
    return calls


def test_load_dataset_passes_the_source_to_the_loader(fake_polis):
    datasets.load_dataset("polis:35bmpjr8um", n=None, seed=0)
    assert fake_polis["source"] == "35bmpjr8um"


def test_load_dataset_forwards_n_and_seed(fake_polis):
    datasets.load_dataset("polis:abc", n=10, seed=7)
    assert fake_polis["n"] == 10 and fake_polis["seed"] == 7


def test_load_dataset_forwards_the_resolved_color_not_none(fake_polis):
    datasets.load_dataset("polis:abc", color=None)
    assert fake_polis["color"] == "polis:group-id"


def test_load_dataset_returns_x_y_rs_and_meta(fake_polis):
    X, y, rs, meta = datasets.load_dataset("polis:abc")
    assert X.shape == (12, 5)
    assert len(y) == 12
    assert isinstance(rs, np.random.RandomState)
    assert isinstance(meta, dict)


def test_load_dataset_meta_carries_slug_and_cmap(fake_polis):
    _X, _y, _rs, meta = datasets.load_dataset("polis:35bmpjr8um", color="polis:n-votes")
    assert meta["slug"] == "polis-35bmpjr8um"
    assert meta["cmap"] == datasets.CONTINUOUS_CMAP
    assert meta["color"] == "polis:n-votes"
    assert meta["dataset"] == "polis"


def test_load_dataset_rejects_a_color_the_dataset_does_not_support(fake_polis):
    with pytest.raises(ValueError):
        datasets.load_dataset("polis:abc", color="label")


# --- dataset_meta (resolved without loading anything) ---

def test_dataset_meta_resolves_without_calling_a_loader(monkeypatch):
    def boom():
        raise AssertionError("dataset_meta must not load data")

    monkeypatch.setitem(datasets.DATASETS, "polis", boom)
    meta = datasets.dataset_meta("polis:abc")
    assert meta == {"dataset": "polis", "source": "abc", "slug": "polis-abc",
                    "color": "polis:group-id", "cmap": datasets.CATEGORICAL_CMAP}


def test_dataset_meta_rejects_a_color_the_dataset_does_not_support():
    with pytest.raises(ValueError):
        datasets.dataset_meta("mnist", color="polis:n-votes")


# --- colour as a filename marker ---

def test_color_marker_is_empty_for_a_datasets_default_scheme():
    assert datasets.color_marker("mnist", "label") == ""
    assert datasets.color_marker("polis", "polis:group-id") == ""


def test_color_marker_names_a_non_default_scheme_without_its_namespace():
    assert datasets.color_marker("polis", "polis:n-votes") == "_colorn-votes"


def test_color_marker_is_filename_safe():
    marker = datasets.color_marker("polis", "polis:n-votes")
    assert "/" not in marker and ":" not in marker
