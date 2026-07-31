"""Which valency-anndata entry point a `--dataset polis:<source>` value reaches.

Pure dispatch tests against a stub standing in for the package, so they run
without the submodule, the `polis` extra, or a network - unlike
test_datasets_polis.py, which exercises the real loader end to end.
"""

import types

import pytest

from _loader import cli

polis = cli.datasets_polis


def _stub_val(reference_names=("japanchoice", "aufstehen"), calls=None):
    """A stand-in for the valency_anndata package: the same `datasets`
    surface the real one exposes (an `__all__`, `polis.load`, and one
    callable per named reference conversation), recording what got called."""
    calls = {} if calls is None else calls

    def record(name):
        def fn(*args, **kwargs):
            calls[name] = (args, kwargs)
            return f"adata:{name}"
        return fn

    datasets = types.SimpleNamespace(
        __all__=["load", "translate_statements", *reference_names],
        load=record("load"),
        translate_statements=record("translate_statements"),
        polis=types.SimpleNamespace(load=record("polis.load")),
        **{name: record(name) for name in reference_names},
    )
    return types.SimpleNamespace(datasets=datasets), calls


def test_a_bare_conversation_id_goes_to_the_generic_loader():
    val, calls = _stub_val()
    polis._load_source(val, "35bmpjr8um")
    assert calls["polis.load"] == (("35bmpjr8um",), {})


def test_a_url_source_goes_to_the_generic_loader_intact():
    val, calls = _stub_val()
    polis._load_source(val, "https://pol.is/report/r6xd526vyjyjrj9navxrj")
    assert calls["polis.load"] == (("https://pol.is/report/r6xd526vyjyjrj9navxrj",), {})


def test_an_hf_slug_keeps_its_colon_and_goes_to_the_generic_loader():
    # `hf:user/dataset` is a source *form* the generic loader understands, not
    # a reference-dataset name, so the split must not eat its colon.
    val, calls = _stub_val()
    polis._load_source(val, "hf:patcon/polis-aufstehen-2018")
    assert calls["polis.load"] == (("hf:patcon/polis-aufstehen-2018",), {})


def test_a_reference_name_with_a_variant_calls_that_loader():
    val, calls = _stub_val()
    polis._load_source(val, "japanchoice:2025_foreign_affairs_security")
    assert calls["japanchoice"] == (("2025_foreign_affairs_security",), {})
    assert "polis.load" not in calls


def test_a_reference_name_without_a_variant_calls_it_with_no_arguments():
    val, calls = _stub_val()
    polis._load_source(val, "aufstehen")
    assert calls["aufstehen"] == ((), {})


def test_reference_names_are_read_off_the_package_not_a_local_copy():
    # A dataset added upstream must work here with no change on this side.
    val, calls = _stub_val(reference_names=("brand_new_dataset",))
    polis._load_source(val, "brand_new_dataset:variant")
    assert calls["brand_new_dataset"] == (("variant",), {})


def test_the_generic_loader_is_not_treated_as_a_reference_dataset():
    # `load` and `translate_statements` are in the package's __all__ but are
    # not conversations; `polis:load` must not call load("").
    val, calls = _stub_val()
    polis._load_source(val, "load")
    assert calls["polis.load"] == (("load",), {})


def test_a_reference_dataset_needing_a_variant_says_so():
    def needs_a_topic(topic):
        raise TypeError("missing 1 required positional argument: 'topic'")

    val, _calls = _stub_val()
    val.datasets.japanchoice = needs_a_topic
    with pytest.raises(ValueError, match="japanchoice"):
        polis._load_source(val, "japanchoice")
