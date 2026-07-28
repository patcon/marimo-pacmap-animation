"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolate_cwd(monkeypatch, tmp_path):
    """Run every test from its own scratch directory.

    Fit caching is on by default and its default cache_dir (".cache/fits") is
    relative, so any test calling main() would otherwise write real cache
    entries into the repo. Chdir-ing keeps that default path exercised as-is
    - tests can still assert against ".cache/fits" - while landing the files
    somewhere disposable.
    """
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
