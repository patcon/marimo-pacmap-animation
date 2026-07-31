"""The Polis loader.

These tests run against a real Polis CSV export - the vendored submodule's own
`polis_real` fixture - because the point of this loader is that it produces a
matrix pacmap can actually fit, and a fake AnnData would prove nothing about
that. They skip themselves when the submodule isn't checked out or the `polis`
extra isn't installed, following the same precedent as the fastplotlib canvas
tests skipping without a GPU (keeps CI green).

No network is involved: a local export directory is one of the sources
valency-anndata's loader accepts.
"""

import numpy as np
import pytest

from _loader import REPO_ROOT, cli

FIXTURE = REPO_ROOT / "vendor" / "valency-anndata" / "tests" / "fixtures" / "polis_real"

pytestmark = pytest.mark.skipif(
    not FIXTURE.is_dir(),
    reason="valency-anndata submodule not checked out (git submodule update --init)",
)


@pytest.fixture(scope="module")
def loaded():
    """Load the fixture once - `recipe_polis` runs a PCA + k-means sweep, so
    it's a few seconds, too slow to repeat per test."""
    pytest.importorskip("valency_anndata", reason="polis extra not installed")
    polis = cli.datasets_polis
    return polis.load_polis(str(FIXTURE), color="polis:group-id")


# --- the matrix pacmap gets ---

def test_x_has_one_row_per_participant_and_one_column_per_statement(loaded):
    X, _y, _rs = loaded
    assert X.shape == (138, 46)


def test_x_is_float32(loaded):
    X, _y, _rs = loaded
    assert X.dtype == np.float32


def test_x_is_c_contiguous(loaded):
    # pacmap's numba kernels want a contiguous array, as load_mnist ensures.
    X, _y, _rs = loaded
    assert X.flags["C_CONTIGUOUS"]


def test_x_has_no_missing_values(loaded):
    # ~70% of the raw vote matrix is unvoted; the imputation step is what makes
    # it fittable at all, so a NaN here means that step was skipped.
    X, _y, _rs = loaded
    assert not np.isnan(X).any()


def test_x_stays_within_the_vote_range(loaded):
    # Statement-wise means of votes in {-1, 0, +1} can't leave [-1, 1].
    X, _y, _rs = loaded
    assert X.min() >= -1.0 and X.max() <= 1.0


def test_x_is_not_constant(loaded):
    # A loader bug that imputed everything to one value would still satisfy
    # every check above.
    X, _y, _rs = loaded
    assert X.std() > 0


# --- colors ---

def test_group_id_gives_one_label_per_participant(loaded):
    X, y, _rs = loaded
    assert len(y) == len(X)


def test_group_id_labels_are_integers(loaded):
    _X, y, _rs = loaded
    assert np.issubdtype(np.asarray(y).dtype, np.integer)


def test_group_id_uses_minus_one_for_participants_polis_did_not_cluster(loaded):
    # recipe_polis only clusters participants above its 7-vote threshold; the
    # rest come back NaN and must still render rather than being dropped.
    _X, y, _rs = loaded
    assert -1 in set(np.asarray(y).tolist())


def test_group_id_has_at_least_two_real_groups(loaded):
    _X, y, _rs = loaded
    groups = {g for g in np.asarray(y).tolist() if g >= 0}
    assert len(groups) >= 2


def test_n_votes_color_counts_each_participants_actual_votes():
    pytest.importorskip("valency_anndata", reason="polis extra not installed")
    X, y, _rs = cli.datasets_polis.load_polis(str(FIXTURE), color="polis:n-votes")
    y = np.asarray(y)
    assert len(y) == len(X)
    # Every participant in a Polis export voted at least once, and nobody can
    # vote more times than there are statements.
    assert y.min() >= 1
    assert y.max() <= X.shape[1]


def test_n_votes_color_varies_between_participants():
    pytest.importorskip("valency_anndata", reason="polis extra not installed")
    _X, y, _rs = cli.datasets_polis.load_polis(str(FIXTURE), color="polis:n-votes")
    assert len(set(np.asarray(y).tolist())) > 1


# --- subsampling, matching load_mnist's contract ---

def test_absolute_n_subsamples_that_many_participants():
    pytest.importorskip("valency_anndata", reason="polis extra not installed")
    X, y, _rs = cli.datasets_polis.load_polis(str(FIXTURE), n=50)
    assert len(X) == 50 and len(y) == 50


def test_fractional_n_is_a_proportion_of_the_participants():
    pytest.importorskip("valency_anndata", reason="polis extra not installed")
    X, _y, _rs = cli.datasets_polis.load_polis(str(FIXTURE), n=0.5)
    assert len(X) == round(0.5 * 138)


def test_n_of_none_keeps_every_participant(loaded):
    X, _y, _rs = loaded
    assert len(X) == 138


def test_returns_a_random_state_seeded_by_seed():
    pytest.importorskip("valency_anndata", reason="polis extra not installed")
    _X, _y, rs_a = cli.datasets_polis.load_polis(str(FIXTURE), n=20, seed=3)
    _X2, _y2, rs_b = cli.datasets_polis.load_polis(str(FIXTURE), n=20, seed=3)
    assert isinstance(rs_a, np.random.RandomState)
    assert rs_a.rand() == rs_b.rand()


def test_same_seed_subsamples_the_same_participants():
    pytest.importorskip("valency_anndata", reason="polis extra not installed")
    Xa, ya, _ = cli.datasets_polis.load_polis(str(FIXTURE), n=20, seed=5)
    Xb, yb, _ = cli.datasets_polis.load_polis(str(FIXTURE), n=20, seed=5)
    assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)


# --- the import guard ---

def test_missing_dependency_raises_a_message_naming_the_submodule_and_extra(monkeypatch):
    """Without valency-anndata importable, the failure should tell you how to
    fix it rather than surfacing a bare ImportError."""
    polis = cli.datasets_polis
    monkeypatch.setattr(polis, "_import_valency", polis._raise_missing)
    with pytest.raises(SystemExit) as e:
        polis.load_polis(str(FIXTURE))
    msg = str(e.value)
    assert "submodule" in msg and "--extra polis" in msg
