import numpy as np

from _loader import cli


def test_resolve_proportion_leaves_int_unchanged():
    assert cli.resolve_proportion(500, 70_000) == 500


def test_resolve_proportion_leaves_none_unchanged():
    assert cli.resolve_proportion(None, 70_000) is None


def test_resolve_proportion_converts_fraction_to_count():
    assert cli.resolve_proportion(0.1, 1000) == 100


def test_resolve_proportion_treats_1_0_as_all():
    assert cli.resolve_proportion(1.0, 1234) == 1234


def test_resolve_proportion_rounds_to_nearest_int():
    assert cli.resolve_proportion(1 / 3, 10) == 3


def test_subsample_pairs_draws_absolute_count():
    pairs = np.arange(20).reshape(10, 2)
    rs = np.random.RandomState(0)
    out = cli.subsample_pairs(pairs, 4, rs)
    assert out.shape == (4, 2)


def test_subsample_pairs_draws_proportion_of_pool():
    pairs = np.arange(40).reshape(20, 2)
    rs = np.random.RandomState(0)
    out = cli.subsample_pairs(pairs, 0.5, rs)
    assert out.shape == (10, 2)


def test_subsample_pairs_caps_at_pool_size():
    pairs = np.arange(6).reshape(3, 2)
    rs = np.random.RandomState(0)
    out = cli.subsample_pairs(pairs, 100, rs)
    assert out.shape == (3, 2)


def test_subsample_pairs_draws_from_within_pool_without_repeats():
    pairs = np.arange(20).reshape(10, 2)
    rs = np.random.RandomState(0)
    out = cli.subsample_pairs(pairs, 5, rs)
    rows = [tuple(row) for row in out]
    assert len(set(rows)) == len(rows)
    assert all(tuple(row) in [tuple(p) for p in pairs] for row in out)


def test_pair_dist_is_1_plus_squared_distance():
    Y = np.array([[0.0, 0.0], [3.0, 4.0]])
    pairs = np.array([[0, 1]])
    d = cli.pair_dist(Y, pairs)
    assert d == 1.0 + 25.0


def test_pair_dist_is_zero_offset_when_points_coincide():
    Y = np.array([[1.0, 1.0], [1.0, 1.0]])
    pairs = np.array([[0, 1]])
    d = cli.pair_dist(Y, pairs)
    assert d == 1.0


def test_pacmap_force_decreases_as_distance_grows():
    near = cli.pacmap_force(np.array([1.0]), w=1.0, kind="nb")
    far = cli.pacmap_force(np.array([100.0]), w=1.0, kind="nb")
    assert far < near


def test_pacmap_force_scales_linearly_with_weight():
    d = np.array([5.0])
    f1 = cli.pacmap_force(d, w=1.0, kind="mn")
    f2 = cli.pacmap_force(d, w=2.0, kind="mn")
    assert np.isclose(f2, 2 * f1)


def test_pacmap_force_differs_by_pair_type_constants():
    d = np.array([5.0])
    f_nb = cli.pacmap_force(d, w=1.0, kind="nb")
    f_mn = cli.pacmap_force(d, w=1.0, kind="mn")
    f_fp = cli.pacmap_force(d, w=1.0, kind="fp")
    assert len({round(float(f_nb[0]), 6), round(float(f_mn[0]), 6), round(float(f_fp[0]), 6)}) == 3
