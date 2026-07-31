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


def test_subsample_pairs_indices_returns_shape_matching_count():
    pairs = np.arange(20).reshape(10, 2)
    rs = np.random.RandomState(0)
    idx = cli.subsample_pairs_indices(pairs, 4, rs)
    assert idx.shape == (4,)


def test_subsample_pairs_indices_has_no_repeats_and_is_in_range():
    pairs = np.arange(20).reshape(10, 2)
    rs = np.random.RandomState(0)
    idx = cli.subsample_pairs_indices(pairs, 5, rs)
    assert len(set(idx.tolist())) == len(idx)
    assert all(0 <= i < len(pairs) for i in idx)


def test_subsample_pairs_indices_caps_at_pool_size():
    pairs = np.arange(6).reshape(3, 2)
    rs = np.random.RandomState(0)
    idx = cli.subsample_pairs_indices(pairs, 100, rs)
    assert idx.shape == (3,)


def test_subsample_pairs_matches_indexing_by_subsample_pairs_indices():
    pairs = np.arange(40).reshape(20, 2)
    seed = 7
    out = cli.subsample_pairs(pairs, 6, np.random.RandomState(seed))
    idx = cli.subsample_pairs_indices(pairs, 6, np.random.RandomState(seed))
    assert np.array_equal(out, pairs[idx])


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


def test_count_drawn_reports_node_count_as_given_point_count():
    PN = np.arange(6).reshape(3, 2)
    PM = np.arange(4).reshape(2, 2)
    PF = np.arange(2).reshape(1, 2)
    counts = cli.count_drawn(1000, PN, PM, PF)
    assert counts["nodes"] == 1000


def test_count_drawn_reports_edge_count_per_pair_type():
    PN = np.arange(6).reshape(3, 2)
    PM = np.arange(4).reshape(2, 2)
    PF = np.arange(2).reshape(1, 2)
    counts = cli.count_drawn(1000, PN, PM, PF)
    assert counts["edges_neighbor"] == 3
    assert counts["edges_midnear"] == 2
    assert counts["edges_further"] == 1


def test_count_drawn_reports_total_edges_as_sum_of_pair_types():
    PN = np.arange(6).reshape(3, 2)
    PM = np.arange(4).reshape(2, 2)
    PF = np.arange(2).reshape(1, 2)
    counts = cli.count_drawn(1000, PN, PM, PF)
    assert counts["edges_total"] == 6


def test_count_drawn_handles_empty_pair_types():
    empty = np.empty((0, 2), dtype=int)
    counts = cli.count_drawn(50, empty, empty, empty)
    assert counts == {
        "nodes": 50,
        "edges_neighbor": 0,
        "edges_midnear": 0,
        "edges_further": 0,
        "edges_total": 0,
    }


# --- subsample_indices: what --n resolves to at load time ---

def test_subsample_indices_none_keeps_every_row_in_order():
    import numpy as np

    idx = cli.subsample_indices(5, None, np.random.RandomState(0))
    assert list(idx) == [0, 1, 2, 3, 4]


def test_subsample_indices_draws_the_requested_count():
    import numpy as np

    idx = cli.subsample_indices(100, 10, np.random.RandomState(0))
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10


def test_subsample_indices_resolves_a_fraction_against_the_total():
    import numpy as np

    assert len(cli.subsample_indices(100, 0.25, np.random.RandomState(0))) == 25


def test_subsample_indices_clamps_a_count_larger_than_the_dataset():
    # Polis conversations are far smaller than MNIST, so the default --n 5000
    # routinely exceeds the population. That's a smaller dataset, not an error
    # - the same clamp subsample_pairs_indices() already applies to --n-lines.
    import numpy as np

    idx = cli.subsample_indices(138, 5000, np.random.RandomState(0))
    assert list(idx) == list(range(138))
