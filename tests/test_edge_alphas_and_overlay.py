import numpy as np

from _loader import cli


def test_v1_mid_near_dominates_when_w_mn_is_high():
    a_nb, a_mn, a_fp = cli.compute_edge_alphas(w_NB=3, w_MN=1000, w_FP=1, preset="v1")
    assert a_mn > a_nb
    assert a_mn > a_fp


def test_v1_further_pairs_stay_near_invisible():
    _, _, a_fp = cli.compute_edge_alphas(w_NB=3, w_MN=1000, w_FP=1, preset="v1")
    assert a_fp <= 0.05


def test_v2_further_pairs_stay_visible_despite_low_weight():
    a_nb, a_mn, a_fp = cli.compute_edge_alphas(w_NB=3, w_MN=1000, w_FP=1, preset="v2", gamma=0.2)
    assert a_fp > 0.05  # meaningfully more visible than v1's near-zero floor


def test_v2_respects_per_type_alpha_ceilings():
    a_nb, a_mn, a_fp = cli.compute_edge_alphas(w_NB=1000, w_MN=1000, w_FP=1000, preset="v2")
    assert a_nb <= cli.EDGE_ALPHA_MAX_V2["nb"] + 1e-9
    assert a_mn <= cli.EDGE_ALPHA_MAX_V2["mn"] + 1e-9
    assert a_fp <= cli.EDGE_ALPHA_MAX_V2["fp"] + 1e-9


def test_v3_edge_already_pushed_apart_fades_relative_to_nearby_edge():
    Y = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.0], [50.0, 0.0]])
    pairs_near = np.array([[0, 1]])
    pairs_far = np.array([[2, 3]])
    _, _, a_fp_near = cli.compute_edge_alphas(
        w_NB=3, w_MN=1000, w_FP=1, preset="v3", Y=Y, pairs=(pairs_near, pairs_near, pairs_near))
    _, _, a_fp_far = cli.compute_edge_alphas(
        w_NB=3, w_MN=1000, w_FP=1, preset="v3", Y=Y, pairs=(pairs_far, pairs_far, pairs_far))
    assert a_fp_far[0] < a_fp_near[0]


def test_overlay_v1_omits_w_fp():
    text = cli.compute_overlay_text(10, 450, 1, w_MN=500.3, w_NB=2.5, w_FP=1.0, preset="v1")
    assert "w_FP" not in text
    assert "w_MN" in text and "w_NB" in text


def test_overlay_v2_stacks_all_three_weights_as_integers():
    text = cli.compute_overlay_text(10, 450, 1, w_MN=500.6, w_NB=2.5, w_FP=1.0, preset="v2")
    assert "w_MN  501" in text
    assert "w_NB    2" in text or "w_NB    3" in text
    assert "w_FP    1" in text


def test_overlay_includes_title_prefix():
    text = cli.compute_overlay_text(0, 450, 1, w_MN=1000, w_NB=3, w_FP=1, title_prefix="localmap ", preset="v1")
    assert text.startswith("localmap ")
