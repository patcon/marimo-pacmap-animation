"""Pair subsampling and the per-pair-type/per-edge alpha math used by rendering."""

import numpy as np

from .data import resolve_proportion


def subsample_pairs(pairs, m, rs):
    """Draw `m` pairs (or a proportion of `len(pairs)` if `m` is a float in
    (0, 1)) for rendering."""
    m = resolve_proportion(m, len(pairs))
    return pairs[rs.choice(len(pairs), min(m, len(pairs)), replace=False)]


def pair_dist(Y, pairs):
    """Per-pair 1 + squared low-dim distance, matching the `d_ij` PaCMAP's
    own gradient is computed from (see pacmap.pacmap.pacmap_grad)."""
    y_ij = Y[pairs[:, 0]] - Y[pairs[:, 1]]
    return 1.0 + (y_ij ** 2).sum(axis=1)


# (offset, numerator) constants per pair type, copied from
# pacmap.pacmap.pacmap_grad's gradient terms - these are intrinsic to PaCMAP's
# loss shape, not something this project chooses.
_FORCE_CONST = {"nb": (10.0, 20.0), "mn": (10000.0, 20000.0), "fp": (1.0, 2.0)}


def pacmap_force(d, w, kind):
    """Per-pair gradient magnitude (unsigned) for pair type `kind`, given its
    weight `w` and its 1 + squared low-dim distance `d`. This is the actual
    force PaCMAP applies to a pair right now - unlike the raw weight, it
    saturates (nb/mn) or decays (fp) as a function of how far apart the pair
    already is in the embedding."""
    a, k = _FORCE_CONST[kind]
    return w * k / (a + d) ** 2


# Per-type opacity ceilings used by the "v2"/"v3" edge-style presets (see
# compute_edge_alphas). Tuned so each pair type can reach a comparable peak
# visibility instead of mid-near's raw weight (up to 1000) drowning out
# neighbour (2-3) and further (always 1).
EDGE_ALPHA_MAX_V2 = {"nb": 0.5, "mn": 0.6, "fp": 0.45}


def compute_edge_alphas(w_NB, w_MN, w_FP, preset="v1", gamma=0.2, Y=None, pairs=None):
    """Per-frame LineCollection alpha for (neighbour, mid-near, further).

    "v1" is the original mapping: each type's alpha is a fixed function of
    its own raw weight, independent of the others. Because w_MN ranges from
    1000 down to 0 while w_NB and w_FP barely move (2-3 and 1 respectively),
    mid-near dominates visually for most of phase 1 and further pairs (fixed
    weight of 1) never rise above alpha 0.05 - effectively invisible.

    "v2" normalizes the three weights against their per-frame max (the
    "combined force" for that frame) and applies a gamma < 1 to compress the
    resulting ratio, so a type that is orders of magnitude weaker than the
    frame's dominant force still gets a visible (if faint) share instead of
    being crushed to ~0. Each type is then scaled by its own opacity ceiling
    so mid-near can still read as the strongest without hiding the others.

    "v3" shades each *individual* drawn edge by its actual instantaneous
    PaCMAP gradient magnitude (see pacmap_force) rather than just its type's
    frame-level weight, so e.g. a further pair that has already been pushed
    apart (and so is contributing ~0 repulsion right now) visibly fades,
    while one still tangled up nearby stays bright. Requires `Y` (current
    embedding) and `pairs` (a (PN, PM, PF) tuple of drawn pair-index arrays).
    Returns one alpha array per pair type instead of a scalar.
    """
    if preset == "v1":
        a_nb = 0.10 * w_NB / 3
        a_mn = 0.55 * w_MN / (w_MN + 3)
        a_fp = 0.05 * w_FP
        return a_nb, a_mn, a_fp

    if preset == "v3":
        PN, PM, PF = pairs
        f_nb = pacmap_force(pair_dist(Y, PN), w_NB, "nb")
        f_mn = pacmap_force(pair_dist(Y, PM), w_MN, "mn")
        f_fp = pacmap_force(pair_dist(Y, PF), w_FP, "fp")
        f_max = max(f_nb.max(initial=0.0), f_mn.max(initial=0.0), f_fp.max(initial=0.0), 1e-9)
        a_nb = EDGE_ALPHA_MAX_V2["nb"] * np.clip(f_nb / f_max, 0.0, 1.0) ** gamma
        a_mn = EDGE_ALPHA_MAX_V2["mn"] * np.clip(f_mn / f_max, 0.0, 1.0) ** gamma
        a_fp = EDGE_ALPHA_MAX_V2["fp"] * np.clip(f_fp / f_max, 0.0, 1.0) ** gamma
        return a_nb, a_mn, a_fp

    w_max = max(w_NB, w_MN, w_FP, 1e-9)
    r_nb = (w_NB / w_max) ** gamma
    r_mn = (w_MN / w_max) ** gamma
    r_fp = (w_FP / w_max) ** gamma
    a_nb = EDGE_ALPHA_MAX_V2["nb"] * r_nb
    a_mn = EDGE_ALPHA_MAX_V2["mn"] * r_mn
    a_fp = EDGE_ALPHA_MAX_V2["fp"] * r_fp
    return a_nb, a_mn, a_fp
