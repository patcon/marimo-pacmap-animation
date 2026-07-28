"""Building the pair-weight schedule as an explicit per-iteration array.

pacmap computes its own schedule inline during the fit, via
`pacmap.pacmap.find_weight`. Materializing it as a `(sum(num_iters), 3)`
array up front lets the same array both *drive* the fit (by patching that
function - see `override_weight_schedule`) and *display* it in the overlay
and schedule strip, so the two can't disagree.

`PRESETS` is the extension point: a new schedule is a new builder returning
the same array shape, and nothing else has to change.
"""

import numpy as np


def _vanilla(num_iters):
    """pacmap's own schedule, replayed. Calls `find_weight` rather than
    reimplementing it so this is bit-identical by construction - the fit must
    be indistinguishable from an unpatched run."""
    from pacmap.pacmap import find_weight

    return np.array(
        [find_weight(1000.0, itr, num_iters=num_iters) for itr in range(sum(num_iters))],
        dtype=float,
    )


def _cycle(num_iters, period=100, mn_min=0.05, mn_max=100.0):
    """Sweep w_MN back and forth between global- and local-structure
    emphasis, forever, instead of converging once.

    Only the *ratios* between the three forces matter - scaling all of them
    scales the gradient, which Adam's per-parameter normalization largely
    absorbs - so w_MN alone moves and w_NB/w_FP are held.

    The sweep is log-spaced because w_MN is a scale parameter: perceptually
    even steps are multiplicative, not additive. Keeping `mn_min` above zero
    also sidesteps vanilla's w_MN = 0 endpoint, which is at infinite
    log-ratio distance and so has no place on a continuous sweep.

    Anchored on cos rather than sin so a run opens at `mn_max` - global
    structure first, as vanilla does - rather than starting mid-sweep.
    """
    if period <= 0:
        raise ValueError(f"schedule period must be positive, got {period}")
    if mn_min <= 0 or mn_max <= 0:
        raise ValueError(f"log-spaced w_MN bounds must be positive, got mn_min={mn_min}, mn_max={mn_max}")

    t = np.arange(sum(num_iters))
    u = (1 + np.cos(2 * np.pi * t / period)) / 2  # 1 = global, 0 = local
    w_MN = np.exp((1 - u) * np.log(mn_min) + u * np.log(mn_max))
    return np.column_stack([w_MN, np.full_like(w_MN, 2.0), np.full_like(w_MN, 1.0)])


PRESETS = {"vanilla": _vanilla, "cycle": _cycle}


def build_schedule(preset, num_iters, **params):
    """`(sum(num_iters), 3)` array of `(w_MN, w_NB, w_FP)`, one row per
    optimization iteration. Note this has no init row - `weight_schedule()`
    adds one for display, so that display array's index matches the snapshot
    index.

    `params` are the preset's own knobs; each preset declares the ones it
    understands, so passing a knob a preset has no use for is an error rather
    than a silent no-op."""
    if preset not in PRESETS:
        raise ValueError(f"unknown schedule preset {preset!r}; valid presets: {', '.join(sorted(PRESETS))}")
    return PRESETS[preset](num_iters, **params)
