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


PRESETS = {"vanilla": _vanilla}


def build_schedule(preset, num_iters):
    """`(sum(num_iters), 3)` array of `(w_MN, w_NB, w_FP)`, one row per
    optimization iteration. Note this has no init row - `weight_schedule()`
    adds one for display, so that display array's index matches the snapshot
    index."""
    if preset not in PRESETS:
        raise ValueError(f"unknown schedule preset {preset!r}; valid presets: {', '.join(sorted(PRESETS))}")
    return PRESETS[preset](num_iters)
