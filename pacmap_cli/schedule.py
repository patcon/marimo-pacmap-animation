"""Building the pair-weight schedule as an explicit per-iteration array.

pacmap computes its own schedule inline during the fit, via
`pacmap.pacmap.find_weight`. Materializing it as a `(sum(num_iters), 3)`
array up front lets the same array both *drive* the fit (by patching that
function - see `override_weight_schedule`) and *display* it in the overlay
and schedule strip, so the two can't disagree.

`PRESETS` is the extension point: a new schedule is a new builder returning
the same array shape, and nothing else has to change.
"""

import contextlib
import inspect

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


def _log_sweep(t, period, lo, hi, phase, name):
    """One channel's sinusoidal sweep between `lo` and `hi`.

    Log-spaced because these weights are scale parameters: perceptually even
    steps are multiplicative, not additive, and the meaningful quantity is a
    ratio like w_MN/w_NB rather than a difference. Bounds must stay above
    zero - vanilla's w_MN = 0 endpoint sits at infinite log-ratio distance
    and so isn't a point a continuous sweep can pass through.

    Anchored on cos rather than sin, so at `phase=0` a run opens at `hi`
    rather than mid-sweep. `phase` is in cycles: 0.5 is antiphase.
    """
    if lo <= 0 or hi <= 0:
        raise ValueError(f"log-spaced {name} bounds must be positive, got {lo} and {hi}")
    u = (1 + np.cos(2 * np.pi * (t / period + phase))) / 2
    return np.exp((1 - u) * np.log(lo) + u * np.log(hi))


def _cycle(num_iters, period=100, mn_min=0.05, mn_max=100.0,
           fp_min=1.0, fp_max=1.0, fp_phase=0.5):
    """Sweep the force balance back and forth forever instead of converging
    once, so the embedding runs as a limit cycle rather than toward a fixed
    point.

    Only the *ratios* between the three forces matter - scaling all of them
    scales the gradient, which Adam's per-parameter normalization largely
    absorbs - so w_NB is held and the two channels that carry meaning move:

      w_MN, the attractive/global channel, sweeping mn_max (pull the global
      skeleton taut) to mn_min (let the local neighbor term refine detail);

      w_FP, the repulsive channel, sweeping fp_max (spread and untangle) to
      fp_min (let attraction re-condense). Held at 1.0 by default, i.e. off,
      so the default is the attraction-only sweep.

    `fp_phase` is in cycles and defaults to 0.5 - antiphase, so repulsion
    troughs exactly when global attraction peaks. In phase the two would
    partly cancel; in antiphase the condense and expand halves reinforce.
    """
    if period <= 0:
        raise ValueError(f"schedule period must be positive, got {period}")

    t = np.arange(sum(num_iters))
    w_MN = _log_sweep(t, period, mn_min, mn_max, 0.0, "w_MN")
    w_FP = _log_sweep(t, period, fp_min, fp_max, fp_phase, "w_FP")
    return np.column_stack([w_MN, np.full_like(w_MN, 2.0), w_FP])


def _breathe(num_iters, period=100, mn_min=3.0, mn_max=3.0,
             fp_min=0.2, fp_max=5.0, fp_phase=0.0):
    """The repulsive twin of `cycle`: w_FP sweeps while w_MN is held.

    Same builder, different defaults. Repulsion is the visually legible
    channel - an inhale where the embedding spreads and untangles, an exhale
    where attraction re-condenses it - whereas cycling attraction alone
    changes structure without moving the extent much, since Adam absorbs a
    good deal of the magnitude change.

    w_MN is held at 3.0, the value vanilla settles on for phases 2-3, so the
    attractive side stays at a sane baseline rather than being switched off.
    `fp_phase` is 0 here (not antiphase) because with w_MN held there is
    nothing to be out of phase with, and 0 opens the run spread out.
    """
    return _cycle(num_iters, period=period, mn_min=mn_min, mn_max=mn_max,
                  fp_min=fp_min, fp_max=fp_max, fp_phase=fp_phase)


PRESETS = {"vanilla": _vanilla, "cycle": _cycle, "breathe": _breathe}


def preset_defaults(preset):
    """A preset's own knob defaults, read off its builder's signature.

    Each preset defines the *shape* of a schedule through these - `breathe`
    holds w_MN and sweeps w_FP purely by defaulting them that way - so they
    can't be duplicated in DEFAULT_CONFIG without the config silently
    overriding every preset back into the same shape. DEFAULT_CONFIG holds
    None for each knob instead, meaning "whatever this preset wants".
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown schedule preset {preset!r}; valid presets: {', '.join(sorted(PRESETS))}")
    return {
        name: p.default
        for name, p in inspect.signature(PRESETS[preset]).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


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


@contextlib.contextmanager
def override_weight_schedule(W):
    """Drive the fit from `W` instead of pacmap's own schedule, for the
    duration of the `with` block.

    Monkey-patches `pacmap.pacmap.find_weight`, which both `pacmap()` and
    `localmap()` call by bare module-level name once per iteration - so
    patching the name is enough, and neither optimization loop has to be
    duplicated. Unlike the far-pair sampler `capture_fp_history()` patches,
    `find_weight` is plain Python rather than numba-jitted, so its signature
    is inspected directly (no `.py_func`).

    The original is restored in `finally` even if the fit raises, so a failed
    fit can't leak the patch into the next one - an `--algorithm both` run
    fits twice.
    """
    import pacmap

    original = pacmap.pacmap.find_weight
    # Guard against a future pacmap upgrade silently changing this
    # function's shape out from under the patch.
    if len(inspect.signature(original).parameters) != 3:
        raise RuntimeError(
            "pacmap.pacmap.find_weight's signature has changed; "
            "override_weight_schedule()'s monkey-patch needs updating for "
            "this pacmap version."
        )

    def wrapped(w_MN_init, itr, *, num_iters):
        if len(W) != sum(num_iters):
            raise ValueError(
                f"schedule has {len(W)} rows but the fit runs {sum(num_iters)} "
                f"iterations ({num_iters}); it was built for different num_iters."
            )
        w_MN, w_NB, w_FP = W[itr]
        return float(w_MN), float(w_NB), float(w_FP)

    pacmap.pacmap.find_weight = wrapped
    try:
        yield
    finally:
        pacmap.pacmap.find_weight = original
