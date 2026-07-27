"""Capturing LocalMAP's per-iteration far-pair (FP) resampling history.

LocalMAP resamples its `pair_FP` graph every 10 iterations once phase 3
begins (`pacmap.pacmap.localmap`, pacmap==0.9.1), but only the final
resampled set survives as `reducer.pair_FP` after `fit_transform()`. The
functions here reconstruct the full per-checkpoint history so rendering can
show the far-pair graph as it actually evolved.
"""

import contextlib
import inspect

import numpy as np


def fp_resample_iterations(num_iters):
    """0-indexed itr values at which LocalMAP resamples pair_FP - pure
    mirror of localmap()'s condition `itr > num_iters[0]+num_iters[1] and
    itr % 10 == 0` (pacmap.pacmap.localmap, pacmap==0.9.1)."""
    boundary = num_iters[0] + num_iters[1]
    total = sum(num_iters)
    return [itr for itr in range(total) if itr > boundary and itr % 10 == 0]


def checkpoint_index_for_frame(frame, checkpoint_frames):
    """Index into an ascending array of checkpoint frame numbers giving the
    latest checkpoint whose frame <= `frame`."""
    idx = np.searchsorted(checkpoint_frames, frame, side="right") - 1
    return max(idx, 0)


@contextlib.contextmanager
def capture_fp_history():
    """Monkey-patch `pacmap.pacmap.sample_FP_pair_nearby` for the duration of
    the `with` block, so every LocalMAP far-pair resample is recorded
    instead of only the final one surviving as `reducer.pair_FP`.

    Yields a list that is filled in as the fit runs: the first captured call
    records its `old_pair_FP` argument (the pre-resample set, i.e. frame 0's
    far-pair graph) as well as its own result, so the yielded list ends up
    as `[(0, initial_pair_FP), (frame_1, resampled_1), ...]` with `frame_i`
    filled in by the caller (this function only knows call order, not
    iteration numbers - see `fp_resample_iterations`). The original function
    is restored in `finally`, even if `fit_transform` raises, so no patched
    state leaks past this call.
    """
    import pacmap

    original = pacmap.pacmap.sample_FP_pair_nearby
    # Guard against a future pacmap upgrade silently changing this
    # function's shape out from under the patch.
    if len(inspect.signature(original.py_func).parameters) != 5:
        raise RuntimeError(
            "pacmap.pacmap.sample_FP_pair_nearby's signature has changed; "
            "capture_fp_history()'s monkey-patch needs updating for this "
            "pacmap version."
        )

    calls = []

    def wrapped(X, pair_neighbors, old_pair_FP, Y, low_dist_thres):
        if not calls:
            calls.append(("initial", old_pair_FP.copy()))
        result = original(X, pair_neighbors, old_pair_FP, Y, low_dist_thres)
        calls.append(("resample", result.copy()))
        return result

    pacmap.pacmap.sample_FP_pair_nearby = wrapped
    try:
        yield calls
    finally:
        pacmap.pacmap.sample_FP_pair_nearby = original
