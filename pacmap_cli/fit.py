"""Running PaCMAP/LocalMAP while capturing every optimization iteration."""

import contextlib
import time

from .cache import fit_key, load_fit, save_fit
from .fp_history import capture_fp_history, fp_resample_iterations
from .schedule import override_weight_schedule


def fit_trace(X, algorithm, n_neighbors, mn_ratio, fp_ratio, num_iters, seed=42, n_components=2,
              low_dist_thres=10.0, schedule=None, schedule_params=None, dataset=None,
              cache_dir=None):
    """Run PaCMAP or LocalMAP, capturing the embedding at every iteration.

    `low_dist_thres` is LocalMAP-only (ignored for PaCMAP): the acceptance
    distance for the "local" far pairs it resamples every 10 iterations.

    With a `cache_dir`, the result is read from / written to an on-disk cache
    keyed by the data and these params (see cache.py); `cache_dir=None`
    always refits and writes nothing. `dataset` names the data's origin for
    the entry's readable metadata only - it never enters the key.

    Returns `pair_FP_history`, a list of `(frame, pair_FP)` checkpoints
    sorted by ascending frame - `[(0, pair_FP)]` for PaCMAP (which never
    resamples FP) or a run where LocalMAP's phase 3 never triggers a
    resample, otherwise `[(0, initial), (frame_1, resample_1), ...]` for
    LocalMAP, one entry per resample event it actually performed."""
    import pacmap

    params = dict(
        n_neighbors=n_neighbors, mn_ratio=mn_ratio, fp_ratio=fp_ratio,
        num_iters=tuple(num_iters), seed=seed, n_components=n_components,
        low_dist_thres=low_dist_thres,
    )
    # PaCMAP ignores low_dist_thres, so it must stay out of its key - otherwise
    # varying the knob would needlessly refit the pacmap half of a `both` run.
    key_params = {k: v for k, v in params.items() if algorithm == "localmap" or k != "low_dist_thres"}
    # Same treatment for the schedule: vanilla leaves the fit unpatched, so its
    # params can't affect the result and must stay out of the key - otherwise
    # landing this feature would invalidate every fit cached before it existed.
    key_params.update(_schedule_key_params(schedule_params))
    key = None
    if cache_dir is not None:
        key = fit_key(X, {**key_params, "algorithm": algorithm, "pacmap_version": pacmap.__version__})
        cached = load_fit(cache_dir, algorithm, key)
        if cached is not None:
            print(f"{algorithm}: cache hit ({key})")
            return cached

    result = _fit_uncached(X, algorithm, **params, schedule=schedule)

    if cache_dir is not None:
        # `dataset` is recorded but deliberately not keyed on: fit_key already
        # hashes X itself, so two datasets can't collide, and adding it to the
        # key would invalidate every fit cached before datasets existed.
        meta_params = {**key_params, "pacmap_version": pacmap.__version__,
                       **({"dataset": dataset} if dataset is not None else {})}
        path = save_fit(cache_dir, algorithm, key, meta_params, result)
        print(f"{algorithm}: cached fit -> {path}")
    return result


def _schedule_key_params(schedule_params):
    """The schedule params that belong in the cache key. Empty for the vanilla
    preset (which never patches the fit), so its knobs - which vanilla has no
    use for anyway - can't trigger a refit that would change nothing."""
    if not schedule_params or schedule_params.get("schedule_preset", "vanilla") == "vanilla":
        return {}
    return dict(schedule_params)


def _fit_uncached(X, algorithm, n_neighbors, mn_ratio, fp_ratio, num_iters, seed, n_components,
                  low_dist_thres, schedule=None):
    """The fit itself. Split out from fit_trace() so the caching layer around
    it can be tested (and bypassed) without running a real fit."""
    import pacmap

    reducer_cls = {"pacmap": pacmap.PaCMAP, "localmap": pacmap.LocalMAP}[algorithm]
    total = sum(num_iters)

    print(f"Running {algorithm}...")
    t0 = time.time()
    reducer = reducer_cls(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=mn_ratio,
        FP_ratio=fp_ratio,
        num_iters=num_iters,
        intermediate=True,
        intermediate_snapshots=list(range(total + 1)),
        random_state=seed,
        verbose=False,
        # LocalMAP-only: PaCMAP.__init__ doesn't accept it.
        **({"low_dist_thres": low_dist_thres} if algorithm == "localmap" else {}),
    )
    with contextlib.ExitStack() as stack:
        # A schedule drives the fit by patching find_weight; None leaves
        # pacmap's own schedule in place, untouched.
        if schedule is not None:
            stack.enter_context(override_weight_schedule(schedule))
        calls = stack.enter_context(capture_fp_history()) if algorithm == "localmap" else []
        trace = reducer.fit_transform(X)  # (total+1, N, n_components) float32
    print("%s fit %.1fs" % (algorithm, time.time() - t0), trace.shape, trace.nbytes / 1e6, "MB")

    pair_neighbors = reducer.pair_neighbors
    pair_MN = reducer.pair_MN
    if calls:
        resample_iters = fp_resample_iterations(num_iters)
        # calls[0] is the initial ("frame 0") checkpoint; calls[1:] are one
        # per resample event, in the same order as resample_iters - the
        # resample at loop-iteration `itr` takes effect for trace frame
        # `itr + 1` (that iteration's embedding update produces frame
        # itr+1, and the new FP set is what's used for gradients from then
        # on).
        pair_FP_history = [(0, calls[0][1])] + [
            (itr + 1, arr) for itr, (_kind, arr) in zip(resample_iters, calls[1:])
        ]
    else:
        pair_FP_history = [(0, reducer.pair_FP)]
    print(f"{algorithm} pairs:", pair_neighbors.shape, pair_MN.shape, pair_FP_history[-1][1].shape,
          f"({len(pair_FP_history)} FP checkpoint(s))")
    return trace, pair_neighbors, pair_MN, pair_FP_history
