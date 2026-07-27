"""Running PaCMAP/LocalMAP while capturing every optimization iteration."""

import time

from .fp_history import capture_fp_history, fp_resample_iterations


def fit_trace(X, algorithm, n_neighbors, mn_ratio, fp_ratio, num_iters, seed=42):
    """Run PaCMAP or LocalMAP, capturing the embedding at every iteration.

    Returns `pair_FP_history`, a list of `(frame, pair_FP)` checkpoints
    sorted by ascending frame - `[(0, pair_FP)]` for PaCMAP (which never
    resamples FP) or a run where LocalMAP's phase 3 never triggers a
    resample, otherwise `[(0, initial), (frame_1, resample_1), ...]` for
    LocalMAP, one entry per resample event it actually performed."""
    import pacmap

    reducer_cls = {"pacmap": pacmap.PaCMAP, "localmap": pacmap.LocalMAP}[algorithm]
    total = sum(num_iters)

    print(f"Running {algorithm}...")
    t0 = time.time()
    reducer = reducer_cls(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=mn_ratio,
        FP_ratio=fp_ratio,
        num_iters=num_iters,
        intermediate=True,
        intermediate_snapshots=list(range(total + 1)),
        random_state=seed,
        verbose=False,
    )
    if algorithm == "localmap":
        with capture_fp_history() as calls:
            trace = reducer.fit_transform(X)  # (total+1, N, 2) float32
    else:
        trace = reducer.fit_transform(X)  # (total+1, N, 2) float32
        calls = []
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
