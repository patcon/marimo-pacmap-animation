"""Running PaCMAP/LocalMAP while capturing every optimization iteration."""

import time


def fit_trace(X, algorithm, n_neighbors, mn_ratio, fp_ratio, num_iters, seed=42):
    """Run PaCMAP or LocalMAP, capturing the embedding at every iteration."""
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
    trace = reducer.fit_transform(X)  # (total+1, N, 2) float32
    print("%s fit %.1fs" % (algorithm, time.time() - t0), trace.shape, trace.nbytes / 1e6, "MB")

    pair_neighbors = reducer.pair_neighbors
    pair_MN = reducer.pair_MN
    pair_FP = reducer.pair_FP
    print(f"{algorithm} pairs:", pair_neighbors.shape, pair_MN.shape, pair_FP.shape)
    return trace, pair_neighbors, pair_MN, pair_FP
