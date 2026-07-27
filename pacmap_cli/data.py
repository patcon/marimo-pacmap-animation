"""Loading and subsampling MNIST."""

import numpy as np


def resolve_proportion(value, total):
    """If `value` is a float in (0, 1], treat it as a proportion of `total`
    and round to an absolute count (1.0 -> all of `total`); otherwise return
    it unchanged. Shared by --n (proportion of the full dataset) and
    --n-lines (proportion of a pair type's available pool)."""
    if isinstance(value, float) and 0 < value <= 1:
        return round(value * total)
    return value


def load_mnist(n=None, seed=0):
    """Load MNIST, optionally subsampled to `n` points (or a proportion of
    the full ~70_000 if `n` is a float in (0, 1)). n=None loads all of it."""
    try:
        from tensorflow.keras.datasets import mnist
        (Xtr, ytr), _ = mnist.load_data()
        Xfull = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
        yfull = ytr.astype(int)
    except Exception as e:
        print("keras unavailable (%s), falling back to openml" % type(e).__name__)
        from sklearn.datasets import fetch_openml
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        Xfull = d.data.astype(np.float32) / 255.0
        yfull = d.target.astype(int)

    n = resolve_proportion(n, len(Xfull))
    rs = np.random.RandomState(seed)
    sel = np.arange(len(Xfull)) if n is None else rs.choice(len(Xfull), n, replace=False)

    X, y = np.ascontiguousarray(Xfull[sel]), yfull[sel]
    print(X.shape, X.dtype, np.bincount(y))
    return X, y, rs
