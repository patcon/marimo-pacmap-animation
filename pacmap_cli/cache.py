"""On-disk caching of fit_trace() results.

A fit is a pure function of the input data and the reducer params (the seed
makes it deterministic), so re-running one - LocalMAP in particular, the slow
half of `--algorithm both` - only to change a render setting is wasted time.
Each fit is stored as a single `.npz` under the cache dir, keyed by a hash of
the data and params, alongside a `.json` sidecar recording those params in
readable form so a directory listing stays inspectable.

Nothing here evicts old entries: a trace is ~18MB at `--n 5000` but ~250MB at
`--n all`, so the cache dir is cleared by hand when it gets too big.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def fit_key(X, params):
    """Content hash identifying a fit: the input data plus every param that
    changes its result. Hashing X itself (rather than just `--n`/`--seed`)
    means a different MNIST source or a changed loader can't silently serve
    a trace fit on different data; `params` carries pacmap's version for the
    same reason, so an upgrade doesn't reuse the old implementation's output."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _paths(cache_dir, algorithm, key):
    stem = Path(cache_dir) / f"{algorithm}_{key}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def load_fit(cache_dir, algorithm, key):
    """Return the cached `(trace, pair_neighbors, pair_MN, pair_FP_history)`
    tuple for `key`, or None if it isn't cached - including when the entry
    exists but can't be read, since a damaged entry is only ever worth
    treating as a miss to be refit and overwritten."""
    npz_path, _ = _paths(cache_dir, algorithm, key)
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as z:
            fp_frames = z["fp_frames"]
            pair_FP_history = [(int(frame), z[f"fp_{i}"]) for i, frame in enumerate(fp_frames)]
            return z["trace"], z["pair_neighbors"], z["pair_MN"], pair_FP_history
    except Exception as e:
        print(f"cache: ignoring unreadable entry {npz_path} ({type(e).__name__}: {e})")
        return None


def save_fit(cache_dir, algorithm, key, params, result):
    """Write `result` (a fit_trace() return tuple) to the cache, returning the
    `.npz` path. Written to a temp file and renamed into place, so an
    interrupted write can't leave a half-file a later run reads as a hit."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _paths(cache_dir, algorithm, key)
    trace, pair_neighbors, pair_MN, pair_FP_history = result

    arrays = {
        "trace": trace,
        "pair_neighbors": pair_neighbors,
        "pair_MN": pair_MN,
        "fp_frames": np.array([frame for frame, _ in pair_FP_history], dtype=np.int64),
    }
    # One array per checkpoint rather than one stacked array: nothing
    # guarantees every LocalMAP resample yields the same number of far pairs.
    arrays.update({f"fp_{i}": arr for i, (_frame, arr) in enumerate(pair_FP_history)})

    tmp_path = npz_path.with_suffix(".npz.tmp")
    # Written through a file handle: given a path, np.savez appends its own
    # ".npz" to anything not already ending in it, so the temp file would
    # land somewhere other than where os.replace() looks for it.
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, npz_path)
    meta_path.write_text(json.dumps({"algorithm": algorithm, **params}, sort_keys=True, indent=2, default=str) + "\n")
    return npz_path
