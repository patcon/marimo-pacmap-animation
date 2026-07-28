"""On-disk caching of fit_trace() results.

A fit is a pure function of the input data and the reducer params (the seed
makes it deterministic), so re-running one - LocalMAP in particular, the slow
half of `--algorithm both` - only to change a render setting is wasted time.

Each fit is one directory of `.npy` files, keyed by a hash of the data and
params, with a `meta.json` recording those params in readable form so a
directory listing stays inspectable. Plain `.npy` rather than a single `.npz`
because the zip container CRCs every byte on the way in and out (measured at
`--n all --n-components 3`: 1.1s save / 1.0s load for npz vs 0.29s / 0.01s
for mmap'd .npy, identical bytes on disk), and because `.npy` can be
memory-mapped - a 379MB trace is paged in as frames are touched instead of
being read into RAM up front. Compression was measured and rejected: float32
embedding coordinates are near-incompressible, and `savez_compressed` cost
115s to save the same fit for an 18% size reduction.

Nothing here evicts entries: a trace is ~3MB at `--n 2000` but ~380MB at
`--n all --n-components 3`, so the cache dir is cleared by hand.
"""

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
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


def _entry_dir(cache_dir, algorithm, key):
    return Path(cache_dir) / f"{algorithm}_{key}"


class _DedupedFPHistory(Sequence):
    """The `(frame, pair_FP)` checkpoint list, rebuilt from a source column
    shared by every checkpoint plus one target column each.

    Rebuilt per access rather than up front: `render.py` reduces each
    checkpoint to `n_lines` rows (`arr[fp_idx]`) and drops the rest, so
    materializing one full checkpoint at a time keeps peak memory at a single
    checkpoint instead of the whole history (~280MB at `--n all`). Behaves
    like the plain list a fresh fit returns - indexable, iterable, sized.
    """

    def __init__(self, frames, sources, targets):
        self._frames = frames
        self._sources = sources
        self._targets = targets

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        return int(self._frames[i]), np.column_stack([self._sources, self._targets[i]])


def _shared_source_column(pair_FP_history):
    """The far-pair source column if every checkpoint shares it, else None.

    LocalMAP's resample only ever changes a far pair's *target* endpoint - the
    same invariant `subsample_pairs_indices()` relies on to track one source
    point's edge across checkpoints - so storing that column once rather than
    per checkpoint removes ~25% of an entry. Checked rather than assumed, so a
    pacmap change that broke it would fall back instead of corrupting a fit.
    """
    first = pair_FP_history[0][1]
    if any(arr.shape != first.shape for _frame, arr in pair_FP_history):
        return None
    if any(not np.array_equal(arr[:, 0], first[:, 0]) for _frame, arr in pair_FP_history):
        return None
    return first[:, 0]


def load_fit(cache_dir, algorithm, key):
    """Return the cached `(trace, pair_neighbors, pair_MN, pair_FP_history)`
    tuple for `key`, or None if it isn't cached - including when the entry
    exists but can't be read, since a damaged entry is only ever worth
    treating as a miss to be refit and overwritten.

    Arrays are memory-mapped and therefore read-only; nothing downstream
    writes to a trace or a pair array.
    """
    path = _entry_dir(cache_dir, algorithm, key)
    if not path.is_dir():
        return None
    try:
        load = lambda name: np.load(path / f"{name}.npy", mmap_mode="r")
        frames = load("fp_frames")
        if (path / "fp_sources.npy").exists():
            pair_FP_history = _DedupedFPHistory(frames, load("fp_sources"), load("fp_targets"))
        else:
            pair_FP_history = [(int(frame), load(f"fp_{i}")) for i, frame in enumerate(frames)]
        return load("trace"), load("pair_neighbors"), load("pair_MN"), pair_FP_history
    except Exception as e:
        print(f"cache: ignoring unreadable entry {path} ({type(e).__name__}: {e})")
        return None


def save_fit(cache_dir, algorithm, key, params, result):
    """Write `result` (a fit_trace() return tuple) to the cache, returning the
    entry's directory. Built in a temp directory and renamed into place, so an
    interrupted write can't leave a partial entry a later run reads as a hit."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_dir(cache_dir, algorithm, key)
    trace, pair_neighbors, pair_MN, pair_FP_history = result

    arrays = {
        "trace": trace,
        "pair_neighbors": pair_neighbors,
        "pair_MN": pair_MN,
        "fp_frames": np.array([frame for frame, _ in pair_FP_history], dtype=np.int64),
    }
    sources = _shared_source_column(pair_FP_history)
    if sources is None:
        # One array per checkpoint: nothing guarantees every resample yields
        # the same far pairs, so a history that broke the shared-source
        # invariant is stored verbatim.
        arrays.update({f"fp_{i}": arr for i, (_frame, arr) in enumerate(pair_FP_history)})
    else:
        arrays["fp_sources"] = sources
        arrays["fp_targets"] = np.stack([arr[:, 1] for _frame, arr in pair_FP_history])

    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()
    try:
        for name, arr in arrays.items():
            np.save(tmp_path / f"{name}.npy", arr)
        (tmp_path / "meta.json").write_text(
            json.dumps({"algorithm": algorithm, **params}, sort_keys=True, indent=2, default=str) + "\n")
        # os.replace() can't swap one non-empty directory for another, so an
        # entry being rewritten (after a corrupt read, say) is cleared first.
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp_path, path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
    return path
