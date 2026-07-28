"""End-to-end smoke test: does main() actually wire load -> fit -> render
together and produce a real output file? Real pacmap fit and real ffmpeg
render, kept tiny (small n, short num_iters) so it stays fast. This is not a
substitute for the unit tests above - it exists to catch wiring mistakes
(wrong function imported into the wrong module, args dropped on the way
through) that only show up when the whole pipeline runs together, which
matters most once the file gets split into multiple modules.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from _loader import cli


@pytest.fixture
def synthetic_mnist(monkeypatch):
    def fake_load_mnist(n=None, seed=0):
        rs = np.random.RandomState(seed)
        n_points = 60 if n is None else int(n)
        X = rs.rand(n_points, 784).astype(np.float32)
        y = rs.randint(0, 10, size=n_points)
        return X, y, rs

    # load_mnist is called as `load_mnist(...)` inside orchestrate.py, which
    # resolves it as a global in orchestrate's own module namespace - not the
    # entry shim's re-exported copy - so it must be patched there.
    monkeypatch.setattr(cli.orchestrate, "load_mnist", fake_load_mnist)


def test_main_renders_mp4_for_pacmap(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    out_file = out_dir / "pacmap_mnist.mp4"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_main_renders_single_frame_png_for_iter(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--iter", "3",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    out_file = out_dir / "pacmap_mnist_iter3.png"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_fit_trace_captures_localmap_fp_history(synthetic_mnist):
    import pacmap

    X, y, rs = cli.orchestrate.load_mnist(n=60, seed=0)
    original_sample_fn = pacmap.pacmap.sample_FP_pair_nearby
    num_iters = (5, 5, 25)

    trace, pair_neighbors, pair_MN, pair_FP_history = cli.fit_trace(
        X, "localmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=num_iters, seed=0,
    )

    # one checkpoint for the initial pair_FP plus one per resample event
    assert isinstance(pair_FP_history, list)
    assert len(pair_FP_history) == len(cli.fp_resample_iterations(num_iters)) + 1
    frames = [f for f, _arr in pair_FP_history]
    assert frames == sorted(frames)
    assert frames[0] == 0
    assert all(arr.shape == pair_FP_history[0][1].shape for _f, arr in pair_FP_history)
    # distinct resample events actually produced different pair sets
    assert not np.array_equal(pair_FP_history[0][1], pair_FP_history[-1][1])
    assert pacmap.pacmap.sample_FP_pair_nearby is original_sample_fn


def test_fit_trace_pacmap_has_single_checkpoint_fp_history(synthetic_mnist):
    X, y, rs = cli.orchestrate.load_mnist(n=60, seed=0)

    trace, pair_neighbors, pair_MN, pair_FP_history = cli.fit_trace(
        X, "pacmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=(2, 2, 2), seed=0,
    )

    assert isinstance(pair_FP_history, list)
    assert len(pair_FP_history) == 1
    assert pair_FP_history[0][0] == 0


@pytest.fixture
def reducer_kwargs_spy(monkeypatch):
    """Records the kwargs fit_trace() constructs each reducer with, while
    still running the real fit underneath."""
    import pacmap

    seen = {}

    def spy(name):
        real = getattr(pacmap, name)

        def wrapper(**kwargs):
            seen[name] = kwargs
            return real(**kwargs)

        monkeypatch.setattr(pacmap, name, wrapper)

    spy("PaCMAP")
    spy("LocalMAP")
    return seen


def test_fit_trace_passes_low_dist_thres_to_localmap(synthetic_mnist, reducer_kwargs_spy):
    X, y, rs = cli.orchestrate.load_mnist(n=60, seed=0)

    cli.fit_trace(
        X, "localmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=(2, 2, 2), seed=0, low_dist_thres=3.5,
    )

    assert reducer_kwargs_spy["LocalMAP"]["low_dist_thres"] == 3.5


def test_fit_trace_defaults_low_dist_thres_to_pacmaps_own_default(synthetic_mnist, reducer_kwargs_spy):
    X, y, rs = cli.orchestrate.load_mnist(n=60, seed=0)

    cli.fit_trace(
        X, "localmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=(2, 2, 2), seed=0,
    )

    assert reducer_kwargs_spy["LocalMAP"]["low_dist_thres"] == 10.0


def test_fit_trace_does_not_pass_low_dist_thres_to_pacmap(synthetic_mnist, reducer_kwargs_spy):
    # PaCMAP.__init__ has no such param - passing it would be a TypeError, so
    # the LocalMAP-only kwarg has to be dropped for the pacmap algorithm.
    X, y, rs = cli.orchestrate.load_mnist(n=60, seed=0)

    cli.fit_trace(
        X, "pacmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=(2, 2, 2), seed=0, low_dist_thres=3.5,
    )

    assert "low_dist_thres" not in reducer_kwargs_spy["PaCMAP"]


def test_low_dist_thres_changes_the_localmap_far_pair_graph(synthetic_mnist):
    """The knob actually does something: a much tighter acceptance distance
    yields a different resampled far-pair set from the default."""
    X, y, rs = cli.orchestrate.load_mnist(n=60, seed=0)
    kwargs = dict(n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0, num_iters=(5, 5, 25), seed=0)

    *_, default_history = cli.fit_trace(X, "localmap", **kwargs, low_dist_thres=10.0)
    *_, tight_history = cli.fit_trace(X, "localmap", **kwargs, low_dist_thres=0.01)

    assert not np.array_equal(default_history[-1][1], tight_history[-1][1])


def test_main_renders_png_for_n_components_3(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--n-components", "3",
        "--iter", "3",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    out_file = out_dir / "pacmap_mnist_3d_iter3.png"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_main_renders_png_for_n_components_3_with_v3_edges(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "localmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--n-components", "3",
        "--edge-style-preset", "v3",
        "--iter", "3",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    out_file = out_dir / "localmap_mnist_3d_iter3.png"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_main_n_components_3_filename_gets_3d_marker(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--n-components", "3",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    out_file = out_dir / "pacmap_mnist_3d.mp4"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_main_n_components_2_filename_unchanged(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    out_file = out_dir / "pacmap_mnist.mp4"
    assert out_file.exists()


def test_main_n_components_3_iter_png_gets_3d_marker(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--n-components", "3",
        "--iter", "3",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    out_file = out_dir / "pacmap_mnist_3d_iter3.png"
    assert out_file.exists()


def test_fit_trace_n_components_3_produces_3_column_trace(synthetic_mnist):
    X, y, rs = cli.orchestrate.load_mnist(n=60, seed=0)
    num_iters = (2, 2, 2)

    trace, pair_neighbors, pair_MN, pair_FP_history = cli.fit_trace(
        X, "pacmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=num_iters, seed=0, n_components=3,
    )

    assert trace.shape == (sum(num_iters) + 1, 60, 3)


def _cache_argv(out_dir, cache_dir, *extra):
    return [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--iter", "3",
        "--output-dir", str(out_dir),
        *(["--cache-dir", str(cache_dir)] if cache_dir else []),
        *extra,
    ]


def test_main_caches_the_fit_by_default(tmp_path, synthetic_mnist):
    cache_dir = tmp_path / "fits"
    cli.main(_cache_argv(tmp_path / "run", cache_dir))

    assert len(list(cache_dir.glob("pacmap_*/trace.npy"))) == 1


def test_main_default_cache_dir_is_dot_cache_fits(tmp_path, synthetic_mnist):
    # conftest's isolate_cwd fixture puts us in a scratch cwd, so the real
    # relative default is exercised without touching the repo.
    from pathlib import Path

    cli.main(_cache_argv(tmp_path / "run", None))

    assert len(list(Path(".cache/fits").glob("pacmap_*/trace.npy"))) == 1


def test_main_second_identical_run_does_not_refit(tmp_path, synthetic_mnist, monkeypatch):
    cache_dir = tmp_path / "fits"
    cli.main(_cache_argv(tmp_path / "run", cache_dir))

    def boom(*args, **kwargs):
        raise AssertionError("refit despite a warm cache")

    monkeypatch.setattr(cli.fit, "_fit_uncached", boom)
    cli.main(_cache_argv(tmp_path / "run2", cache_dir))

    assert (tmp_path / "run2" / "pacmap_mnist_iter3.png").exists()


def test_main_no_cache_flag_writes_no_cache_entry(tmp_path, synthetic_mnist):
    cache_dir = tmp_path / "fits"
    cli.main(_cache_argv(tmp_path / "run", cache_dir, "--no-cache"))

    assert not cache_dir.exists()


def test_main_no_cache_flag_still_refits_with_a_warm_cache(tmp_path, synthetic_mnist, monkeypatch):
    cache_dir = tmp_path / "fits"
    cli.main(_cache_argv(tmp_path / "run", cache_dir))

    calls = []
    real_fit = cli.fit._fit_uncached
    monkeypatch.setattr(cli.fit, "_fit_uncached",
                        lambda *a, **kw: (calls.append(1), real_fit(*a, **kw))[1])
    cli.main(_cache_argv(tmp_path / "run2", cache_dir, "--no-cache"))

    assert len(calls) == 1


def test_main_changing_a_fit_param_adds_a_second_cache_entry(tmp_path, synthetic_mnist):
    cache_dir = tmp_path / "fits"
    cli.main(_cache_argv(tmp_path / "run", cache_dir))
    cli.main(_cache_argv(tmp_path / "run2", cache_dir, "--seed", "7"))

    assert len(list(cache_dir.glob("pacmap_*/trace.npy"))) == 2


def test_main_low_dist_thres_reaches_the_localmap_fit(tmp_path, synthetic_mnist, reducer_kwargs_spy):
    argv = _cache_argv(tmp_path / "run", tmp_path / "fits", "--low-dist-thres", "2.5")
    argv[argv.index("pacmap")] = "localmap"
    cli.main(argv)

    assert reducer_kwargs_spy["LocalMAP"]["low_dist_thres"] == 2.5


def test_main_renders_multiple_outputs_for_comma_separated_iter(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    argv = [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--iter", "3,5,2-4",
        "--output-dir", str(out_dir),
    ]
    cli.main(argv)

    png_3 = out_dir / "pacmap_mnist_iter3.png"
    png_5 = out_dir / "pacmap_mnist_iter5.png"
    mp4_2_4 = out_dir / "pacmap_mnist_iter2-4.mp4"
    for out_file in (png_3, png_5, mp4_2_4):
        assert out_file.exists()
        assert out_file.stat().st_size > 0
