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


def _synthetic_data(n=None, seed=0, n_features=784, n_labels=10):
    """MNIST-shaped data without MNIST: same (X, y, rs) contract every loader
    returns, small enough for a real fit in a test."""
    rs = np.random.RandomState(seed)
    n_points = 60 if n is None else int(n)
    X = rs.rand(n_points, n_features).astype(np.float32)
    y = rs.randint(0, n_labels, size=n_points)
    return X, y, rs


@pytest.fixture
def synthetic_mnist(monkeypatch):
    def fake_load_dataset(spec, n=None, seed=0, color=None):
        X, y, rs = _synthetic_data(n=n, seed=seed)
        return X, y, rs, cli.datasets.dataset_meta(spec, color)

    # load_dataset is called as `load_dataset(...)` inside orchestrate.py, which
    # resolves it as a global in orchestrate's own module namespace - not the
    # entry shim's re-exported copy - so it must be patched there.
    monkeypatch.setattr(cli.orchestrate, "load_dataset", fake_load_dataset)


@pytest.fixture
def synthetic_polis(monkeypatch):
    """A Polis-shaped run with no submodule and no network: a small vote-like
    matrix and integer group ids, loaded through the same seam."""
    def fake_load_dataset(spec, n=None, seed=0, color=None):
        X, y, rs = _synthetic_data(n=n, seed=seed, n_features=25, n_labels=3)
        return np.round(X * 2 - 1).astype(np.float32), y, rs, cli.datasets.dataset_meta(spec, color)

    monkeypatch.setattr(cli.orchestrate, "load_dataset", fake_load_dataset)


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

    out_file = out_dir / "mnist" / "pacmap.mp4"
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

    out_file = out_dir / "mnist" / "pacmap_iter3.png"
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_fit_trace_captures_localmap_fp_history(synthetic_mnist):
    import pacmap

    X, y, rs = _synthetic_data(n=60, seed=0)
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
    X, y, rs = _synthetic_data(n=60, seed=0)

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
    X, y, rs = _synthetic_data(n=60, seed=0)

    cli.fit_trace(
        X, "localmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=(2, 2, 2), seed=0, low_dist_thres=3.5,
    )

    assert reducer_kwargs_spy["LocalMAP"]["low_dist_thres"] == 3.5


def test_fit_trace_defaults_low_dist_thres_to_pacmaps_own_default(synthetic_mnist, reducer_kwargs_spy):
    X, y, rs = _synthetic_data(n=60, seed=0)

    cli.fit_trace(
        X, "localmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=(2, 2, 2), seed=0,
    )

    assert reducer_kwargs_spy["LocalMAP"]["low_dist_thres"] == 10.0


def test_fit_trace_does_not_pass_low_dist_thres_to_pacmap(synthetic_mnist, reducer_kwargs_spy):
    # PaCMAP.__init__ has no such param - passing it would be a TypeError, so
    # the LocalMAP-only kwarg has to be dropped for the pacmap algorithm.
    X, y, rs = _synthetic_data(n=60, seed=0)

    cli.fit_trace(
        X, "pacmap", n_neighbors=5, mn_ratio=0.5, fp_ratio=2.0,
        num_iters=(2, 2, 2), seed=0, low_dist_thres=3.5,
    )

    assert "low_dist_thres" not in reducer_kwargs_spy["PaCMAP"]


def test_low_dist_thres_changes_the_localmap_far_pair_graph(synthetic_mnist):
    """The knob actually does something: a much tighter acceptance distance
    yields a different resampled far-pair set from the default."""
    X, y, rs = _synthetic_data(n=60, seed=0)
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

    out_file = out_dir / "mnist" / "pacmap_3d_iter3.png"
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

    out_file = out_dir / "mnist" / "localmap_3d_iter3.png"
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

    out_file = out_dir / "mnist" / "pacmap_3d.mp4"
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

    out_file = out_dir / "mnist" / "pacmap.mp4"
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

    out_file = out_dir / "mnist" / "pacmap_3d_iter3.png"
    assert out_file.exists()


def test_fit_trace_n_components_3_produces_3_column_trace(synthetic_mnist):
    X, y, rs = _synthetic_data(n=60, seed=0)
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

    assert (tmp_path / "run2" / "mnist" / "pacmap_iter3.png").exists()


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

    png_3 = out_dir / "mnist" / "pacmap_iter3.png"
    png_5 = out_dir / "mnist" / "pacmap_iter5.png"
    mp4_2_4 = out_dir / "mnist" / "pacmap_iter2-4.mp4"
    for out_file in (png_3, png_5, mp4_2_4):
        assert out_file.exists()
        assert out_file.stat().st_size > 0


# --- dataset -> directory, colour -> filename ---

def _iter_argv(out_dir, *extra):
    return [
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--iter", "3",
        "--output-dir", str(out_dir),
        *extra,
    ]


def test_main_puts_a_polis_run_in_its_own_dataset_directory(tmp_path, synthetic_polis):
    out_dir = tmp_path / "run"
    cli.main(_iter_argv(out_dir, "--dataset", "polis:35bmpjr8um"))

    assert (out_dir / "polis-35bmpjr8um" / "pacmap_iter3.png").exists()


def test_main_dataset_directory_nests_above_the_param_tag(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    cli.main(_iter_argv(out_dir, "--tag-output"))

    tag_dirs = [p for p in (out_dir / "mnist").iterdir() if p.is_dir()]
    assert len(tag_dirs) == 1
    assert (tag_dirs[0] / "pacmap_iter3.png").exists()


def test_main_default_color_leaves_the_filename_unmarked(tmp_path, synthetic_polis):
    out_dir = tmp_path / "run"
    cli.main(_iter_argv(out_dir, "--dataset", "polis:abc", "--color", "polis:group-id"))

    assert (out_dir / "polis-abc" / "pacmap_iter3.png").exists()


def test_main_non_default_color_marks_the_filename(tmp_path, synthetic_polis):
    out_dir = tmp_path / "run"
    cli.main(_iter_argv(out_dir, "--dataset", "polis:abc", "--color", "polis:n-votes"))

    assert (out_dir / "polis-abc" / "pacmap_colorn-votes_iter3.png").exists()


def test_main_rejects_focus_label_against_a_continuous_color(tmp_path, synthetic_polis):
    # --focus-label compares y for equality, which is meaningless against a
    # magnitude - it must say so rather than silently matching a float.
    argv = _iter_argv(tmp_path / "run", "--dataset", "polis:abc",
                      "--color", "polis:n-votes", "--focus-label", "1")
    with pytest.raises(ValueError, match="focus-label"):
        cli.main(argv)


def test_main_focus_label_still_works_with_a_categorical_color(tmp_path, synthetic_polis):
    out_dir = tmp_path / "run"
    cli.main(_iter_argv(out_dir, "--dataset", "polis:abc", "--focus-label", "1"))

    assert next((out_dir / "polis-abc").glob("pacmap*.png"), None) is not None


def test_main_records_the_dataset_in_the_cache_entry_metadata(tmp_path, synthetic_polis):
    import json

    cache_dir = tmp_path / "fits"
    cli.main(_iter_argv(tmp_path / "run", "--dataset", "polis:abc", "--cache-dir", str(cache_dir)))

    meta = json.loads(next(cache_dir.glob("pacmap_*/meta.json")).read_text())
    assert meta["dataset"] == "polis:abc"


# --- ogl web export ---

def test_main_ogl_export_round_trips_through_a_real_fit(tmp_path, synthetic_mnist):
    """The one end-to-end pass over the ogl backend: real fit, real trace,
    real .pcmp read back. No optional deps and no GPU, so unlike the
    fastplotlib backend this runs in CI."""
    out_dir = tmp_path / "run"
    cli.main([
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--renderer", "ogl",
        "--output-dir", str(out_dir),
    ])

    out_file = out_dir / "mnist" / "pacmap_ogl.pcmp"
    assert out_file.exists() and out_file.stat().st_size > 0

    header, arrays = cli.pcmp.read_pcmp(out_file)
    assert header["frames"] == 7          # sum(num_iters) + 1, at the default --step 1
    assert header["points"] == 60
    assert header["dims"] == 2
    assert header["num_iters"] == [2, 2, 2]
    assert arrays["positions"].shape == (7, 60, 2)
    assert arrays["colors"].shape == (60, 3)


def test_main_ogl_export_honors_step(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    cli.main([
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--step", "3",
        "--renderer", "ogl",
        "--output-dir", str(out_dir),
    ])

    header, _ = cli.pcmp.read_pcmp(out_dir / "mnist" / "pacmap_ogl.pcmp")
    assert header["iters"] == [0, 3, 6]


def test_main_ogl_export_carries_the_datasets_colormap(tmp_path, synthetic_polis):
    """cmap is threaded from dataset_meta through run_algorithm, so a
    continuous color scheme must reach the exported header."""
    out_dir = tmp_path / "run"
    cli.main([
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--dataset", "polis:abc",
        "--color", "polis:n-votes",
        "--renderer", "ogl",
        "--output-dir", str(out_dir),
    ])

    header, _ = cli.pcmp.read_pcmp(out_dir / "polis-abc" / "pacmap_ogl_colorn-votes.pcmp")
    assert header["cmap"] == "viridis"


def test_main_ogl_export_supports_3d(tmp_path, synthetic_mnist):
    out_dir = tmp_path / "run"
    cli.main([
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-components", "3",
        "--renderer", "ogl",
        "--output-dir", str(out_dir),
    ])

    header, arrays = cli.pcmp.read_pcmp(out_dir / "mnist" / "pacmap_3d_ogl.pcmp")
    assert header["dims"] == 3
    assert arrays["positions"].shape[2] == 3
