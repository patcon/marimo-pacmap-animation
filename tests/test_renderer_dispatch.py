"""Tests for the --renderer flag and the renderer-backend dispatch registry."""

import numpy as np
import pytest

from _loader import cli


# --- config / flag parsing ---

def test_build_config_renderer_defaults_to_matplotlib():
    args = cli.parse_args([])
    cfg = cli.build_config(args)
    assert cfg["renderer"] == "matplotlib"


def test_build_config_renderer_cli_flag_overrides_default():
    args = cli.parse_args(["--renderer", "fastplotlib"])
    cfg = cli.build_config(args)
    assert cfg["renderer"] == "fastplotlib"


def test_parse_args_renderer_rejects_unknown_choice():
    with pytest.raises(SystemExit):
        cli.parse_args(["--renderer", "gnuplot"])


def test_param_tag_excludes_renderer():
    # renderer is disambiguated via a filename marker (like n_components'
    # _3d), not a --tag-output slug entry.
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["renderer"] = "fastplotlib"
    assert cli.param_tag(cfg) == "default"


# --- dispatch registry ---

def test_renderers_registry_has_matplotlib_and_fastplotlib():
    assert set(cli.render.RENDERERS) >= {"matplotlib", "fastplotlib"}


def test_get_backend_unknown_renderer_raises_value_error():
    with pytest.raises(ValueError, match="unknown renderer"):
        cli.render.get_backend("gnuplot")


def test_get_backend_matplotlib_resolves_to_callables():
    backend = cli.render.get_backend("matplotlib")
    assert callable(backend["animation"])
    assert callable(backend["frame"])


def test_render_dispatch_uses_registry_backend(monkeypatch):
    calls = []
    fake_backend = {
        "animation": lambda **kw: calls.append(("animation", kw)) or "anim.mp4",
        "frame": lambda **kw: calls.append(("frame", kw)) or "frame.png",
    }
    monkeypatch.setitem(cli.render.RENDERERS, "fastplotlib", lambda: fake_backend)
    assert cli.render.render_animation(renderer="fastplotlib", out_path="x.mp4") == "anim.mp4"
    assert cli.render.render_frame(renderer="fastplotlib", out_path="x.png", frame=0) == "frame.png"
    assert [name for name, _ in calls] == ["animation", "frame"]


# --- fastplotlib + 3D rejection (until 3D support lands, plan Task 7) ---

def test_run_algorithm_rejects_fastplotlib_with_n_components_3(monkeypatch):
    def fail_if_fit_runs(*args, **kwargs):
        raise AssertionError("fit_trace ran before the renderer/3D validation")

    monkeypatch.setattr(cli.orchestrate, "fit_trace", fail_if_fit_runs)
    cfg = dict(cli.DEFAULT_CONFIG)
    cfg["renderer"] = "fastplotlib"
    cfg["n_components"] = 3
    cfg["num_iters"] = (2, 2, 2)
    with pytest.raises(ValueError, match="fastplotlib"):
        cli.run_algorithm(
            X=np.zeros((10, 4)), y=np.zeros(10, dtype=int),
            rs=np.random.RandomState(0), algorithm="pacmap", cfg=cfg,
            iter_out_paths=[(None, "out.mp4")],
        )


# --- filename marker ---

def test_main_fastplotlib_filename_gets_fpl_marker(tmp_path, monkeypatch):
    def fake_load_mnist(n=None, seed=0):
        rs = np.random.RandomState(seed)
        n_points = 60 if n is None else int(n)
        X = rs.rand(n_points, 784).astype(np.float32)
        y = rs.randint(0, 10, size=n_points)
        return X, y, rs

    monkeypatch.setattr(cli.orchestrate, "load_mnist", fake_load_mnist)

    def fake_backend():
        def write(out_path=None, **kwargs):
            with open(out_path, "wb") as f:
                f.write(b"fake")
            return out_path
        return {"animation": write, "frame": write}

    monkeypatch.setitem(cli.render.RENDERERS, "fastplotlib", fake_backend)

    out_dir = tmp_path / "run"
    cli.main([
        "--algorithm", "pacmap",
        "--n", "60",
        "--n-neighbors", "5",
        "--num-iters", "2,2,2",
        "--n-lines", "5",
        "--renderer", "fastplotlib",
        "--output-dir", str(out_dir),
    ])

    assert (out_dir / "pacmap_mnist_fpl.mp4").exists()
