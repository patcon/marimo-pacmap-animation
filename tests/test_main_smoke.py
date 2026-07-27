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

    monkeypatch.setattr(cli, "load_mnist", fake_load_mnist)


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
