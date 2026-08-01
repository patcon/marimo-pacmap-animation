"""The .pcmp format has two implementations -- pacmap_cli/pcmp.py writes it,
app/pcmp.js reads it -- and nothing else forces them to agree. This runs the
JavaScript reader under node against a file the Python writer just produced,
so a change to the byte layout on one side fails loudly rather than silently
breaking the player.

Skipped when node is unavailable; node is not a dependency of this project.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _loader import cli
from _synthetic import synthetic_render_inputs

APP_DIR = Path(__file__).resolve().parent.parent / "app"

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")


def _run_reader(pcmp_path, tmp_path):
    """Parse `pcmp_path` with app/pcmp.js and return what it saw as JSON."""
    script = tmp_path / "read.mjs"
    script.write_text(f"""
        import {{ readFileSync }} from 'node:fs';
        import {{ parsePcmp }} from {json.dumps(str(APP_DIR / "pcmp.js"))};

        const buf = readFileSync({json.dumps(str(pcmp_path))});
        // Copy into a standalone ArrayBuffer: Node pools Buffer memory, so
        // buf.buffer carries an arbitrary byteOffset that would break the
        // absolute offsets in the header.
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        const {{ header, positions, colors }} = parsePcmp(ab);

        console.log(JSON.stringify({{
          header,
          positionsLength: positions.length,
          colorsLength: colors.length,
          firstPositions: Array.from(positions.subarray(0, 6)),
          firstColors: Array.from(colors.subarray(0, 3)),
        }}));
    """)
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.fixture
def exported(tmp_path):
    inputs = synthetic_render_inputs(n_points=12, num_iters=(2, 2, 2))
    out = tmp_path / "out.pcmp"
    cli.render_animation(renderer="ogl", out_path=str(out), n_lines=5, step=1,
                         cmap="tab10", **inputs)
    return out, inputs


@requires_node
def test_js_reader_agrees_with_the_python_writer(exported, tmp_path):
    out, inputs = exported
    seen = _run_reader(out, tmp_path)

    assert seen["header"]["frames"] == 7
    assert seen["header"]["points"] == 12
    assert seen["header"]["dims"] == 2
    assert seen["positionsLength"] == 7 * 12 * 2
    assert seen["colorsLength"] == 12 * 3


@requires_node
def test_js_reader_recovers_the_exact_float_values(exported, tmp_path):
    """Catches an endianness or alignment mistake, which a length check alone
    would not."""
    out, inputs = exported
    seen = _run_reader(out, tmp_path)

    expected = inputs["trace"][0].reshape(-1)[:6]
    np.testing.assert_allclose(seen["firstPositions"], expected, rtol=1e-6)

    _, arrays = cli.pcmp.read_pcmp(out)
    np.testing.assert_allclose(seen["firstColors"], arrays["colors"][0], rtol=1e-6)


@requires_node
def test_js_reader_rejects_a_corrupt_file(tmp_path):
    bad = tmp_path / "bad.pcmp"
    bad.write_bytes(b"NOPE1" + b"\0" * 32)

    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_reader(bad, tmp_path)
    assert "not a .pcmp file" in exc.value.stderr
