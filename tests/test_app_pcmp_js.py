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
        import {{ parsePcmp, dequantizeFrame }} from {json.dumps(str(APP_DIR / "pcmp.js"))};

        const buf = readFileSync({json.dumps(str(pcmp_path))});
        // Copy into a standalone ArrayBuffer: Node pools Buffer memory, so
        // buf.buffer carries an arbitrary byteOffset that would break the
        // absolute offsets in the header.
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        const parsed = parsePcmp(ab);
        const {{ header, positions, colors }} = parsed;

        console.log(JSON.stringify({{
          header,
          positionsType: positions.constructor.name,
          positionsLength: positions.length,
          colorsLength: colors.length,
          firstColors: Array.from(colors.subarray(0, 3)),
          frame0: Array.from(dequantizeFrame(parsed, 0)),
          lastFrame: Array.from(dequantizeFrame(parsed, header.frames - 1)),
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
    assert seen["positionsType"] == "Uint16Array"
    assert seen["positionsLength"] == 7 * 12 * 2
    assert seen["colorsLength"] == 12 * 3


@requires_node
def test_js_reader_recovers_the_exact_float_values(exported, tmp_path):
    """Catches an endianness or alignment mistake, which a length check alone
    would not."""
    out, inputs = exported
    seen = _run_reader(out, tmp_path)

    _, arrays = cli.pcmp.read_pcmp(out)
    np.testing.assert_allclose(seen["firstColors"], arrays["colors"][0], rtol=1e-6)


@requires_node
def test_js_dequantize_agrees_with_the_python_reference(exported, tmp_path):
    """The decode now exists twice -- quantize_positions' inverse in Python, and
    dequantizeFrame in JS -- so this pins them to each other. A wrong scale or
    an off-by-one on QUANT_MAX would still *look* plausible in the player."""
    out, inputs = exported
    seen = _run_reader(out, tmp_path)

    header, arrays = cli.pcmp.read_pcmp(out)
    expected = cli.pcmp.dequantize_positions(
        arrays["positions"], header["pos_min"], header["pos_extent"])

    np.testing.assert_allclose(seen["frame0"], expected[0].reshape(-1), rtol=1e-6)
    np.testing.assert_allclose(seen["lastFrame"], expected[-1].reshape(-1), rtol=1e-6)
    # And the decode has to land back on the trace it came from.
    np.testing.assert_allclose(seen["frame0"], inputs["trace"][0].reshape(-1), atol=1e-4)


@requires_node
def test_js_reader_still_reads_a_pre_quantization_float32_export(exported, tmp_path):
    """Files exported before positions were quantized carry float32 positions
    and no pos_min/pos_extent; the player must keep opening them rather than
    stranding whatever is already in app/data/."""
    out, inputs = exported
    header, arrays = cli.pcmp.read_pcmp(out)

    legacy = tmp_path / "legacy.pcmp"
    old_header = {k: v for k, v in header.items()
                  if k not in ("pos_min", "pos_extent", "arrays", "version", "payload_offset")}
    cli.pcmp.write_pcmp(legacy, old_header, {
        "positions": np.ascontiguousarray(inputs["trace"][:7], dtype=np.float32),
        "colors": arrays["colors"],
    })

    seen = _run_reader(legacy, tmp_path)
    assert seen["positionsType"] == "Float32Array"
    np.testing.assert_allclose(seen["frame0"], inputs["trace"][0].reshape(-1), rtol=1e-6)


@requires_node
def test_js_reader_rejects_a_corrupt_file(tmp_path):
    bad = tmp_path / "bad.pcmp"
    bad.write_bytes(b"NOPE1" + b"\0" * 32)

    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_reader(bad, tmp_path)
    assert "not a .pcmp file" in exc.value.stderr
