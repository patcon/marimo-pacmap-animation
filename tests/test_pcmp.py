"""The .pcmp container: a JSON header plus a little-endian float32 payload in
one file, written by --renderer ogl and read by app/.

These tests are the spec the JS reader in app/app.js is written against, so
they assert the byte layout directly rather than only round-tripping through
this module's own reader.
"""
import json
import struct

import numpy as np
import pytest

from _loader import cli

pcmp = cli.pcmp


def _arrays():
    return {
        "positions": np.arange(24, dtype=np.float32).reshape(4, 3, 2),
        "colors": np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3),
    }


def test_round_trips_header_and_arrays(tmp_path):
    path = tmp_path / "out.pcmp"
    arrays = _arrays()
    pcmp.write_pcmp(path, {"dataset": "mnist", "frames": 4}, arrays)

    header, out = pcmp.read_pcmp(path)
    assert header["dataset"] == "mnist"
    assert header["frames"] == 4
    for name, arr in arrays.items():
        assert out[name].dtype == np.float32
        assert out[name].shape == arr.shape
        np.testing.assert_array_equal(out[name], arr)


def test_header_describes_each_arrays_shape_offset_and_bytes(tmp_path):
    path = tmp_path / "out.pcmp"
    arrays = _arrays()
    pcmp.write_pcmp(path, {}, arrays)

    header, _ = pcmp.read_pcmp(path)
    positions = header["arrays"]["positions"]
    colors = header["arrays"]["colors"]
    assert positions["dtype"] == "float32"
    assert positions["shape"] == [4, 3, 2]
    assert positions["offset"] == 0
    assert positions["bytes"] == 24 * 4
    # Arrays are laid back to back in insertion order.
    assert colors["offset"] == positions["bytes"]
    assert colors["bytes"] == 9 * 4


def test_magic_and_header_length_are_readable_without_this_module(tmp_path):
    """app/app.js parses these first 9 bytes by hand."""
    path = tmp_path / "out.pcmp"
    pcmp.write_pcmp(path, {"dataset": "mnist"}, _arrays())

    raw = path.read_bytes()
    assert raw[:5] == b"PCMP1"
    (header_len,) = struct.unpack("<I", raw[5:9])
    header = json.loads(raw[9:9 + header_len].rstrip(b"\0").decode("utf-8"))
    assert header["dataset"] == "mnist"


def test_payload_starts_4_byte_aligned(tmp_path):
    """Float32Array views on the ArrayBuffer need a byteOffset that is a
    multiple of 4, so the header is padded rather than packed tight."""
    for dataset in ["a", "ab", "abc", "abcd"]:  # vary the header length mod 4
        path = tmp_path / f"{dataset}.pcmp"
        pcmp.write_pcmp(path, {"dataset": dataset}, _arrays())
        header, _ = pcmp.read_pcmp(path)
        assert header["payload_offset"] % 4 == 0


def test_rejects_dtypes_outside_the_supported_set_rather_than_casting(tmp_path):
    path = tmp_path / "out.pcmp"
    with pytest.raises(ValueError, match="float64"):
        pcmp.write_pcmp(path, {}, {"positions": np.arange(6, dtype=np.float64)})


def test_round_trips_a_uint16_array(tmp_path):
    """Quantized positions ride in the payload as uint16; the container has to
    carry them as faithfully as it carries float32."""
    path = tmp_path / "out.pcmp"
    q = np.array([[0, 1], [65535, 32768]], dtype=np.uint16)
    pcmp.write_pcmp(path, {}, {"positions": q, "colors": np.zeros((1, 3), np.float32)})

    header, out = pcmp.read_pcmp(path)
    assert header["arrays"]["positions"]["dtype"] == "uint16"
    assert header["arrays"]["positions"]["bytes"] == 4 * 2
    assert out["positions"].dtype == np.uint16
    np.testing.assert_array_equal(out["positions"], q)


def test_every_array_starts_4_byte_aligned_even_after_an_odd_uint16_one(tmp_path):
    """`new Float32Array(buf, byteOffset, n)` throws unless byteOffset is a
    multiple of 4. An odd number of uint16s is an odd number of pairs of bytes,
    so without padding the array after it would land 2 mod 4 -- which happens
    for real whenever frames*points*dims is odd."""
    path = tmp_path / "out.pcmp"
    pcmp.write_pcmp(path, {}, {
        "positions": np.zeros((1, 5, 3), np.uint16),  # 15 elements -> 30 bytes
        "colors": np.zeros((5, 3), np.float32),
    })

    header, out = pcmp.read_pcmp(path)
    for spec in header["arrays"].values():
        assert (header["payload_offset"] + spec["offset"]) % 4 == 0
    np.testing.assert_array_equal(out["colors"], np.zeros((5, 3), np.float32))


def test_mixed_dtypes_are_laid_back_to_back_by_their_own_itemsize(tmp_path):
    """A uint16 array is 2 bytes per element, so the array after it starts at
    a different offset than a float32 array of the same shape would imply."""
    path = tmp_path / "out.pcmp"
    pcmp.write_pcmp(path, {}, {
        "positions": np.zeros((4, 3, 2), np.uint16),
        "colors": np.zeros((3, 3), np.float32),
    })

    header, out = pcmp.read_pcmp(path)
    assert header["arrays"]["positions"]["bytes"] == 24 * 2
    assert header["arrays"]["colors"]["offset"] == 24 * 2
    assert out["colors"].dtype == np.float32


def test_writes_little_endian_regardless_of_platform(tmp_path):
    path = tmp_path / "out.pcmp"
    pcmp.write_pcmp(path, {}, {"positions": np.array([1.0], dtype=np.float32)})

    header, _ = pcmp.read_pcmp(path)
    raw = path.read_bytes()[header["payload_offset"]:]
    assert raw[:4] == struct.pack("<f", 1.0)


def test_truncated_payload_raises(tmp_path):
    path = tmp_path / "out.pcmp"
    pcmp.write_pcmp(path, {}, _arrays())
    raw = path.read_bytes()
    path.write_bytes(raw[:-8])

    with pytest.raises(ValueError, match="truncated"):
        pcmp.read_pcmp(path)


def test_wrong_magic_raises(tmp_path):
    path = tmp_path / "out.pcmp"
    pcmp.write_pcmp(path, {}, _arrays())
    raw = bytearray(path.read_bytes())
    raw[:5] = b"XXXX1"
    path.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="magic"):
        pcmp.read_pcmp(path)


# --- position quantization -------------------------------------------------
#
# Positions are the whole file (a full-MNIST 3D trace is ~380 MB as float32),
# so they ship as uint16 against a per-frame, per-axis range. These tests are
# the spec app/pcmp.js and the player's vertex shader decode against.


def test_quantized_positions_round_trip_within_one_step(tmp_path):
    rs = np.random.RandomState(0)
    positions = rs.uniform(-5, 5, size=(3, 50, 2)).astype(np.float32)

    q, mins, extents = pcmp.quantize_positions(positions)
    back = pcmp.dequantize_positions(q, mins, extents)

    assert q.dtype == np.uint16
    assert q.shape == positions.shape
    # One step is the frame's extent spread over the 65535 intervals uint16
    # offers, so the error bound is a property of the data, not a magic number.
    step = extents.max() / 65535
    assert np.abs(back - positions).max() <= step


def test_quantization_preserves_each_frames_extremes_exactly(tmp_path):
    """The endpoints anchor the range, so they must not drift -- otherwise the
    embedding would visibly shrink or grow by a fraction of a step."""
    positions = np.array([[[-3.0, 1.0], [7.0, 4.0]]], dtype=np.float32)

    q, mins, extents = pcmp.quantize_positions(positions)
    back = pcmp.dequantize_positions(q, mins, extents)

    np.testing.assert_allclose(back.min(axis=1), positions.min(axis=1), rtol=1e-6)
    np.testing.assert_allclose(back.max(axis=1), positions.max(axis=1), rtol=1e-6)


def test_each_frame_gets_its_own_range_so_early_frames_keep_precision(tmp_path):
    """The whole point of quantizing per frame: the embedding expands ~30x over
    a run, and the camera zooms in on the early frames, so a single global
    range would spend almost all its resolution on the final frame."""
    tiny = np.linspace(0, 0.001, 40, dtype=np.float32).reshape(1, 40, 1)
    huge = np.linspace(0, 1000.0, 40, dtype=np.float32).reshape(1, 40, 1)
    positions = np.concatenate([tiny, huge], axis=0)

    q, mins, extents = pcmp.quantize_positions(positions)
    back = pcmp.dequantize_positions(q, mins, extents)

    tiny_error = np.abs(back[0] - positions[0]).max()
    # Precision tracks each frame's own extent. Against one global range the
    # tiny frame's error would be ~1000/65535, a million times coarser.
    assert tiny_error <= 0.001 / 65535


def test_a_degenerate_axis_round_trips_exactly_rather_than_dividing_by_zero(tmp_path):
    """A frame whose points share an x (or a 2D trace viewed as flat) has zero
    extent on that axis; the encoding must not produce NaN."""
    positions = np.array([[[2.0, 0.0], [2.0, 5.0]]], dtype=np.float32)

    q, mins, extents = pcmp.quantize_positions(positions)
    back = pcmp.dequantize_positions(q, mins, extents)

    assert np.isfinite(back).all()
    np.testing.assert_allclose(back[:, :, 0], positions[:, :, 0], rtol=1e-6)
