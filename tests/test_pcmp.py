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


def test_rejects_non_float32_rather_than_casting(tmp_path):
    path = tmp_path / "out.pcmp"
    with pytest.raises(ValueError, match="float32"):
        pcmp.write_pcmp(path, {}, {"positions": np.arange(6, dtype=np.float64)})


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
