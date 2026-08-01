"""The .pcmp container -- one self-describing file holding an embedding trace.

Written by the `ogl` renderer backend (render_ogl.py) and read by the static
WebGL player in app/. The layout is deliberately trivial to parse from
JavaScript with a DataView and a few typed-array views, so it is documented
here as the byte layout rather than as this module's API:

    b"PCMP1"    5 bytes, magic
    u32 LE      header length in bytes, including its zero padding
    <header>    UTF-8 JSON, zero-padded so the payload starts 4-byte aligned
    <payload>   little-endian float32, arrays back to back in header order

Everything small enough to be cheap as JSON (per-frame weights, camera path,
iteration indices) lives in the header; only the bulk arrays -- positions and
baked per-point colors -- go in the payload. `header["arrays"]` maps each name
to {dtype, shape, offset, bytes}, where `offset` is relative to
`header["payload_offset"]`.

Two constraints exist for the JS reader's sake, and are enforced rather than
assumed:

- The payload is 4-byte aligned, because `new Float32Array(buf, byteOffset, n)`
  throws on an unaligned offset.
- Arrays must already be float32. Casting silently would let a float64 trace
  double the file size without anyone noticing, so it is an error instead.
"""
import json
import struct

import numpy as np

MAGIC = b"PCMP1"
# Version lives in the header rather than the magic so a reader can report a
# useful mismatch instead of "not a .pcmp file".
VERSION = 1

_HEADER_LEN_STRUCT = struct.Struct("<I")
_PREAMBLE_BYTES = len(MAGIC) + _HEADER_LEN_STRUCT.size
_MAX_U32 = 2 ** 32 - 1


def write_pcmp(path, header, arrays):
    """Write `arrays` (name -> float32 ndarray) with `header` metadata.

    `header` is copied and extended with `version`, `payload_offset` and the
    `arrays` table; callers own every other key.
    """
    described = {}
    offset = 0
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.dtype != np.float32:
            raise ValueError(
                f"array {name!r} has dtype {arr.dtype}, expected float32 -- cast it "
                f"deliberately at the call site rather than silently here")
        described[name] = {
            "dtype": "float32",
            "shape": list(arr.shape),
            "offset": offset,
            "bytes": arr.nbytes,
        }
        offset += arr.nbytes

    full = dict(header)
    full["version"] = VERSION
    full["arrays"] = described

    # payload_offset depends on the header's own length, which depends on
    # payload_offset. Encode once with a placeholder to learn the length, then
    # again for real. The placeholder is the widest value the field can take
    # (a 10-digit u32), so the real encoding is never longer and the zero
    # padding absorbs the difference -- no fixed-point loop needed.
    full["payload_offset"] = _MAX_U32
    probe = _encode_header(full)
    padded_len = _padded_header_len(len(probe))
    full["payload_offset"] = _PREAMBLE_BYTES + padded_len

    encoded = _encode_header(full)
    if len(encoded) > padded_len:  # pragma: no cover - defensive
        raise ValueError("header grew when payload_offset was filled in")
    encoded += b"\0" * (padded_len - len(encoded))

    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(_HEADER_LEN_STRUCT.pack(padded_len))
        fh.write(encoded)
        for name, arr in arrays.items():
            fh.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
    return path


def read_pcmp(path):
    """Read a .pcmp back as `(header, arrays)`.

    Exists for tests and debugging; the app/ player is the real consumer and
    implements this independently in JavaScript.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    if raw[:len(MAGIC)] != MAGIC:
        raise ValueError(f"bad magic {raw[:len(MAGIC)]!r}: not a .pcmp file")
    if len(raw) < _PREAMBLE_BYTES:
        raise ValueError("truncated .pcmp: file ends inside the preamble")

    (header_len,) = _HEADER_LEN_STRUCT.unpack(raw[len(MAGIC):_PREAMBLE_BYTES])
    header_end = _PREAMBLE_BYTES + header_len
    if len(raw) < header_end:
        raise ValueError("truncated .pcmp: file ends inside the header")
    header = json.loads(raw[_PREAMBLE_BYTES:header_end].rstrip(b"\0").decode("utf-8"))

    base = header["payload_offset"]
    arrays = {}
    for name, spec in header["arrays"].items():
        start = base + spec["offset"]
        end = start + spec["bytes"]
        if len(raw) < end:
            raise ValueError(f"truncated .pcmp: array {name!r} needs bytes {start}-{end}, "
                             f"file is {len(raw)} bytes")
        arrays[name] = np.frombuffer(raw[start:end], dtype="<f4").reshape(spec["shape"])
    return header, arrays


def _encode_header(header):
    # separators drop JSON's default spaces: at ~450 frames of weights and
    # camera path this is a meaningful fraction of the header.
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def _padded_header_len(length):
    """Round up so the payload lands on a 4-byte boundary."""
    total = _PREAMBLE_BYTES + length
    return length + (-total % 4)
