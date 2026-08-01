"""The .pcmp container -- one self-describing file holding an embedding trace.

Written by the `ogl` renderer backend (render_ogl.py) and read by the static
WebGL player in app/. The layout is deliberately trivial to parse from
JavaScript with a DataView and a few typed-array views, so it is documented
here as the byte layout rather than as this module's API:

    b"PCMP1"    5 bytes, magic
    u32 LE      header length in bytes, including its zero padding
    <header>    UTF-8 JSON, zero-padded so the payload starts 4-byte aligned
    <payload>   little-endian arrays back to back in header order

Everything small enough to be cheap as JSON (per-frame weights, camera path,
iteration indices) lives in the header; only the bulk arrays -- positions and
baked per-point colors -- go in the payload. `header["arrays"]` maps each name
to {dtype, shape, offset, bytes}, where `offset` is relative to
`header["payload_offset"]` and `dtype` is one of DTYPES.

Two constraints exist for the JS reader's sake, and are enforced rather than
assumed:

- The payload is 4-byte aligned, because `new Float32Array(buf, byteOffset, n)`
  throws on an unaligned offset.
- Arrays must already be one of the supported dtypes. Casting silently would
  let a float64 trace double the file size without anyone noticing, so it is
  an error instead.

Positions specifically are written as uint16 against a per-frame, per-axis
range (see quantize_positions): they are essentially the whole file, and
float32 puts a full-MNIST 3D trace at ~380 MB. The decode is a scale and an
offset, which the player gets for free from a normalized vertex attribute.
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

# Every dtype the payload may hold, mapped to its explicit little-endian numpy
# code. The JS reader keys off the same names.
DTYPES = {"float32": "<f4", "uint16": "<u2"}

# Quantized positions span the full uint16 range, so a value is q/QUANT_MAX of
# the way across its frame's extent -- which is exactly what WebGL's normalized
# UNSIGNED_SHORT attribute hands the vertex shader, at no cost.
QUANT_MAX = 65535


def write_pcmp(path, header, arrays):
    """Write `arrays` (name -> ndarray of a DTYPES dtype) with `header` metadata.

    `header` is copied and extended with `version`, `payload_offset` and the
    `arrays` table; callers own every other key.
    """
    described = {}
    offset = 0
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        dtype = _dtype_name(arr.dtype)
        if dtype is None:
            raise ValueError(
                f"array {name!r} has dtype {arr.dtype}, expected one of "
                f"{sorted(DTYPES)} -- cast it deliberately at the call site "
                f"rather than silently here")
        described[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "offset": offset,
            "bytes": arr.nbytes,
        }
        # Pad to the next 4-byte boundary rather than packing tight: a typed
        # array view needs an offset that is a multiple of its itemsize, and an
        # odd-length uint16 array would otherwise leave the next one at 2 mod 4
        # (which happens whenever frames*points*dims is odd).
        offset += arr.nbytes + (-arr.nbytes % 4)

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
            code = DTYPES[described[name]["dtype"]]
            payload = np.ascontiguousarray(arr, dtype=code).tobytes()
            fh.write(payload)
            fh.write(b"\0" * (-len(payload) % 4))  # the alignment pad, see above
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
        code = DTYPES.get(spec["dtype"])
        if code is None:
            raise ValueError(f"array {name!r} has unsupported dtype {spec['dtype']!r}")
        arrays[name] = np.frombuffer(raw[start:end], dtype=code).reshape(spec["shape"])
    return header, arrays


def quantize_positions(positions):
    """Encode `positions` (frames, points, dims) as uint16 against each frame's
    own per-axis range.

    Returns `(q, mins, extents)`, where `mins`/`extents` are (frames, dims) and
    a value decodes as `mins + q / QUANT_MAX * extents`.

    Per frame rather than once globally because the embedding expands ~30x over
    a run *and* the camera zooms in on the early frames: a single range would
    spend nearly all its resolution on the final frame, exactly where it is
    least needed. Per frame, 65536 levels across the frame's own extent is ~30x
    finer than a screen pixel at the default framing.
    """
    positions = np.asarray(positions, dtype=np.float32)
    mins = positions.min(axis=1)
    extents = positions.max(axis=1) - mins

    # An axis where every point coincides has zero extent; dividing would give
    # NaN, and every value there is `min` anyway.
    divisor = np.where(extents > 0, extents, 1.0)
    scaled = (positions - mins[:, None, :]) / divisor[:, None, :]
    q = np.rint(scaled * QUANT_MAX).astype(np.uint16)
    return q, mins.astype(np.float32), extents.astype(np.float32)


def dequantize_positions(q, mins, extents):
    """Inverse of quantize_positions. The player does this in the vertex shader
    (and, for depth sorting, in app/pcmp.js); this exists for tests and
    debugging, and as the reference both are checked against."""
    q = np.asarray(q, dtype=np.float32) / QUANT_MAX
    return (np.asarray(mins)[:, None, :] + q * np.asarray(extents)[:, None, :]).astype(np.float32)


def _dtype_name(dtype):
    for name, code in DTYPES.items():
        if dtype == np.dtype(code):
            return name
    return None


def _encode_header(header):
    # separators drop JSON's default spaces: at ~450 frames of weights and
    # camera path this is a meaningful fraction of the header.
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def _padded_header_len(length):
    """Round up so the payload lands on a 4-byte boundary."""
    total = _PREAMBLE_BYTES + length
    return length + (-total % 4)
