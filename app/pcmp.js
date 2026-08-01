// Reader for the .pcmp container written by `--renderer ogl`.
// See pacmap_cli/pcmp.py for the authoritative byte layout:
//
//   b"PCMP1"   5 bytes, magic
//   u32 LE     header length, including its zero padding
//   <header>   UTF-8 JSON, zero-padded so the payload starts 4-byte aligned
//   <payload>  little-endian arrays back to back, each padded to 4 bytes
//
// Kept free of any import so it can be exercised outside a browser (see
// tests/test_app_pcmp_js.py, which runs it under node against a real export).

const MAGIC = 'PCMP1';

/** Payload dtypes, by the name the header uses. */
const VIEWS = { float32: Float32Array, uint16: Uint16Array };

/** Quantized positions span the full uint16 range. Must match pcmp.py's
 *  QUANT_MAX -- and 65535 is also what a normalized UNSIGNED_SHORT vertex
 *  attribute divides by, which is what makes the GPU decode free. */
export const QUANT_MAX = 65535;

/**
 * Parse a .pcmp ArrayBuffer.
 * @returns {{header: object, positions: Float32Array, colors: Float32Array}}
 */
export function parsePcmp(buffer) {
  if (buffer.byteLength < 9) throw new Error('truncated .pcmp: shorter than its preamble');

  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 5));
  if (magic !== MAGIC) throw new Error(`not a .pcmp file (magic ${JSON.stringify(magic)})`);

  const headerLen = new DataView(buffer).getUint32(5, true);
  if (buffer.byteLength < 9 + headerLen) throw new Error('truncated .pcmp: header is incomplete');

  const raw = new TextDecoder().decode(bytes.subarray(9, 9 + headerLen));
  // The header is zero-padded for payload alignment; JSON.parse chokes on the
  // trailing NULs.
  const header = JSON.parse(raw.replace(/\0+$/, ''));

  const out = { header };
  for (const [name, spec] of Object.entries(header.arrays)) {
    const View = VIEWS[spec.dtype];
    if (!View) throw new Error(`${name}: unsupported dtype ${spec.dtype}`);
    const start = header.payload_offset + spec.offset;
    if (buffer.byteLength < start + spec.bytes) {
      throw new Error(`truncated .pcmp: array "${name}" is incomplete`);
    }
    // Every array starts 4-byte aligned by construction (the writer pads),
    // which is what a typed-array view over the buffer requires.
    out[name] = new View(buffer, start, spec.bytes / View.BYTES_PER_ELEMENT);
  }
  return out;
}

/**
 * Decode frame `f`'s positions to world space as a Float32Array of
 * `points * dims` values.
 *
 * Positions ship as uint16 against a per-frame, per-axis range (halving the
 * file), so reading them on the CPU -- which the player only does to depth
 * sort -- means undoing that. The GPU gets the same decode for free from a
 * normalized attribute plus the scale/offset uniforms, so this is not on the
 * drawing path.
 *
 * Exports predating quantization carry float32 positions and no pos_min, and
 * are returned as a view rather than a copy.
 */
export function dequantizeFrame(parsed, f) {
  const { header, positions } = parsed;
  const stride = header.points * header.dims;
  const slice = positions.subarray(f * stride, (f + 1) * stride);
  if (!header.pos_min) return slice;

  const min = header.pos_min[f];
  const extent = header.pos_extent[f];
  const out = new Float32Array(stride);
  for (let i = 0; i < stride; i++) {
    const axis = i % header.dims;
    out[i] = min[axis] + (slice[i] / QUANT_MAX) * extent[axis];
  }
  return out;
}
