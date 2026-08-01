// Reader for the .pcmp container written by `--renderer ogl`.
// See pacmap_cli/pcmp.py for the authoritative byte layout:
//
//   b"PCMP1"   5 bytes, magic
//   u32 LE     header length, including its zero padding
//   <header>   UTF-8 JSON, zero-padded so the payload starts 4-byte aligned
//   <payload>  little-endian float32, arrays back to back
//
// Kept free of any import so it can be exercised outside a browser (see
// tests/test_app_pcmp_js.py, which runs it under node against a real export).

const MAGIC = 'PCMP1';

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
    if (spec.dtype !== 'float32') throw new Error(`${name}: unsupported dtype ${spec.dtype}`);
    const start = header.payload_offset + spec.offset;
    if (buffer.byteLength < start + spec.bytes) {
      throw new Error(`truncated .pcmp: array "${name}" is incomplete`);
    }
    // byteOffset is a multiple of 4 by construction, which Float32Array requires.
    out[name] = new Float32Array(buffer, start, spec.bytes / 4);
  }
  return out;
}
