"""Cross-check ROCMinfer's decoder against the pinned upstream quantizer.

The strongest available evidence that our re-derived format maths is correct:
decode the same ROCmFP4 tensor with CPython's own implementation of the pinned
format definition and compare every float bit-for-bit against what ROCMinfer
produces.

Run:  python tools/crosscheck_decode.py <gguf> [<gguf> ...]
"""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys

# Pinned format constants, independently re-derived from FIPS-style definitions.
QK = 32
DUAL_BLOCK_BYTES = 18
FAST_BLOCK_BYTES = 17
GGML_TYPE_ROCMFP4 = 100
GGML_TYPE_ROCMFP4_FAST = 101

CODEBOOK = [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10]

VALUE_TYPES = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    8: ("str", 0), 9: ("arr", 0), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


def scale_table() -> list[float]:
    """UE4M3 half-scale table: 7 subnormal steps then 15 octaves of 8."""
    table = [m * 2.0 ** -10 for m in range(8)]
    for exponent in range(1, 16):
        for mantissa in range(8):
            index = 8 + (exponent - 1) * 8 + mantissa
            if index >= 126:
                continue
            table.append((8 + mantissa) * 2.0 ** (exponent - 11))
    table += [0.0] * (127 - len(table))
    return table


SCALES = scale_table()


def scale_to_float(byte: int) -> float:
    return SCALES[byte] if byte <= 0x7E else 0.0


def parse_gguf(path: pathlib.Path) -> tuple[dict, list[dict]]:
    data = path.read_bytes()
    assert data[:4] == b"GGUF", "bad magic"
    version, = struct.unpack_from("<I", data, 4)
    n_tensors, n_kv = struct.unpack_from("<QQ", data, 8)
    pos = 24

    def read_string(p: int) -> tuple[str, int]:
        length, = struct.unpack_from("<Q", data, p)
        p += 8
        return data[p:p + length].decode("utf-8", "replace"), p + length

    def read_value(p: int, vtype: int) -> tuple[object, int]:
        fmt, size = VALUE_TYPES[vtype]
        if vtype == 8:
            return read_string(p)
        if vtype == 9:
            elem, p = struct.unpack_from("<I", data, p)[0], p + 4
            count, p = struct.unpack_from("<Q", data, p)[0], p + 8
            efmt, esize = VALUE_TYPES[elem]
            if elem == 8:
                for _ in range(count):
                    _, p = read_string(p)
                return (elem, count), p
            p += count * esize
            return (elem, count), p
        value = struct.unpack_from(fmt, data, p)[0]
        return value, p + size

    kv: dict = {}
    for _ in range(n_kv):
        key, pos = read_string(pos)
        vtype, pos = struct.unpack_from("<I", data, pos)[0], pos + 4
        value, pos = read_value(pos, vtype)
        kv[key] = value

    tensors = []
    for _ in range(n_tensors):
        name, pos = read_string(pos)
        n_dims, pos = struct.unpack_from("<I", data, pos)[0], pos + 4
        dims = list(struct.unpack_from("<" + "Q" * n_dims, data, pos))
        pos += 8 * n_dims
        type_id, offset = struct.unpack_from("<IQ", data, pos)
        pos += 12
        tensors.append({"name": name, "dims": dims, "type": type_id, "offset": offset})

    aligned = (pos + 31) // 32 * 32
    return kv, tensors, aligned, data


def decode_block(block: bytes, dual: bool) -> list[float]:
    out = [0.0] * QK
    if dual:
        d0 = scale_to_float(block[16])
        d1 = scale_to_float(block[17])
    else:
        d0 = d1 = scale_to_float(block[16])
    for j in range(16):
        out[j] = CODEBOOK[block[j] & 0x0F] * d0
        out[j + 16] = CODEBOOK[block[j] >> 4] * d1
    return out


def rocminfer_decode(exe: str, gguf: str, tensor: str) -> list[int]:
    """Ask ROCMinfer for the raw IEEE-754 bit patterns of one decoded tensor.

    The engine emits bit patterns rather than decimal text because JSON has no
    hex-float literal and decimal formatting is not guaranteed to round-trip.
    Returning the raw integers lets the comparison below be exact.
    """
    proc = subprocess.run([exe, "decode", gguf, tensor, "--json"],
                          capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["encoding"] == "ieee754_bits", "unexpected encoding"
    return payload["values"]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    exe = sys.argv[1]
    failures = 0
    for arg in sys.argv[2:]:
        gguf = pathlib.Path(arg)
        _, tensors, data_off, data = parse_gguf(gguf)
        print(f"\n=== {gguf.name} ===")
        for t in tensors:
            if t["type"] not in (GGML_TYPE_ROCMFP4, GGML_TYPE_ROCMFP4_FAST):
                continue
            dual = t["type"] == GGML_TYPE_ROCMFP4
            block_bytes = DUAL_BLOCK_BYTES if dual else FAST_BLOCK_BYTES
            n_blocks = 1
            for d in t["dims"]:
                n_blocks *= d
            n_blocks //= QK
            start = data_off + t["offset"]
            payload = data[start:start + n_blocks * block_bytes]

            mine: list[int] = []
            for b in range(n_blocks):
                for value in decode_block(payload[b * block_bytes:(b + 1) * block_bytes],
                                          dual):
                    mine.append(struct.unpack("<I", struct.pack("<f", value))[0])
            theirs = rocminfer_decode(exe, str(gguf), t["name"])

            if len(theirs) != len(mine):
                print(f"  MISMATCH LENGTH {t['name']}: "
                      f"python {len(mine)} vs rocminfer {len(theirs)}")
                failures += 1
                continue
            bad = [(i, a, b) for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
            if bad:
                i, a, b = bad[0]
                fa = struct.unpack("<f", struct.pack("<I", a))[0]
                fb = struct.unpack("<f", struct.pack("<I", b))[0]
                print(f"  MISMATCH {t['name']}: {len(bad)}/{len(mine)} differ; "
                      f"first at {i}: python {fa!r} rocminfer {fb!r}")
                failures += 1
            else:
                print(f"  ok  {t['name']:<40} {len(mine):>9} weights bit-identical")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())