"""Emit the engine-internal signed-int4 GGUF artifact used by the e2e test.

ggml defines no raw signed-int4 type, so this container is not produced by any
upstream quantizer: ROCMinfer owns its type id (`GgmlType::kQ4_0Int4` = 200)
and therefore owns its writer. This script exists so the artifact that
`tests/test_int4_format_e2e.cpp` loads is a real file on disk, produced by a
writer that is separate from the reader under test — the same separation
`tools/crosscheck_decode.py` keeps for the ROCmFP4 goldens.

The payload packing is derived from the engine's int4 kernels
(`src/kernels/int4_matvec.hip`, `src/kernels/int4_wmma.hip`), which is what
makes this file loadable: element `k` of a row occupies the low nibble of byte
`k/2` when `k` is even and the high nibble when odd, two nibbles per byte, no
scales, no codebook. Nothing here is copied from ROCmFPX.

Usage:
    python tools/gen_int4_artifact.py [output_path]

Writes one tensor per shape and prints each row's exact expected dot products
as a JSON oracle, which the test recomputes on the C++ side independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

GGML_TYPE_INT4 = 200
ALIGNMENT = 32
VALUE_TYPES = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    8: ("str", 0), 9: ("arr", 0), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

# Shapes written into the artifact. The wide shape crosses the WMMA dispatch
# threshold and the narrow shape exercises the row-wise kernel.
SHAPES = [("blk.0.ffn_gate_sq.weight", 256, 512),
          ("blk.0.ffn_up_sq.weight", 64, 512)]

SEED = 0x9E3779B97F4A7C15


def nibble(value: int) -> int:
    """Two's-complement int4 encoding of one value in -8..7."""
    if not -8 <= value <= 7:
        raise ValueError(f"int4 out of range: {value}")
    return value & 0x0F


class SplitMix64:
    """Fixed-seed generator so every emitted byte is reproducible."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def int4(self) -> int:
        return (self.next() % 16) - 8


def put_string(out: bytearray, value: str) -> None:
    raw = value.encode("utf-8")
    out += struct.pack("<Q", len(raw))
    out += raw


def put_value(out: bytearray, vtype: int, value: object) -> None:
    """Append one GGUF value: its type tag, then its payload."""
    out += struct.pack("<I", vtype)
    if vtype == 8:
        put_string(out, str(value))
        return
    fmt, _ = VALUE_TYPES[vtype]
    out += struct.pack(fmt, value)


def build_payload(rng: SplitMix64, rows: int, cols: int) -> tuple[bytes, list[list[int]], list[int]]:
    """Pack `rows` x `cols` signed int4 weights row-major, two per byte."""
    values: list[list[int]] = []
    weights = bytearray()
    for _ in range(rows):
        row = [rng.int4() for _ in range(cols)]
        values.append(row)
        packed = bytearray()
        for k in range(0, cols, 2):
            packed.append(nibble(row[k]) | (nibble(row[k + 1]) << 4))
        weights += packed
    x = [rng.int4() for _ in range(cols)]
    return bytes(weights), values, x


def oracle_row(weight_row: list[int], x: list[int]) -> int:
    total = 0
    for w, xv in zip(weight_row, x):
        total += w * xv
    return total


def build_file(shapes, out_path: pathlib.Path) -> dict:
    """Assemble a GGUF v3 container and return the oracle for its tensors."""
    rng = SplitMix64(SEED)
    tensors = []
    oracle = {}
    for name, rows, cols in shapes:
        payload, values, x = build_payload(rng, rows, cols)
        tensors.append({"name": name, "dims": [rows, cols],
                        "type": GGML_TYPE_INT4, "payload": payload})
        oracle[name] = {
            "rows": rows,
            "cols": cols,
            "x": x,
            "expected": [oracle_row(values[r], x) for r in range(rows)],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }

    out = bytearray()
    out += struct.pack("<I", 0x46554747)          # "GGUF"
    out += struct.pack("<I", 3)                   # version
    out += struct.pack("<Q", len(tensors))        # tensor count
    out += struct.pack("<Q", 1)                   # kv count

    put_string(out, "general.architecture")
    put_value(out, 8, "rocmint4")

    offset = 0
    for tensor in tensors:
        put_string(out, tensor["name"])
        out += struct.pack("<I", len(tensor["dims"]))
        for dim in tensor["dims"]:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", tensor["type"])
        out += struct.pack("<Q", offset)
        offset += len(tensor["payload"])
        offset = (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    while len(out) % ALIGNMENT != 0:
        out += b"\x00"

    for tensor in tensors:
        out += tensor["payload"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out))
    return {
        "artifact": str(out_path),
        "bytes": len(out),
        "sha256": hashlib.sha256(bytes(out)).hexdigest(),
        "type_id": GGML_TYPE_INT4,
        "tensors": oracle,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?",
                        default="docs/goldens/artifacts/int4-e2e.gguf",
                        help="destination path for the GGUF artifact")
    parser.add_argument("--oracle", help="write the JSON oracle to this path")
    args = parser.parse_args()

    result = build_file(SHAPES, pathlib.Path(args.output))

    if args.oracle:
        pathlib.Path(args.oracle).write_text(json.dumps(result, indent=2),
                                             encoding="utf-8")

    print(f"artifact {result['artifact']}")
    print(f"  bytes   {result['bytes']}")
    print(f"  sha256  {result['sha256']}")
    for name, info in result["tensors"].items():
        print(f"  tensor  {name}  rows={info['rows']} cols={info['cols']}  "
              f"payload_sha256={info['payload_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
