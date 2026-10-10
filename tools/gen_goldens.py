"""Regenerate and verify ROCMinfer golden digests against an external oracle.

The point of this script is to break the circularity that let a wrong SHA-256
constant survive: the digests committed under docs/goldens are produced by
CPython's hashlib (OpenSSL-backed), not by this project's own implementation.
If the two ever disagree, exactly one of them is wrong and the disagreement is
the finding.

Usage:
    python tools/gen_goldens.py --verify    # compare, exit nonzero on mismatch
    python tools/gen_goldens.py --update    # rewrite docs/goldens/README.md

Model paths come from the command line or from the defaults below. Files that
are absent are skipped, so this is safe to run on a machine without models.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

DEFAULT_MODELS = [
    r"G:\More-models\gemma-4-E2B-it-Q4_0_ROCMFP4_COHERENT.gguf",
    r"G:\More-models\Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf",
    r"G:\More-models\Spark-X2.5-4B-Q4_0_ROCMFP4_STRIX_LEAN.gguf",
]

GOLDEN_DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "goldens" / "README.md"

# Tensor digests this project pins, as (artifact basename, tensor name).
PINNED_TENSORS = [
    ("gemma-4-E2B-it-Q4_0_ROCMFP4_COHERENT.gguf", "blk.0.attn_k.weight"),
    ("Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf", "output.weight"),
]


def file_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(models: list[pathlib.Path]) -> dict[str, str]:
    """Map artifact basename -> SHA-256 for every readable model."""
    found: dict[str, str] = {}
    for model in models:
        if model.is_file():
            found[model.name] = file_digest(model)
        else:
            print(f"skip (absent): {model}", file=sys.stderr)
    return found


def parse_doc_digests(text: str) -> dict[str, str]:
    """Extract every `artifact` ... `sha` pair recorded in the golden tables.

    The artifact table interleaves a byte-count column, and the tensor table
    interleaves format, size and payload columns, so any number of cells may sit
    between the artifact cell and the digest cell.
    """
    digests: dict[str, str] = {}
    row = re.compile(r"^\|[^|]*`([^`]+\.gguf)`[^|]*\|[^|]*\|\s*`([0-9a-f]{64})`", re.MULTILINE)
    for name, digest in row.findall(text):
        digests.setdefault(name, digest)
    return digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="compare against the committed table and fail on mismatch")
    parser.add_argument("--update", action="store_true",
                        help="rewrite the committed table with freshly computed digests")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    models = [pathlib.Path(m) for m in args.models]
    found = collect(models)

    if not found:
        print("no model files present; nothing to do", file=sys.stderr)
        return 0

    if not GOLDEN_DOC.is_file():
        print(f"missing {GOLDEN_DOC}", file=sys.stderr)
        return 1

    text = GOLDEN_DOC.read_text(encoding="utf-8")

    if args.update:
        for name, digest in found.items():
            pattern = re.compile(r"(\| `" + re.escape(name) + r"` \| [\d,]+ \| `)[0-9a-f]{64}(`)")
            replacement = r"\g<1>" + digest + r"\g<2>"
            text, count = pattern.subn(replacement, text)
            if count == 0:
                print(f"warning: no table row for {name}", file=sys.stderr)
        GOLDEN_DOC.write_text(text, encoding="utf-8")
        print(f"updated {GOLDEN_DOC}")
        return 0

    if args.verify:
        recorded = parse_doc_digests(text)
        failures = 0
        for name, digest in sorted(found.items()):
            expected = recorded.get(name)
            if expected is None:
                print(f"NOT RECORDED: {name}", file=sys.stderr)
                failures += 1
            elif expected != digest:
                print(f"MISMATCH {name}\n  recorded {expected}\n  actual   {digest}",
                      file=sys.stderr)
                failures += 1
            else:
                print(f"ok  {name}  {digest}")
        if failures:
            print(f"{failures} digest(s) disagree with the external oracle",
                  file=sys.stderr)
            return 1
        print("all recorded digests match the external oracle")
        return 0

    parser.error("choose --verify or --update")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())