# Golden artifacts

Format goldens are pinned to specific upstream artifacts. Regenerating a digest
here requires re-reading the exact file bytes, so a mismatch means the artifact
changed, the decoder changed, or a reader is wrong. Any of those three must be
investigated; none of them may be papered over by re-recording the value.

## Artifacts

Digests are SHA-256 over the entire file, produced by `rocminfer digest`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `gemma-4-E2B-it-Q4_0_ROCMFP4_COHERENT.gguf` | 3,360,161,632 | `f2b5e2d1e14abfe0f121cc4169fb9e169c0d7c4dc06aeb38f5a9555507d377b9` |
| `Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf` | 4,957,544,640 | `2d28718985e62510b0304240da1e0af200c06cc2e68ca283916ed4d87119cd70` |
| `Spark-X2.5-4B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` | 2,260,702,144 | `f3c3a526d5ef7b8e249ca1b1a092878d047bfefc9e3cb432d21643452cf94699` |

## Tensor payload digests

SHA-256 over a single tensor's packed bytes as stored in the file, produced by
`rocminfer decode <file> [tensor]`. These pin the *quantized bytes* our decoder
consumes, so a layout or offset regression is caught even if the numeric output
happens to look plausible.

| Artifact | Tensor | Format | Payload bytes | Payload SHA-256 |
|---|---|---|---:|---|
| gemma-4-E2B COHERENT | `blk.0.attn_k.weight` | Q4_0_ROCMFP4 | 221,184 | `b632a3bf1e0ecd376f4b2c237a6ed76a0641ad8bc91152b4bd2d13c1abd06dcf` |
| Ornith-1.0-9b STRIX_LEAN | `output.weight` | Q4_0_ROCMFP4_FAST | 540,344,320 | `c7234c4c91ed2e75c4c79ec8c8c06bb27026e883226f452fb165ea2a96489a7b` |

## Decoded ranges

| Artifact | Tensor | Weights decoded | Min | Max | Mean |
|---|---|---:|---:|---:|---:|
| gemma-4-E2B COHERENT | `blk.0.attn_k.weight` | 393,216 | -0.273438 | 0.195312 | 2.07052e-05 |
| Ornith-1.0-9b STRIX_LEAN | `output.weight` | 33,554,432 | -0.234375 | 0.214844 | -2.88829e-05 |

Weight ranges are consistent with a per-block UE4M3 scale in the low
single-digit range, which is what a trained projection layer produces. A decode
that yielded F16-scale magnitudes would indicate a byte-offset or layout error,
not a model property.

## Provenance

Both ROCmFP4 artifacts were produced by the pinned ROCmFPX revision recorded in
`THIRD_PARTY_SOURCES.md` (`fb08d7cdb67082ddda15a25e64dc7ec413afc41c`). They were
obtained from their publishers and are not redistributed here; only the digests
above are committed. Regenerating the tensor digests requires the same artifact,
so a holder of the file can reproduce them exactly with:

```
rocminfer decode gemma-4-E2B-it-Q4_0_ROCMFP4_COHERENT.gguf blk.0.attn_k.weight
rocminfer decode Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf output.weight
```