# Third-Party Sources

ROCMinfer is an independent, from-scratch AMD-first inference engine. It does
**not** link against, vendor, or run the upstream projects below. They are
pinned here as *references* for formats, quantization recipes, model metadata
and prior-art concepts, per `NINFER_ROCmFPX_HaloFPX_INTEGRATION_SPEC.md` Phase 0.

Any code copied or adapted from a source listed here must preserve that source's
license notice in the file header and be recorded in the "Adapted code" table.

## Pinned references

| Source | Remote | Pinned commit | Date | License | Local checkout |
|---|---|---|---|---|---|
| ROCmFPX | `charlie12345/ROCmFPX` | `fb08d7cdb67082ddda15a25e64dc7ec413afc41c` | 2026-09-23 | MIT | `H:\ports\ROCmFPX` |
| HaloFPX | `julianmb/halofpx` | `d9ec975b059a5895224fdedfe451cdf270a598d9` | 2026-09-24 | Apache-2.0 | `H:\ports\halofpx` |
| DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash` | `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | 2026-10-11 | MIT | remote reference only |

DeepSeek-V4.1-Flash is pinned as a **format and architecture reference only**. It
is a 552B-backbone MoE with ~763B stored parameters and roughly 196B engram
tables, so it cannot be served on this project's `gfx1031` / `gfx1201` targets
with 16 GB of VRAM; nothing here authorises a claim that it can. What it does
provide is a machine-readable specification for the FP4 KV-cache layouts and the
cross-layer KV-sharing design discussed in arXiv:2609.19969, which is the part
Phase 6 needs. Only these files are treated as authoritative:

| Path | What it specifies |
|---|---|
| `inference/config.json` | `kv_source_layers`, `index_source_layers`, `compress_ratios`, `index_topk`, `index_head_dim`, `window_size`, expert and attention constants |
| `inference/kernel.py` | `fp4_act_quant` — the FP4 quantizer for both KV stores |
| `config.json` | `num_key_value_heads`, `max_position_embeddings`, the transformers-side equivalents |

No local checkout is taken; the pinned commit is cited so a claim can be traced
back to an exact revision. Re-fetch with
`git clone https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash && git checkout <sha>`
only if a byte-exact artifact is needed (see "Reviewed manifests").

Commit subjects for the pinned revisions:

- ROCmFPX `fb08d7c`: "Merge pull request #38 from ciru-ai/agent/vulkan-deep-context-stability-20260724"
- HaloFPX `d9ec975`: "fix: unload engine off the event loop on shutdown"

Both checkouts were obtained with `git clone --depth 1`; the pinned SHA is the
shallow HEAD at capture time. Re-fetch a specific SHA with
`git clone https://github.com/<remote>.git && git checkout <sha>` if history is
needed.

## License notices

### ROCmFPX - MIT

Upstream is a `llama.cpp` / `ggml` fork. The checkout's `LICENSE` is the MIT
license, and the copyright line in this revision reads
`Copyright (c) 2023-2026 The ggml authors`.

Bundled third-party components declared by the upstream tree
(`THIRD_PARTY_NOTICES.md`):

| Component | License | Copyright |
|---|---|---|
| llama.cpp / ggml | MIT | 2023-2026 The ggml authors |
| cpp-httplib | MIT | (c) 2017 yhirose |
| nlohmann/json | MIT | (c) 2013-2025 Niels Lohmann |
| gguf-py | MIT | (c) 2023 Georgi Gerganov |

Upstream states its ROCmFP4 source files, scripts and documentation additions
are provided under the same MIT license as the rest of the source tree unless a
file says otherwise.

### HaloFPX - Apache License 2.0

The checkout's `LICENSE` is the Apache License, Version 2.0, and the copyright
line reads `Copyright 2026 ROCmFPX & Strix Halo Open Source Community`.

Apache-2.0 requires that redistributed derivative works carry the license, a
NOTICE if one exists, and prominent statements of changes. ROCMinfer does not
redistribute HaloFPX code; it re-implements concepts. If any HaloFPX file is
later adapted, add its notice here and in the adapted-code table.

## Reviewed manifests (frozen artifacts)

Raw registry data is snapshotted under `docs/manifests/` so the model inventory
does not depend on a live upstream endpoint. These copies are byte-exact.

| Artifact | Source path | SHA-256 |
|---|---|---|
| `docs/manifests/halofpx-models.json` | `halofpx/registry/models.json` @ `d9ec975` | `a6dc8f90a667756090127fcc6b6bbd9d4cc006be033e2fed10b6730e8c7e4c88` |
| `docs/manifests/halofpx-presets.json` | `halofpx/registry/presets.json` @ `d9ec975` | `b49c42e43fe6242dfd2c1202ce5288d578c4d3a8d4387dcefa275558cef2bbe3` |
| `docs/manifests/rocmfpx-api-inventory.md` | extracted from `H:\ports\ROCmFPX` @ `fb08d7c` | see file header |

Model weights are **not** included and remain subject to each model publisher's
own license and terms.

## DeepSeek-V4.1-Flash — MIT

Repo-root `LICENSE` (MIT, `Copyright (c) 2023 DeepSeek`). There is **no** license
file inside `inference/`; the repo-level MIT notice covers it. Pinned at
`2cba9e42`. Nothing from this tree has been copied, vendored, or adapted yet: the
engine re-derives from the published layout parameters, and if any upstream code
is later adapted it must be entered in the adapted-code table below with its
license notice preserved.

Recorded here because it is the only upstream in this file whose *spec* is
executable rather than prose. Two limits are binding:

- The FP4 nibble order inside a byte is **not stated** anywhere in the released
  files; it is left to the `torch.float4_e2m1fn_x2` dtype convention. A bit-exact
  golden therefore requires running upstream's quantizer, not writing a packing
  loop from an assumed order.
- The CED prefill/decode parameter split, the SWA Bounded Replay policy, and the
  "Full / Reindex / Reuse" mode labels are README prose only. They have no field
  in either `config.json` and no implementation in `model.py`, which is a single
  flat 40-layer stack with no encoder/decoder split. Any support for them would
  be an invented design and is recorded as unresolved instead.

### Derived fact, recomputed not quoted

The paper's "890 bytes per token" global KV footprint is independently
reproducible from the pinned config plus the quantizer's group sizes:

- compressed KV: `head_dim` 512 x 4 bit = 256 B payload + 512/16 = 32 E4M3 scale
  bytes = 288 B per position; ratios 2, 2, 2 (sources 2, 8, 14) give 3 x 144 B
  and ratio 1 (source 20) gives 288 B, subtotal **720 B/token**
- indexer K: `index_head_dim` 128 x 4 bit = 64 B + 128/32 = 4 E8M0 scale bytes =
  68 B per position; 3 x 34 B at ratio 2 and 68 B at ratio 1, subtotal
  **170 B/token**
- total **890 B/token**

This excludes the per-layer sliding-window ring (a per-sequence constant, which
is why SWA Bounded Replay matters only for the persistent figure) and the int32
Top-K index arrays. The layer-to-source assignment is inferred from sequential
execution order, not read from a published field.

## Adapted code

| ROCMinfer file | Upstream source | Upstream commit | License | Nature of adaptation |
|---|---|---|---|---|
| _(none yet)_ | | | | |

## Provisional status of upstream claims

Per the integration spec, HaloFPX performance numbers and registry entries are
informational only: they are measured mainly on Strix Halo `gfx1151` under a
specific RADV Wave64 environment. They are **not** expected ROCMinfer results on
`gfx1031` or `gfx1201`, and some registry variants are explicitly marked
unvalidated or parked by their authors. See `docs/manifests/halofpx-models.json`
and the upstream compatibility inventory for per-entry caveats.
