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
