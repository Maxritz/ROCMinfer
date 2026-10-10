# NInfer Integration Spec: ROCmFPX Formats + HaloFPX Model Zoo
## Native NInfer support for AMD-focused GGUF quants and large MoE models

**Status:** Architecture and implementation plan based on repository inspection. Not an assertion that the integrations have already been implemented or tested in NInfer.

**NInfer GPU targets:** RDNA2 `gfx1031` (RX 6700 XT class) and RDNA4 `gfx1201` (RX 9070 XT class).  
**Reported local toolchain:** ROCm 10.1 / HIP 7.16.x. The project owner reports that their setup supports `gfx1031` as well as `gfx1201`, with rocBLAS in their `gfx1031` environment and hipBLAS on the RX 9070 XT. Record exact local versions during bring-up.  
**Reviewed upstream projects:**
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- [julianmb/halofpx](https://github.com/julianmb/halofpx)

**Architecture policy:** NInfer remains the inference engine. ROCmFPX contributes quant format definitions, exact reference maths, quantisation recipes, model artefacts, and implementation lessons. HaloFPX contributes a model registry/zoo, compatible downloadable model packages, per-model run configuration concepts, hardware/model-fit logic, serving-profile ideas and operational tooling. Do not embed HaloFPX's Python server or forked `llama.cpp` runtime inside NInfer merely to access the models.

---

## 1. What these two projects actually contribute

### 1.1 ROCmFPX: weights and execution-format donor

The checked `charlie12345/ROCmFPX` README describes a `llama.cpp`-based experimental family of 2-, 3-, 4-, 6-, and 8-bit GGUF model-weight formats, with CPU reference paths and HIP/ROCm and Vulkan kernels. The repository distinguishes ROCmFP2/ROCmFP4 as more mature on its tested Strix Halo paths from ROCmFP3/FP6/FP8 development previews. It also has a separate signed-integer `ROCmI4` format and an explicitly lossy, opt-in W4A4 WMMA experiment guarded for `gfx1151` only.

Important implementation paths in that repo:

| Upstream path | What it contains | NInfer reuse approach |
|---|---|---|
| `ggml/rocmfp4/rocmfp4.h` | ROCmFP4 block/type definitions | Port as a format descriptor, not by copying ggml type IDs |
| `ggml/rocmfp4/rocmfp4.c` | CPU quantise/dequantise/reference math | Reuse maths and tests or independently port under NInfer interfaces |
| `ggml/rocmfp4/rocmfp4_hip.cu`, scale/codebook headers | HIP conversion/decode helpers | Adapt isolated maths; still implement NInfer's own kernels and dispatch |
| `ggml/rocmfpx/rocmfpx.h` | Block layouts for FP2/FP3/FP6/FP8 and I4 | Port layouts, row-size rules and block validation |
| `ggml/rocmfpx/rocmfpx.c` | Reference quantisation, unpack/dequantisation and validation | Use as reference oracle, especially for packed-code order and scale maths |
| `ggml/rocmfpx/test_rocmfpx.c`, `test_rocmfp2_reference.c` | Reference and codebook tests | Adapt to NInfer unit tests before tuning kernels |
| `ggml/src/ggml-cuda/vecdotq.cuh` | Format-specific packed code extraction and vector-dot math | Port relevant functions, then replace CUDA-specific constructs |
| `ggml/src/ggml-cuda/mmq.cuh` | Quantised matrix-matrix tile/load/dispatch patterns | Use as an algorithm reference; not a standalone HIP module |
| `ggml/src/ggml-cuda/mma.cuh`, `common.cuh` | AMD WMMA branches and type-trait patterns in the upstream fork | Study for `gfx1201`; inspect every architecture guard, don't inherit a `gfx1151` guard |
| `ggml/src/ggml-hip/CMakeLists.txt` | HIP compilation and rocBLAS/hipBLAS linkage | Build-system reference only |

ROCmFPX follows a clear ggml kernel coverage contract for each custom model-weight type: copy/convert (`CPY`), row lookup (`GET_ROWS`), row updates (`SET_ROWS`), matrix/vector multiply (`MUL_MAT`), and routed/expert matmul (`MUL_MAT_ID`), with CPU reference checks and backend-op tests. NInfer must implement the equivalent operators it uses, including expert/routed matrix multiply, not just a decode function.

### 1.2 HaloFPX: model zoo and operation recipes

The checked `julianmb/halofpx` README describes HaloFPX as a model server/zoo wrapper around a pinned ROCmFPX/`llama.cpp` engine. Its advertised product features include model discovery/download, model presets/aliases, per-model `run_config`, automatic hardware architecture detection, memory-fit checks, hot load/unload, OpenAI-compatible serving, vision-projector download, and suggested MTP/KV-cache run profiles.

The checked tree contains, among other files:

- `registry/models.json`: model IDs, source Hugging Face repositories, quant variants, file names, advertised model sizes, minimum-memory estimates, optional checksums, multimodal projector assets, and `run_config`.
- `registry/presets.json`: named quant presets and a Strix Halo hardware profile.
- `halofpx/hardware.py`: architecture detection and a family classifier including `gfx1031` family and `gfx1200`/`gfx1201` RDNA4. Its fallback to `gfx1151` is unsafe to copy blindly into NInfer; unknown detection must not silently select Strix-specific kernels.
- `halofpx/registry.py`: model aliases, known quant names, local model discovery and cache integration.
- `halofpx/model_manager.py`: pulls model files from Hugging Face and optional projector assets.
- `halofpx/engine_manager.py`: process lifecycle and engine load/unload; it starts a separate `llama-server` process.
- `docs/HARDWARE_SUPPORT.md`: model-size/memory sizing claims by hardware tier.
- `docs/QUANTIZATION_RECIPES.md`: conversion commands and model-preservation recipes.
- `scripts/build_engine.sh`: builds or downloads its inference engine. Its prebuilt binary is explicitly intended for Strix Halo; other architectures need a source build. Its source URL/pin should be treated as repository revision-specific.

The model zoo contains example large models and variants such as Qwen 3.8 27B ROCmFP4_FAST/ROCmFP2/ROCmFP8, Nemotron 3.5 30B A3B ROCmFP4_FAST, Ornith 1.5 35B A3B ROCmFP4, Ornith 1.0 ROCmFPX, Qwen 3.8 Flash Next 125B MoE, Ling 3.0 Flash 124B MoE, DeepSeek V4 Flash 284B MoE, Laguna S 2.1 and other community artifacts. Names, files, checksums, sizes and supported architectures must be read from a pinned registry revision; this document does not guarantee continued availability or the correctness of every entry.

**Key distinction:** a model zoo registry entry proves that the wrapper knows a model file and its expected engine invocation. It does not prove that NInfer understands its model architecture, GGUF tensor types, tensor names, tokenizer, chat template, MTP head, vision projector or specialised attention layers.

---

## 2. Integration goal: load these models directly in NInfer

The required result is:

```text
HaloFPX / ROCmFPX model registry + model artifacts
                       |
                       v
           NInfer model package resolver
                       |
                       v
         GGUF metadata + tensor-type parser
                       |
                       v
     format registry / exact quant block decoders
                       |
                       v
 model architecture graph / MoE routing / MTP & vision support
                       |
                       v
     NInfer scheduler + streamed weight residency
                       |
           +-----------+------------+
           |                        |
           v                        v
   gfx1031 kernel path       gfx1201 kernel path
   RDNA2 vector/dot          RDNA4 vector/dot/WMMA
           |                        |
           +------------+-----------+
                        v
        NInfer CLI / native server / OpenAI API
```

No `llama-server` subprocess should be required at inference time for the target product. A separate conversion utility can be used offline if upstream metadata or formats need transformation, but the preferred path is that NInfer reads the original GGUF and executes it natively.

---

## 3. Model compatibility is a matrix, not a single checkbox

Each candidate model needs independent compatibility records.

| Layer | Required check | Failure response |
|---|---|---|
| Artifact | URL/repository and exact file/shards resolve; file hash matches if known | Reject incomplete/tampered download and preserve partial status |
| Container | GGUF version, metadata, alignment, tensor offsets, type IDs, endianness | Fail with exact unsupported/corrupt field; never reinterpret an unknown type |
| Tokenizer/template | tokenizer vocabulary, merges, special tokens, chat template, stop tokens | Do not claim chat compatibility until tokenisation/template tests pass |
| Architecture | exact model architecture, layer configuration and tensor-name mapping | Reject unsupported architecture instead of falling through to a similarly named model |
| Tensor formats | every tensor type and block layout supported, including mixed-quant recipe tensors | Report unsupported tensor types by name and count |
| Compute operators | dense matmul, vector matvec, expert/routed `MUL_MAT_ID`, attention, norms, RoPE, gating, recurrent/hybrid attention as applicable | Use only numerically valid fallback or reject model |
| Memory | full/partial offload, file mapping, hot/cold expert residency, KV and scratch budgets | Refuse load or choose a documented lower-memory profile |
| Speculation | embedded MTP/NextN head or compatible external drafter detected and supported | Disable speculation, do not pass invalid MTP flags |
| Multimodal | model architecture plus projector format and preprocessing path | Run text-only only if supported; report vision unavailable |
| Quality | fixed prompt/logit/perplexity/agentic test suite against an upstream baseline | Keep format/model marked experimental |
| Performance | benchmark exact GPU, OS, driver, ROCm, kernel path, context and batch | Do not import Strix Halo throughput numbers as expectations for RDNA2/4 |

### 3.1 First audit the exact model architecture

HaloFPX model entries include dense, MoE, hybrid-attention, vision, MTP-capable and non-MTP models. Support for a similar Qwen variant does not imply support for a newer Qwen hybrid/MoE model. In particular:

- **MoE:** need top-k router, expert index selection, shared experts if present, routed expert GEMM, expert weight paging and correct per-token routing semantics.
- **Hybrid/recurrent attention:** must preserve model state across decode steps, implement convolution/recurrent state updates exactly and handle speculative rollback if MTP is supported. A dense Transformer attention implementation is not enough.
- **MTP/NextN:** parse embedded draft/next-token head tensors and metadata, implement acceptance/rollback, and benchmark enabled versus disabled. Do not enable because the registry says MTP true without checking the actual file.
- **Vision/multimodal:** parse the separate `mmproj` artifact and include image preprocessing, projector execution and multimodal token handling; downloading an `mmproj` is not equivalent to implementing vision.
- **Large MoE file layouts:** support split/sharded GGUF files, large offsets, safe mapping, and selected-expert paging. The total file size can greatly exceed resident VRAM, but only if unused tensors are not eagerly materialised.

---

## 4. Quant format registry: exact storage identities

Create a NInfer-owned registry. Do not hardcode GGUF numeric enum values without tying them to a pinned upstream version.

Suggested record:

```cpp
enum class QuantFamily : uint8_t {
    NativeFloat, SymmetricInteger, CodebookFloat,
    ScaledSignedInteger, CustomAMD
};

struct QuantFormatDescriptor {
    QuantFormatId id;             // NInfer stable internal identifier
    const char * stable_name;
    uint32_t upstream_type_id;    // valid only for pinned GGUF source revision
    uint32_t block_elements;
    uint32_t block_bytes;
    QuantFamily family;
    ScaleEncoding scale_encoding;
    PackLayout pack_layout;
    DecodeBlockFn decode_block;
    ReferenceDotFn reference_dot;
    KernelDispatchTable kernels;
    bool supported_on_gfx1031;
    bool supported_on_gfx1201;
};
```

Keep binary storage type separate from recipe/preset. For example, `Q4_0_ROCMFP4_FAST` and dual-scale `Q4_0_ROCMFP4` have different bytes per block. A `STRIX_LEAN` or `COHERENT` recipe can assign multiple quant types to different tensors, and should not be treated as one tensor encoding.

### 4.1 Format table and cautions

| Format | On-disk layout documented by upstream | Decode semantics / caution | Recommended first execution path |
|---|---|---|---|
| `Q2_0_ROCMFPX` / ROCmFP2 | 32-element blocks; 2.50 bpw native weight block described in upstream README | Validate exact packed code and dual UE4M3 scale layout from pinned reference; do not assume it is ordinary `Q2_K` or `IQ2_XXS` | CPU reference, then vector packed decoder |
| `Q3_0_ROCMFPX` / ROCmFP3 | 32 codes, 12 payload bytes + 2 scale bytes = 14 bytes, 3.50 bpw | 3-bit codes cross byte boundaries; code set `0, ±1, ±2, ±4`; two 16-element scales | Bit extraction + code map + vector dot |
| `Q4_0_ROCMFP4` | 32 weights, 16 packed nibble bytes + two scales = 18 bytes, 4.50 bpw | ROCmFP4-specific Codebook10/E2M1-derived code representation and half-block scales; **not NVIDIA NVFP4** | Packed decode + scale; use integer dot only if mathematical equivalence proven |
| `Q4_0_ROCMFP4_FAST` | 32 weights, 16 packed nibble bytes + one scale = 17 bytes, 4.25 bpw | Different scale layout from dual-scale variant; cannot share the same block stride or half-scale decoder | FAST-specific fused decoder and dot path |
| `Q6_0_ROCMFPX` / ROCmFP6 | 32 codes, 24 packed bytes + two scale bytes = 26 bytes, 6.50 bpw | Six-bit signed-magnitude values up to 31; code crosses byte boundaries; two scales | Bit unpack to registers, then vector/dot transform if exact |
| `Q8_0_ROCMFPX` / ROCmFP8 name | 32 signed 8-bit values + one custom scale byte = 33 bytes, 8.25 bpw | Scaled signed integer, values clamped `[-127,127]`; **not FP8 E4M3/E5M2** | `V_DOT4`/IU8 route if scale and activation maths fit; vector fallback |
| `Q4_0_ROCMI4` | 32 signed two's-complement four-bit integer codes plus UE4M3 scale = 17 bytes | Distinct format from ROCmFP4; upstream W4A4 path adds lossy activation quantisation and is guarded to `gfx1151` | Exact int8/vector route first; target-specific IU4/W4A4 only opt-in |
| `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ2_XXS`, other stock GGUF | Upstream ggml-defined block formats | Must use exact standard GGUF decoder/operator support; these can coexist in a mixed quant recipe | Existing supported decoder or explicit fallback |

For ROCmFP2 the public README documents a 2.50-bpw block format and an S40 `{−4, −1, +1, +4}` codebook with dual UE4M3 scales in the newer README description. Because format design changed during upstream development, pin the exact commit and port the reference implementation at that revision, rather than relying on a prose-only description. Its block struct in a different commit may show a different packing expression; source plus tests for the chosen pin are authoritative.

**Scale encoding:** ROCmFPX source comments describe finite unsigned UE4M3 scale bytes and reject invalid values including `0x7f` and sign-bit scale values. Keep the scale decoder distinct from standard floating-point FP8 decode.

---

## 5. Match each format to the supplied RDNA2 / RDNA4 ISA

The provided ISA files are the instruction reference for the NInfer targets. The upstream ROCmFPX/HaloFPX kernel paths, however, were extensively tuned on `gfx1151`; they cannot be copied under the assumption that the same exact instructions exist on `gfx1031` or `gfx1201`.

### 5.1 RDNA2 `gfx1031`

Use a correctness-first fused decoder and the documented target instructions where operand semantics permit:

- `V_DOT4_I32_I8` / `V_DOT4_U32_U8`: candidate for signed/unsigned packed 8-bit integer dot products.
- `V_DOT8_I32_I4` / `V_DOT8_U32_U4`: candidate for integer four-bit operand dot products.
- `V_DOT2_F32_F16` and/or `V_FMA_F16`: candidate for FP16 dot/arithmetic paths.
- FP32 FMA: baseline where scales/format conversions or precision requirements make integer dot transformations unattractive.

Mappings must preserve scale math. Example: a signed-code times per-block scale times activation can be rearranged into an integer dot followed by a block-scale multiply only if activation scaling/quantisation and accumulated range are mathematically identical. Do not directly feed ROCmFP4 codebook nibbles into signed-I4 dot instructions. First decode Codebook10 to the actual signed integer level, then use an integer route only if the level representation and scale handling are exact.

### 5.2 RDNA4 `gfx1201`

Probe and use target-specific WMMA operations where the installed ROCm/LLVM compiler exposes the documented instructions. The user's supplied RDNA4 ISA includes 16x16 WMMA families for FP16, BF16, IU8/IU4 and FP8/BF8 variants. This establishes ISA-level candidates, not the exact HIP C++ builtin signature or the performance of the chosen layout.

- FP16/BF16: WMMA for matching floating operands and accumulation.
- `ROCmI4`: IU4 WMMA is a candidate only after confirming its signedness/modifier contract and packing; do not reuse the upstream `__gfx1151__` implementation unmodified.
- `ROCmFP8`/`Q8_0_ROCMFPX`: signed scaled INT8 should be evaluated as IU8/signed integer-dot maths, not fed to the FP8 matrix path.
- `ROCmFP4`: unpack Codebook10, apply the scale layout, then use an exact supported operand encoding or vector math. FP8 WMMA is not a direct FP4 replacement.
- `ROCmFP2/3/6`: custom-code decode can be fused into each tile; any integer/FP16/WMMA transform must be tested against the reference.
- Maintain a vector fallback for every dtype and matrix shape.

### 5.3 Formats not made equivalent by shared bit-width

- **ROCmFP4 is not NVFP4.**
- **ROCmFP8 is not IEEE FP8.**
- **ROCmI4 is not ROCmFP4.**
- **Generic integer INT4 does not mean the tensor has the ROCmI4 block layout.**
- WMMA integer instructions require correctly encoded integer operands; they do not directly interpret custom scaled quantised GGUF blocks.
- A conversion pipeline that materialises all weights as FP16 may make the model run but defeats the memory-efficiency target. Convert only the active block/tile in registers/LDS unless measured evidence favours a limited cache of expanded hot weights.

---

## 6. Weight streaming + expert residency: extend beyond HaloFPX's model serving

HaloFPX presents large models with per-model context/KV and serving profiles. NInfer's distinguishing goal is to stream hot MoE experts on low-memory systems, so incorporate the registry but keep residency/runtime management inside NInfer.

### 6.1 Required storage tiers

1. **Backing store:** local SSD/mmap or another file-backed source of truth. Keep GGUF bytes packed in their original format.
2. **Host cache:** optional bounded page cache for active model ranges, expert blocks, tokenizer, and metadata. Do not read all expert weights eagerly.
3. **GPU resident experts:** recent/likely experts under a strict budget.
4. **GPU staging buffers:** bounded double/triple buffers for asynchronous upload and compute overlap.
5. **Optional decoded hot-tile cache:** only if a performance/space calculation shows that repeatedly decoding hot experts costs more than storing the decoded form. Scope to hot blocks rather than whole tensor/model by default.

### 6.2 Expert streaming design

- Parse and validate expert tensor offset ranges before execution.
- For each MoE layer, router returns expert IDs and weights. Aggregate token/expert assignments; do not start one disk read per token if a batch can share the same expert.
- Map the selected expert IDs to backing-store ranges and cache keys.
- Prioritise experts by immediate demand, reuse frequency and expected next-step probability; treat next-layer prefetch as speculative work that may be cancelled or deprioritised.
- Asynchronously stage packed weight blocks into GPU-visible memory.
- Start GEMM only once the correct block is resident and the dependency is satisfied.
- Keep separate per-GPU byte budgets for resident, staging, KV, scratch and graph buffers.
- Evict only when no in-flight kernel still references a block; use explicit events/refcounts/versioned residency handles.
- For quantised GEMM, decode the active packed block inside the tile. Do not decode all experts before knowing which experts are used.
- Maintain a deterministic fallback when async loading, graph capture, or a hot expert cache is unavailable.
- Instrument I/O latency, cache hit rate, bytes uploaded, expert reuse distance, stall time, time in decode vs GEMM, and eviction churn.

### 6.3 Avoid a common mistake

A smaller quant lowers storage/transfer bytes, but an elaborate decode may lower effective math throughput. The policy must be benchmark-based per target, model and workload. For low memory, a 2.50-bpw model may be necessary; for speed or quality, a 4.25/4.50-bpw model may win. Quant choice is not a universal fixed order.

---

## 7. Model-zoo ingestion from HaloFPX

Consume `registry/models.json` conceptually as a metadata source, but do not make NInfer import HaloFPX Python modules or depend on its launch manager.

### 7.1 NInfer model manifest

Create a stable JSON manifest with NInfer-owned schema:

```json
{
  "schema_version": 1,
  "model_id": "example-model",
  "display_name": "Example model",
  "source": {
    "kind": "huggingface",
    "repo_id": "publisher/repository",
    "files": [
      {
        "filename": "model.gguf",
        "role": "weights",
        "sha256": null,
        "required": true
      }
    ]
  },
  "variants": [
    {
      "variant_id": "rocmfp4-fast",
      "quant_preset": "Q4_0_ROCMFP4_FAST",
      "estimated_weight_bytes": null,
      "estimated_min_device_bytes": null
    }
  ],
  "architecture": {
    "family": "detect-from-model-metadata",
    "mtp": "detect-from-model",
    "vision": "detect-from-model"
  },
  "run_profiles": {
    "gfx1031": {},
    "gfx1201": {}
  }
}
```

This is a schema sketch only; the manifest must be validated by an actual parser. Use explicit null/unknown values rather than treating missing size/hash as known values.

### 7.2 Model sync/import tool

Implement an offline `ninfer-model` helper (or equivalent subcommands):

- `list`: show registered models and local availability.
- `info <model>`: show architecture, format, file size/hash, quant preset, MTP/vision detection and supported NInfer capability status.
- `pull <model> --variant <variant>`: download required files, resume shards safely, verify hash when available, then atomically promote complete downloads.
- `inspect <file.gguf>`: print every metadata key, tensor type and count, per-type byte sizes, total mapped bytes, unsupported types and architecture.
- `validate <file.gguf>`: check header, tensors, block alignment, offsets and format validators.
- `quantize`: only where NInfer supports writing the format correctly; initially, use the pinned ROCmFPX quantizer as an offline external tool if appropriate, then validate its output with NInfer's independent reader.
- `bench`: create an explicit report per GPU/backend/model/context/batch/quant.

Do not trust Hugging Face repository/file strings or invoke model-specific shell fragments taken directly from JSON. Validate filenames, escape process arguments, avoid shell interpolation and write downloads to temporary locations until hashes/format checks pass.

### 7.3 Run profiles must be target-qualified

Do not blindly copy `run_config` from HaloFPX. Profiles tuned on `gfx1151`/Mesa RADV Wave64 may be wrong on `gfx1031` or `gfx1201`.

NInfer should resolve a profile through:
1. model architecture capability;
2. installed device target/capabilities;
3. exact quant format and kernel support;
4. current free memory and memory budget;
5. measured profile for that target/model family;
6. safe generic fallback.

MTP should be enabled only after parsing the actual model and verifying its next-token head. KV types should remain a separate runtime setting from the model-weight quant format.

---

## 8. Portability and integration boundaries

### 8.1 What not to copy wholesale

- HaloFPX Python FastAPI wrapper, CLI and `llama-server` subprocess management if NInfer already owns CLI/server.
- HaloFPX's `gfx1151` performance assumptions, Strix-specific profile defaults and environment overrides such as `HSA_OVERRIDE_GFX_VERSION=11.5.1`.
- ROCmFPX ggml type enum values, tensor abstractions, CUDA template instantiations or backend-op dispatch code without adaptation.
- The exact `__gfx1151__` W4A4 path and the reported benchmark results as if they were validated on `gfx1031`/`gfx1201`.
- Format names that hide distinct block bytes, scale policies or codebooks.

### 8.2 What to preserve as concepts

- deterministic CPU reference quantise/dequantise/dot path;
- exhaustive packed-code validation;
- custom decoder per format;
- per-format MMVQ and MMQ kernels;
- explicit backend-op coverage matrix;
- per-model quant recipes that protect sensitive tensors;
- model zoo metadata with file hashes and exact variants;
- explicit MTP capability detection, rather than enabling flags on model name alone;
- hardware-aware but target-qualified model-fit and launch profiles;
- server load/unload and model switching, implemented natively in NInfer;
- a clear separation between weight quant and KV-cache quant.

---

## 9. Implementation phases

### Phase 0: Pin the external data

- Pin exact commits for ROCmFPX and HaloFPX in `THIRD_PARTY_SOURCES.md`.
- Capture required licenses and third-party notices. HaloFPX advertises Apache 2.0; ROCmFPX advertises MIT in its repository metadata. Preserve notices on code copied or adapted.
- Save raw model manifests/metadata as reviewed artifacts, not as an uncontrolled live dependency.
- Produce the initial architecture/type compatibility inventory.

**Exit gate:** source revision, licensing, model file identities and quant format versions are reproducible.

### Phase 1: Parser and reference maths

- Register ROCmFP4 dual-scale and FAST separately.
- Port CPU/reference decoders and row sizes.
- Port reference tests and validate against several files generated by the pinned upstream quantizer.
- Implement model manifest parser and `inspect`/`validate` utility.

**Exit gate:** every tensor using the selected format decodes with expected values; malformed/truncated/unknown data fails safely.

### Phase 2: First GPU execution path

- Implement fused packed decode + vector/dot matvec for `gfx1031`.
- Implement vector path and then `gfx1201` WMMA candidates for compatible operand types/shapes.
- Implement copy/convert, row lookup, dense matmul and routed/expert matmul support as required by NInfer.
- Use the ISA docs supplied with NInfer and the installed compiler for the target-specific probes.

**Exit gate:** kernel tests pass on both GPUs with a CPU oracle and the same model produces acceptable logits.

### Phase 3: Model zoo and artifact tooling

- Import a limited set of models and variants from HaloFPX's registry.
- Download/verify with manifest-driven tooling.
- Add size estimators for weight, KV, scratch, staging and residency pools separately.
- Implement `info`, `pull`, `inspect`, `validate` and `bench`.

**Exit gate:** artifacts can be managed and loaded without launching an external engine; missing capabilities are reported by tensor/operator.

### Phase 4: MoE streaming and scheduling

- Introduce expert-keyed cache handles, bounded GPU residency, async staging and safe eviction.
- Batch tokens by expert when appropriate; measure dispatch overhead and occupancy.
- Integrate graph replay only after stream/event dependencies and buffer lifetimes are proven.
- Record when an inference stalls on storage, host copies, cache misses or compute.

**Exit gate:** larger-than-VRAM MoE can load and execute within explicit memory budgets; no use-after-free, stale block, incomplete transfer or silent wrong expert.

### Phase 5: RDNA4 acceleration and optional paths

- Test WMMA for FP16/BF16 and valid integer operand representations.
- Evaluate IU4 for ROCmI4 only after deriving the exact operand contract for `gfx1201`.
- For ROCmFP4, test codebook decode plus matching operand path rather than pretending it is native NVFP4.
- Add cache/profile optimisations only when a benchmark demonstrates benefit.

**Exit gate:** every accelerated path has a compile result, generated ISA proof, numerical proof and workload-specific performance result.

### Phase 6: Model architecture coverage

Start with models whose architecture NInfer already implements. Only then add:
1. dense model(s) with ROCmFP4 variants;
2. a compatible small Qwen for format-smoke validation;
3. MoE model with standard routing;
4. hybrid/recurrent MoE with persistent state and rollback;
5. MTP-capable model;
6. vision model/projector path.

Do not combine all new format, architecture, speculation and streaming variables in one first test. That makes failures difficult to isolate.

---

## 10. Qualification test matrix

For every format/model/target combination, record `unsupported`, `reference-only`, `GPU-fallback`, `accelerated-verified`, or `measured`.

| Test layer | Required tests |
|---|---|
| Format | exhaustive code decode, invalid scales, row size, block alignment, tails, random deterministic reference comparisons |
| Operation | copy/quantize/dequantize, row lookup/update, matvec, GEMM, routed/expert GEMM |
| Target | `gfx1031` build + runtime and `gfx1201` build + runtime independently |
| Numerical | tensor output max/mean error, logit diff, top-token agreement, perplexity |
| Model | fixed prompt, deterministic token sequence, stop tokens, tool/JSON/coding prompts for agent presets |
| MoE | routing correctness, selected expert IDs, expert paging hit/miss, boundary/cancel tests, concurrent in-flight references |
| Memory | smaller VRAM than model, fragmented cache, near-budget load, KV growth, unload/reload, failed transfer and recovery |
| MTP | file metadata/tensor detection, acceptance rates, target equivalence, rollback and draft-state handling |
| Vision | image preprocessing, mmproj load, multi-image and text-only behaviour as supported |
| Performance | prompt prefill, one-token decode, batched decode, routed-MoE decode, load time, peak host/GPU memory, I/O stall and p50/p95 latency |
| Regression | repeat on reference kernels, forced fallback, different target dispatch, repeated sessions and long soak |

### Per-result record

- source commits and model SHA256;
- GPU model, target ID, driver, OS, ROCm/HIP/LLVM version, Vulkan driver if used;
- build flags, compiler command, generated ISA excerpt;
- quant preset and actual per-tensor type counts;
- context, batch, number of sequences, output tokens and prompt;
- latency/throughput, peak host/GPU memory, quantised bytes read/uploaded, cache hit ratio;
- error metrics and fixed output comparison;
- status label and any known caveat.

---

## 11. Model selection and rollout strategy

Do not start with the largest advertised MoE just because it demonstrates the capacity goal. Start with an artifact that isolates format correctness, then scale up.

Recommended order:

1. **Tiny dense model in ROCmFP4/FPX format:** prove parser, dequant and matvec.
2. **Matched BF16/F16 baseline model:** compare output/logits/perplexity to quantify the quantisation difference.
3. **ROCmFP4_FAST vs dual-scale ROCmFP4:** ensure distinct 17-byte vs 18-byte block parsing.
4. **ROCmFP2:** measure whether reduced I/O outweighs extra unpack/math on both targets.
5. **ROCmFP3/FP6/FP8 preview formats:** only after exact tests exist and mark as experimental.
6. **MoE file with all experts larger than VRAM:** prove streaming and cache semantics independent of MTP.
7. **MTP-capable model:** prove accepted tokens, rollback correctness and performance after the base route works.
8. **Hybrid/recurrent MoE and multimodal:** only when the relevant architecture path is already implemented.

HaloFPX's published benchmarks and registry are useful to choose model artifacts and starting quant variants. They are measured principally on Strix Halo `gfx1151` and a specific RADV Wave64 environment; do not use those numbers as expected NInfer results on RX 6700 XT-class `gfx1031` or RX 9070 XT `gfx1201`.

---

## 12. Acceptance criteria

- [ ] NInfer reads pinned upstream ROCmFP4/ROCmFPX GGUF files directly.
- [ ] Each storage format is registered under a NInfer-owned stable identifier.
- [ ] All type IDs are tied to a pinned source format revision.
- [ ] Dual-scale ROCmFP4 and FAST format are parsed differently.
- [ ] `Q8_0_ROCMFPX` is handled as scaled signed-int8, not standard floating FP8.
- [ ] ROCmFP4 is not treated as NVIDIA NVFP4.
- [ ] CPU reference decoder matches upstream tests for selected formats.
- [ ] `gfx1031` and `gfx1201` both have a correct fallback implementation.
- [ ] Any RDNA4 WMMA/integer-dot path has compiler, ISA, correctness and performance evidence for `gfx1201`.
- [ ] No `gfx1151`-only path is silently enabled on `gfx1031` or `gfx1201`.
- [ ] Model zoo tooling can list, pull, verify, inspect and report unsupported features.
- [ ] NInfer runs model artifacts without launching a `llama-server` subprocess.
- [ ] MoE expert weights can remain packed in backing storage and be staged on demand within a defined memory budget.
- [ ] Expert eviction respects asynchronous readers and in-flight kernels.
- [ ] MTP and vision are detected from model files and only enabled when genuinely supported.
- [ ] Prefill, decode, memory, and quality are measured separately.
- [ ] Benchmark results make no unsupported extrapolation from Strix Halo numbers.
- [ ] Third-party source revisions and licence notices are retained.

---

## 13. Decision summary

**Use ROCmFPX for the quant formats, quantisation algorithms, reference tests, format-specific dot products, and model-weight files. Use HaloFPX for the model registry, variants, file acquisition, model metadata and model-specific run-profile concept. Keep NInfer's compute runtime, target-specific kernels, MoE expert pager, scheduler, KV memory, graph/event system and server as the authoritative execution system.**

Implement `Q4_0_ROCMFP4_FAST` and dual-scale `Q4_0_ROCMFP4` first, because they provide the clearest immediate path into the AMD-specific weight formats. Then add ROCmFP2, followed by FP3/FP6/FP8 previews. Treat ROCmI4/W4A4 as a separate lossy integer experiment. Use the supplied RDNA2/RDNA4 ISA docs to build per-target dot and WMMA paths, but never assume a `gfx1151` experiment proves support on either target.

### Sources inspected

- ROCmFPX repository overview and quant family: https://github.com/charlie12345/ROCmFPX
- ROCmFPX format references: https://github.com/charlie12345/ROCmFPX/tree/main/ggml/rocmfp4 and https://github.com/charlie12345/ROCmFPX/tree/main/ggml/rocmfpx
- ROCmFPX quantised compute templates: https://github.com/charlie12345/ROCmFPX/blob/main/ggml/src/ggml-cuda/vecdotq.cuh, https://github.com/charlie12345/ROCmFPX/blob/main/ggml/src/ggml-cuda/mmq.cuh, https://github.com/charlie12345/ROCmFPX/blob/main/ggml/src/ggml-cuda/mma.cuh
- HaloFPX model registry: https://github.com/julianmb/halofpx/tree/main/registry
- HaloFPX hardware support matrix: https://github.com/julianmb/halofpx/blob/main/docs/HARDWARE_SUPPORT.md
- HaloFPX quant recipes: https://github.com/julianmb/halofpx/blob/main/docs/QUANTIZATION_RECIPES.md
- HaloFPX engine builder: https://github.com/julianmb/halofpx/blob/main/scripts/build_engine.sh
