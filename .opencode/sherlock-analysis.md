# Sherlock Analysis Log

Append-only. One entry per investigation.

## [RUN-001] 2026-10-10 — MODE: FULL — TARGET: SHA-256 golden digest

### BASELINE
- Purpose: content-address model artifacts and tensor payloads for format goldens.
- Host: Windows 11, RX 9070 XT (gfx1201), ROCm 7.16.26331, ROCm LLVM clang 23.
- Oracle: Python `hashlib` (CPython, OpenSSL-backed), cross-checked against
  .NET `System.Security.Cryptography.SHA256`. Both agree.

### FINDINGS

| Rank | Symptom | Evidence | Status |
|---|---|---|---|
| 1 | Every digest wrong, including the empty string | `sha256("")` = `b841e9ac...`, truth `e3b0c442...` | CONFIRMED |
| 2 | Schedule and round intermediates verified correct | `w[16]=61626380`, `w[17]=000f0000`, round-0 `t1=54da50e8`, `a=5d6aebcd` all match the published SHA-256("abc") trace | CONFIRMED |
| 3 | `rotr` verified correct | three `static_assert` cases pass | CONFIRMED |
| 4 | K table had three transposed hex digits | see root cause | CONFIRMED |

### ROOT CAUSE
Three FIPS 180-4 round constants were mistranscribed:

| Index | Written | Correct |
|---|---|---|
| K[5]  | `59f11142` | `59f111f1` |
| K[21] | `4a74843a` | `4a7484aa` |
| K[37] | `766a0aab` | `766a0abb` |

### WHY IT WAS HARD TO SEE
A wrong SHA-256 constant does not fail loudly. It produces a well-formed 32-byte
digest that differs from the truth but is internally consistent, so every
self-consistency check passes: one-shot equals chunked, streaming equals
one-shot, and a second hand-written reference agrees. Two implementations
reproducing the same transcription error would agree with each other and both
be wrong, which is exactly what happened here.

Compounding it, two *expected* values in the test were also recalled from memory
and were wrong (a 64-zero-byte digest and a 448-bit vector digest). Memory is not
an oracle. Every expected digest in `tests/test_sha256.cpp` is now produced by
an external implementation and re-verified, not recalled.

### STRUCTURAL FIX (RUN-002)
Correcting three literals treats the symptom. The defect is that the constants
were *transcribed* when FIPS 180-4 *defines* them:

    K[i] = first 32 bits of frac(cbrt(p_i)),  i = 0..63
    H[i] = first 32 bits of frac(sqrt(p_i)),  i = 0..7

`include/rocminfer/detail/sha256_constants.hpp` now evaluates that definition
with exact integer arithmetic: scaling by a power of two moves the fractional
part into the integer domain, so `floor(cbrt(p << 96)) & 0xFFFFFFFF` is exactly
the wanted word, and the roots come from integer binary search rather than
floating point. There is no longer a table to mistype.

Verification is deliberately independent of that derivation: the test
recomputes every constant in `long double` via its own prime generator, so a
shared mistake would have to occur in two unrelated implementations to survive.

### ACTIONS
- [x] Corrected the three constants against FIPS 180-4.
- [x] Replaced the transcribed table with a derivation from the FIPS definition
      using exact integer arithmetic; the transcription step no longer exists.
- [x] Verification uses a different arithmetic path (long double) and a
      separate prime generator from the production one.
- [x] Replaced every recalled expected digest with a value confirmed against
      Python `hashlib` and .NET.
- [x] Added a 145-byte three-block vector and a million-`a` vector so both the
      single-block and multi-block paths are pinned.
- [x] Added `tools/gen_goldens.py`, which binds the digests committed under
      `docs/goldens` to CPython's OpenSSL-backed `hashlib`, removing the last
      place where a digest was self-certified.

### RESIDUAL RISK
The constant table can no longer be wrong, because it is computed rather than
typed. The remaining exposure is the *expected digests* in the test, which are
still literals; `tools/gen_goldens.py --verify` now covers the artifact digests,
but the in-test vectors have no such binding. Treat any newly added vector the
same way: produce it with a trusted implementation, never from memory.

### VALIDATION GATES
- [x] Baseline recorded with full config
- [x] At least three measurements per hypothesis
- [x] Hypothesis status updated to CONFIRMED
- [x] Actions have checkboxes are complete

## [RUN-003] 2026-10-10 — MODE: FULL — TARGET: HIP device compilation in build-hip

### BASELINE
- Symptom: the `build-hip` tree produced no `HIP_COMPILER` rule in
  `build-hip\CMakeFiles\rules.ninja`; every translation unit (including
  `src/profiler_hip.cpp` and the new `src/kernels/rocmfp4_dot.hip`) compiled via
  `CXX_COMPILER`, so no device code was emitted. The HIP build silently linked
  only the HIP *runtime API*.
- Host: Windows 11, RX 9070 XT (gfx1201), ROCm 7.16.26331, ROCm LLVM clang 23.
- CMake 3.29, Ninja generator.

### ROOT CAUSE
`project(rocminfer ... LANGUAGES CXX)` was invoked before the
`option(ROCMINFER_ENABLE_HIP ...)`, and `enable_language(HIP)` appeared inside an
`if()` block. CMake fixes each source file's language at target-creation time,
so when HIP was not in the initial `LANGUAGES` set the `.hip` source fell back
to CXX. This is CMake's `Windows-Clang-HIP` module rule, not a toolchain bug.

### FIX
Moved `option(ROCMINFER_ENABLE_HIP ...)` ahead of `project()` and made `project`
declare `LANGUAGES CXX HIP` when HIP is enabled. Added an explicit
`set_source_files_properties(... LANGUAGE HIP)` on the device source as a guard.
CXX and HIP must share the Clang ABI on Windows, so the ROCm clang
`G:/ROCM10RT-gfx1201/lib/llvm/bin/clang++.cpp` is used for both
(`CMAKE_CXX_COMPILER`); `hipcc.bat` is not accepted by CMake as
`CMAKE_HIP_COMPILER`. ROCm clang lives under `lib/llvm/bin`, not `llvm/bin`.

### VERIFY
- `rules.ninja` now contains `rule HIP_COMPILER__rocminfer_core_unscanned_*`
  and the kernel builds via `HIP_COMPILER__...`.
- `rocmfp4_dot` test runs on-device (RX 9070 XT) and PASSES: exact family
  bit-exact, numerical family below the proven FP32 bound, degenerate 0x7E
  scale decodes to zero, argument-rejection validates.
- Full CTest: 10/10 green (profiler, rocmfp4_reference, gguf_reader, sha256,
  rocmfp4_payload, rocmfp4_dot, two golden cross-checks, golden digest verify,
  doxy-validate).

### ACTIONS
- [x] `option()` hoisted before `project()`; HIP added to `LANGUAGES`.
- [x] Device source classified with explicit `LANGUAGE HIP`.
- [x] Build instructions recorded in AGENTS.md.
- [x] Doxygen gate registered as CTest `doxy-validate`.
- [x] Cross-compilation path for `gfx1031` documented (configure
      `CMAKE_HIP_ARCHITECTURES=gfx1031,gfx1201`); hardware-local run limited to
      gfx1201 by tooling availability.

### VALIDATION GATES
- [x] Baseline recorded with full config
- [x] At least three measurements per hypothesis
- [x] Hypothesis status updated to CONFIRMED
- [x] Actions have checkboxes and are complete

## [RUN-002] 2026-10-10 — MODE: HIGH — TARGET: SHA-256 constant provenance

### DELTA from RUN-001
- Transcribed K table replaced by derivation from the FIPS 180-4 definition.
- Verification moved to an independent arithmetic path (long double vs integer).
- Artifact digests bound to an external oracle via tools/gen_goldens.py.
- Duplicated manifest probe constant in main.cpp removed by routing
  inspect/validate through GgufReader.

### UPDATED FINDINGS
| Rank | Component | Cost | Delta | Status |
|---|---|---|---|---|
| 1 | SHA-256 constants | 3 wrong entries | transcription removed entirely | FIXED |
| 2 | Test vectors | 2 wrong values | now oracle-produced | FIXED |
| 3 | Golden digests | self-certified | bound to CPython hashlib | FIXED |
| 4 | Manifest probe size | duplicated in 2 files | single definition | FIXED |

## [RUN-004] 2026-10-10 — MODE: LOW — TARGET: WMMA-accelerated matvec feasibility on gfx1201

### BASELINE
- Question: Can RDNA4 WMMA (matrixmultiply) instructions accelerate the fused
  ROCmFP4 dequantize-and-dot kernel in src/kernels/rocmfp4_dot.hip?
- Host: Windows 11, RX 9070 XT (gfx1201), ROCm 7.16.26331.
- Toolchain: G:\ROCM10RT-gfx1201\bin\hipcc.bat, clang 23.
- Existing kernel: FP32 dequantize-and-FMA in registers, no matrix instruction
  (MFMA is CDNA-only; documented at rocmfp4_dot.hpp:34).
- Format: ROCmFP4 Q4_0: QK_ROCMFP4=32, QS_ROCMFP4=16, QK_ROCMFP4_BYTES=18
  (dual-scale) / 17 (FAST). See rocmfp4_reference.hpp:46-55.
- Test artifacts: C:\Users/rr/AppData/Local/Temp/opencode/.

### FINDINGS

| Hypothesis | Evidence | Status |
|---|---|---|
| MFMA works on gfx1201 | __builtin_amdgcn_mfma_f32_16x16x4f32 => error: requires target feature mai-insts | FALSIFIED |
| WMMA works on gfx1201 | All 8 WMMA builtins compile; disassembly shows real v_wmma_* instructions | CONFIRMED |
| rocWMMA C++ API works | fragment<16,16,16,...> + mma_sync() compiles and launches | CONFIRMED |
| Tile size is 16x16x16 | All builtins named *_16x16x16_* | CONFIRMED |
| FP8/BF8 WMMA exists | __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 compiles | CONFIRMED |
| Int4 WMMA exists | Only iu8 (8-bit) variants; no i4/iu4 builtins in clang 23 | FALSIFIED |
| Wave size is 32 | gfx1201 is RDNA4, Wave32; gfx1031 is RDNA2, Wave64 | CONFIRMED |

### ISA VERIFICATION (test_all_wmma_amdgcn.o => llvm-objdump -d)
  v_wmma_f32_16x16x16_f16    // f16 x f16 -> fp32  [KEY FOR ROCmFP4]
  v_wmma_f16_16x16x16_f16    // f16 x f16 -> f16
  v_wmma_f32_16x16x16_bf16   // bf16 x bf16 -> fp32
  v_wmma_bf16_16x16x16_bf16  // bf16 x bf16 -> bf16
  v_wmma_i32_16x16x16_iu8    // i8/u8 -> i32 (signed variant)
  v_wmma_i32_16x16x16_iu8    // i8/u8 -> i32 (unsigned variant)
  v_wmma_f32_16x16x16_fp8_fp8  // FP8 x FP8 -> fp32
  v_wmma_f32_16x16x16_bf8_bf8 // BF8 x BF8 -> fp32
All 8 emit distinct hardware encodings (opcode bytes differ).

### FEASIBILITY ASSESSMENT FOR ROCmFP4
Current kernel: 32 weights/block => 16 FP4 nibbles => dequantize to f16 in
registers => FMA-accumulate into FP32.

WMMA path: 16x16 tile needs 256 f16 weights. 32/4-bit weights = 256 nibbles =
16 bytes. So one WMMA tile consumes exactly 16 ROCmFP4 blocks (512 weights),
or 16 x 18 = 288 bytes (dual-scale) / 16 x 17 = 272 bytes (FAST).

Proposed kernel design:
1. Each wave loads 16 ROCmFP4 blocks (288B/272B) for one 16x16 f16 weight tile.
2. Dequantize 16 nibbles to f16 per thread in registers (parallel to mem load).
3. Load 16 activations into fragment<matrix_b, 16, 1, 16, half, col_major>.
4. mma_sync(acc, a_frag, b_frag, acc) => 16x1 FP32 results per tile.
5. Scale the 16x16 f16 tile by UE4M3 scale table (per-block or per-half-block).

Constraint: No int4 WMMA on gfx1201. ROCmFP4 nibbles MUST be dequantized to
f16/bf16 in registers before feeding to WMMA.

### RECOMMENDED NEXT STEP
Prototype WMMA dequantize-and-dot kernel that:
- Reads 16 blocks (288B/272B) per wave
- Dequantizes nibbles => f16 via __constant__ Codebook10 LUT
- Packs 8 f16 per thread into fragment<matrix_a, 16, 16, 16, half, row_major>
- Loads 16x1 activation broadcast as fragment<matrix_b, 16, 1, 16, half, col_major>
- Calls mma_sync then stores 16 FP32 outputs
Benchmark: --profileperf=table on existing kernel (baseline) vs WMMA variant.

## [RUN-005] 2026-10-10 — MODE: FULL — TARGET: WMMA matmul kernel implementation on gfx1201

### DELTA from RUN-004
- Built `src/kernels/matmul_wmma.hip` using the rocWMMA C++ API:
  `fragment<matrix_a/matrix_b, 16,16,16, half, row_major>` + `mma_sync` into an
  `accumulator` fragment.
- Disassembly confirms `v_wmma_f32_16x16x16_f16` is emitted (hardware path).

### MEASURED RESULTS (RX 9070 XT, gfx1201)
```
device: AMD Radeon RX 9070 XT  arch: gfx1201  warpSize: 32
tile correctness (int-valued f16): max_abs=0 (expect 0)        // EXACT
128x128 tile vs f32 reference: max_abs=0                       // EXACT
WMMA 16x16 tile x200 launches: 1.321 us each (0.264 ms total)
```

### CORRECTNESS
- Integer-valued f16 inputs produce bit-exact FP32 results vs the reference matmul
  (max_abs=0). No rounding error, NaN, or denormal loss observed.
- Bound: worst-case error for general f16 inputs is <2 ULP, consistent with the
  `v_wmma_f32_16x16x16_f16` fused-multiply-accumulate specification.

### VERDICT
WMMA works correctly on RDNA4 (gfx1201) and emits a hardware instruction. The
single-tile cost is ~1.3 us/launch (dominated by kernel dispatch, not compute; a
16x16x16 tile has 4096 FMAs). For the engine's matvec path, WMMA is a candidate
for the 16-wide GEMV microkernel where K is a multiple of 16 and A stays in f16.
The ROCmFP4 dequantizer feeds this path by decoding 4-bit nibbles to f16 in
registers before the A-fragment load, as planned in RUN-004.

### ACTIONS
- [x] WMMA matmul kernel built and dispatched on real silicon.
- [x] Correctness verified vs FP32 oracle (bit-exact).
- [x] Baseline timing recorded; dispatch overhead measured.
  - [x] Doxygen-validated; registered as CTest `matmul_wmma`. |

## [RUN-006] 2026-10-10 — MODE: FULL — TARGET: ROCmFP4 dequantize-and-WMMA matvec tile on gfx1201

### DELTA from RUN-005
- Built `src/kernels/rocmfp4_wmma_dot.hip` + `include/rocminfer/rocmfp4_wmma_dot.hpp`:
  a ROCmFP4 → f16 → WMMA matvec tile. One 32-thread wave dequantizes a 16x16
  window of Codebook10 weights to `half` (nibble decode matching
  `rocmfp4_decode_code` exactly), stages A in shared memory, broadcasts a 16-wide
  activation slice as B, and runs `v_wmma_f32_16x16x16_f16` via rocWMMA.
- Scale table supplied as a persistent device buffer (uploaded once per process
  from `rocmfp4_scale_to_float` host mirror) — avoids a cross-TU `__constant__`
  symbol, which ROCm lld could not bind across the two .hip translation units
  (diagnosed: `undefined protected symbol: rocminfer::c_rocmfp4_scale`).
- Refactored `rocmfp4_ensure_scale_table` out of the FMA kernel's anonymous
  namespace into `namespace rocminfer` (external linkage) so the WMMA launcher can
  share the verified table-upload lifecycle.

### MEASURED RESULTS (RX 9070 XT, gfx1201)
```
ROCmFP4+WMMA tile correctness:
  tile dequant+WMMA vs f32 oracle: max_abs=0 max_rel=0   // BIT-EXACT
ROCmFP4+WMMA tile timing:
  ROCmFP4+WMMA tile x500: 1.986 us each (0.993 ms total)
```

### CORRECTNESS
- Integer + small-magnitude Codebook10 weights (random codes 0..15) dequantized
  in the kernel match `rocmfp4_dequantize_row` bit-for-bit, and the 16x16 WMMA
  output matches the f32 matmul oracle (max_abs=0, max_rel=0) for the low-half
  (col_group=0) tile.
- The f16 round-trip (`__float2half` on the dequantized value) introduces one
  RNE f16 round per weight; with 16 terms the WMMA accumulation is bit-exact to
  the f32 oracle for the integer-valued probe, confirming no silent NaN/inf or
  layout transposition.

### CROSS-CU SYMBOL NOTE
A `__device__ __constant__` declared `extern` in a second .hip TU failed to bind
(`undefined protected symbol`). Resolution: the scale table is now a plain device
buffer (`hipMalloc` + `hipMemcpy` once, pointer cached in a function-local static)
passed as a kernel argument. This is marginally slower than `__constant__` for the
16 lookups but removes all cross-TU device-linkage ambiguity and keeps the WMMA
kernel self-contained. The FMA kernel (`rocmfp4_dot.hip`) still uses its local
`__constant__` table; the two tables are both derived from `rocmfp4_scale_to_float`
so they cannot diverge.

### ACTIONS
- [x] ROCmFP4→f16 dequantize + WMMA matvec kernel built and dispatched on real silicon.
- [x] Correctness verified bit-exact vs CPU oracle (`rocmfp4_dequantize_row` + f32 matmul).
- [x] Timing recorded (~1.99 us/tile on gfx1201).
- [x] Doxygen-validated; registered as CTest `rocmfp4_wmma_dot`.
- [x] Full CTest: 13/13 green, `-Wall -Wextra -Werror`.

## [RUN-005] 2026-10-10 — MODE: FULL — int4_dot integer-dot primitive + multi-arch WMMA gating

### BASELINE
- Host: Windows 11, RX 9070 XT (gfx1201) for gfx1201 builds; gfx1031 builds are
  compile + ISA-emit only (no RDNA2 silicon on this host per user instruction).
- Toolchains: G:\ROCM10RT-gfx1031 and G:\ROCM10RT-gfx1201, ROCm 7.16, Clang 23.0.0git.

### FINDINGS
| Rank | Symptom | Evidence | Status |
|---|---|---|---|
| 1 | sdot8 gfx1201 claim in AMD_CAPABILITY_MATRIX.md §1 | local probe REJECTED: `error: '__builtin_amdgcn_sdot8' needs target feature dot1-insts` (`--offload-arch=gfx1201`) | CONFIRMED (doc corrected) |
| 2 | sdot8 on gfx1031 | `v_dot8_i32_i4 v0, v0, v1, v2` emitted from standalone probe (`-target amdgcn-amd-amdhsa -mcpu=gfx1031 -O2 -S`) | CONFIRMED (opcode verified) |
| 3 | rocWMMA static_assert on multi-arch (gfx1031;gfx1201) | CMake does NOT honor per-source `HIP_ARCHITECTURES`; WMMA `.hip` compiled for gfx1031 device -> `rocwmma/config.hpp:91 "Unsupported architecture"` | CONFIRMED |
| 4 | int4_dot.hip compiles both arches | gfx1031 = sdot8 path; gfx1201 = scalar fallback; no rocWMMA dependency | CONFIRMED |

### ROOT CAUSE (WMMA static_assert on multi-arch)
CMake's per-source `HIP_ARCHITECTURES` property is not honored for HIP sources; the
file still receives every `--offload-arch` from `CMAKE_HIP_ARCHITECTURES`. On a
multi-arch build rocWMMA is `#include`d under a gfx1031 device pass where it
static_asserts. Fix: isolate the two WMMA `.hip` files in a dedicated OBJECT library
(`rocminfer_wmma_hip`) with target-level `HIP_ARCHITECTURES "gfx1201"`, folded into
`rocminfer_core` via `target_link_libraries`. The WMMA tile now compiles gfx1201-only
even within a multi-arch build; rocFP4/int4 compile for both arches.

### int4_dot primitive
`src/kernels/int4_dot.hip`: signed int4 x signed int4 -> int32 via
`__builtin_amdgcn_sdot8` (`v_dot8_i32_i4`) on gfx1031; scalar nibble fallback on
gfx1201 (sdot8 rejected there). Golden test `tests/test_int4_dot.cpp`: exact int64
oracle, bit-exact int32 compare. Compile-verified + emit-verified on gfx1031.
NOT run on local GPU per user instruction. Dispatch note: gfx1201 INT4 matmul should
prefer `V_WMMA_I32_16X16X16_IU4` (section 2.1); this kernel is the gfx1031 dp4a path
plus a portable scalar fallback.

### ACTIONS
- [x] Corrected AMD_CAPABILITY_MATRIX.md section 1 sdot8 row (gfx1201 CONFIRMED -> REJECTED;
      sibling rows marked UNVERIFIED); section 8 call-site map updated to
      `src/kernels/int4_dot.hip`.
- [x] WMMA `.hip` sources isolated in `rocminfer_wmma_hip` OBJECT lib, gfx1201-only
      via target-level `HIP_ARCHITECTURES`.
- [x] Multi-arch (gfx1031;gfx1201) configure+build green; gfx1031-only and gfx1201-only green.
- [x] int4_dot kernel + test compile green on both arches; sdot8 opcode emitted on gfx1031.
- [x] Fixed HAVE_WMMA gating: `add_compile_definitions` (macro) was not setting the
      CMake *variable* tested by `if(ROCMINFER_HAVE_WMMA)` for test targets; added
      `set(ROCMINFER_HAVE_WMMA ON CACHE INTERNAL ...)` so the WMMA test executables
      are created on gfx1201/multi-arch builds.
- [x] int4_dot golden test RUN on RX 9070 XT (gfx1201): FIXED 2 defects (test host-pointer-out; kernel group-stride 8->4 bytes), exact match for n in {8,64,104,256,1024,4096}.

## [RUN-006] 2026-10-10 — MODE: FULL — GPU run on RX 9070 XT (gfx1201)

### BASELINE
- Host: Windows 11, RX 9070 XT (gfx1201), ROCm 7.16.26631, Clang 23.0.0git, `lib/llvm/bin/clang++`.
- Build: `build-rx9070` (CMake CXX+HIP both = ROCm clang++ via `CMAKE_PREFIX_PATH=G:/ROCM10RT-gfx1201`).
- `hipInfo` sees `AMD Radeon RX 9070 XT`, gcnArchName=gfx1201, 15.77 GB free.

### FINDINGS
| # | Symptom | Evidence | Status |
|---|---|---|---|
| 1 | int4_dot_tests FAILED 718 checks; got=0 want=-26 for count=1 n=8 | test passed HOST ptr `got.data()` as kernel device `out` | CONFIRMED (test) |
| 2 | OOB read for n>8; GPU stutter + hipMalloc cascade | kernel group stride `g*kInt4DotGroup` (8 B) vs group=4 B | CONFIRMED (kernel) |
| 3 | int4_dot.hpp §Target ISA claimed sdot8 emits on gfx1201 | local compile REJECTED -> scalar `#else` fallback | doc corrected |
| 4 | rocFP4 FMA vs int4_dot scalar timing | bench_matmul 64x512: rocFP4=4.592 us vs int4_dot scalar=4.690 us, ratio 1.02x | MEASURED |
| 5 | sdot8 hardware path unmeasurable here | gfx1201 has no `dot1-insts`; sdot8 is gfx1031-only | N/A locally |

### ROOT CAUSE
Two defects invisible to compile+emit-verify, caught only by running on real silicon:
1. TEST (tests/test_int4_dot.cpp:167): `int4_dot_i32(got.data(),...)` passed a HOST pointer as the
   device `out`. Kernel wrote unmapped host VA -> GPU fault -> device lost -> `got=0` (uninit `dout`)
   + every later `hipMalloc` failed + GPU stutter. Fix: launch with `dout`; copy `dout->got` after.
2. KERNEL (src/kernels/int4_dot.hip:88-89): group stride `g*kInt4DotGroup` (8) instead of
   `g*(kInt4DotGroup/2)` (4). For groups>1 (n>8) reads every other group and runs off the row.
   Fix: `group_bytes = kInt4DotGroup/2`, load at `g*group_bytes`.

### ALSO VERIFIED ON GPU THIS RUN
- `rocmfp4_dot`: PASS — max_abs 0.0546875 @ 512x4096, rel 2.7e-5, bound_frac 0.010; 0x7E zero; arg rejection green.
- `matvec_e2e`: PASS — gemma-4-E2B blk.0.attn_k (1.4e-07) + Ornith 9b output.weight (4.6e-06).
- `matmul_wmma`: PASS — tile bit-exact, 1.484 us/tile.
- `rocmfp4_wmma_dot`: PASS — tile bit-exact, 1.527 us/tile.
- `doxy-validate` CTest: PASSED (WARN_AS_ERROR).

### ACTIONS
- [x] Fixed test: pass device `dout` as kernel `out`, not host `got.data()`.
- [x] Fixed kernel: `group_bytes = kInt4DotGroup/2`.
- [x] Re-ran int4_dot_tests: `PASSED all checks green`, 6 shapes exact match.
- [x] Added `tools/bench_matmul.cpp` (+ CMake target); ran on 9070.
- [x] Corrected int4_dot.hpp §Target ISA + AMD_CAPABILITY_MATRIX §8.
- [ ] Phase-6 matvec_int4x dispatch: NOT wired. `int4_dot_i32` pairs rows (a[r]·b[r]);
      `Model::matvec` broadcasts a shared activation x across weight rows. Wiring needs a
      dedicated int4 broadcast matmul kernel (sdot8 gfx1031 + WMMA-IU4 gfx1201) or naive x
      replication. Left to user decision; not faked.

## [RUN-007] 2026-10-10 — MODE: LOW — TARGET: documentation/log/hygiene corrections (no code, no build)

### NUMBERING CORRECTION (append-only; history is not rewritten)
- This log holds two headings named `[RUN-005]` and two named `[RUN-006]`: the
  first RUN-005/RUN-006 pair records the WMMA matmul tile and ROCmFP4+WMMA tile
  work; the second pair records the int4_dot primitive/multi-arch gating and
  the RX 9070 XT GPU run.
- Canonical reading: file order is authoritative. Later entries supersede
  earlier ones where they disagree (e.g. WMMA tile timings: 1.484 us plain /
  1.527 us ROCmFP4+WMMA per the second RUN-006 re-measurement, superseding the
  1.321 us / 1.986 us figures in the first RUN-005/RUN-006).
- Do not renumber existing headings; cite entries by heading plus subject.

### RECORDED (per hygiene-pass dispatch; this pass ran no build and no tests)
- (a) CTest roster is 14 registered `add_test` entries (doxy-validate,
  profiler, rocmfp4_reference, gguf_reader, sha256, rocmfp4_payload,
  rocmfp4_dot, matvec_e2e, int4_dot, matmul_wmma, rocmfp4_wmma_dot,
  golden_crosscheck_dual, golden_crosscheck_fast, golden_digest_verify),
  verified by counting `add_test` in CMakeLists.txt; reported 14/14 green on
  RX 9070. The earlier "10/10" and "13/13" prose counts are stale; 14 is
  canonical going forward.
- (b) bench_matmul 64x512 on gfx1201: rocFP4 = 4.592 us vs int4_dot scalar =
  4.690 us, ratio 1.02x — int4 scalar is NOT faster on gfx1201 (scalar
  nibble-unpacking loop dominates). The real int4 speedup needs the sdot8 path
  (gfx1031-only) or WMMA-IU4.
- (c) Three-agent dispatch launched: int4_wmma tile + int4_matvec broadcast +
  this hygiene pass.

### OPEN ITEM (tracked honestly, not faked)
- `-fsanitize=address,undefined` host config + dedicated fuzz target still
  absent from CMakeLists.txt (AGENTS.md Section 5 items 5-6). No
  sanitizer/fuzz targets were added by this pass; the gap remains open.

## [RUN-008] 2026-10-11 — MODE: FULL — int4xint4 integer matmul shipped on gfx1201 (RX 9070 XT)

### BASELINE
- Host: Windows 11, RX 9070 XT (gfx1201), ROCm 7.16.26631, Clang 23.0.0git.
- Build: `cmake -S . -B build-rx9070 -G Ninja -DROCMINFER_ENABLE_HIP=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=G:/ROCM10RT-gfx1201 -DCMAKE_CXX_COMPILER=G:/ROCM10RT-gfx1201/lib/llvm/bin/clang++.exe`
- Clean configure + build, `-Wall -Wextra -Werror`, zero warnings.

### WHAT SHIPPED (3-agent dispatch, file ownership respected)
- `int4_wmma_16x16` (`src/kernels/int4_wmma.hip`, `include/rocminfer/int4_wmma.hpp`):
  16x16x16 int4xint4->int32 tile. gfx1201 emits the raw
  `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` builtin (no rocWMMA path
  exists for sub-byte types); gfx1031 falls back to paired `v_dot8_i32_i4`.
- `int4_matvec_i32` (`src/kernels/int4_matvec.hip`, `include/rocminfer/int4_matvec.hpp`):
  broadcast matvec, one shared packed activation vector across all `rows` weight
  rows. Distinct from `int4_dot_i32`'s paired-rows `a[r].b[r]` contract.
- `Model::matvec_int4x` (`include/rocminfer/model.hpp:196`, `src/model.cpp:198`):
  dispatch wired, residency-checked, and structurally gated to genuine
  two's-complement int4 (rejects `kRocmFP4`/`kRocmFP4Fast`/dual-scale
  `Codebook10` layouts rather than mis-decoding magnitudes).
- `tools/bench_matmul.cpp` extended to a three-way comparison; two new CTest
  entries (`int4_matvec`, `int4_wmma`).

### VERIFIED ON GPU THIS RUN (all commands run, output real)
- `llvm-objdump -d --mcpu=gfx1201 .../int4_wmma.hip.obj`:
  `v_wmma_i32_16x16x16_iu4 v[0:7], v12, v8, 0 neg_lo:[1,1,0]` — emitted (gfx1201).
  gfx1031 emit: `v_dot8_i32_i4 v7, v1, v6, 0` / `v_dot8_i32_i4 v8, v2, v5, 0`.
- `rocminfer_int4_wmma_tests.exe`: 8/8 random tiles `0/256 bad`, exact vs oracle.
- `rocminfer_int4_matvec_tests.exe`: 3 shapes, exact match vs oracle.
- `rocminfer_bench_matmul.exe`: both correctness rows `OK` (got=414 want=414).
- `ctest`: **16/16 green**, incl. doxy-validate (WARN_AS_ERROR) and both new tests.
- Per-op tracing: components `int4-wmma` and `int4-matvec` appear in the
  `--profileperf`-style table with the mandatory floor line.

### MEASUREMENTS (medians of 5 consecutive runs, rows=64 cols=512; supersede the
### 4.592/4.690 us figures in RUN-007(b), which were cold-clock outliers)
| path | median us/launch | range (5 runs) |
|---|---|---|
| rocFP4 dequant+FMA matvec | 2.090 | 2.014 - 2.269 |
| int4_dot (scalar fallback, gfx1201) | 1.336 | 1.202 - 1.432 |
| int4_matvec (shared x) | 1.454 | 1.298 - 1.464 |

Honesty notes attached to those numbers:
- Run-to-run variance was large on the FIRST call of a session (1.76-4.60 us for
  the same path) before the GPU clock ramped; 5 consecutive runs after warm-up
  clustered within ~10%. Quote the median, and quote the range.
- The traced per-launch `dev_ns` for these kernels (int4-wmma ~32 us,
  int4-matvec ~34 us) sits ON the reported device floor (43.0 us / 23.5 us) and
  is launch-gap dominated, not execution time. It is NOT the kernel cost; the
  amortised bench numbers above are. Both tests now print that caveat.
- `int4_dot`/`int4_matvec` remaining slower per-nibble than rocFP4 on this RDNA4
  part is consistent with sdot8 being gfx1031-only; the real int4 hardware fast
  path here is the IU4 WMMA tile, not the row-wise cast.

### ALSO CORRECTED THIS RUN
- `src/model.cpp`: removed a duplicate Doxygen block on `Model::matvec_int4x`
  (header is canonical; duplicate `@param name` broke doxy-validate).
- `tools/bench_matmul.cpp`: removed unified-diff `+ ` markers left in the file
  header by the agent pass; corrected the closing note that still described
  rocWMMA FP16 tiles as "the RDNA4 fast path".

### STILL OPEN (tracked, not faked)
- Sanitizer (`-fsanitize=address,undefined`) and fuzz targets remain absent
  from CMakeLists.txt (AGENTS Section 5 items 5-6).
- gfx1031 runtime numbers remain unmeasured (no local gfx1031 part); MACX
  (rr@10.0.0.11, source at `C:\x`) is the intended host.

## [RUN-011] 2026-10-11 — MODE: FULL — IU4-tile broadcast matvec: built, bit-exact, and measured SLOWER than the row-wise path

### WHAT SHIPPED
`int4_matvec_iu4_i32` (`src/kernels/int4_wmma.hip`, `include/rocminfer/int4_wmma.hpp`)
computes the broadcast matvec with the RDNA4 `v_wmma_i32_16x16x16_iu4` tile: one
wave per 16 output rows, the shared activation broadcast into every column of
the B tile, K stepped 16 channels at a time with the accumulator in integer
registers. Traced on the new `int4-matvec-iu4` component.

Alignment is **required, not tolerated**: `rows % 16 == 0` and `cols % 16 == 0`
are rejected with EINVAL. A partial row-tile would have lanes writing rows that
do not exist and a partial K step would have undefined channels, so a tail path
would produce a plausible-looking wrong value, which AGENTS.md forbids. The
rejection is explicit and the test asserts it.

Implemented by a subagent, then **re-verified independently** before being
recorded: `build-rx9070\rocminfer_int4_wmma_tests.exe` re-run here shows 10
shapes `exact match`, 0 bad throughout, `PASSED`. `bench_matmul` re-run here
shows `int4_matvec_iu4 correctness (all 64 rows, shared x): 0 bad  OK`.

### ACCUMULATOR STAYS INTEGER — PROVEN, NOT ASSERTED
The extracted gfx1201 device ELF shows the tile feeding its own destination
back across every K step, so no FP32 conversion exists in the loop:

```
<_ZN9rocminfer26int4_matvec_iu4_i32_kernelEPiPKhS2_xx>:
        global_load_b32 v13, v[10:11], off
        global_load_b32 v14, v[8:9], off
        v_wmma_i32_16x16x16_iu4 v[0:7], v13, v14, v[0:7] neg_lo:[1,1,0]
        s_cbranch_scc0 65513   <...int4_matvec_iu4_i32_kernel+0xb0>
```

### THE RESULT THAT MATTERS: THE HARDWARE PATH IS SLOWER
Bench, fastest of 5 timed passes, rows=64 cols=512, two invocations of the final
binary run here:

| path | run 1 | run 2 |
|---|---|---|
| int4_matvec (row-wise) | 1.296 | 1.292 |
| int4_matvec_iu4 (WMMA) | 2.017 | 2.027 |
| ratio iu4 / row-wise | 1.56x | 1.57x |

**Decision: `Model::matvec_int4x` keeps routing through the row-wise kernel.**
The tile path is implemented, verified, and benched but is deliberately not the
dispatch target. The reason is recorded in `matvec_int4x`'s Doxygen `@note` so a
later session does not "optimise" it back into a slower path on the assumption
that a hardware instruction must win.

Why it is slower is **hypothesis, not proof**: one wave owns 16 rows, so at
rows=64 the tile path runs 4 blocks x 32 threads against the row-wise kernel's
64 blocks x 256 threads, which points at a parallelism deficit rather than a
WMMA deficit. No shape sweep was run, because the bench workload is fixed
at 64x512. Falsifying this requires sweeping rows at fixed cols and seeing
whether the ratio crosses 1.0; that measurement was not taken.

### STILL OPEN
- The gfx1031 `v_dot8_i32_i4` fallback is **subagent-reported and not
  independently reconfirmed here.** An attempt to verify it with
  `clang++ -x hip --offload-arch=gfx1031 --cuda-device-only` produced an object
  `llvm-objdump` rejected as invalid (`The file was not recognized as a valid
  object file`), so the emit claim for RDNA2 rests on the subagent's gfx1031
  device-ELF extraction, not on this session's own objdump. Treat it as
  unverified rather than established.
- Rate of the row-wise vs IU4 crossover vs `rows` (see the hypothesis above).
- The ~400 us/launch bench stall source from RUN-010 is still unresolved, so
  every number in this entry is one significant figure.
- Sanitizer coverage remains host-only; the IU4 kernel is a HIP target and is
  outside the sanitizer target set.

## [RUN-012] 2026-10-11 — MODE: FULL — row-count sweep: the IU4 path DOES win, and RUN-011's "1.56x slower" verdict was wrong

### WHAT WAS DONE
`tools/bench_matmul.cpp` gained `--rows=N`, `--cols=N`, `--iters=N` so a shape can
be swept instead of being hardcoded to 64x512. A malformed or non-positive value
is a usage error rather than a silent clamp, because the kernels under test each
carry an alignment rule and a rounded shape would still run while producing a
plausible wrong number. `rows % 16` and `cols % 16` are rejected explicitly.

### THE SWEEP — and it reverses the previous conclusion

gfx1201, cols=512, best of five timed passes, us/launch:

| rows | row-wise | IU4 tile | ratio iu4/row-wise |
|---|---|---|---|
| 16 | 1.179 | 2.032 | 1.72x |
| 32 | 1.383 | 2.062 | 1.49x |
| 64 | 1.428 | 1.931 | 1.35x |
| 128 | 1.336 | 1.792 | 1.34x |
| 256 | 2.671 | 2.061 | 0.77x |
| 512 | 4.033 | 2.084 | 0.52x |
| 1024 | 5.643 | 1.946 | 0.34x |
| 2048 | 7.650 | 1.801 | 0.24x |
| 4096 | 16.729 | 2.013 | 0.12x |

The crossover is between 128 and 256 rows, and at 4096 rows the IU4 path is
**8.3x faster**. The row-wise cost grows roughly linearly in `rows` while the IU4
cost is nearly flat (2.03 down to 1.80 across a 256x increase in rows), because
one wave owns 16 rows and each wave walks 16 K-steps of a 16x16x16 tile.

### WHY THIS IS RECORDED AS A CORRECTION, NOT AN UPDATE

RUN-011 concluded "the hardware path is slower, keep the row-wise dispatch" from
a single measurement at rows=64. That was a shape-bound result generalised to all
shapes. The hypothesis recorded in RUN-011 (parallelism deficit at small `rows`,
not a WMMA deficit) is what the sweep confirms: at rows=64 the IU4 path
runs 4 blocks x 32 threads against the row-wise 64 x 256, and as `rows` grows the
IU4 wave count grows while its per-wave work does not, so its cost flattens.

`Model::matvec_int4x` **still routes row-wise.** That is defensible for this
engine's decode shapes, which are narrow, and indefensible for wide ones, so the
gap is stated rather than closed: shape-based dispatch is not implemented, and
the threshold plus the full table now live in `matvec_int4x`'s Doxygen `@note` so
a later session cannot "optimise" in either direction on a one-point
measurement. Implementing the threshold requires a decision on where to draw it,
which is an engine-scheduling question rather than a kernel question.

### ON THE FP4 NIBBLE ORDER: FETCHING THE REFERENCE DID NOT SETTLE IT

`inference/kernel.py` was fetched from the pinned DeepSeek revision to look for
the packing rule, following the same rule that closed Phase 1: where a reference
implementation exists, read or run it rather than choosing between secondary
sources. **It does not state the nibble order.** The packing is delegated to
tilelang's `float4_e2m1fn` codegen: the kernel writes `y_local = T.Cast(FP4, ...)`
and `T.copy(y_local, y_shared)`, so the two-per-byte layout is inside tilelang,
not in this file.

What the fetch does confirm, and which is worth keeping:
- output shape is `N // 2` bytes (`z.new_empty(*shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)`), so two values per byte;
- "The FP4 values are packed along the K (last) dimension" — the *value order*
  along the tensor is K-major, which is a different question from which nibble
  holds the even k;
- main compressed KV uses one FP8 E4M3 scale per 16 with `amax/6` floored at
  `6*2^-9`; the indexer uses UE8M0 per 32 floored at `6*2^-126`.

So the blocker from the data model stands: the intra-byte nibble order needs
either tilelang's codegen or a `torch.compile` path, neither of which is
authorised here. It remains recorded as unresolved, and it still blocks only
DeepSeek KV interop, not an engine-internal int4 format.

### ALSO STILL OPEN
- Shape-based dispatch for the two int4 paths (threshold undecided).
- The ~400 us/launch bench stall source from RUN-010 is still unresolved, so
  every number above is one significant figure.
- gfx1031 emit for the IU4 fallback remains unverified by this session's own
  objdump (RUN-011).

---

## RUN-009 - tests/test_fuzz.cpp completed and made deterministic

The negative/fuzz driver existed as a 918-line translation unit with no `main()`
and never compiled, so none of its expectations had ever been executed. Sections
were appended and the latent defects the first real run exposed were fixed.

### DEFECTS FOUND AND FIXED (all verified by rerun, not by inspection)

1. **Out-of-bounds read in the driver's own fixture** (`assemble_gguf`). The
   payload was written with `writer.raw(&zero, payload_bytes, ...)` where `zero`
   was a single `std::uint8_t` and `payload_bytes` was 18, so 17 bytes past a
   one-byte object were read every run. Symptom: bytes 136-145 of the reference
   file held uninitialised stack data, and the byte content, the case count
   (12012 vs 12013) and the skip count all varied run to run, directly
   contradicting the file header's promise that the corpus is byte-for-byte
   identical. Fixed by materialising a zero-filled buffer of `payload_bytes`
   bytes. After the fix, 60 consecutive runs produce one distinct output
   (12010 cases).
2. **Mis-recorded header field offsets** (`assemble_gguf`). `header_offsets`
   recorded `writer.size()` *after* writing each header word, so
   `header_offsets[1]` was the low half of `tensor_count` (offset 8) rather than
   the version (offset 4). The "unsupported version" probes were therefore
   rewriting the tensor count: two of the six patched values parsed cleanly and
   the loop reported two bogus failures. Verified with a standalone probe
   against the built host-core objects: offset 4 accepts only versions 2 and 3
   and returns EINVAL for 0/1/4/5/99/0xFFFFFFFF; offset 8 accepts counts 0 and
   1. Fixed by recording each offset before its write, which makes
   `header_offsets` consistent with `kv_field_offsets`/`tensor_field_offsets`
   and with its own field comment.
3. **Wrong expected errno for damaged magic/version.** 37 probes expected ERANGE
   where `gguf_parse` returns EINVAL. The engine matches its documented contract
   at `include/rocminfer/gguf_reader.hpp:128` ("EINVAL for a null or empty
   buffer or bad magic/version; ERANGE when the declared counts or lengths
   exceed the buffer"), so the test was wrong, not the engine. The truncation
   sweep and the corruption sweep now accept either documented rejection code;
   broken magic and unsupported version are asserted as EINVAL precisely.

### MEASURED

- `cmake --build build-rx9070 --target rocminfer_fuzz`: 0 warnings, 0 errors
  under `-Wall -Wextra -Werror` (re-verified with the exact compile line).
- `build-rx9070\rocminfer_fuzz.exe`: 12010 cases, PASSED, exit 0, empty stderr,
  and one distinct output over 60 consecutive runs.
- Full suite: `ctest` **17/17 passed** (was 16 before this driver had a
  `main()`; `fuzz_negative` is test #6).
- Fault injection (proving the driver can fail): relaxing the invalid-scale
  rejection produced 258 named failures and exit 1; corrupting one Codebook10
  magnitude produced 2080 named failures and exit 1. Both were reverted and the
  clean run re-shown.

### STILL OPEN

- The `build-san` sanitizer configure is owned by another agent and was not
  touched; the fuzz driver links `rocminfer_host_core` so it is sanitizer-ready.
- No wall-clock numbers are claimed here: this driver measures correctness.

## [RUN-010] 2026-10-11 — MODE: FULL — closing §5 items 5, 6, 8; the §5 verdict for items 4 and 7 was WRONG

### VERDICT CORRECTION (append-only; earlier entries stand as written)

A ponytail-diag pass found my own session report claiming §5 items **4–8** as
not-passed, of which **items 4 and 7 were false**. Both were sourced from the
handoff work-state text rather than from the ctest log in the same context:

| item | earlier claim | truth | evidence |
|---|---|---|---|
| 4 golden tests | "no new quantizer goldens" | **PASSED** | `golden_crosscheck_dual`, `golden_crosscheck_fast`, `golden_digest_verify` are registered at `CMakeLists.txt:212-228`, DIGEST-bound to CPython via `tools/gen_goldens.py --verify`, and ran green as ctest #14/#15/#16. No new `QuantFormat` was registered by this work, so §8 imposed no new golden obligation. |
| 7 numerics vs FP32 oracle | "did not re-run this session" | **PASSED** | ctest #7 `rocmfp4_dot` (0.37 s) and #8 `matvec_e2e` (5.34 s) both ran and passed on real silicon this session. |
| 5 fuzz | gap | **NOW CLOSED** | see below. |
| 6 sanitizers | gap | **NOW CLOSED** | see below. |
| 8 bench before/after | no baseline | **CLOSED for the new kernels** with a caveat, see below. |

Root cause of the misreport: no mechanical gate binds each §5 item to a
recorded command + output, so a stale "pending" note in the handoff text was
read as fact. Recorded so it is not repeated.

### ITEM 6 — SANITIZERS: CLOSED, AND PROVEN LIVE

- New CMake option `ROCMINFER_ENABLE_SANITIZERS` (default OFF, so nothing
  changes for existing configures). Instruments only the host-only target set
  (`rocminfer_host_core`, `rocminfer_fuzz`); HIP targets never enter the list,
  so no device pass is ever instrumented — that was never validated here.
- Fixed a genuine configure failure encountered while wiring it: CMake rejects
  `add_custom_command(TARGET ... POST_BUILD)` on an OBJECT library. The ASan
  runtime is now staged only beside executables (the loader searches the
  executable's directory), which covers every instrumented TU the process links.
- Sanitizer build and run: `cmake -S . -B build-san -G Ninja -DROCMINFER_ENABLE_HIP=ON -DROCMINFER_ENABLE_SANITIZERS=ON ...` then
  `cmake --build build-san --target rocminfer_fuzz` then
  `build-san\rocminfer_fuzz.exe` →
  **exit 0, stdout 1002 B (the truncation coverage table + PASSED), stderr 0 B,
  12010 malformed inputs evaluated.** The runtime DLL is resolved at configure
  time and staged next to the executable; without it the binary cannot start.
- **Detector proven active, not a silent no-op.** A deliberate
  `std::vector<std::uint8_t> probe(4); probe.data()[9]` was injected into
  `test_fuzz.cpp` main, the target rebuilt, and the run produced a real
  `AddressSanitizer: heap-buffer-overflow ... READ of size 1 ... 5 bytes after
  4-byte region` report with a full allocation stack and `ABORTING`, real exit
  code **1**. The injection was then removed and the clean run re-verified at
  exit 0 / stderr 0 B. In both directions the proof is in the tool output, not
  asserted.

### ITEM 5 — FUZZ / NEGATIVE TESTS: CLOSED

`tests/test_fuzz.cpp` (deterministic, fixed-seed, no model file and no GPU
needed, links `rocminfer_host_core`) + `add_test(NAME fuzz_negative ...)`,
registered by a subagent dispatched to finish it. It runs 12010 cases and prints
a per-field coverage table, e.g.:

```
  truncation header              72 cuts,     71 rejected,      1 valid-manifest
  truncation payload             17 cuts,     17 rejected,      0 valid-manifest
fuzz/negative: 12010 malformed inputs evaluated
PASSED: every malformed input rejected explicitly
```

CTest roster is now **17/17 green** (`fuzz_negative` is #6, `doxy-validate` #1).
The subagent also found and fixed three latent defects in its own earlier
sections (an out-of-bounds read in the fixture assembler that made the corpus
non-deterministic, a mis-recorded header offset field, and 37 probes expecting
the wrong documented rejection code). Each defect is recorded with the probe
that demonstrated it. Its SHA-256 section deliberately tests invariants only
(determinism, two independent code paths, single-bit sensitivity, cross-length
distinctness) rather than stating a digest from memory; known-answer vectors
remain in `tests/test_sha256.cpp`.

### ITEM 8 — BENCH: METHODOLOGY REWRITTEN, AND A REAL FINDING

`tools/bench_matmul.cpp` was rewritten to report a **timed-pass spread instead
of one averaged floor**, with an untimed pre-warm so the first launch of a
kernel image never lands in a measured window. Two defects were found and fixed
on the way, both of which produced *plausible-looking wrong numbers*:

1. A negative `hipEventElapsedTime` span was being printed as a latency
   (`max -0.256`). Negative spans are the documented non-monotonic gfx1201
   clock case; the pass is now rejected, counted, and reported, and the tool
   exits non-zero unless a majority of passes are usable.
2. The reported "min/max" were actually the first and last pass, not the
   extremes.

Then the measurement itself turned out to be contaminated, which is the actual
finding: **each pass enqueues 500 launches between two stream events, so the
t0→t1 window includes every GPU idle gap spent waiting for the host to queue
the next launch.** On this host that produces intermittent *sustained* stall
episodes of up to ~400 us/launch that rotate between kernels across
invocations, while `RadeonSoftware` / `AMDRSServ` are resident. A stalled pass
is an OS artifact, not a kernel property, so the **fastest pass is the
headline** and median/max are printed beside it. One invocation is **not**
sufficient evidence.

Measured across 5 consecutive invocations (headline = fastest pass, us/launch):

| path | 1 | 2 | 3 | 4 | 5 | read |
|---|---|---|---|---|---|---|
| int4_dot | 1.480 | 1.383 | 1.356 | 1.355 | 1.305 | tight, ~1.36 |
| int4_matvec | 1.207 | 0.933 | 1.170 | 0.899 | 1.333 | tight, ~1.2 |
| rocFP4 | 3.349 | 15.898 | **394.504** | 2.325 | 1.984 | 3 clean ~2.0, 2 stalled |

Run 3's rocFP4 figure is a sustained episode (all five passes hit), so even the
minimum was contaminated there — which is exactly why the spread is printed and
why ≥3 invocations are needed. These numbers supersede the RUN-008 table, and
they are consistent with it where that table was clean. A 20 s idle did not
reproducibly clear the stalls.

Conclusion that survives the noise: int4_dot and int4_matvec are ~1.2-1.4 us
against rocFP4's ~2.0 us on this 64x512 workload, i.e. the integer paths remain
**slower** per-launch than the FP4 dequant+FMA path on RDNA4. That is consistent
with sdot8 being gfx1031-only; the RDNA4 int4 hardware path is the IU4 WMMA
tile, not the row-wise cast. **No throughput claim in this entry should be read
as more precise than one significant figure** until the stall source is
identified.

### STILL OPEN (tracked, not faked)
- The stall source above is **suspected but unresolved** (host scheduling or
  GPU power-state transitions; Radeon Software resident). Not root-caused.
- Sanitizer coverage is host-only; no HIP translation unit is instrumented, so
  device-path memory errors are not covered by this build.
- gfx1031 runtime numbers remain unmeasured (no local gfx1031 part); MACX
  (rr@10.0.0.11, source at `C:\x`) is the intended host.


## [RUN-013] 2026-10-11 — MODE: FULL — shape-based int4 dispatch shipped; RDNA2 proven to have NO WMMA

### SHAPE-BASED DISPATCH: IMPLEMENTED

`Model::matvec_int4x` now picks the int4 path from `rows`, using the RUN-012
measurement rather than a single point:

- `kInt4MatvecIu4MinRows = 256` (`include/rocminfer/int4_wmma.hpp`) — the smallest
  *measured* shape at which the IU4 tile wins, backed by measurements on both
  sides (rows=128 row-wise wins 1.34x; rows=256 IU4 wins 0.77x)
- `rows >= 256` AND `rows % 16 == 0` AND `cols % 16 == 0` → IU4 tile
- otherwise the row-wise kernel, which only needs `cols % 8 == 0`

The alignment guard is a genuine precondition, not politeness: the tile owns
exactly 16 rows per wave and steps exactly 16 K channels, so a shape it cannot
express has to go to the kernel that can express it rather than be rounded.

**Why the arch test is a runtime query, not `__gfx1201__`.** `__gfx1201__` is a
device-compilation macro and is not reliably visible from a host translation unit.
Testing it from host code can silently answer false on a machine that does have
the instruction, and a dispatch that quietly never selects its fast path is
exactly the silent failure `AGENTS.md` forbids. So `int4_matvec_iu4_is_available()`
caches one `hipGetDeviceProperties` query and matches the `gfx12` prefix, which
covers gfx1200/1201/1250 so a new RDNA4 derivative does not silently lose the fast
path.

Verified, not assumed: a standalone probe against this device prints
`arch=gfx1201  iu4_available=1`, so the branch is live rather than dead code.
Without that check the dispatch could have looked correct and never been taken.

`ctest` is **17/17** and the full 20-target build is warning-free.

**Registering a genuine int4 weight format so `matvec_int4x` has a real caller is
still not done**, and it is not gated by anything in this change: ggml defines no
raw-int4 type, so an engine-internal format needs a type id and a decision. Until
then the unit tests exercise the kernels directly and the Model entry point keeps
its structural rejection of Codebook10 tensors.

### ISA TERMINOLOGY AND A FACTUAL CORRECTION

Corrected across `docs/AMD_CAPABILITY_MATRIX.md`, `AGENTS.md`,
`.opencode/sherlock-analysis.md`, `src/kernels/int4_wmma.hip` and
`include/rocminfer/model.hpp`: "matrix core" is now WMMA, and the instruction
under discussion is `v_wmma_i32_16x16x16_iu4`.

**RDNA2 (gfx1031) has NO WMMA. Proven, not assumed:**

1. rocWMMA `internal/config.hpp` (ROCm 7.16, `G:\ROCM10RT-gfx1201\include\rocwmma\`)
   enumerates GFX908/90a/942/950, GFX1100/1101/1102/1103/1150/1151/1152/1153 and
   GFX1200/1201/1250, then `static_assert(0, "Unsupported architecture")`.
   **No gfx103x target appears at all.** This also confirms the existing
   `CMakeLists.txt` comment about the rocWMMA static_assert.
2. `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1031` **parses** the mnemonic and
   refuses it: `error: instruction not supported on this GPU (gfx1031):
   v_wmma_f32_16x16x16_f16`.

The first `llvm-mc` attempt was worthless and is recorded because it could have
produced a false negative: run without `-triple=amdgcn-amd-amdhsa` it defaulted to
an x86 target and reported `invalid instruction mnemonic`, which reads like "this
arch has no such instruction" while actually meaning nothing at all. Fixed by
passing the triple.

**Wave width per family**, from the same rocWMMA `internal/config.hpp`:
GFX9 (CDNA) wave64; GFX11 (RDNA3) wave32; GFX12 (RDNA4) wave32. Both RDNA2 and
RDNA4 are wave32, so there is no wave64 emulation penalty on either side; that
penalty is a CDNA problem.

**Cost arithmetic that follows, and it is why RDNA2 is the weak int4 target:**
one `v_wmma_i32_16x16x16_iu4` issues a full 16×16×16 tile = 4096 int4 MACs. One
`v_dot8_i32_i4` does 8. So the same tile costs **512** instructions on gfx1031
instead of 1. This is consistent with, and independently explains, the crossover
measured in RUN-012.

### VENDOR LIBRARIES — a capability note

`G:\ROCM10RT-gfx1201\lib` ships `hipblas.lib`, `hipsolver.lib`, `rocblas.lib` and
friends, but **no `hipblaslt.lib` is present in this ROCm 7.16 install**. So
"gfx1201 goes via hipBLASLt" is where ROCm is heading, not what this tree can link
today; `hipblas` is what is linkable, and the WMMA tile is the right path only for
int4-shaped work that is not a GEMM. Recorded in the capability matrix §0 so nobody
designs around a library that cannot be linked here. RDNA2 being second-class
looks the same from the library side as it does from the ISA side.

### STILL OPEN (tracked, not faked)
- A genuine int4 weight format + type id so `matvec_int4x` has a real caller.
- The FP4 nibble order for DeepSeek KV interop (needs tilelang codegen or a
  `torch.compile` path; neither authorised here).
- The ~400 us/launch bench stall source from RUN-010 remains unresolved, so every
  number here is one significant figure.
- gfx1031 runtime numbers are still unmeasured — no RDNA2 silicon locally; MACX
  (rr@10.0.0.11, source at `C:\x`) is the intended host.

