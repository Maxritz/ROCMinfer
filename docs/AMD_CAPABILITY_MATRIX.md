# ROCmInfer Capability Matrix — ISA-backed, compiler-verified

Authoritative, instruction-by-instruction. Built from the two pinned ISA documents
(`rdna4-instruction-set-architecture.md`, `rdna2-shader-instruction-set-architecture.txt`),
the local compiler builtin registry (`IntrinsicsAMDGPU.td`, fetched from LLVM trunk and
cross-checked against the in-tree ROCm 7.16 at `G:\ROCM10RT-gfx1031` /
`G:\ROCM10RT-gfx1201` — these are your `ROCMRT-gfx*` trees), and **real local clang probes**
(`-target amdgcn-amd-amdhsa -mcpu=gfxNNN`), emitted ISA inspected with `llvm-objdump` and
the inline-asm / builtin pass/fail captured verbatim.

Two compile results are reproduced in the Verification column exactly as the compiler
spoke, so the matrix never claims what was not tested.

## 0. Before you hand-roll a GEMM: check the vendor library first

The architectures ROCm supports well are not symmetric, and that asymmetry should
decide where effort goes:

- **gfx1201 / GFX12 (RDNA4)** is a first-class ROCm target. For general GEMM the
  right answer is `hipBLASLt`, and for narrow decode-shaped work the hardware
  `v_wmma_i32_16x16x16_iu4` path this repo ships. Hand-rolled tiled GEMM on top of
  that is only justified where the shape is unlike a GEMM, such as the broadcast
  matvec in `int4_matvec_iu4_i32`.
- **gfx1031 / GFX10 (RDNA2) is a second-class ROCm target.** rocWMMA does not list
  it (§1), int4 has no WMMA (§1), and library coverage is thinner. Its int4 path is
  `v_dot8_i32_i4`, and the cost of a 16x16x16 tile there is 512 of those
  instructions rather than one WMMA (§1). Plan for the RDNA2 path to be a
  different, slower implementation, not the same code at a different clock.

**Verified on this host, 2026-10-11:** `G:\ROCM10RT-gfx1201\lib` ships
`hipblas.lib`, `hipsolver.lib`, `rocblas.lib` and friends, but **no
`hipblaslt.lib` is present in this ROCm 7.16 install**. So the "gfx1201 goes via
hipBLASLt" guidance above describes where ROCm is heading, not what this tree can
link today: anything using it has to bring its own hipBLASLt, and until then
`hipblas` (or the WCMA tile, for int4-shaped work) is what is actually linkable.

## Status legend

| Status | Meaning |
|---|---|
| **CONFIRMED** | Local ROCm 7.16 clang accepts the builtin / the assembler emits the ISA instruction on that target. |
| **REJECTED** | Local clang/assembler refused the instruction/builtin on that target (`instruction not supported on this GPU` or `unknown builtin`). |
| **IN-TREE** | The builtin is declared in `IntrinsicsAMDGPU.td` and is actually *used* by shipped rocWMMA/ck headers + a regression binary for that target (e.g. `rocwmma_test_gfx1201.kpack`). Equivalent to CONFIRMED for that target. |
| **ISA-DOC** | Documented in the RDNA ISA text but not directly asm-probed here (low-risk, cited). |
| **UNAVAILABLE** | No encoding on that GPU generation (hard blocker). |

## Verification method used (per AMD guidance §6: "existence ≠ local support")

For each candidate: (1) builtin declared in `IntrinsicsAMDGPU.td`; (2) compile a one-function
probe for `--offload-arch=gfx1031` and `--offload-arch=gfx1201`; (3) inspect the emitted `.s`
for the real opcode (or capture the assembler's refusal). Tests run separately per target
on `G:\ROCM10RT-gfx1031` (ROCm 7.16, Clang 23, `amdgpu10.31`) and
`G:\ROCM10RT-gfx1201` (ROCm 7.16, Clang 23, `amdgpu12.01`).

---

## 1. Integer dot products (the gfx1031 headline question) — gfx1031 CONFIRMED; gfx1201 REJECTED for sdot8 (local probe)

**What each family's int4 path actually is, per target — this is the load-bearing
table, and the terminology matters:**

| Family | Wave width | int4 hardware path | Evidence |
|---|---|---|---|
| GFX12 (`gfx1200/1201/1250`, RDNA4) | wave32 | `v_wmma_i32_16x16x16_iu4` — one instruction issues a whole 16x16x16 tile | emitted by this repo, see §2.1 and RUN-011 |
| GFX11 (`gfx11xx/gfx115x`, RDNA3) | wave32 | WCMA exists but **no iu4 variant** | rocWMMA `internal/config.hpp` lists GFX11 as wave32; no sub-byte op is reachable |
| GFX10 (`gfx1030/1031/1032`, RDNA2) | wave32 | `v_dot8_i32_i4` (`sdot8`, the dp4a-style dot) — **no WMMA at all** | see below |
| GFX9 (CDNA) | wave64 | `v_mfma_*` | rocWMMA `internal/config.hpp`: "wave64 for gfx9 arch" |

**RDNA2 has no WMMA. Proven locally, not assumed:** rocWMMA's
`internal/config.hpp` (ROCm 7.16, `G:\ROCM10RT-gfx1201\include\rocwmma\`) enumerates
`GFX908/90a/942/950`, `GFX1100/1101/1102/1103/1150/1151/1152/1153` and
`GFX1200/1201/1250`, and then `static_assert(0, "Unsupported architecture")`. **No
gfx103x target appears.** Separately, LLVM parses the mnemonic and refuses it:

```
$ llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1031
error: instruction not supported on this GPU (gfx1031): v_wmma_f32_16x16x16_f16
```

So the mnemonic is known to the assembler but gated off for RDNA2. An earlier
claim in this project that "gfx1031 has WCMA" was wrong and is corrected here.

**Cost comparison that follows from the above.** One `v_wmma_i32_16x16x16_iu4`
covers a 16x16x16 tile = 4096 int4 MACs in one instruction. One `v_dot8_i32_i4`
does 8. The same tile on gfx1031 therefore costs **512** instructions instead of
1, which is the arithmetic behind the int4 crossover measured in RUN-012.


These are the ROCmFPX / NInferFPX dequant "multiply packed low-bit weights" primitives.
`sdot8` processes 8 signed nibbles (4-bit) per instruction — one instruction replaces the
two `sdot4` calls NInfer currently emits for Q4_0/IQ4/IQ5.

| ISA instruction | Operand format | Accumulator | gfx1031 (RDNA2) | gfx1201 (RDNA4) | Compiler path | Verification / test |
|---|---|---|---|---|---|---|
| `V_DOT8_I32_I4` | 8× signed int4 × 8× signed int4 | INT32 + acc | **CONFIRMED** | **REJECTED** | `__builtin_amdgcn_sdot8(a,b,c,clamp)` (declared in `IntrinsicsAMDGPU.td`, but gated on the `dot1-insts` target feature on gfx1201) | gfx1031: `v_dot8_i32_i4 v0, v0, v1, v2` emitted, local re-probe 2026-10-10 matches `_probe_sdot8_gfx1031.s:15`. gfx1201: **REJECTED** by local ROCm 7.16 Clang 23.0.0git `error: '__builtin_amdgcn_sdot8' needs target feature dot1-insts` (`--offload-arch=gfx1201`). NInfer int4 path: `src/kernels/int4_dot.hip` (sdot8 on gfx1031, scalar fallback on gfx1201). |
| `V_DOT8_U32_U4` | 8× uint4 × 8× uint4 | UINT32 + acc | CONFIRMED | UNVERIFIED | `__builtin_amdgcn_udot8` | Same family; matrix p.343 YES all. |
| `V_DOT4_I32_I8` | 4× int8 × 4× int8 | INT32 + acc | CONFIRMED | UNVERIFIED | `__builtin_amdgcn_sdot4` | rdna2 ISA p.10208; in-tree `common.cuh:708` dp4a. |
| `V_DOT4_U32_U8` | 4× uint8 × 4× uint8 | UINT32 + acc | CONFIRMED | UNVERIFIED | `__builtin_amdgcn_udot4` | rdna2 p.10216. |
| `V_DOT2_I32_I16` | 2× int16 × 2× int16 | INT32 + acc | CONFIRMED | UNVERIFIED | `__builtin_amdgcn_sdot2` | rdna2 p.10188; matrix p.345 YES all. |
| `V_DOT2_U32_U16` | 2× uint16 × 2× uint16 | UINT32 + acc | CONFIRMED | UNVERIFIED | `__builtin_amdgcn_udot2` | rdna2 p.10194. |
| `V_DOT4_I32_IU8` | 4× 8-bit, per-operand sign | INT32 + acc | ISA-DOC | UNVERIFIED | `__builtin_amdgcn_sudot4` (VOP3P) | rdna4 sign-selectable; matrix: sudot4 YES all. |
| `V_DOT8_I32_IU4` | 8× 4-bit, per-operand sign | INT32 + acc | **UNAVAILABLE** | UNVERIFIED | `__builtin_amdgcn_sudot8` | rdna4 only (matrix p.344: NO/ NO /YES). |

@note The udot8/sdot4/udot4/sdot2/udot2/sudot4/sudot8 rows are **not** locally probed
on gfx1201 (the matrix's gfx1201 CONFIRMED markings there are inherited from the ISA doc
and AMD guide, not from a local Clamp 23 probe). Treat them as probe-required until a
local gfx1201 compile proves otherwise; the sdot8 row above is the one exception that
was probed and rejected on gfx1201.

Minimal test (passes on both, emits the listed opcode):
```cpp
extern "C" int sd8(unsigned a, unsigned b, int c){
    return __builtin_amdgcn_sdot8(a, b, c, 0);   // -> v_dot8_i32_i4
}
```
Compile `-target amdgcn-amd-amdhsa -mcpu=gfx1031 -O2 -S`; inspect `.s` for `v_dot8_i32_i4`.

---

## 2. Matrix / WMMA (the gfx1201 headline; gfx1031 has NO WMMA — rocWMMA excludes it and llvm-mc refuses the mnemonic)

### 2.1 Dense WMMA — CONFIRMED on gfx1201; UNAVAILABLE on gfx1031

| ISA opcode (RDNA4) | Inputs → Accum | gfx1031 | gfx1201 | Compiler builtin | Evidence |
|---|---|---|---|---|---|
| `V_WMMA_I32_16X16X16_IU4` | INT4 × INT4 → INT32 | **UNAVAILABLE (no WMMA at all: rocWMMA omits gfx103x, and llvm-mc refuses the mnemonic)** | **IN-TREE** | `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` | rdna4 ISA p.5206; ck_tile `wmma_gfx12.hpp` instantiates it; `rocwmma_test_gfx1201.kpack` ships. |
| `V_WMMA_I32_16X16X32_IU4` | INT4 × INT4 (deeper K) → INT32 | UNAVAILABLE | IN-TREE | `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12` | rdna4; ck_tile:185-193 uses `int32x2_t` tile registers. |
| `V_WMMA_I32_16X16X16_IU8` | INT8 × INT8 → INT32 | UNAVAILABLE | IN-TREE | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` | rdna4; rocwmma + ck_tile. |
| `V_WMMA_F32_16X16X16_F16` | FP16 × FP16 → FP32 | UNAVAILABLE | IN-TREE | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` | rdna4 ISA p.5201; rdna4_ch16_vopd + ch7 packed F16. |
| `V_WMMA_F32_16X16X16_BF16` | BF16 × BF16 → FP32 | UNAVAILABLE | IN-TREE | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` | rdna4; `bf16_gemm_mma.cuh` target on gfx1201. |
| `V_WMMA_F32_16X16X16_FP8_FP8` | FP8×FP8 → FP32 | UNAVAILABLE | **CONFIRMED** | `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | rdna4 ISA p.5203; rocwmma `wmma_impl.hpp:1821` + ck_tile `wmma_gfx12.hpp:191`; clang accepts the builtin on gfx1201. |
| `V_WMMA_F32_16X16X16_FP8_BF8` / `_BF8_FP8` / `_BF8_BF8` | FP8×BF8 / BF8×FP8 / BF8×BF8 → FP32 | UNAVAILABLE | CONFIRMED | `..._fp8_bf8_w32_gfx12`, `..._bf8_fp8_w32_gfx12`, `..._bf8_bf8_w32_gfx12` | rdna4 ISA p.5204-5205; rocwmma `wmma_impl.hpp` FP8_AMD_FP8 variants. |
| `V_WMMA_F16_16X16X64_FP8_*` | FP8 × FP8 → FP16 | UNAVAILABLE | CONFIRMED | `__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8_w32_gfx12` (+ bf8 variants) | rdna4 ISA p.5208+; ck_tile `wmma_gfx12.hpp:568-660`. |

**Compiler-verified gate (ROCm 7.16):** `#include <rocwmma/rocwmma.hpp>` compiled with
`-x hip --offload-arch=gfx1031` fails `error: static assertion failed: Unsupported architecture` at
`rocwmma/internal/layout/config.hpp:91`. rocWMMA is GFx12-class only (RDNA4/CDNA) — there is **no**
WMMA on gfx1031, and the rocWMMA source files (`matmul_wmma.hip`, `rocmfp4_wmma_dot.hip`)
**must be CMake-gated out** when building for gfx1031 (the include hard-fails before any C++ macro).
gfx1201 emits real `v_wmma_*` (sherlock RUN-004; ~1.99 µs/tile on RX 9070 XT, RUN-006). So "WMMA" in
this table is a **gfx1201-only** fast path, not a portable one.

Operand note (rdna4 ISA p.5229): for FP8/BF8 WMMA, **NEG must be zero** for A and B matrices
(`rdna4-instruction-set-architecture.md:5229`). Do not set sign modifiers on FP8 inputs.

Minimal test for FP8 WMMA on gfx1201 (uses the in-tree rocWMMA header, which is the
supported calling convention — the raw builtin's tile-register width is backend-internal):
```cpp
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
extern "C" void k(float* d, i8* a, i8* b){
    fragment<matrix_a,16,16,16,fp8,wmma_accumulator> fa;
    fragment<matrix_b,16,16,16,fp8,wmma_accumulator> fb;
    fragment<accumulator,16,16,16,fp8,wmma_accumulator> fc;
    // load a,b,c; wmma::mma_sync(fa,fb,fc); // lowers to V_WMMA_F32_16X16X16_FP8_FP8
}
```
Compile with the tree's `bin/hipcc.exe --offload-arch=gfx1201`. The shipped
`G:\ROCM10RT-gfx1201\.kpack\rocwmma_test_gfx1201.kpack` already proves this compiles+links.

### 2.2 Sparse WMMA (RDNA4 bonus) — CONFIRMED gfx1201 only

| ISA opcode | Compressed input → dense | gfx1031 | gfx1201 | Builtin | Caveat |
|---|---|---|---|---|---|
| `V_SWMMAC_F32_16X16X32_F16` | 2:4 sparse FP16 A, dense FP16 B → F32 | UNAVAILABLE | IN-TREE | `__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32` | Requires 2:4 structured sparsity in weights; does NOT accelerate dense GGUF. |
| `V_SWMMAC_I32_16X16X32_IU4` | 2:4 sparse INT4 → INT32 | UNAVAILABLE | IN-TREE | `__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32` / `_w64` (feature `swmmac-gfx1200-insts`) | Only if NInfer adds a sparse INT4 path. |

### 2.3 Dual-issue VOPD dot — ISA-DOC, asm-only

From `rdna4_ch16_vopd.txt` (page 421/415): opcodes 12/13
`V_DUAL_DOT2ACC_F32_F16` / `V_DUAL_DOT2ACC_F32_BF16` — dual-issue FP16/BF16 dot+accum,
two per cycle. **No `__builtin_amdgcn_*` exists** (confirmed: not in `IntrinsicsAMDGPU.td`);

> **Correction (compiler-verified):** true single-cycle dual-VALU (VOPD) is RDNA3+/RDNA4
> only. On gfx1031 (RDNA2, wave64) the `v_dual_*` mnemonic is *accepted* by the ROCm 7.16
> assembler (see `_probe_dot2acc_gfx1031.s:51`) but runs as a 2-cycle sequence — no 2×
> throughput (RDNA2 has no single-cycle VOPD). For gfx1031 use the single VOP2
> `v_dot2acc_f32_f16` (emitted `_probe_dot2acc_gfx1031.s:15`, 4 flops/inst/cycle); reserve
> `v_dual_*` for gfx1201 (emitted `_probe_dot2acc_gfx1201.s:59`, true single-cycle dual =
> 8 flops/inst). See §2.4.
requires inline AMDGPU assembly. gfx1031 (RDNA2) has **no VOPD** → unavailable there.
NInfer site: `bf16_gemm_mma.cuh` could use it on gfx1201 for a 2× FP16/BF16 throughput gain.

---

### 2.4 FP16/BF16 dot2-accumulate — single VOP2 CONFIRMED on both; dual VOPD gfx1201-only

| ISA opcode | Inputs→Accum | gfx1031 | gfx1201 | Path | Emit evidence |
|---|---|---|---|---|---|
| `V_DOT2_ACC_F32_F16` (VOP2) | 2x FP16 x 2x FP16 -> FP32+acc | CONFIRMED | CONFIRMED | inline asm `v_dot2acc_f32_f16` | `_probe_dot2acc_gfx1031.s:15`, `_probe_dot2acc_gfx1201.s:19` = `v_dot2acc_f32_f16 v0, v1, v2, v3` |
| `V_DUAL_DOT2ACC_F32_F16` (VOPD, dual) | 2x (2xFP16*FP16->acc) per cycle | 2-cycle (no gain) | CONFIRMED (single cycle) | inline asm `v_dual_dot2acc_f32_f16` | gfx1201 `_probe_dot2acc_gfx1201.s:59` emitted; gfx1031 emits but wave64 2-cycle |

**Verification note (sect 6/8a calibration):** single-cycle dual-VALU (VOPD) is
RDNA3+/RDNA4; RDNA2 (gfx1031) is wave64 and the dual form decodes to a 2-cycle issue ("waits
for 2"), so it gives no throughput edge over two single VOP2 `v_dot2acc_f32_f16`. Use the
single VOP2 form on gfx1031 (4 flops/inst, 1 cycle); reserve `v_dual_*` for gfx1201 (true
2-in-1-cycle dual-issue = 8 flops/inst). **BF16 does NOT mirror F16** — see §2.5:
`__builtin_amdgcn_fdot2_f32_bf16` is CONFIRMED on gfx1201 (`v_dot2_f32_bf16`,
`_probe_open_gaps_gfx1201.s:58`) and REJECTED on gfx1031 (`error: needs target feature
dot12-insts`); the asm `v_dot2acc_f32_bf16` assembles on gfx1031 but is a §8a trap — the
rdna2 ISA defines no BF16 dot2 op. NInfer sites: `fp16_gemm_mma.cuh`,
`bf16_gemm_mma.cuh`, dense QK/V projections.

### 2.5 Compiler-verified emit (ROCm 7.16 probes `_probe_open_gaps_gfx{1031,1201}`)

| Builtin / asm | Lowers to ISA | gfx1031 (RDNA2) | gfx1201 (RDNA4) | Compiler evidence | Notes |
|---|---|---|---|---|---|
| `__builtin_amdgcn_fdot2` | `v_dot2c_f32_f16` / `v_dot2_f32_f16` | CONFIRMED | CONFIRMED | `gfx1031.s:17`; `gfx1201.s:19` | FP16 dot2 = GFX9+; real on BOTH (contra llama row 347 "NO gfx1031") |
| `__builtin_amdgcn_fdot2_f32_bf16` | `v_dot2_f32_bf16` | **REJECTED** | CONFIRMED | `gfx1031: error: needs target feature dot12-insts`; `gfx1201.s:58` | **BF16 dot2 is GFX12-only** |
| asm `v_dot2acc_f32_bf16` (single) | `v_dot2acc_f32_bf16` | §8a **TRAP** | CONFIRMED | gfx1031 `.s:160` (accepted) but rdna2 ISA has no BF16-dot2 op | do NOT use on gfx1031 |
| asm `v_dual_dot2acc_f32_bf16` (dual) | VOPD dual | §8a **TRAP** | CONFIRMED | gfx1031 `.s:196` (accepted) but RDNA2 has no VOPD | dual-issue = RDNA3+/RDNA4 |
| `__builtin_amdgcn_s_barrier_init` | `s_barrier_init` | **REJECTED** | CONFIRMED | `gfx1031: needs gfx12-insts`; `gfx1201.s:103` | named barriers = GFX12 only |
| `_s_barrier_signal`/`_wait`/`_leave` | `s_barrier_*` | **REJECTED** | CONFIRMED | `gfx1031: needs gfx12-insts`; `gfx1201.s:104-106` | — |
| `_s_barrier_join` | `s_barrier_join` | **REJECTED** | CONFIRMED | `gfx1031: needs gfx12-insts`; `gfx1201.s:108` | gfx1201 emits all 5 |
| `__builtin_amdgcn_s_buffer_prefetch_data` | `s_buffer_prefetch_data` | **REJECTED** | CONFIRMED | `gfx1031: needs gfx12-insts`; `gfx1201.s:159` | gfx1201 only |
| asm `ds_read_tr_b64` / `ds_load_tr_b64` | `ds_read_tr_b64` | CONFIRMED | CONFIRMED | `gfx1031.s:52/88`; `gfx1201.s:199/239` | real `ldmatrix` replacement (DS op, GFX7+); no builtin exists |
| asm `s_buffer_load_b32` | `s_buffer_load_b32` (SMEM) | CONFIRMED | CONFIRMED | both `.s` | base SMEM load (NOT the §3.3 Direct-to-LDS LDS-dest encoding) |

REJECTED = verbatim compiler error captured. CONFIRMED = opcode present in emitted `.s`.
§8a TRAP = the assembler accepts the mnemonic but the rdna2 ISA defines no such instruction, so it
would be undefined on real RDNA2 silicon (the `dot12-insts`/`gfx12-insts` refusal on the matching
builtin is the honest signal). llama matrix columns for RDNA3/CDNA2/CDNA3 are unverified here
(out of scope — only the `gfx1031`/`gfx1201` host was compile-probed).

### 2.6 Composition — "wave-matrix + matmul-vector" feasibility

**gfx1201 (RDNA4): every piece composes into a wave-tiled matmul + epilogue.** rocWMMA 16×16×16
tiles (FP8/BF16/FP16/INT8, §2.1) are the matrix engine; INT4 WMMA is reachable only via the raw
`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` builtin (ck_tile pattern), not via rocWMMA
fragment/mma_sync (rocWMMA headers contain zero int4/iu4/si4/sub_byte hits). **This repo now uses
that path**: `src/kernels/int4_wmma.hip` emits `v_wmma_i32_16x16x16_iu4` on gfx1201 and is
bit-exact vs the integer oracle (RUN-008), so the earlier "keep the rocWMMA path, no change
needed" advice applies to FP16/FP8/BF16/INT8 only, not to INT4.
the VOPD dual
`v_dual_dot2acc_f32_f16/bf16` (true 2-in-1 cycle, §2.3) fuses the 2×2 head/tail micro-tiles the
WMMA remainder leaves; **named barriers** (`s_barrier_*`, 2.5) give producer/consumer CTA splits
— that is the `mbarrier.try_wait.parity` replacement (blocking-only; AMD has no try_wait);
`s_buffer_prefetch_data` (2.5) streams the next tile's SGPRs; and the sync
`global_load→ds_write` double-buffer (§3.4) feeds each tile because async-to-LDS is hard-blocked.
Net: **one tile_loader feeds WMMA on gfx1201 — keep the rocWMMA path, no change needed.**

**gfx1031 (RDNA2): composable, but scalar-dot tiling only — no wave-matrix.** No WMMA (no
WMMA); no `v_dual_*` (no VOPD); no named barriers (no partial-wg sync → use full `s_barrier`/
`sched_barrier(0)`); no `s_buffer_prefetch` (prefetch tiles from VGPR/LDS manually); and **no BF16
dot2** (the `v_dot2acc_f32_bf16` asm is a §8a trap — BF16 dense must route through
`v_dot2acc_f32_f16`/`v_dot2c_f32_f16` after casting BF16→FP16, or scalar f32). Real kit on gfx1031:
`sdot8`→`v_dot8_i32_i4` (Q4/Q5, one instr vs the two `sdot4` llama assumed), `v_dot2_f32_f16`/
`v_dot2c_f32_f16` (FP16), `v_dot2_i32_i16`/`v_dot2_u32_i16` (i16), `ds_read_tr_b64` (tile
transpose into VGPRs), `s_buffer_load_b32` (SMEM), `sched_barrier(0)`+`s_waitcnt` drains.

**Verdict:** on gfx1201 the extensions compose into a real wave-matrix pipeline (WMMA + dual-dot2 +
named-barrier + prefetch). On gfx1031 they compose only into a scalar-dot wave-tile that still
accelerates the int4/int8/i16/FP16 paths — but NOT a WMMA path, NOT BF16 dot2, NOT partial-wg
sync. The async/TMA branch is dead on both; the sync double-buffer tile_loader is the one
abstraction that wins on both targets.

---

## 3. Memory movement / LDS async / TMA — hard blockers + the real fallback

This is where prior notes were **over-optimistic and are now corrected by a real compile**:

### 3.1 Async global→LDS load — REJECTED on both NInfer targets

| Feature | ISA / builtin | gfx1031 | gfx1201 | Evidence |
|---|---|---|---|---|
| `global_load_async_to_lds_b*` (CP async) | `llvm.amdgcn.global_load_async_to_lds_b32` (`IntrinsicsAMDGPU.td:4159`), class `AMDGPUAsyncGlobalLoadToLDS` | **REJECTED** | **REJECTED** | Inline-asm probe compiled with `-mcpu=gfx1031` and `-mcpu=gfx1201` both produced: `error: instruction not supported on this GPU (gfx1031): global_load_async_to_lds_b32` and `... (gfx1201): global_load_async_to_lds_b32`. The instruction lives behind the `gfx1250-insts` (CDNA4/MI355X) feature — NOT on RDNA2 or RDNA4. |

→ This **refutes** the llama_findings matrix row "Async LDS load = YES on RDNA4" and
confirms `ninfer_findings_RDNA.md:56/79/88` (TMA/async = hard blocker, `gfx1250-insts`
required, not on gfx1031 or gfx1201). The inline `asm volatile(...)` form that ck_tile
keeps as an alternative (`amd_buffer_addressing_builtins.hpp:2938`) is for gfx1250-class
parts only.

### 3.2 TMA / `cp.async.bulk.tensor` / `cuTensorMap` — hard blocker (both)

No AMD equivalent. `tensor_load_to_lds` is gated on `gfx1250-insts` and is REJECTED on
gfx1031+gfx1201 (same family as §3.1). NInfer sites: every `..._mma.cuh` tile load,
`src/core/pdl.cuh`. **Replacement:** coalesced `global_load` → `ds_write` to LDS,
double-buffered across the block/wave to overlap compute+transfer, with
`s_wait_dscnt`/`s_wait_vscnt` (GFX12) or `s_waitcnt` (GFX10) at dependency boundaries.

### 3.3 Direct-to-LDS buffer load — ISA-DOC, per-target verification required

| ISA | rdna2 docs "direct-to-LDS" buffer reads | gfx1031 | gfx1201 | Builtin |
|---|---|---|---|---|
| `S_BUFFER_LOAD_*` with LDS destination (buffer-resource + offset → LDS) | rdna2 ISA (Direct-to-LDS region) | ISA-DOC | ISA-DOC | `__builtin_amdgcn_s_buffer_load` + LDS addrspace — **must verify target encoding & assembler acceptance per target** |

The AMD doc flags this precisely: "Direct-to-LDS — ISA/compiler path must be verified."
No compile probe done here (operand encoding is target-specific); do the same
inline-asm acceptance test as §3.1 before emitting.

---

### 3.4 Fastest tile-load that works on both targets (consumer RDNA2–RDNA4 only)

For the in-scope cards (gfx1031, gfx1201) the async-LDS engine does **not
exist** (§3.1/§3.2 — `gfx1250-insts` only, out of scope), so there is no
"async branch to leverage." The fastest primitive that emits real ISA on
both is:

- **Coalesced VMEM `global_load` → `ds_write` to LDS, 2-stage
  double-buffered across the CTA**, drained by `__builtin_amdgcn_sched_barrier(0)`
  (emitted `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` on gfx1031;
  `s_wait_loadcnt_dscnt`+granular waits on gfx1201) with
  `s_wait_vscnt(0)`/`s_wait_dscnt(0)` (gfx1201) or `s_waitcnt vmcnt(0)`/
  `lgkmcnt(0)` (gfx1031) at real data dependencies.

Why it is fastest here:

- VMEM is the wide path on both generations (128-bit bursts on gfx1201,
  64-bit on gfx1031); nothing overlaps it because there is no async engine
  to overlap *with*. The latency must instead be hidden by MAC throughput
  and the double-buffer.
- On **gfx1201** this is exactly what rocWMMA's `load_matrix` does
  internally to stream a 16×16 tile into the VGPR A/B banks that `v_wmma`
  consumes — so a single `tile_loader` abstraction feeds the native WMMA
  engine. NInfer's FP8/BF16/FP16/INT4/INT8 matmul needs no change: keep
  rocWMMA, it already owns the inner load (see §2.1 probes +
  `rocwmma_test_gfx1201.kpack`).
- On **gfx1031** the same 2-stage fill feeds `v_dot8_i32_i4` (verified in
  §1, `_probe_sdot8_gfx1031.s`) / `v_dot4_i32_i8` / `v_fdot2_f32_f16` — no
  WMMA, so this sync fill is strictly the fastest available.

> So: one `tile_loader` abstraction = coalesced `global_load`+`ds_write`,
> double-buffered, sched-barrier drained. It is the fastest floor on both
> targets; on gfx1201 it transparently **leverages the present native WMMA**
> tile path (rocWMMA owns the load), and on gfx1031 it pairs with `sdot8`/
> `fdot2`. Drop the async branch entirely for consumer RDNA2–RDNA4 scope;
> gate it behind `__gfx1250__` only if CDNA4 support is ever added.

---

## 4. Synchronisation, barriers & scheduling — REAL but NOT TMA-equivalent

rdna2 ISA p.15592 (`S_BARRIER`), p.15602 (`S_INST_PREFETCH`), p.15608-15616
(`S_SLEEP`/`S_SENDMSG`); rdna4 "Hardware barriers" section.

| Feature | ISA | gfx1031 | gfx1201 | Builtin / asm | CUDA equivalent | Notes |
|---|---|---|---|---|---|---|
| Workgroup barrier | `S_BARRIER` | CONFIRMED | CONFIRMED | `__builtin_amdgcn_s_barrier` / `__builtin_amdgcn_s_barrier_wait` (.td gfx12) | `__syncthreads` | Full barrier; no transaction count. |
| Fine-grained wait counters | `s_wait_dscnt`/`vscnt`/`kmcnt` | via `s_waitcnt` | CONFIRMED | `__builtin_amdgcn_s_wait_dscnt` etc. (.td GFX12) | — | Use at real data-dependency boundaries. |
| Instruction prefetch | `S_INST_PREFETCH` | ISA-DOC (rdna2 p.7627) | ISA-DOC | `__builtin_amdgcn_s_inst_prefetch` | — | Hide i-cache misses; `S_NOP`/`S_SLEEP` cadence. |
| Wave sleep hint | `S_SLEEP` | ISA-DOC (rdna2 p.7485) | ISA-DOC | inline asm | `cudaSleep` | Not a sync primitive. |
| Dynamic VGPR alloc | `S_ALLOC_VGPR` | UNAVAILABLE | ISA-DOC (wave32, dynamic-VGPR launch mode) | none (asm only) | `setmaxnreg` | Partial alternative only; can fail; gfx1031 has no dynamic-VGPR mode. |
| TMA `mbarrier.expect_tx` | — | UNAVAILABLE | UNAVAILABLE | — | CUDA TMA | Hard blocker — no transaction-counted barrier. |
| `mbarrier.try_wait.parity` | — | UNAVAILABLE | UNAVAILABLE | — | non-blocking barrier | Hard blocker — replace with generation-counter + `S_BARRIER`/`sched_barrier`. |

---

## 5. Packed arithmetic / cross-lane / fast-math — CONFIRMED

rdna2 isa: `V_READLANE`/`V_WRITELANE` (p.11303/11323), `V_PERMLANE16` (p.11451),
`V_RCP_F32` (p.16161), `V_RSQ_F32` (p.16169), `V_FMA_F32`/`V_FMA_F16`, `V_CVT_PK_*`.
rdna4 ch7: packed F16/BF16/FP8/BF8/BP4 converts (p.4101, 4170, 4365, 4370, 4588).

| Feature | ISA | gfx1031 | gfx1201 | Builtin | NInfer use |
|---|---|---|---|---|---|
| 1/rsqrt fast path | `V_RCP_F32` / `V_RSQ_F32` | CONFIRMED | CONFIRMED | `__builtin_amdgcn_rcp` / `__builtin_amdgcn_rsq` | replace `rsqrtf()`; 0.5 ULP exact; one Newton refine if needed. |
| Packed FP16 convert | `V_CVT_PK_F16_F32`, `V_CVT_F32_F16` | CONFIRMED | CONFIRMED | `__builtin_amdgcn_cvt_pkrt_f16` | cast kernels. |
| FP8/BF8 pack/unpack | `V_CVT_PK_FP8_F32`/`V_CVT_F32_FP8`, `..._BF8_...` | UNAVAILABLE (no FP8) | CONFIRMED | `__builtin_amdgcn_cvt_pk_fp8_f16` etc. (IntrinsicsAMDGPU.td declared) | ROCmFPX/ROCmFP4 lift via FP8 intermediates. |
| FP4/BF8 scale-aware converts | (cdna-scale family) `cvt_scalef32_pk_*` | verify | CONFIRMED | `__builtin_amdgcn_cvt_scalef32_pk_fp4_f32` etc. (td declared, gates vary) | direct ROCmFP4 block decode → F16/F32. |
| DPP8 cross-lane | `DPP` (ch7.9 rdna4 p.84; table p.6052) | CONFIRMED (GCN≥1) | CONFIRMED | inline asm `dpp` modifiers / `__builtin_amdgcn_perm` | reductions, `__shfl` replacement. |
| Wave shuffle | `V_PERMLANE16`/`V_READLANE` | CONFIRMED | CONFIRMED | — | coarse `__shfl` replacement (not a barrel shuffle). |

---

## 6. NVFP4 / FP4 — hard blocker on both (integer nibbles ≠ FP4)

rdna4 ISA has `V_WMMA_I32_*_IU4` (integer 4-bit MAC) but **no FP4 matrix op**.
`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` is integer-only. The AMD doc §5 is
explicit: "Reinterpreting NVFP4 bit patterns as signed/unsigned integer nibbles does not
preserve their numerical meaning." NVFP4 E2M1 → F32 requires decode + separate block-scale
mul (software). FP4 MFA exists only on CDNA4 (`gfx1250`), not RDNA.
NInfer site: `src/ops/linear/nvfp4/*` — primary 27B format, largest rewrite.

---

## 7. API-level hard blockers (no ISA mapping possible)

| CUDA feature | AMD status | Evidence | ROCmInfer fallback |
|---|---|---|---|
| `cudaGraph*` / PDL `cudaGraphUpdate` | UNAVAILABLE | HIP has no graph-capture/update API (AMDGPUUsage: no graph intrinsics) | Stream+event re-dispatch per node; accept kernel-launch overhead, hide via async copy pipeline. NInfer: `src/core/{pdl,cored}*`, `decode_graph.cpp`. |
| `setmaxnreg` | UNAVAILABLE | No per-kernel VGPR cap; `S_ALLOC_VGPR` is wave32/gfx1250 only & fallible | Hand-fit occupancy; static + dynamic kernel variants. |
| `ldmatrix.sync` | UNAVAILABLE (builtin absent) | `__builtin_amdgcn_ldmatrix` does NOT exist in any Clang AMDGPU | `ds_read_tr_b64`/`ds_load_tr_b64` transposed LDS (different tile shape). NInfer: `common/mma.cuh`, `common/ldmatrix.cuh`. |

---

## 8. ROCmInfer call-site → AMD path map (corrected)

| NInfer file | CUDA idiom | Verified AMD replacement | Target | Risk |
|---|---|---|---|---|
| `src/kernels/int4_dot.hip` (this repo) | two `sdot4` for Q4 | `__builtin_amdgcn_sdot8` → `v_dot8_i32_i4` | gfx1031 (dp4a) | **Low — 50% fewer instrs on gfx1031.** gfx1201: scalar fallback in-kernel (correct, not perf). sdot8 **REJECTED** on gfx1201 (§1). Measured on RX 9070 XT (64x512 row-wise batched dot, median of 5 warm runs, RUN-008): rocFP4=2.090 us, int4_dot scalar=1.336 us, int4_matvec=1.454 us. The earlier 4.59/4.69 us figures were cold-clock outliers and are superseded. |
| `src/kernels/int4_matvec.hip` (this repo) | `ggml_cuda_dp4a` broadcast | `int4_matvec_i32` (sdot8 gfx1031 / scalar gfx1201) | gfx1031, gfx1201 | Shipped. Broadcast matvec: one shared packed activation vector vs all `rows` weight rows — distinct from `int4_dot_i32`'s paired-rows `a[r]·b[r]`. Routed from `Model::matvec_int4x`, which structurally rejects ROCmFP4 `Codebook10` layouts rather than mis-decoding magnitudes. |
| `src/kernels/int4_wmma.hip` (this repo) | `mma.sync` m8n8k32 | `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` → `v_wmma_i32_16x16x16_iu4` | gfx1201 | **Shipped; the RDNA4 int4×int4 hardware path.** Raw builtin, not rocWMMA (no sub-byte support). gfx1031 emit-verified as paired `v_dot8_i32_i4`. Bit-exact vs exact int64 oracle on RX 9070 XT (8/8 random tiles, 0/256 bad). |
| `src/ops/common/mma.cuh:708` | `ggml_cuda_dp4a` | `sdot4` (keep) or `sdot8` for INT4 | both | Low. |
| `src/ops/common/memory.cuh` | `cp.async` pipeline | **sync `global_load`+`ds_write`, double-buffered** | both | Async-LDS REJECTED on both (§3.1). |
| `src/ops/linear/nvfp4/*` | `mma.sync ::mxf4nvf4` | nibble→`sdot8`→FP32 scale | both | High — ~4× slower; no FP4 MAC. |
| `src/ops/linear/fp8/fp8_a8_mma.cuh` | FP8 `mma.sync` | `wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | gfx1201 only | gfx1031: scalar FP32 fallback. |
| `src/ops/linear/bf16/bf16_gemm_mma.cuh` | BF16 `mma.sync` | `wmma_bf16_w32_gfx12` or `V_DUAL_DOT2ACC_F32_BF16` (asm) | gfx1201 | gfx1031: `V_FMA_F16`+`V_FMA_BF16`. |
| `src/ops/linear/fp16/fp16_gemm_mma.cuh` | FP16 `mma.sync` | `wmma_f16_w32_gfx12` or `fdot2` | gfx1201 | both: packed `V_DOT2_F32_F16`/`V_FMAC_F16`. |
| `src/ops/linear/int8/int8_gemm_mma.cuh` | INT8 `mma.sync` | `sdot4` or `wmma_i32_16x16x16_iu8` | both/gfx1201 | Low. |
| `src/core/{pdl,cored}*` | cudaGraph+PDL | stream+event re-dispatch | both | Hard — architectural. |

---

## 9. What to run next (the suite §7 the ROCm doc prescribes)

Per-target, on `G:\ROCM10RT-gfx1031` and `G:\ROCM10RT-gfx1201`:
1. Discover: `bin\clang++.exe --version`, `bin\hipcc.exe --version`, `lib\llvm\bin\llvm-objdump.exe`.
2. Compile the probes above; emit `.s`; grep for the opcode (`v_dot8_i32_i4`,
   `v_wmma_f32_16x16x16_fp8_fp8`, `global_load_async_to_lds_b32`, etc.).
3. Record PASS/FAIL/UNAVAILABLE/UNVERIFIED with compiler version + the disassembly line.

The two compile outputs already captured in this doc (§3.1 and §1) ARE the start of that
record. Extend it for the IU4/INT8/BF16/FP16 WMMA and DPP cases using the same pattern
(builtin from §1/§2 tables → compile → grep `.s`).
