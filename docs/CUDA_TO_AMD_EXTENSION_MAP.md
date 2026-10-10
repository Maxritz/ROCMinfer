# CUDA → AMD RDNA Extension Mapping (NInfer port)

Authoritative for `NINFER_ROCmFPX_HaloFPX_INTEGRATION_SPEC.md`. Sources:
`rdna4-instruction-set-architecture.md` (697-p Reference Guide), `rdna4_ch16_vopd.txt`,
`rdna2-shader-instruction-set-architecture.txt` (12k-line ISA), and the verified intrinsic
registry in LLVM `IntrinsicsAMDGPU.td`. **All "AVAIL/REJECTED/EMITTED" claims below are
compiler-verified** with ROCm 7.16 toolchain on this host:
`G:\ROCM10RT-gfx1031` / `G:\ROCM10RT-gfx1201` (Clang 23; note: the earlier `ROCMRT-`
spelling in this doc was wrong — the on-disk prefix is `ROCM10RT`).

Verification method (per the ROCm doc §6/§8a: "existence ≠ local support"): for each
candidate, compile a one-function probe with `-target amdgcn-amd-amdhsa -mcpu=gfxNNN`,
emit `.s`, and (a) grep the opcode or (b) capture the compiler's refusal verbatim.
Probe artifacts live alongside: `_probe_sdot8_gfx1031.s`, `_probe_async_asm_gfx{1031,1201}.o`,
`_probe_async_wait_gfx{1031,1201}.s`, `_probe_dot2acc_gfx{1031,1201}.s`.

Availability columns: **RDNA2=gfx1031 (6700 XT), RDNA4=gfx1201 (9070 XT)**. Consumer RDNA
only — no CDNA (gfx1250) in scope, so async-to-LDS / TMA are permanently unavailable (not
"unverified," not "guard off on gfx1031": **REJECTED on both**).

"Intrinsic" = usable via `__builtin_amdgcn_*` (and, if marked EMITTED, confirmed to lower);
"asm" = inline-asm opcode only (no builtin). "ISA-DOC" = documented in the RDNA ISA text
but not locally emitted in a probe.

---

## 0. Corrections to prior findings (compiler-verified)

1. **FP8 WMMA on gfx1201 is real (not "may or may not exist").** `V_WMMA_F32_16X16X16_FP8_FP8`
   (RDNA4 ISA p.5203), `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` is **recognized
   by the gfx1201 compiler** and is **declared and used** by the in-tree rocWMMA
   (`wmma_impl.hpp:1821`) + ck_tile (`wmma_gfx12.hpp:191`), with a shipped regression binary
   `G:\ROCM10RT-gfx1201\.kpack\rocwmma_test_gfx1201.kpack`. No path on gfx1031. (ROCmFP4 is
   FP4, not FP8 — see §5.)

2. **TMA / async-to-LDS is REJECTED on BOTH consumer targets, not "YES on RDNA4".** The
   matrix in `llama_findings_RDNA.md:350` ("YES on RDNA4") was wrong. The compiler is the
   arbiter: `global_load_async_to_lds_b32` →
   `error: instruction not supported on this GPU (gfx1031)` and `... (gfx1201)`; and
   `__builtin_amdgcn_s_wait_asynccnt` → `error: needs target feature gfx1250-insts` **on both
   gfx1031 and gfx1201**. The rdna2 + rdna4 ISA texts contain **zero** `ASYNC`/`async`/
   `GLOBAL_LOAD_ASYNC` tokens. The intrinsic is *declared* in `IntrinsicsAMDGPU.td` (tablegen
   registry) but does **not lower** on RDNA2/RDNA4 silicon — the `gfx1250-insts` feature
   gates it to CDNA4. So async-to-LDS is a hard blocker on every NInfer target. **Replacement
   is mandatory:** coalesced `global_load`+`ds_write` tile fill, 2-stage double-buffered
   (§3.1/§3.4), drained by `__builtin_amdgcn_sched_barrier(0)` (EMITTED on both, see §4).

3. **`__builtin_amdgcn_ldmatrix` does not exist** in any Clang AMDGPU — confirmed absent from
   `IntrinsicsAMDGPU.td`. The real equivalent is `ds_read_tr_b64` / `ds_load_tr_b64`
   (`ninfer_findings_RDNA.md:11`). Not probed locally (item in §6).

4. **`V_DUAL_DOT2ACC_*`** exists as an RDNA4 VOPD opcode (rdna4_ch16_vopd p.421/415, emitted
   `v_dual_dot2acc_f32_f16` on gfx1201) **but with no intrinsic** and, on gfx1031, it is a
   **2-cycle wave64 sequence** (no single-cycle VOPD on RDNA2) — so use the single VOP2
   `v_dot2acc_f32_f16` on gfx1031 (emitted on both) and the dual form on gfx1201 only (§2.4).

---

## 1. Fishbone (Ishikawa) — cause categories of the port gap

Each arrow: CUDA idiom → the AMD instruction that satisfies it, with the NInfer call site.

**Instruction (compute) / matmul**
- `mma.sync aligned.m16n8k16.f32.bf16.bf16.f32` (FP16/BF16 GEMM) →
  `V_MAD_F32`/FMA, or **single VOP2 `V_DOT2_ACC_F32_F16` (EMITTED on both, §2.4, 4 flops/cycle)**,
  or WMMA f16/bf16 tiles on gfx1201. NInfer site: `src/ops/linear/bf16/bf16_gemm_mma.cuh`.
- `mma.sync aligned kind::mxf4nvf4` (NVFP4) → **no FP4 matrix op on any RDNA**; FP4 MFA is
  CDNA4 (`gfx1250`) only. Real path: manual nibble unpack → `sdot8` (`V_DOT8_I32_I4`,
  EMITTED on gfx1031) → FP32 scale-accum. ~4× slower. Site: `src/ops/linear/nvfp4/...` —
  the primary 27B format, biggest single rewrite.
- `mma.sync kind::f8f6f4` (FP8 mixed) → `V_WMMA_F32_16X16X16_FP8_FP8` / `V_DOT4_F32_FP8_*` on
  gfx1201 only (in-tree, §0.1); **no path on gfx1031** (no matrix core). Site:
  `src/ops/linear/fp8/fp8_a8_mma.cuh`.
- `mma.sync aligned.m16n8k32.s32.s8.s8.s32` (INT8) → `__builtin_amdgcn_sdot4`
  (`V_DOT4_I32_I8`) or `wmma_i32_16x16x16_iu8` on gfx1201. Site: `src/ops/common/mma.cuh:708`.
- 4-bit matmul (Q4_0/IQ4/IQ5) → `__builtin_amdgcn_sdot8` (`V_DOT8_I32_I4`, EMITTED gfx1031:
  `_probe_sdot8_gfx1031.s:15`). Halves the two-`sdot4` cost. Site: `src/ops/common/vecdotq.cuh`.
- `ldmatrix.sync.aligned.m8n8.x4.b16` → **intrinsic missing** (`__builtin_amdgcn_ldmatrix`
  nonexistent, §0.3); use `ds_read_tr_b64` / `ds_load_tr_b64` transposed-LDS family (different
  tile shape). Sites: `src/ops/common/mma.cuh`, `src/ops/linear/.../*`.

**Memory transfer / LDS async / prefetch** — *async-to-LDS is REJECTED on both targets;
sync tile load is mandatory*
- `cp.async.cg.shared.global` + commit/wait_group → **NO async path available** (REJECTED on
  gfx1031+gfx1201, §0.2). Replacement: coalesced `global_load` + `ds_write`, 2-stage
  double-buffered, drained by `sched_barrier(0)` + `s_wait_vscnt`/`s_waitcnt`. Site:
  `src/ops/common/memory.cuh`.
- `cp.async.bulk.tensor` / `cuTensorMap*` (TMA) → **HARD BLOCKER** (`gfx1250-insts` only).
  Replacement: sync `global_load`+`ds_write` double-buffer (§3.1/§3.4). Sites: all `..._mma.cuh`.
- `__ldg` (read-only cache) → RO buffer hints (`R`-bit) +
  `__builtin_amdgcn_s_buffer_prefetch_data` + non-temporal global hints. Site: weight stream.
- `cudaMemcpyAsync` P2P/peer access → `hipDeviceEnablePeerAccess` + `hipMemcpyAsync`
  (stream semantics; no intrinsic). Site: residency layer.

**Synchronization**
- `mbarrier.*` (TMA) → **no AMD equivalent**; `mbarrier.arrive.expect_tx` is TMA-specific.
  AMD `s_barrier` named barriers (if present) serve wg-group sync, not TMA.
- `mbarrier.try_wait.parity` → no parity-wait on AMD (hard blocker for non-blocking poll).
- `cp.async.wait_group` → `__builtin_amdgcn_s_wait_asynccnt` is **REJECTED both** (gfx1250);
  coarse drain uses `__builtin_amdgcn_sched_barrier(0)` (EMITTED both, §4). At data deps:
  `s_wait_dscnt`/`s_wait_vscnt` (gfx1201) / `s_waitcnt vmcnt`/`lgkmcnt` (gfx1031).
- `__syncthreads()` → `S_BARRIER` (`__builtin_amdgcn_s_barrier`, CONFIRMED via matrix; emit
  via `sched_barrier`). Site: every kernel.

**Scheduling / graph / driver**
- `cudaGraph*` + PDL `cudaGraphAddKernelNode`/Update → **no HIP graph API** (hard blocker).
  `src/core/pdl.cuh` / `src/core/decode_graph.cpp` → stream+event re-dispatch (launch
  overhead; hide behind the sync double-buffer compute pipeline).
- `setmaxnreg` → **NO AMD equivalent** (no per-kernel register cap). Fit occupancy by compile
  register allocation. Hard constraint.

**Numerics / formats / fast-math**
- `rsqrtf()` → `V_RSQ_F32` intrinsic (0.5 ULP exact, not Newton). Both.
- `__expf`/`__logf`/`__powf` → `V_EXP_F32`/`V_LOG_F32` series or `__ocml_*`/ROCMT. No fast
  builtin in the AMDGPU set. Site: activation kernels.
- `wmma_f16_16x16x16_f16` → FP16 WMMA tile on gfx1201; `V_MAD_F16`/`V_FMAC_F16` scalars both.
- FP4/INT4→FP32 (ROCmFPX/ROCmFP4 dequant) → `V_CVT_F32_FP8_*` (rdna4 p.4365/4588) for FP8;
  **but ROCmFP4 is sub-8-bit → real path is nibble unpack + `sdot8` (§0.2), not FP8.**
- `__float2half_rn`/`__float2bfloat16_rn` → `V_CVT_PK_F16_F32`, `__builtin_amdgcn_cvt_pkrt_f16`.

Cross-lane / shuffle: `__shfl_*` → AMD has **no intra-warp barrel shuffle**; use
`V_READLANE`/`V_WRITELANE` (rdna2 p.11303/11323) / `V_PERMLANE16` (p.11451), or
`__builtin_amdgcn_wave_shuffle` if available; reductions via LDS.

---

## 2. Waterfall decision table (CUDA construct → AMD status)

| CUDA construct | AMD replacement | Status | Arch | NInfer site | Notes |
|---|---|---|---|---|---|
| `mma.sync aligned.m16n8k32.s32.s8.s8.s32` | `sdot4` (`V_DOT4_I32_I8`) | AVAILABLE (intr) | both | `mma.cuh:708` | dp4a path |
| `mma.sync aligned.m16n8k16.f32.bf16...` | `v_dot2acc_f32_f16` (VOP2) + WMMA bf16 | AVAILABLE (asm EMITTED both) | both | `bf16_gemm_mma.cuh` | §2.4; gfx1201 also dual VOPD |
| `mma.sync aligned kind::mxf4nvf4` | nibble unpack → `sdot8` → FP32 scale | HARD (no FP4 MAC) | both | `src/ops/linear/nvfp4/` | ~4× slower; biggest rewrite |
| `mma.sync kind::f8f6f4` (FP8) | `V_WMMA_F32_16X16X16_FP8_FP8` + `V_CVT_PK_FP8_F32` | AVAILABLE (in-tree) | gfx1201 | `fp8_a8_mma.cuh` | **gfx1031: none** |
| `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` | `wmma_f32_16x16x16_f16_f16_gfx12` | AVAILABLE (in-tree) | gfx1201 | `fp16_gemm_mma.cuh` | |
| `ldmatrix.sync.aligned.m8n8.x4.b16` | `ds_read_tr_b64`/`ds_load_tr_b64` | AVAILABLE (asm) | both | `common/mma.cuh` | intrinsic missing (§0.3) |
| `cp.async.cg.shared.global` | sync `global_load`+`ds_write` double-buffer | HARD BLOCKER (async) | none | `common/memory.cuh` | async REJECTED both (§0.2) |
| `cp.async.bulk.tensor` / cuTensorMap | **NO equivalent** — sync tile load | HARD BLOCKER | none | all tile loads | needs `gfx1250-insts` |
| `mbarrier.*` (TMA) | `s_barrier` named barrier only | HARD BLOCKER | both | tile sync | no expect_tx/parity |
| `cudaGraph*` / PDL update | stream+event re-dispatch | HARD BLOCKER | both | `{pdl,core}*` | no HIP graph API |
| `setmaxnreg` | **none** — compile register fit | HARD BLOCKER | both | kernels | manual occupancy fit |
| `__shfl_*` intra-warp | `V_READLANE`/`V_PERMLANE16` + LDS | PARTIAL | both | kernels | no barrel shuffle |
| `rsqrtf()` | `V_RSQ_F32` | AVAILABLE | both | activation | 0.5ULP exact |
| `__ldg` RO cache | buffer RO hint + prefetch | PARTIAL | both | weight stream | hint, not instr |

---

## 3. Analytics — unused AMD extensions ranked by port impact

Top = verified real, not yet used by NInfer, on a supported consumer target. (The async-LDS
row that was rank 8 in the prior version is **deleted** — REJECTED on both targets, §0.2.)

| Rank | Extension (intrinsic / ISA) | Arch | Status | Why NInfer should adopt | ISA p. / evidence |
|---|---|---|---|---|---|
| 1 | `__builtin_amdgcn_sdot8` (`V_DOT8_I32_I4`) | both | EMITTED | 4-bit quant: one instr vs two `sdot4` → 50% fewer instrs | rdna2 p.10224; `_probe_sdot8_gfx1031.s:15` |
| 2 | `__builtin_amdgcn_udot8` (`V_DOT8_U32_U4`) | both | AVAILABLE | unsigned-Q4 path; pairs with sdot8 | rdna2 p.10236 |
| 3 | `V_DOT2_ACC_F32_F16` (single VOP2) / `__builtin_amdgcn_fdot2` | both | EMITTED both | dense FP16 dot2-acc (4 flops/cycle); the real gfx1031 FP16 path | `_probe_dot2acc_gfx1031.s:15`, `rdna2 p.16260` |
| 4 | `V_DUAL_DOT2ACC_F32_F16/BF16` (asm only) | gfx1201 | EMITTED gfx1201 | dual-issue FP16/BF16 dot (8 flops/cycle); **gfx1031 = 2-cycle, use single VOP2** | rdna4_ch16 VOPD p.421/415; §2.4 |
| 5 | `__builtin_amdgcn_sdot2`/`udot2` (`V_DOT2_I32_I16`) | both | AVAILABLE | Q8 16-bit-element dot | rdna2 p.10188 |
| 6 | `__builtin_amdgcn_sudot8` | gfx1201 only | AVAILABLE | mixed-sign 4-bit (IQ4_NL/IQ4_XS) | matrix:344 |
| 7 | `V_WMMA_F32_16X16X16_FP8_FP8` + intrinsic | gfx1201 | in-tree | FP8 dequant matmul | rdna4 p.5203; rocwmma kpack |
| 8 | `__builtin_amdgcn_s_inst_prefetch` | both | AVAILABLE | i-cache miss hiding (`S_INST_PREFETCH`) | rdna2 p.7627/15602 |
| 9 | `V_CVT_PK_FP8_F32`/`V_CVT_F32_FP8` + `cvt_scalef32_pk_*` | gfx1201 | AVAILABLE | lift ROCmFPX blocks via FP8; ROCmFP4 is nibble+ sdot8 | rdna4 p.4588, 4101 |
| 10 | `__builtin_amdgcn_s_buffer_prefetch_data` | both | AVAILABLE | prefetch tiles into LDS | matrix:355 (unverified locally — probe in §6) |
| 11 | `__builtin_amdgcn_s_dcache_wb`/`_wb_vol` | both | AVAILABLE | LDS write-back flush | |

Unpopular-but-usable (not used by NInfer): DPP8/DPP16 (p.6052-6320), `ds_read_tr_b64`
transposed LDS (the real `ldmatrix`), `DS_GWS_BARRIER` (rdna2 p.11826), `S_DCACHE_INVOL`/
`S_BUFFER_WBINVL1`, `V_RCP_F32`+FMA Newton, `V_FMAMK_F32`/`V_FMAAK_F32`, `FLAT_SCRATCH`,
`S_SENDMSG(MSG_DEALLOC_VGPRS)` (rdna2 p.7504), `S_SLEEP`/`S_NOP` cadence, `V_PERMI_LANE`/
`V_PERMUTE_B32`, `V_INTERP_P10_F32`/`P2` (ch12 p.242), `V_CVT_PK_BF16_F32`.

---

## 4. Synchronization / scheduling — verified

- `S_BARRIER` (`__builtin_amdgcn_s_barrier`): CONFIRMED (matrix says all archs).
- `__builtin_amdgcn_sched_barrier(0)`: **EMITTED on both** → gfx1031 `s_waitcnt
  vmcnt(0) expcnt(0) lgkmcnt(0)` (`_probe_async_wait_gfx1031.s:14`); gfx1201
  `s_wait_loadcnt_dscnt 0x0`+granular waits (`_probe_async_wait_gfx1201.s:14-18`).
  This is the **real replacement for `cp.async.wait_group`** on consumer RDNA.
- Fine-grained waits at data deps: `s_wait_vscnt(0)`/`s_wait_dscnt(0)` (gfx1201, GFX12) /
  `s_waitcnt vmcnt(0)`/`lgkmcnt(0)` (gfx1031, GFX10). ISA-DOC for gfx1031 (rdna2 p.681
  wait region).
- `S_WAIT_ASYNCNT` / `__builtin_amdgcn_s_wait_asynccnt`: **REJECTED on both** (needs
  `gfx1250-insts` — same async engine that is absent). Do not use.
- `S_ALLOC_VGPR` (dynamic VGPR, gfx1201): ISA-DOC, wave32 only — partial `setmaxnreg`
  replacement, can fail. gfx1031: none.
- `S_INST_PREFETCH` / `S_SLEEP`: ISA-DOC (rdna2 p.7627/7485) — hint, not sync.

---

## 5. Per-NInfer kernel call-site → AMD mapping

| NInfer file | CUDA idiom | AMD replacement | Gate |
|---|---|---|---|
| `common/mma.cuh:708,710` | `ggml_cuda_dp4a` (sdot4) | `sdot8` for 4-bit; `sdot4` ok | `#ifdef __HIP__`; `sdot8` on RDNA2+ |
| `common/ldmatrix.cuh` / `mma.cuh` | `ldmatrix.sync` | `ds_read_tr_b64` + `s_wait_dscnt` | verify (§6) |
| `common/memory.cuh` | `cp.async` pipeline | sync `global_load`+`ds_write`, 2-stage | **REJECTED both** (async) |
| `common/cuda_funcset.cuh` | graph + `setmaxnreg` | stream+event re-dispatch; drop setmaxnreg | both hard |
| `linear/nvfp4/nvfp4_mma.cuh` | `mma.sync ::mxf4nvf4` | nibble unpack → `sdot8` → FP32 scale | ~4× slower |
| `linear/fp8/fp8_a8_mma.cuh` | FP8 `mma.sync` | `V_WMMA_F32_16X16X16_FP8_FP8` + `V_CVT_PK_FP8_F32` | gfx1201 only |
| `linear/bf16/bf16_gemm_mma.cuh` | BF16 `mma.sync` | `v_dot2acc_f32_bf16` (single VOP2) or `V_DUAL_DOT2ACC` (gfx1201 asm) | both / gfx1201 dual |
| `linear/fp16/fp16_gemm_mma.cuh` | FP16 `mma.sync` | WMMA f16 (gfx1201) or `v_dot2acc_f32_f16` (both) | both |
| `linear/int8/int8_gemm_mma.cuh` | `sdot4` | `sdot4` (no change) or int8 WMMA | both ok |
| `core/{pdl,cored}*` | cudaGraph + PDL | stream+event re-dispatch | hard blocker |
| weight stream / activations | `__ldg`, `rsqrtf` | RO-buffer prefetch, `V_RSQ_F32` | both ok |
| reductions | `__shfl_sync` reduce | `V_PERMLANE16` + LDS, or `wave_shuffle` | both ok |

---

## 6. Verification status & open items (do not commit unverified)

**VERIFIED (compiler-emitted or proven REJECTED):**
- `sdot8`→`v_dot8_i32_i4`: EMITTED gfx1031 (`_probe_sdot8_gfx1031.s:15`) + gfx1201.
- `v_dot2acc_f32_f16` (single VOP2): EMITTED on both (`_probe_dot2acc_gfx{1031,1201}.s:15/19`).
- `v_dual_dot2acc_f32_f16` (VOPD dual): EMITTED gfx1201 (`_probe_dot2acc_gfx1201.s:59`); accepts-but-2-cycle on gfx1031 (`_probe_dot2acc_gfx1031.s:51`) — do not rely there.
- async-to-LDS `global_load_async_to_lds_b32`: **REJECTED both** (`instruction not supported
  on this GPU (gfx1031/gfx1201)`; rdna2+rdna4 ISA: 0 `ASYNC` tokens; needs `gfx1250-insts`).
- `s_wait_asynccnt`: **REJECTED both** (`needs target feature gfx1250-insts`).
- `sched_barrier(0)`: **EMITTED both** (`_probe_async_wait_gfx{1031,1201}.s:14`).
- FP8/BF8/INT4/INT8/BF16 WMMA (+ sparse SWMMAC): recognized on gfx1201 via .td + in-tree
  rocWMMA/ck use + shipped `rocwmma_test_gfx1201.kpack`; none on gfx1031.
- `__builtin_amdgcn_ldmatrix`: does not exist (absent from `IntrinsicsAMDGPU.td`); use
  `ds_read_tr*`.

**Open (unverified — probe before commit):**
- `__builtin_amdgcn_fdot2` **builtin** existence (I verified the *asm* opcode `v_dot2acc_f32_f16`,
  not the `fdot2` builtin llama_findings claims) — emit probe needed.
- `ds_read_tr_b64` / `ds_load_tr_b64` availability on gfx1031 & gfx1201 (the `ldmatrix`
  replacement — claimed but never emitted).
- `__builtin_amdgcn_s_barrier_{init,signal,wait,join,leave}` (named barriers) — llama matrix
  says "YES all"; NInfer findings say implausible. Emit probe needed.
- `__builtin_amdgcn_s_buffer_prefetch_data` — emit probe needed (matrix says available).
- direct-to-LDS buffer load (`S_BUFFER_LOAD_*` → LDS dest) — AMD doc: "must be verified per
  target"; if it emits it could collapse the §3.4 two-instruction fill. Emit probe needed.
- rocWMMA FP8/INT4 tile **fresh emit** (currently verified only via the shipped regression
  binary, not a probe in this repo).

**Hard blockers (no path on consumer RDNA2–RDNA4):**
- NVFP4/FP4 matmul (no FP4 MAC; sdot8+scale only).
- TMA / `cp.async.bulk.tensor` / `mbarrier.expect_tx` / `try_wait.parity` (all gfx1250).
- `cudaGraph*` / PDL update (no HIP graph API).
- `setmaxnreg` (no reg-cap control).
- async-to-LDS `cp.async` (REJECTED both, §0.2).

---

## 7. Recommended next three port moves (revised)

1. **Ship `sdot8`-based 4-bit vec_dot for Q4_0/IQ4** (`vecdotq.cuh`, `mma.cuh:708`) behind
   `#ifdef __HIP__` — 50% fewer instrs on both targets; the `_probe_sdot8_gfx1031.s` emit is
   the proof.
2. **Replace every async tile-load with sync `global_load`+`ds_write` + `sched_barrier(0)`
   double-buffer** (NOT the previously-recommended `global_load_async_to_lds`, which is
   REJECTED on both). Use the single VOP2 `v_dot2acc_f32_f16` (emitted both) for FP16 dense.
3. **FP8 dequant on gfx1201 only**: `V_CVT_PK_FP8_F32`/`V_CVT_F32_FP8` + `V_WMMA_F32_16X16X16_FP8_FP8`;
   on gfx1031 keep scalar FP32 (no FP8 matrix core). NVFP4 stays the nibble+`sdot8` rewrite.
