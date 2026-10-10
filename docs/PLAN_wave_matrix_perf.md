# PLAN: Wave-matrix + matmul-vector — max perf (gfx1201) with reliable fallbacks (gfx1031)

Goal: one kernel family that compiles for `gfx1031` (RDNA2) **and** `gfx1201` (RDNA4), uses the
fastest available ISA on each, and never relies on a §8a trap. Capability facts are all
compiler-verified in `AMD_CAPABILITY_MATRIX.md` §2.5 (ROCm 7.16, `G:\ROCM10RT-gfx{1031,1201}`)
— no capability is claimed here without that evidence.

This plan deliberately stays in C++/HIP with the existing ROCm toolchain (no Python runtime,
per AGENTS.md §1). TileLang/ROCmFPX/HaloFPX are reference specs only (THIRD_PARTY_SOURCES.md).

## 0. Gating (verified: rocWMMA is a CMake-level, not C++-macro, gate)

`--offload-arch=gfx1031 -x hip -fsyntax-only` on `#include <rocwmma/rocwmma.hpp>` fails with
`error: static assertion failed: Unsupported architecture` at
`rocwmma/internal/layout/config.hpp:91` — rocWMMA is **GFx12-class only** (RDNA4/CDNA) and has NO
macro that lets a `gfx1031` TU include it. Therefore the WMMA kernels **must not be compiled** for
gfx1031; the gate is CMake source-level, not a C++ `#ifdef`:

```cmake
# CMakeLists.txt — the two rocWMMA sources are gfx1201-only (rocWMMA config.hpp:91 REJECTED gfx1031).
# The scalar FP32 dequant-and-FMA kernel (rocmfp4_dot.hip) compiles for BOTH — keep it for gfx1031.
set(_has_wmma OFF)
if("gfx1201" IN_LIST CMAKE_HIP_ARCHITECTURES)
    list(APPEND _has_wmma ON)
endif()
target_sources(rocminfer_core PRIVATE src/kernels/rocmfp4_dot.hip)            # both targets
if(_has_wmma)
    target_sources(rocminfer_core PRIVATE
        src/kernels/matmul_wmma.hip            # WMMA FP16 tile, gfx1201
        src/kernels/rocmfp4_wmma_dot.hip)      # WMMA int4 matvec, gfx1201
    target_compile_definitions(rocminfer_core PRIVATE ROCMINFER_HAVE_WMMA=1)
endif()
```
A `__GFX12__`/`__GFX11__` macro *can* exist (AMD emits family macros), but it is **not** the
right tool here because the rocWMMA header itself hard-fails before any `#ifdef` body runs.
`ROCMINFER_HAVE_WMMA` is then the C++ gate for the launcher (`matmul_wmma.hpp`,
`rocmfp4_wmma_dot.hpp`): on gfx1031 the launchers return `ENOTSUP` and the engine takes the
scalar `rocmfp4_dot_f32` path — which is the correct RDNA2 floor (no matrix core exists on gfx1031).
For the int4/gfx1031 fast path that the matrix lists (§1: `v_dot8_i32_i4` CONFIRMED on gfx1031 but
absent from the current tree), add a dedicated `sdot8`-tiled matvec kernel guarded by
`#if !ROCMINFER_HAVE_WMMA && defined(__GFX10__)` — its emit was verified in §2.5; wiring it is the
next perf gate for the RDNA2 host.

## 1. Dispatch matrix (per-target fast path)

| Layer | gfx1201 fast path (CONFIRMED) | gfx1031 fallback (CONFIRMED) | Fallback ladder (slow→fast) |
|---|---|---|---|
| **Tile MAC** | rocWMMA 16×16×16 (`V_WMMA_*`, §2.1) — matrix core | `v_dot2acc_f32_f16`/`v_dot2c_f32_f16` F16 dot2 (§2.5) + `sdot8`→`v_dot8_i32_i4` (int4/5) + `sdot2`/`udot2` (i16) | scalar f32 |
| **Dual-issue** | `v_dual_dot2acc_f32_f16/bf16` (real VOPD, §2.3/2.5) | none (RDNA2 has no VOPD; `v_dual_*` asm is a §8a trap) | single VOP2 dot2 |
| **BF16 dense** | `v_dot2_f32_bf16` (builtin EMITTED, §2.5) | **cast BF16→FP16, use `v_dot2_f32_f16` — never the bf16-asm trap** | `v_pk_fma_f16` / scalar f32 |
| **Tile sync** | `s_barrier_init/signal/wait/leave/join` (named, §2.5) | `s_barrier`/`sched_barrier(0)` full-CTA only (no partial-wg) | sched_barrier + s_waitcnt |
| **Tile prefetch** | `s_buffer_prefetch_data` (§2.5) | none (gfx12-insts; use VGPR pre-load) | — |
| **Tile fill** | rocWMMA internal load_matrix (global→VGPR A/B banks) | sync `global_load`→`ds_write`, 2-stage (§3.4); `ds_read_tr_b64` transpose into VGPRs | — |

Relation: gfx1201 fast path ⊃ (rocWMMA ⊕ dual-dot2 ⊕ named-barrier ⊕ prefetch); gfx1031 ⊂ (scalar-dot ⊕ ds_read_tr ⊕ sched_barrier). Every gfx1201 feature has a gfx1031 equivalent *except*: WMMA, dual-issue, named barriers, prefetch, and BF16 dot2 — for these gfx1031 takes the ladder's next rung (single dot2 / f16-cast / full barrier / sync fill).

## 2. The one kernel abstraction that wins both targets: `TileLoader`

`src/kernels/tile_loader.hip` — a single callable that hides the async/dead branch:

```cpp
struct TileLoader {
    // Coalesced global_load -> ds_write, K-buffers deep (default 2 = double-buffered).
    // Drains with sched_barrier(0) + target-appropriate wait at dependency boundaries.
    // On gfx1201 this is exactly what rocWMMA's load_matrix streams (no async engine
    //   exists on consumer RDNA — §3.1/§3.2 REJECTED both), so it feeds WMMA/LDS directly.
    // On gfx1031 it feeds the scalar-dot micro-tile via ds_read_tr_b64.
    void fill(/*global ptr, lds ptr, bytes*/);   // emits sched_barrier(0) inside
    static constexpr uint32_t s_wait_vscnt() {   // gfx1201 granular / gfx1031 s_waitcnt
        return rocminfer_is_gfx12()
            ? __builtin_amdgcn_s_wait_vscnt(...)
            : __asm__("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"); // gfx1031
    }
};
```
Contract: after `fill()` returns, the requested tile is in LDS and all prior VMEM is visible;
callers need no target `#ifdef`. The async-TMA/`cp.async` branch is gated behind
`__gfx1250__` (out of scope; see §8.2) so it can never compile on gfx1031/gfx1201.

## 3. Wave-matrix + matmul-vector composition

- **gfx1201:** rocWMMA 16×16×16 tile → residual micro-tiles (≤2-wide) filled by
  `v_dual_dot2acc_f32_f16/bf16` (2 dot2/cycle). MV epilogue (bias+act): DPP8/perm reductions +
  `v_pk_fma_f16` elementwise. CTA split via `s_barrier_*` (producer/consumer) — this is the
  `mbarrier.try_wait.parity` replacement, **blocking only** (AMD has no try_wait).
- **gfx1031:** no matrix core → micro-tile = 8-wide `v_dot8_i32_i4` (Q4/Q5 weights) or 2×
  `v_dot2_f32_f16` (FP16). Layout into VGPRs via `ds_read_tr_b64` (real on both, §2.5 — the
  ldmatrix replacement). MV epilogue via `v_pk_fma_f16` + DPP8. No partial-wg barriers → whole
  wave/CTA barrier with `sched_barrier(0)`. BF16 dense → cast to FP16, use `v_dot2_f32_f16`
  (the `v_dot2acc_f32_bf16` asm is a §8a trap on gfx1031 — do not emit).

## 4. Validation gates (AGENTS.md §5, ten-iteration)

1. Configure clean: `cmake -S . -B build-hip -G Ninja -DROCMINFER_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=...ROCM10RT-gfx1031/clang++.exe` (cross-compile gfx1031+1201).
2. Compile `-Wall -Wextra -Werror`; the `rocminfer_is_gfx12()` feature-test probe must `#error` on
   an unknown macro (never silent-wrong).
3. Unit tests for `sdot8`/`fdot2`/`TileLoader::fill` on both targets (CPU oracle path).
4. Golden: compare int4/fp16/bf16 tile output vs FP32 reference (max abs rel bound: F16/BF16FMA
   exact; bf16-cast = ≤2 ULP on the cast, ≤1 on FMA).
5. Fuzz: truncated/bad scale bytes, K%tile ≠ 0; must return `EINVAL`, never plausible-wrong.
6. `-fsanitize=address,undefined` on host paths (GGUF parse + TileLoader setup).
7. Numerics: FP32 oracle max-abs/rel reported per op-type (see AGENTS.md §8a on constants).
8. Bench via `--profileperf=json,on` before/after; the only perf claim is a measured
   `--profileperf` line — no unverified throughput numbers.
9. Instrumentation-floor check on any traced bench (PERF_TRACE.md contract).
10. `doxygen docs/Doxyfile` with `WARN_AS_ERROR=YES`, zero warnings; every function in
    `tile_loader.hip` / the dot wrappers gets `@brief/@param/@return/@pre/@post`.

## 5. Risk / non-negotiables

- **§8a trap hard rule:** `v_dot2acc_f32_bf16` and `v_dual_*` are gated behind
  `__builtin_amdgcn_sdot*`? no — behind `rocminfer_is_gfx12() && __builtin_amdgcn_is_dot12()`
  equivalent. Pragmatically: the BF16-dot2 and dual-dot2 paths compile **only** under
  `#if rocminfer_is_gfx12()`; on gfx1031 the BF16 path is the FP16-cast ladder. Add a static
  `static_assert` that `gfx1031 ⇒ ¬dot12` using the feature macro, so a wrong build fails loud.
- Async-to-LDS / TMA / cudaGraph / setmaxnreg stay `__gfx1250__`-gated (REJECTED on both
  targets, §3.1/§7) — they are dead code for consumer RDNA, not "unverified."
- No new Python runtime, no vendored third-party kernel (TileLang/ROCmFPX/HaloFPX reference only).
