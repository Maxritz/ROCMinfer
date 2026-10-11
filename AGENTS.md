# AGENTS.md — ROCMinfer

ROCMinfer is a from-scratch, AMD-first LLM inference engine in C++/HIP.
Authoritative scope: `NINFER_ROCmFPX_HaloFPX_INTEGRATION_SPEC.md`.

Targets: `gfx1031` (RX 6700 XT class, RDNA2) and `gfx1201` (RX 9070 XT, RDNA4).
Build host: Windows 11. No NVIDIA. No CUDA. No Vulkan unless explicitly added.

## 0. Non-negotiables

1. **No porting.** ROCmFPX/HaloFPX are *references* (pinned in `THIRD_PARTY_SOURCES.md`).
   Do not vendor, link, or transcribe their implementation. Re-derive from the
   format spec. If code is genuinely adapted, record it in the adapted-code
   table of `THIRD_PARTY_SOURCES.md` with the upstream commit and license.
2. **No fake code.** No `TODO`, `FIXME`, `HACK`, `XXX`, empty function bodies,
   `unimplemented()`, placeholder returns, demo-only paths, invented numbers.
   If something cannot be implemented, say so *before* writing it, and make the
   code fail loudly (nonzero exit / exception / error code) rather than return a
   plausible wrong value.
3. **Never claim unverified performance.** Every throughput/latency/bottleneck
   claim must come from an actual run in this repo, with the command shown and
   its real output. See §6.
4. **Never claim unverified correctness.** Tests must be run, not written and
   assumed green.

## 1. Architecture policy

- ROCMinfer is the engine. It owns GGUF parsing, graph building, scheduling,
  residency, and the sampling path.
- ROCmFPX contributes *format definitions and reference maths only*.
- HaloFPX contributes *model-zoo metadata and hardware-fit concepts only*.
- No Python runtime, no `llama-server` subprocess, no forked llama.cpp.
- Weight quant and KV-cache quant stay separate subsystems with separate tests.

## 2. Language and build

- C++20 (`-std=c++20`), HIP for device code, CMake >= 3.21.
- Compiler: ROCm LLVM (`amdclang++`) with `CMAKE_HIP_ARCHITECTURES` set per target.
- Warnings: `/W4` equivalent (`-Wall -Wextra`) plus `-Werror` in CI.
- No exceptions across HIP/host ABI boundaries; no RTTI in device code.
- RAII for every resource. No raw `new`/`delete`, no manual `malloc`/`free`.
  Fixed-size scratch goes in caller-owned arenas passed by pointer + capacity.
- Error style: return `Status` / `errno`-like codes from internal APIs; throw
  only at the CLI boundary. Never swallow an error or collapse it to a default.
- Validate every input at the boundary (GGUF fields, byte counts, magic,
  lengths) and reject explicitly.

## 3. Doxygen is mandatory

Every public function, struct, enum, and module gets a Doxygen block. This is
enforced by the `dox-validate` skill (auto-triggered on write/edit).

Required tags:

- `@brief` — one line, mandatory.
- `@param[in]` / `@param[out]` / `@param[in,out]` — direction is mandatory.
- `@return` — mandatory for non-void.
- `@pre` — preconditions (non-NULL, lengths, capacities).
- `@post` — postconditions (what is valid on return).
- `@note` — rationale, non-obvious constraints.
- `@warning` — ownership, lifetime, thread-safety, or misuse hazards.

Format (follow exactly):

```cpp
/**
 * @brief Decode one ROCmFP4 block row to FP32.
 * @param[in] src  Packed blocks, at least `row_bytes()` bytes (non-NULL).
 * @param[out] dst Output floats, caller-owned, at least `k` elements.
 * @param[in] k    Element count, must be a non-zero multiple of QK_ROCMFP4.
 * @param[in] dual_scale True to use the two UE4M3 halves, false for the single
 *                       whole-block scale (FAST layout).
 * @return 0 on success, EINVAL if `src`/`dst` is NULL or `k` is not aligned.
 * @pre  `src` holds exactly `row_bytes(k, dual_scale)` bytes.
 * @post `dst` contains `k` dequantized values; `src` is unmodified.
 * @note  Scale bytes 0x7F and any byte with the sign bit set are invalid and
 *        cause EINVAL rather than a silent zero scale.
 * @warning Non-thread-safe when distinct threads write the same `dst`.
 */
```

Files get a `@file` block. Structs document every field with a trailing comment
naming units, valid range, and alignment.

## 4. Performance tracing in every component (mandatory)

**Every** function, kernel launch, allocation, transfer, and sync point carries
tracing through the shared begin/end path. Not just the hot paths — the point is
that the instrumentation floor is uniform and measurable.

Contract (full spec in `docs/PERF_TRACE.md`):

| Aspect | Rule |
|---|---|
| Switch | `--profileperf[=table\|json\|off]` on the CLI; off by default |
| Scope | `TraceScope(TraceRecorder&, component, domain[, stream])`, RAII |
| Domains | `kHost`, `kDevice`, `kTransfer`, `kAlloc`, `kSync`, `kDispatch` |
| Recorded per record | `host_ns`, `dev_ns`, `idle_ns`, `kind`, `component` |
| Aggregates per component | `ops`, `%dev`, `dev us`, `idle us`, `host us` |
| Floor | Every run with tracing on prints a measured instrumentation floor, calibrated by timing N empty ops through the identical path |
| Overhead honesty | If tracing overhead is not negligible for a measurement, the run is reported as instrumented and the uninstrumented number is quoted alongside |
| State | Explicitly injected `TraceRecorder&`. No hidden globals |
| Off path | When disabled the scope compiles to a null check; no allocation, no event creation |

Rules:

- Never add timing ad hoc. Use `TraceScope`. Do not call the clock directly.
- Never nest two `TraceScope`s on the same component at the same depth.
- Label components by logical role (`attention`, `projections`, `recurrent`,
  `norms`, `embed`, `head+sample`, `transfer`, `ffn-activate`, `residual`,
  `bias`, `alloc`, `sync`), not by function name.
- A device scope must bracket the *launch*, and use HIP events so `dev_ns` is
  GPU execution time, not enqueue time.
- A component that launches GPU work and does host work in between needs
  separate scopes so `idle_ns` is real rather than inferred.
- Benchmark claims in commits, docs, or replies must come from
  `--profileperf` output on a real run, with the command recorded.

## 5. Ten-iteration validation

Every change passes all of these before it is considered done:

1. Configure (clean tree)
2. Compile, `/W4`/`-Wall -Wextra`, `-Werror` in CI
3. Unit tests
4. Golden/reference tests against the pinned upstream quantizer output
5. Fuzz/negative tests (truncated, malformed, unknown GGUF type)
6. Sanitizers (`-fsanitize=address,undefined`) on host paths
7. Numerics check vs FP32 oracle (max abs/rel error bound stated)
8. Benchmarks, before and after, via `--profileperf`
9. Instrumentation-floor check when tracing was enabled
10. Doxygen build with `WARN_AS_ERROR=ON`, zero warnings

Report exactly one of:

```
✅ Validated: <which of the 10 passed>
❌ Failed: <which iteration, actual output>
```

Never report `✅` without having run them.

## 6. Performance analysis discipline (sherlock)

- Persistent analysis log: `.opencode/sherlock-analysis.md`, append-only.
- Never optimise by guesswork. Baseline → decompose → instrument → investigate
  → optimise one variable → verify.
- At least three measurements before declaring a bottleneck.
- Distinguish proven / suspected / unresolved root causes explicitly.
- Compare against *achievable* ceilings for this workload, not theoretical peak.
- Never stop at the first bottleneck while another significant one remains.
- Keep the original baseline; never discard a correctness test to gain speed.
- Success = reproducible improvement, verified correct, explained by evidence.

## 7. Naming and layout

- `snake_case` functions/variables, `PascalCase` types, `SCREAMING_SNAKE` for
  format constants (`QK_ROCMFP4`, `QS_ROCMFP4_FAST`).
- Public headers under `include/rocminfer/`, implementation under `src/`,
  CUDA/HIP kernels under `src/kernels/`, tests under `tests/`.
- Namespace `rocminfer`. C ABI only where a stable external boundary is needed.
- One concept per file. No god translation unit.

## 8. Correctness for formats

- Format definitions live in one registry; no duplicated block layouts.
- Every registered format has: block size in bytes, bits-per-weight, codebook
  or level set, scale encoding, a CPU reference decode, and a negative test for
  every invalid scale byte.
- Dual-scale and single-scale variants of a family are registered as *distinct*
  formats. Never merge them behind one name.
- Golden tests must be generated by the pinned upstream quantizer and committed
  with their SHA.

## 8a. Constants and expected values

A mistyped constant is the highest-severity defect class in this project,
because it fails silently: a wrong SHA-256 round constant still yields a
well-formed 32-byte digest, so every self-consistency check passes. Two
hand-typed copies of the same wrong table agree with each other, and a test
whose expected values were also written from memory agrees with the defect
instead of catching it.

Rules:

- **Derive constants; never transcribe them.** If a specification defines a
  constant (FIPS 180-4 defines SHA-256's K and H as fractional bits of cube
  and square roots of primes), evaluate that definition at compile time. See
  `include/rocminfer/detail/sha256_constants.hpp`.
- **Verify through an independent path.** A derivation checked against itself
  proves nothing. The SHA-256 test recomputes every constant in floating point
  with a separate prime generator.
- **Expected values come from an oracle, never from memory.** Every expected
  digest, checksum, or magic value must be produced by a trusted implementation
  and re-run, not recalled. `tools/gen_goldens.py --verify` binds the committed
  artifact digests to CPython's `hashlib`.
- **A synthetic test cannot certify a layout it invented.** When a container
  format disagrees with the implementation, check the *writer* in the pinned
  reference before trusting the test. A test that encodes the same wrong
  assumption as the code will pass forever.

## 9. Session rules

- Read `THIRD_PARTY_SOURCES.md` before using any upstream artefact.
- Read `.opencode/sherlock-analysis.md` at session start if it exists.
- Record durable findings (measured numbers, blockers, decisions) in repo files.
  Do not rely on conversation memory surviving compaction.

## Build configuration (Windows host)

Configure the HIP build with ROCm LLVM clang as **both** the C++ and HIP compiler.
CMake's `Windows-Clang-HIP` platform module requires both languages to share the
Clang ABI, so mixing MSVC C++ with HIP fails configure.

```bat
cmake -S . -B build-hip -G Ninja ^
  -DROCMINFER_ENABLE_HIP=ON ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 ^
  -DCMAKE_PREFIX_PATH="G:/ROCM10RT-gfx1201" ^
  -DCMAKE_CXX_COMPILER="G:/ROCM10RT-gfx1201/lib/llvm/bin/clang++.exe"
cmake --build build-hip --config RelWithDebInfo
```

- ROCm 7.16.26331 lives at `G:\ROCM10RT-gfx1201`.
- The ROCm clang binary is `G:\...\lib\llvm\bin\clang++.exe` (not `llvm/bin`).
- `hipcc.bat` exists but CMake's HIP support does **not** accept it as
  `CMAKE_HIP_COMPILER`; use clang++ directly.
- A Doxygen file is checked in at `docs/Doxyfile` with `WARN_AS_ERROR=YES`.
  Run `doxygen docs/Doxyfile` to validate documentation.

## Phase status

- Phase 0: complete (pinning, profiler, build).
- Phase 1: complete (GGUF reader, ROCmFPX registry, CPU reference, goldens).
- Phase 2: complete (fused dequantize-and-dot kernel, bit-identical exact family,
  FP32-bound numerical family, real-GPU run on gfx1201, cross-compiled for
  gfx1031).
- Phase 3: model-zoo metadata + CLI tooling (`inspect`, `validate`, `info`,
  `bench`); complete.
- Phase 4: graph execution / scheduler / residency / MoE streaming; **complete**
  (Model/GpuBuffer residency layer, fused dequantize-and-dot matvec, end-to-end
  GPU matvec verified vs CPU oracle).
- Phase 5: RDNA4 WMMA acceleration — **implemented** (`src/kernels/matmul_wmma.hip`,
  `src/kernels/rocmfp4_wmma_dot.hip`, `include/rocminfer/{matmul_wmma,rocmfp4_wmma_dot}.hpp`);
  ROCmFP4→f16 dequantize-and-WMMA tile bit-exact vs CPU oracle on gfx1201,
  1.484 µs (plain matmul tile) and 1.527 µs (ROCmFP4+WMMA tile) per launch (RUN-006 re-measurement; supersedes earlier ~1.3/~2.0 µs figures).
  rocWMMA C++ API emits real `v_wmma_f32_16x16x16_f16` hardware instruction.
- Phase 5b: int4×int4 integer matmul — **implemented and verified on gfx1201**
  (`src/kernels/int4_wmma.hip`, `src/kernels/int4_matvec.hip`,
  `include/rocminfer/{int4_wmma,int4_matvec}.hpp`, `Model::matvec_int4x` dispatch).
  gfx1201 emits the raw `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`
  builtin → `v_wmma_i32_16x16x16_iu4` (rocWMMA has no sub-byte path); gfx1031
   emits paired `v_dot8_i32_i4`. 16/16 CTest green on RX 9070 XT; bench
  (rows=64 cols=512, median of 5, RUN-008): rocFP4 2.090 µs, int4_dot scalar
  1.336 µs, int4_matvec 1.454 µs. Those supersede the cold-clock 4.592/4.690 µs
  figures recorded in RUN-007(b).
- Phase 5c: IU4-tile broadcast matvec `int4_matvec_iu4_i32`
  (`src/kernels/int4_wmma.hip`) — implemented and bit-exact vs an exact int64
  oracle on gfx1201 across 10 shapes, with the accumulator proven to stay in
  integer registers (objdump shows `v_wmma_i32_16x16x16_iu4 v[0:7], v13, v14,
  v[0:7]` feeding the destination back across every K step). **It is NOT wired
  as the dispatch**, because it measured 2.02 µs/launch against the row-wise
  kernel's 1.29 µs (1.56x slower) at 64x512, fastest of 5 timed passes,
  RUN-011. Suspected parallelism-bound, not matrix-core-bound: one wave owns 16
  rows, so 4 blocks x 32 threads vs the row-wise 64 x 256. `Model::matvec_int4x`
  still routes row-wise, and the reason is recorded in its Doxygen `@note` so it
  is not "optimised" back into a slower path.
- Phase 6: architecture coverage (multi-family quant, KV-cache quant, vision,
  MoE); **researched, not yet started**. The intended data model for the KV-quant
  part is written down in `docs/PHASE6_KV_QUANT_DATA_MODEL.md` (three distinct
  KV formats, the cross-layer ownership map, the residency truth table, and what
  is explicitly out of scope). DeepSeek-V4.1-Flash `2cba9e42` (MIT) is pinned in
  `THIRD_PARTY_SOURCES.md` as the spec source, with the FP4 nibble order recorded
  as an unresolved blocker that prevents any registry row or golden being
  written. §6's sanitizer/fuzz and §5 items 5/6 are closed; see RUN-010.

## 10. Evaluated and not adopted

Recorded so these are not re-proposed each session. Re-open only if the stated
reason stops being true.

| Tool | Local at | Verdict | Reason |
|---|---|---|---|
| REA (Reverse Engineer Anything) | `H:\ports\rea`, MCP `io.github.morluto/rea` | Not needed for format work | ROCmFPX and HaloFPX are **full readable source**, and we can build them. REA is for no-source binaries. Also unusable here: Hopper is unavailable on Windows, Ghidra is PE-only P0, and ROCmFPX ships Linux ELF. Golden tests come from building the pinned upstream quantizer, not decompiling. |
| TileLang-ROCm | `G:\tilelang-rocm`, MIT, v0.1.15, `76c95fe15a778d3eb7767c4217ec0c784c2318da` | Not adopted | Python DSL/codegen conflicts with §1 "no Python runtime" for the engine. Revisit only as a **development-time** kernel-authoring aid for Phase 2/5 if it can emit committed HIP for `gfx1031`/`gfx1201`. |
