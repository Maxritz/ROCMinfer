/**
 * @file bench_matmul.cpp
 * @brief Micro-bench: ROCmFP4 fused dequant+FMA matvec vs int4_dot integer matmul.
 *
 * Compares, on the same `rows` x `cols` workload, the two row-wise dot paths the
 * engine ships:
 *
 * - `rocmfp4_dot_f32` : int4 (Codebook10) weights x FP32 activations -> FP32, via a
 *   fused register FMA kernel.
 * - `int4_dot_i32`    : two's-complement int4 x int4 -> int32, via `__builtin_amdgcn_sdot8`
 *   on gfx1031 (RDNA2) or the scalar fallback on gfx1201 (RDNA4).
 * - `int4_matvec_i32` : two's-complement int4 rows x ONE shared int4 vector ->
 *   int32 (broadcast matvec: each row contracts against the same activation
 *   vector, unlike `int4_dot_i32`'s paired-rows `a[r].b[r]`).
 * - `int4_matvec_iu4_i32` : the same broadcast matvec driven by the RDNA4
 *   `v_wmma_i32_16x16x16_iu4` matrix instruction (one wave per 16 output rows),
 *   i.e. the engine's INT4 matmul hardware path on gfx1201, where `sdot8` does
 *   not exist.
 *
 * The int4_dot correctness is pinned to an exact int64 CPU oracle on row 0. Each
 * kernel is timed with HIP events over many launches (min-of-N to suppress
 * noise). Numbers are raw us/launch, not extrapolated.
 *
 * @note sdot8 is an RDNA2 (`gfx1031`) instruction; on RDNA4 (`gfx1201`, e.g. the RX
 *       9070 XT) the compiler rejects it and the int4_dot build falls back to the
 *       scalar C++ path, so the int4_dot timing below is the *scalar* cost on a 9070.
 *       The sdot8 hardware path has no measurement here because this host has no
 *       gfx1031 part; see docs/AMD_CAPABILITY_MATRIX.md section 1.
 */

#include <rocminfer/int4_dot.hpp>
#include <rocminfer/int4_matvec.hpp>
#include <rocminfer/int4_wmma.hpp>
#include <rocminfer/rocmfp4_dot.hpp>
#include <rocminfer/rocmfp4_reference.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

/// Abort with a message if a HIP call fails (no silent errors).
void hip_check(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s failed: %s (code %d)\n", what, hipGetErrorString(err),
                     static_cast<int>(err));
        std::exit(1);
    }
}

/// Simple splitmix64-style PRNG so the synthetic weights are reproducible.
std::uint64_t next_rng(std::uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// Pack `n` signed int4 values (range -8..7) into `n/2` little-endian bytes.
void pack_int4(std::uint8_t* out, const std::int8_t* v, std::int64_t n) {
    for (std::int64_t k = 0; k < n; ++k) {
        const std::uint8_t nibble = static_cast<std::uint8_t>(v[static_cast<std::size_t>(k)] & 0x0Fu);
        if (k % 2 == 0) {
            out[static_cast<std::size_t>(k / 2)] |= nibble;
        } else {
            out[static_cast<std::size_t>(k / 2)] |= (nibble << 4);
        }
    }
}

/// Two's-complement sign-extend a 4-bit nibble (host mirror of the kernel decode).
std::int8_t sext4(std::uint8_t nibble) noexcept {
    const int v = static_cast<int>(nibble & 0x0Fu);
    return static_cast<std::int8_t>(v < 8 ? v : v - 16);
}

/// CPU oracle: sum of `n` products of two's-complement int4 values.
std::int32_t oracle_int4(const std::uint8_t* a, const std::uint8_t* b, std::int64_t n) {
    std::int64_t sum = 0;
    for (std::int64_t k = 0; k < n; ++k) {
        const std::uint8_t ab = a[static_cast<std::size_t>(k / 2)];
        const std::uint8_t bb = b[static_cast<std::size_t>(k / 2)];
        const std::uint8_t an = (k % 2 == 0) ? (ab & 0x0Fu) : (ab >> 4);
        const std::uint8_t bn = (k % 2 == 0) ? (bb & 0x0Fu) : (bb >> 4);
        sum += static_cast<std::int64_t>(sext4(an)) * static_cast<std::int64_t>(sext4(bn));
    }
    return static_cast<std::int32_t>(sum);
}

int run() {
    if (hipSetDevice(0) != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice failed\n");
        return 1;
    }

    hipDeviceProp_t props{};
    hip_check(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties");
#if defined(__gfx1201__)
    static constexpr const char* kArch = "gfx1201 (RDNA4, scalar int4_dot fallback)";
#elif defined(__gfx1031__)
    static constexpr const char* kArch = "gfx1031 (RDNA2, sdot8 hardware path)";
#else
    static constexpr const char* kArch = "host (arch resolved at runtime)";
#endif
    const char* arch = (props.gcnArchName[0] != '\0') ? props.gcnArchName : kArch;
    std::printf("device: %s  arch: %s  warpSize: %d\n", props.name, arch, props.warpSize);

    const std::int64_t rows = 64;
    const std::int64_t cols = 512;          // multiple of 16 (IU4 WMMA) and 8 (int4_dot)
    const int iters = 500;
    const int warm = 10;

    // ---- ROCmFP4 weight tensor: rows x cols as Q4_0_ROCMFP4 (dual scale). ----
    const std::int64_t blocks_per_row = cols / rocminfer::QK_ROCMFP4;               // 16
    const std::int64_t fp4_row_bytes   = blocks_per_row * rocminfer::QK_ROCMFP4_BYTES; // 288
    const std::size_t  fp4_bytes       = static_cast<std::size_t>(rows) * fp4_row_bytes;
    std::vector<std::uint8_t> fp4(fp4_bytes, 0u);
    {
        std::uint64_t rng = 0xDU;
        for (std::int64_t r = 0; r < rows; ++r) {
            for (std::int64_t blk = 0; blk < blocks_per_row; ++blk) {
                std::uint8_t* b = fp4.data()
                    + static_cast<std::size_t>(r) * fp4_row_bytes
                    + static_cast<std::size_t>(blk) * rocminfer::QK_ROCMFP4_BYTES;
                for (std::uint32_t j = 0; j < rocminfer::QS_ROCMFP4; ++j) {        // 16 nibble bytes
                    b[static_cast<std::size_t>(j)] =
                        static_cast<std::uint8_t>(next_rng(rng) & 0x07u);  // code 0..7
                }
                b[16] = 0x01u;   // dual-scale e[0]: valid finite UE4M3
                b[17] = 0x01u;   // e[1]
            }
        }
    }

    // ---- int4 operands: rows x cols signed int4 (two's complement). ----
    const std::int64_t int4_row_bytes = cols / 2;                            // 256
    const std::size_t  int4_bytes     = static_cast<std::size_t>(rows) * int4_row_bytes;
    std::vector<std::uint8_t> a(int4_bytes, 0u);
    std::vector<std::uint8_t> bb(int4_bytes, 0u);
    {
        std::uint64_t rng = 0x1234ULL;
        std::vector<std::int8_t> va(static_cast<std::size_t>(cols));
        std::vector<std::int8_t> vb(static_cast<std::size_t>(cols));
        for (std::int64_t r = 0; r < rows; ++r) {
            for (std::int64_t k = 0; k < cols; ++k) {
                va[static_cast<std::size_t>(k)] =
                    static_cast<std::int8_t>((next_rng(rng) % 16) - 8);       // -8..7
                vb[static_cast<std::size_t>(k)] =
                    static_cast<std::int8_t>((next_rng(rng) % 16) - 8);
            }
            pack_int4(a.data() + static_cast<std::size_t>(r) * int4_row_bytes,
                      va.data(), cols);
            pack_int4(bb.data() + static_cast<std::size_t>(r) * int4_row_bytes,
                      vb.data(), cols);
        }
    }

    // FP32 activation vector for the ROCmFP4 matvec (shared across rows).
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (std::int64_t k = 0; k < cols; ++k) {
        x[static_cast<std::size_t>(k)] =
            static_cast<float>((k * 7 + 3) & 0x07) - 3.0f;                    // -3..3
    }

    // Device buffers.
    void* dfp4 = nullptr;
    void* dx   = nullptr;
    void* dyf  = nullptr;
    void* da   = nullptr;
    void* dbb  = nullptr;
    void* dyi  = nullptr;
    void* dWm  = nullptr;
    void* dxm  = nullptr;
    void* dym  = nullptr;
    void* dyu  = nullptr;
    hip_check(hipMalloc(&dfp4, fp4_bytes), "hipMalloc fp4");
    hip_check(hipMalloc(&dx, sizeof(float) * static_cast<std::size_t>(cols)), "hipMalloc dx");
    hip_check(hipMalloc(&dyf, sizeof(float) * static_cast<std::size_t>(rows)), "hipMalloc dyf");
    hip_check(hipMalloc(&da, int4_bytes), "hipMalloc da");
    hip_check(hipMalloc(&dbb, int4_bytes), "hipMalloc dbb");
    hip_check(hipMalloc(&dyi, sizeof(std::int32_t) * static_cast<std::size_t>(rows)), "hipMalloc dyi");
    hip_check(hipMemcpy(dfp4, fp4.data(), fp4_bytes, hipMemcpyHostToDevice), "hipMemcpy fp4");
    hip_check(hipMemcpy(dx, x.data(), sizeof(float) * static_cast<std::size_t>(cols),
                          hipMemcpyHostToDevice), "hipMemcpy dx");
    hip_check(hipMemcpy(da, a.data(), int4_bytes, hipMemcpyHostToDevice), "hipMemcpy da");
    hip_check(hipMemcpy(dbb, bb.data(), int4_bytes, hipMemcpyHostToDevice), "hipMemcpy dbb");

    // ---- int4 broadcast matvec workload: the SAME packed rows as `a`, but one
    // ---- shared activation vector (row 0 of `bb`) broadcast to every row. ----
    const std::size_t xm_bytes = static_cast<std::size_t>(int4_row_bytes);
    std::vector<std::uint8_t> xm(xm_bytes, 0u);
    std::memcpy(xm.data(), bb.data(), xm_bytes);
    hip_check(hipMalloc(&dWm, int4_bytes), "hipMalloc dWm");
    hip_check(hipMalloc(&dxm, xm_bytes), "hipMalloc dxm");
    hip_check(hipMalloc(&dym, sizeof(std::int32_t) * static_cast<std::size_t>(rows)), "hipMalloc dym");
    hip_check(hipMalloc(&dyu, sizeof(std::int32_t) * static_cast<std::size_t>(rows)), "hipMalloc dyu");
    hip_check(hipMemcpy(dWm, a.data(), int4_bytes, hipMemcpyHostToDevice), "hipMemcpy dWm");
    hip_check(hipMemcpy(dxm, xm.data(), xm_bytes, hipMemcpyHostToDevice), "hipMemcpy dxm");

    // ---- Correctness: int4_dot row 0 vs CPU oracle (device `out`, not host). ----
    const std::int32_t expect = oracle_int4(a.data(), bb.data(), cols);
    std::int32_t got0 = 0;
    const int rc = rocminfer::int4_dot_i32(static_cast<std::int32_t*>(dyi), da, dbb,
                                           1, cols, nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "int4_dot_i32 returned %d\n", rc);
        return 1;
    }
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize (correctness)");
    hip_check(hipMemcpy(&got0, dyi, sizeof(std::int32_t), hipMemcpyDeviceToHost),
              "hipMemcpy got0");
    const bool correct = (got0 == expect);
    std::printf("int4_dot correctness (row 0): got=%d want=%d  %s\n",
                static_cast<int>(got0), static_cast<int>(expect),
                correct ? "OK" : "MISMATCH");

    // ---- Correctness: int4 broadcast matvec row 0 vs CPU oracle. ----
    const std::int32_t expect_mv = oracle_int4(a.data(), xm.data(), cols);
    std::int32_t got_mv0 = 0;
    const int rc_mv =
        rocminfer::int4_matvec_i32(static_cast<std::int32_t*>(dym), dWm, dxm,
                                   1, cols, nullptr);
    if (rc_mv != 0) {
        std::fprintf(stderr, "int4_matvec_i32 returned %d\n", rc_mv);
        return 1;
    }
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize (matvec correctness)");
    hip_check(hipMemcpy(&got_mv0, dym, sizeof(std::int32_t), hipMemcpyDeviceToHost),
              "hipMemcpy got_mv0");
    const bool correct_mv = (got_mv0 == expect_mv);
    std::printf("int4_matvec correctness (row 0, shared x): got=%d want=%d  %s\n",
                static_cast<int>(got_mv0), static_cast<int>(expect_mv),
                correct_mv ? "OK" : "MISMATCH");

    // ---- Correctness: the IU4 WMMA matvec over every row vs the CPU oracle. ----
    const int rc_iu4 =
        rocminfer::int4_matvec_iu4_i32(static_cast<std::int32_t*>(dyu), dWm, dxm,
                                       rows, cols, nullptr);
    if (rc_iu4 != 0) {
        std::fprintf(stderr, "int4_matvec_iu4_i32 returned %d\n", rc_iu4);
        return 1;
    }
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize (iu4 correctness)");
    std::vector<std::int32_t> got_iu4(static_cast<std::size_t>(rows), 0);
    hip_check(hipMemcpy(got_iu4.data(), dyu, static_cast<std::size_t>(rows) * sizeof(std::int32_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy got_iu4");
    int iu4_bad = 0;
    for (std::int64_t r = 0; r < rows; ++r) {
        if (got_iu4[static_cast<std::size_t>(r)] !=
            oracle_int4(a.data() + static_cast<std::size_t>(r) * int4_row_bytes, xm.data(), cols)) {
            if (iu4_bad < 4) {
                std::printf("  int4_matvec_iu4 row %lld: got=%d want=%d\n",
                            static_cast<long long>(r), got_iu4[static_cast<std::size_t>(r)],
                            oracle_int4(a.data() + static_cast<std::size_t>(r) * int4_row_bytes,
                                        xm.data(), cols));
            }
            ++iu4_bad;
        }
    }
    const bool correct_iu4 = (iu4_bad == 0);
    std::printf("int4_matvec_iu4 correctness (all %lld rows, shared x): %d bad  %s\n",
                 static_cast<long long>(rows), iu4_bad,
                 correct_iu4 ? "OK" : "MISMATCH");

    // ---- Timing helpers. ----
    hipEvent_t t0{};
    hipEvent_t t1{};
    hip_check(hipEventCreateWithFlags(&t0, hipEventBlockingSync), "hipEventCreate t0");
    hip_check(hipEventCreateWithFlags(&t1, hipEventBlockingSync), "hipEventCreate t1");
    float ms = 0.0f;

    // Median-of-N. A single averaged floor reports the GPU's boost clock as if
    // it were a kernel property: on this host the FIRST timed pass of a session
    // measured 4.595 us for the same rocmFP4 workload that settles at ~2.1 us
    // once the clocks ramp, so the head line used to be whichever sample
    // happened to run first. Each pass re-times `iters` launches with its own
    // event pair; the median of the passes is the reported figure and the min
    // and max passes are printed beside it so the spread stays visible.
    constexpr int kPasses = 5;
    float fp4_samples[kPasses] = {};
    float i32_samples[kPasses] = {};
    float mv_samples[kPasses] = {};
    float iu4_samples[kPasses] = {};

    auto sample = [&](void (*launch)(void*, const void*, const void*, void*,
                                     std::int64_t, std::int64_t, void*),
                      void* out, const void* a, const void* b, std::int64_t r,
                      std::int64_t c, float* samples) {
        for (int pass = 0; pass < kPasses; ++pass) {
            for (int i = 0; i < warm; ++i) {
                launch(out, a, b, nullptr, r, c, nullptr);
            }
            hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize (sample warmup)");
            hip_check(hipEventRecord(t0), "hipEventRecord t0 (sample)");
            for (int i = 0; i < iters; ++i) {
                launch(out, a, b, nullptr, r, c, nullptr);
            }
            hip_check(hipEventRecord(t1), "hipEventRecord t1 (sample)");
            hip_check(hipEventSynchronize(t1), "hipEventSynchronize t1 (sample)");
            hip_check(hipEventElapsedTime(&ms, t0, t1), "hipEventElapsedTime (sample)");
            samples[pass] = ms * 1000.0f / static_cast<float>(iters);
        }
    };

    // Pre-warm every kernel once, untimed, before any pass is measured. The
    // first launch of a kernel image pays module-load and instruction-cache
    // costs that are not the kernel's cost, and an early version of this tool
    // folded that into pass 0: a fresh process measured 296 us/launch for the
    // same rocmFP4 work whose steady state is single-digit microseconds. Every
    // pass already warms up internally; this handles the process's first launch
    // of each image so no timed pass ever pays it.
    const auto prewarm = [&](void (*launch)(void*, const void*, const void*, void*,
                                           std::int64_t, std::int64_t, void*),
                             void* out, const void* a, const void* b, std::int64_t r,
                             std::int64_t c) {
        for (int i = 0; i < warm; ++i) {
            launch(out, a, b, nullptr, r, c, nullptr);
        }
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize (prewarm)");
    };
    prewarm([](void* out, const void* a, const void* b, void*, std::int64_t r,
               std::int64_t c, void*) {
                (void)rocminfer::rocmfp4_dot_f32(static_cast<float*>(out), a,
                                                 static_cast<const float*>(b), r, c,
                                                 /*dual_scale=*/true, nullptr);
            },
            dyf, dfp4, dx, rows, cols);
    prewarm([](void* out, const void* a, const void* b, void*, std::int64_t r,
               std::int64_t c, void*) {
                (void)rocminfer::int4_dot_i32(static_cast<std::int32_t*>(out), a, b, r,
                                              c, nullptr);
            },
            dyi, da, dbb, rows, cols);
    prewarm([](void* out, const void* a, const void* b, void*, std::int64_t r,
               std::int64_t c, void*) {
                (void)rocminfer::int4_matvec_i32(static_cast<std::int32_t*>(out), a, b,
                                                 r, c, nullptr);
            },
            dym, dWm, dxm, rows, cols);
    prewarm([](void* out, const void* a, const void* b, void*, std::int64_t r,
               std::int64_t c, void*) {
                (void)rocminfer::int4_matvec_iu4_i32(static_cast<std::int32_t*>(out), a, b,
                                                     r, c, nullptr);
            },
            dyu, dWm, dxm, rows, cols);

    // Timing: ROCmFP4 fused matvec.
    sample([](void* out, const void* a, const void* b, void*, std::int64_t r,
              std::int64_t c, void*) {
               (void)rocminfer::rocmfp4_dot_f32(static_cast<float*>(out), a,
                                                static_cast<const float*>(b), r, c,
                                                /*dual_scale=*/true, nullptr);
           },
           dyf, dfp4, dx, rows, cols, fp4_samples);

    // Timing: int4_dot (scalar fallback on gfx1201).
    sample([](void* out, const void* a, const void* b, void*, std::int64_t r,
              std::int64_t c, void*) {
               (void)rocminfer::int4_dot_i32(static_cast<std::int32_t*>(out), a, b, r,
                                             c, nullptr);
           },
           dyi, da, dbb, rows, cols, i32_samples);

    // Timing: int4 broadcast matvec (shared-x workload, rows x cols).
    sample([](void* out, const void* a, const void* b, void*, std::int64_t r,
              std::int64_t c, void*) {
               (void)rocminfer::int4_matvec_i32(static_cast<std::int32_t*>(out), a, b,
                                                r, c, nullptr);
           },
           dym, dWm, dxm, rows, cols, mv_samples);

    // Timing: int4 broadcast matvec on the RDNA4 IU4 WMMA hardware path.
    sample([](void* out, const void* a, const void* b, void*, std::int64_t r,
              std::int64_t c, void*) {
               (void)rocminfer::int4_matvec_iu4_i32(static_cast<std::int32_t*>(out), a, b,
                                                    r, c, nullptr);
           },
           dyu, dWm, dxm, rows, cols, iu4_samples);

    // The headline is the FATest pass, not the median. Every pass enqueues
    // `iters` launches between two stream events, so the t0->t1 window measures
    // GPU execution PLUS any idle gap the GPU spends waiting for the host to
    // queue the next launch. On this host that contamination is large and
    // intermittent: a stall of up to ~250 us/launch shows up in isolated passes
    // (Radeon Software / AMDRSServ are resident), while int4_dot rarely stalls.
    // A stalled pass is an OS scheduling artifact, not a property of the kernel,
    // so the minimum pass is the least-noise estimate of the amortised
    // per-launch cost and the median and max are printed beside it to keep the
    // spread visible. For kernels within a few microseconds of launch overhead,
    // even this figure is an UPPER BOUND on kernel time, not a measurement of it.
    struct TimingSummary {
        int valid = 0;  ///< Passes with a non-negative elapsed span.
        float lo = 0.0f;  ///< Fastest usable pass, us/launch (the headline).
        float hi = 0.0f;  ///< Slowest usable pass, us/launch.
        float median = 0.0f;  ///< Median of the usable passes, us/launch.
    };
    const auto summarize = [](const float* samples) {
        TimingSummary out;
        float tmp[kPasses];
        for (int i = 0; i < kPasses; ++i) {
            if (samples[i] >= 0.0f) {
                tmp[out.valid++] = samples[i];
            }
        }
        if (out.valid != 0) {
            std::sort(tmp, tmp + out.valid);
            out.lo = tmp[0];
            out.hi = tmp[out.valid - 1];
            out.median = tmp[out.valid / 2];
        }
        return out;
    };
    const TimingSummary fp4_stat = summarize(fp4_samples);
    const TimingSummary i32_stat = summarize(i32_samples);
    const TimingSummary mv_stat = summarize(mv_samples);
    const TimingSummary iu4_stat = summarize(iu4_samples);

    // A majority of the passes must be usable; fewer than that and the number
    // below is guesswork, so say so and exit non-zero instead of printing it.
    for (const TimingSummary* s : {&fp4_stat, &i32_stat, &mv_stat, &iu4_stat}) {
        if (s->valid * 2 <= kPasses) {
            std::fprintf(stderr,
                         "FAIL: only %d of %d timing passes were usable\n",
                         s->valid, kPasses);
            return 1;
        }
    }

    std::printf("\n");
    std::printf("workload: rows=%lld cols=%lld  (row-wise batched dot, nibble-aligned)\n",
                static_cast<long long>(rows), static_cast<long long>(cols));
    std::printf("  passes: %d timed passes x %d launches each%s\n", kPasses, iters,
                fp4_stat.valid == kPasses && i32_stat.valid == kPasses &&
                         mv_stat.valid == kPasses && iu4_stat.valid == kPasses
                    ? " (headline = fastest pass)"
                    : " (headline = fastest pass; negative-span passes discarded)");
    std::printf("  rocFP4  dequant+FMA matvec : %9.3f us/launch  (median %.3f, max %.3f)\n",
                 static_cast<double>(fp4_stat.lo), static_cast<double>(fp4_stat.median),
                 static_cast<double>(fp4_stat.hi));
    std::printf("  int4_dot (scalar fallback)  : %9.3f us/launch  (median %.3f, max %.3f)\n",
                 static_cast<double>(i32_stat.lo), static_cast<double>(i32_stat.median),
                 static_cast<double>(i32_stat.hi));
    std::printf("  int4_matvec (shared x)      : %9.3f us/launch  (median %.3f, max %.3f)\n",
                 static_cast<double>(mv_stat.lo), static_cast<double>(mv_stat.median),
                 static_cast<double>(mv_stat.hi));
    std::printf("  int4_matvec_iu4 (WMMA path)  : %9.3f us/launch  (median %.3f, max %.3f)\n",
                  static_cast<double>(iu4_stat.lo), static_cast<double>(iu4_stat.median),
                  static_cast<double>(iu4_stat.hi));
    std::printf("  ratio  int4_dot / rocFP4    : %.2fx\n",
                static_cast<double>(i32_stat.lo) / static_cast<double>(fp4_stat.lo));
    std::printf("  ratio  int4_matvec / rocFP4 : %.2fx\n",
                static_cast<double>(mv_stat.lo) / static_cast<double>(fp4_stat.lo));
    std::printf("  ratio  int4_matvec_iu4 / rocFP4 : %.2fx\n",
                 static_cast<double>(iu4_stat.lo) / static_cast<double>(fp4_stat.lo));
    std::printf("  ratio  int4_matvec_iu4 / int4_matvec : %.2fx\n",
                 static_cast<double>(iu4_stat.lo) / static_cast<double>(mv_stat.lo));
    std::printf("  (headline is the fastest pass: t0->t1 spans 500 back-to-back launches and\n");
    std::printf("   therefore includes any GPU idle gap waiting on the host. Stalled passes\n");
    std::printf("   are discarded by taking the minimum, and the spread is printed above.)\n");
    std::printf("  (sdot8 is gfx1031/RDNA2-only; on this RX 9070 XT the int4_dot path is the\n");
    std::printf("   scalar fallback. The RDNA4 hardware int4 path is the\n");
    std::printf("   v_wmma_i32_16x16x16_iu4 tile behind int4_matvec_iu4_i32: one wave per 16\n");
    std::printf("   output rows, the accumulator in integer registers across every 16-channel\n");
    std::printf("   K step. int4_wmma_16x16 remains the single-tile primitive it was built as.)\n");

    hip_check(hipEventDestroy(t0), "hipEventDestroy t0");
    hip_check(hipEventDestroy(t1), "hipEventDestroy t1");
    hip_check(hipFree(dfp4), "hipFree dfp4");
    hip_check(hipFree(dx), "hipFree dx");
    hip_check(hipFree(dyf), "hipFree dyf");
    hip_check(hipFree(da), "hipFree da");
    hip_check(hipFree(dbb), "hipFree dbb");
    hip_check(hipFree(dyi), "hipFree dyi");
    hip_check(hipFree(dWm), "hipFree dWm");
    hip_check(hipFree(dxm), "hipFree dxm");
    hip_check(hipFree(dym), "hipFree dym");
    hip_check(hipFree(dyu), "hipFree dyu");
    return (correct && correct_mv && correct_iu4) ? 0 : 1;
}

}  // namespace

int main() {
    return run();
}
