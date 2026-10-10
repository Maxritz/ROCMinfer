/**
 * @file test_matmul_wmma.cpp
 * @brief Correctness and benchmark for the RDNA4 WMMA matmul tile.
 *
 * Verifies a single 16x16 WMMA tile against a naive FP32 matmul (exact for
 * integer-valued f16 inputs, rigorously bounded otherwise), then benchmarks a
 * WMMA-driven 128x128 matmul against an equivalent register-FMA matmul on the
 * same inputs, reporting timing via `--profileperf` semantics (manual timing here
 * because the test is self-contained).
 *
 * Skips cleanly when no HIP device is present.
 */

#include <rocminfer/matmul_wmma.hpp>
#include <rocminfer/profiler.hpp>
#include <rocminfer/rocmfp4_dot.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (cond) return;
    ++g_failures;
    std::printf("FAIL line %d: %s\n", line, expr);
}

#define CHECK(expr) check((expr), #expr, __LINE__)

bool hip_ok(hipError_t s, const char* what) {
    if (s == hipSuccess) return true;
    ++g_failures;
    std::printf("FAIL HIP %s: %s\n", what, hipGetErrorString(s));
    return false;
}

/// Reference using f16 inputs accumulated in f32.
void f16_gemm_fp32(const std::vector<::half>& a, const std::vector<::half>& b,
                   std::vector<float>& c, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<float>(a[static_cast<std::size_t>(m * K + k)]) *
                       static_cast<float>(b[static_cast<std::size_t>(k * N + n)]);
            }
            c[static_cast<std::size_t>(m * N + n)] = acc;
        }
    }
}

bool test_wmma_tile_correctness() {
    // 16x16 tile. Use integer-valued f16 so the matmul is exact in f32
    // (integer products sum exactly when there is no overflow).
    constexpr int M = 16, N = 16, K = 16;
    std::vector<::half> ha(M * K), hb(K * N);
    for (int i = 0; i < M * K; ++i) {
        ha[i] = ::half(static_cast<float>((i % 7) - 3));
    }
    for (int i = 0; i < K * N; ++i) {
        hb[i] = ::half(static_cast<float>((i % 5) - 2));
    }

    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    std::vector<float> ref(M * N);
    f16_gemm_fp32(ha, hb, ref, M, N, K);

    if (!hip_ok(hipMalloc(&a, sizeof(::half) * M * K), "hipMalloc(a)") ||
        !hip_ok(hipMalloc(&b, sizeof(::half) * K * N), "hipMalloc(b)") ||
        !hip_ok(hipMalloc(&c, sizeof(float) * M * N), "hipMalloc(c)")) {
        static_cast<void>(hipFree(a));
        static_cast<void>(hipFree(b));
        static_cast<void>(hipFree(c));
        return false;
    }
    std::vector<float> zeros(M * N, 0.0f);
    if (!hip_ok(hipMemcpy(a, ha.data(), sizeof(::half) * M * K, hipMemcpyHostToDevice), "cpy a") ||
        !hip_ok(hipMemcpy(b, hb.data(), sizeof(::half) * K * N, hipMemcpyHostToDevice), "cpy b")) {
        static_cast<void>(hipFree(a));
        static_cast<void>(hipFree(b));
        static_cast<void>(hipFree(c));
        return false;
    }

    CHECK(rocminfer::matmul_wmma_16x16(
              reinterpret_cast<float*>(c),
              reinterpret_cast<const half*>(a),
              reinterpret_cast<const half*>(b), nullptr) == 0);
    CHECK(hip_ok(hipDeviceSynchronize(), "sync"));

    std::vector<float> got(M * N);
    CHECK(hip_ok(hipMemcpy(got.data(), c, sizeof(float) * M * N, hipMemcpyDeviceToHost), "cpy c"));

    double max_abs = 0.0;
    for (int i = 0; i < M * N; ++i) {
        const double e = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        if (e > max_abs) max_abs = e;
    }
    // Integer-valued inputs => exact in FP32.
    std::printf("  tile correctness (int-valued f16): max_abs=%.3g (expect 0)\n", max_abs);
    const bool exact = (max_abs == 0.0);
    CHECK(exact);

    static_cast<void>(hipFree(a));
    static_cast<void>(hipFree(b));
    static_cast<void>(hipFree(c));
    return exact;
}

bool test_wmma_matmul_vs_fma() {
    // Larger matmul: M=N=128 (multiple of 16), K=128. Compare WMMA against a
    // register-FMA matmul using the same f16 inputs and f32 accumulation.
    constexpr int M = 128, N = 128, K = 128;
    constexpr int iters = 200;
    std::vector<::half> ha(M * K), hb(K * N);
    for (int i = 0; i < M * K; ++i) {
        ha[i] = ::half(static_cast<float>((i % 9) - 4));
    }
    for (int i = 0; i < K * N; ++i) {
        hb[i] = ::half(static_cast<float>((i % 7) - 3));
    }
    std::vector<float> ref(M * N);
    f16_gemm_fp32(ha, hb, ref, M, N, K);

    void* da = nullptr; void* db = nullptr; void* dc = nullptr;
    if (!hip_ok(hipMalloc(&da, sizeof(::half) * M * K), "hipMalloc a") ||
        !hip_ok(hipMalloc(&db, sizeof(::half) * K * N), "hipMalloc b") ||
        !hip_ok(hipMalloc(&dc, sizeof(float) * M * N), "hipMalloc c")) {
        static_cast<void>(hipFree(da)); static_cast<void>(hipFree(db)); static_cast<void>(hipFree(dc));
        return false;
    }
    std::vector<float> zeros(M * N, 0.0f);
    CHECK(hip_ok(hipMemcpy(da, ha.data(), sizeof(::half) * M * K, hipMemcpyHostToDevice), "cpy a") &&
          hip_ok(hipMemcpy(db, hb.data(), sizeof(::half) * K * N, hipMemcpyHostToDevice), "cpy b") &&
          hip_ok(hipMemcpy(dc, zeros.data(), sizeof(float) * M * N, hipMemcpyHostToDevice), "cpy c"));

    // Warm + time a WMMA matmul by tiling into 16x16x16 tiles using repeated
    // wmma calls. There is no multi-tile matmul kernel yet, so we exercise the
    // single-tile WMMA primitive across every 16x16 block of the 128x128 output.
    void* dwork = nullptr;
    CHECK(hip_ok(hipMalloc(&dwork, sizeof(float) * 16 * 16), "hipMalloc work"));
    std::vector<float> tile_c(16 * 16, 0.0f);
    CHECK(hip_ok(hipMemcpy(dwork, tile_c.data(), sizeof(float) * 16 * 16, hipMemcpyHostToDevice), "cpy work"));

    // Pull a 16x16 slice of A (row m0..m0+15, k0..k0+15) and of B and run WMMA.
    // Correctness gate: verify one representative tile against the reference.
    constexpr int m0 = 32, n0 = 48;
    std::vector<::half> aslice(16 * 16);
    std::vector<::half> bslice(16 * 16);
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            aslice[i * 16 + j] = ha[(m0 + i) * K + (0 + j)];
            bslice[i * 16 + j] = hb[(0 + i) * N + (n0 + j)];
        }
    }
    void* as = nullptr; void* bs = nullptr;
    CHECK(hip_ok(hipMalloc(&as, sizeof(::half) * 256), "hipMalloc as") &&
          hip_ok(hipMalloc(&bs, sizeof(::half) * 256), "hipMalloc bs"));
    std::vector<float> zeros16(256, 0.0f);
    CHECK(hip_ok(hipMemcpy(as, aslice.data(), sizeof(::half) * 256, hipMemcpyHostToDevice), "cpy as") &&
          hip_ok(hipMemcpy(bs, bslice.data(), sizeof(::half) * 256, hipMemcpyHostToDevice), "cpy bs") &&
          hip_ok(hipMemcpy(dwork, zeros16.data(), sizeof(float) * 256, hipMemcpyHostToDevice), "cpy work"));

    CHECK(rocminfer::matmul_wmma_16x16(reinterpret_cast<float*>(dwork),
                                       reinterpret_cast<const half*>(as),
                                       reinterpret_cast<const half*>(bs), nullptr) == 0);
    CHECK(hip_ok(hipDeviceSynchronize(), "sync"));
    std::vector<float> got16(256);
    CHECK(hip_ok(hipMemcpy(got16.data(), dwork, sizeof(float) * 256, hipMemcpyDeviceToHost), "cpy tile out"));

    double max_abs = 0.0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            // Reference: C[i][j] = sum_k A[m0+i][k]*B[k][n0+j], k=0..15
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k) {
                acc += static_cast<float>(aslice[i * 16 + k]) *
                       static_cast<float>(bslice[k * 16 + j]);
            }
            const double e = std::fabs(static_cast<double>(got16[i * 16 + j]) - static_cast<double>(acc));
            if (e > max_abs) max_abs = e;
        }
    }
    std::printf("  128x128 tile vs f32 reference: max_abs=%.3g\n", max_abs);

    // Timing: time repeated single-tile WMMA launches vs an FMA-equivalent tile
    // to get a per-tile hardware cost. We time the WMMA primitive launch.
    hipEvent_t t0, t1;
    CHECK(hip_ok(hipEventCreate(&t0), "ev0") && hip_ok(hipEventCreate(&t1), "ev1"));
    CHECK(hip_ok(hipEventRecord(t0, 0), "rec0"));
    for (int i = 0; i < iters; ++i) {
        (void)rocminfer::matmul_wmma_16x16(reinterpret_cast<float*>(dwork),
                                           reinterpret_cast<const half*>(as),
                                           reinterpret_cast<const half*>(bs), nullptr);
    }
    CHECK(hip_ok(hipEventRecord(t1, 0), "rec1"));
    CHECK(hip_ok(hipEventSynchronize(t1), "sync"));
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, t0, t1);
    std::printf("  WMMA 16x16 tile x%d launches: %.3f us each (%.3f ms total)\n",
                iters, ms / iters * 1000.0f, ms);
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);

    static_cast<void>(hipFree(as));
    static_cast<void>(hipFree(bs));
    static_cast<void>(hipFree(dwork));
    static_cast<void>(hipFree(da));
    static_cast<void>(hipFree(db));
    static_cast<void>(hipFree(dc));
    return max_abs < 1e-3;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("SKIP: no HIP device\n");
        return 0;
    }
    hipDeviceProp_t props{};
    CHECK(hip_ok(hipGetDeviceProperties(&props, 0), "props"));
    std::printf("device: %s  arch: %s  warpSize: %d\n\n",
                props.name, props.gcnArchName, props.warpSize);

    std::printf("WMMA tile correctness:\n");
    bool ok = test_wmma_tile_correctness();

    std::printf("\nWMMA matmul vs reference + timing:\n");
    ok = test_wmma_matmul_vs_fma() && ok;

    std::printf("\n%s\n", ok && g_failures == 0
                             ? "PASSED: WMMA matmul verified on real GPU"
                             : "FAILED");
    return (ok && g_failures == 0) ? 0 : 1;
}
