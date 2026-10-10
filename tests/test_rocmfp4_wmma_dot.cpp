/**
 * @file test_rocmfp4_wmma_dot.cpp
 * @brief Correctness and timing for the ROCmFP4-dequantize-and-WMMA dot tile.
 *
 * Builds a 16-row window of packed Codebook10 weights, dequantizes a 16x16 tile
 * on the CPU oracle (`rocmfp4_dequantize_row`), computes a reference 16x16 f32
 * matmul against a broadcast 16-wide activation slice, and compares it to the GPU
 * WMMA kernel output. Reports tile timing via HIP events.
 */

#include <rocminfer/rocmfp4_dot.hpp>
#include <rocminfer/rocmfp4_reference.hpp>
#include <rocminfer/rocmfp4_wmma_dot.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>

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

/// Pack 16 random Codebook10 codes plus a scale into an 18-byte dual-scale block.
void pack_block(std::uint8_t* block, const std::uint8_t* codes, std::uint8_t scale_lo) {
    for (int i = 0; i < 16; ++i) {
        block[i] = codes[i];
    }
    block[rocminfer::QS_ROCMFP4] = scale_lo;
    block[rocminfer::QS_ROCMFP4 + 1u] = static_cast<std::uint8_t>(scale_lo + 1u);
}

std::uint64_t splitmix64_next(std::uint64_t* state) {
    std::uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

bool test_tile_correctness() {
    constexpr int ROWS = 16, COLS = 16;
    constexpr std::int64_t row_bytes = static_cast<std::int64_t>(rocminfer::QK_ROCMFP4_BYTES);  // 18

    std::uint64_t rng = 0xC0FFEE;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(ROWS * row_bytes), 0);
    // 16 rows, each one 18-byte dual-scale block holding 16 codes (cols 0..15, low
    // nibble of each block byte).
    for (int r = 0; r < ROWS; ++r) {
        std::uint8_t codes[16];
        for (int i = 0; i < 16; ++i) {
            codes[i] = static_cast<std::uint8_t>(splitmix64_next(&rng) & 0x0F);
        }
        pack_block(packed.data() + r * row_bytes, codes,
                   static_cast<std::uint8_t>(1 + (splitmix64_next(&rng) & 7)));
    }

    // Activation slice in f16.
    std::vector<::half> x_host(COLS);
    for (int i = 0; i < COLS; ++i) {
        x_host[i] = __float2half(static_cast<float>((splitmix64_next(&rng) >> 11) % 15 - 7) / 8.0f);
    }

    // CPU oracle: dequantize a FULL 32-weight block per row (the dequantizer
    // requires multiples of QK_ROCMFP4=32), then the kernel's col_group=0 tile
    // covers the low 16 weights (cols 0..15), so compare against deq32[r][0..15].
    constexpr int BLOCK_W = rocminfer::QK_ROCMFP4;  // 32
    std::vector<float> deq32(static_cast<std::size_t>(ROWS * BLOCK_W));
    for (int r = 0; r < ROWS; ++r) {
        CHECK(rocminfer::rocmfp4_dequantize_row(
                  deq32.data() + r * BLOCK_W,
                  packed.data() + r * row_bytes, static_cast<std::size_t>(row_bytes),
                  static_cast<std::size_t>(BLOCK_W), true) == 0);
    }

    std::vector<float> ref(static_cast<std::size_t>(ROWS * COLS), 0.0f);
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            double acc = 0.0;
            for (int k = 0; k < COLS; ++k) {
                acc += static_cast<double>(deq32[r * BLOCK_W + k]) *
                       static_cast<double>(x_host[k]);
            }
            ref[r * COLS + c] = static_cast<float>(acc);
        }
    }

    // Upload to device and run the WMMA tile (col_group=0 => low half, col 0..15).
    void* dp = nullptr; void* dx = nullptr; void* dout = nullptr;
    CHECK(hip_ok(hipMalloc(&dp, packed.size()), "hipMalloc(w)") &&
          hip_ok(hipMalloc(&dx, sizeof(::half) * COLS), "hipMalloc(x)") &&
          hip_ok(hipMalloc(&dout, sizeof(float) * ROWS * COLS), "hipMalloc(out)"));
    std::vector<float> zeros(ROWS * COLS, 0.0f);
    CHECK(hip_ok(hipMemcpy(dp, packed.data(), packed.size(), hipMemcpyHostToDevice), "cpy w") &&
          hip_ok(hipMemcpy(dx, x_host.data(), sizeof(::half) * COLS, hipMemcpyHostToDevice), "cpy x") &&
          hip_ok(hipMemcpy(dout, zeros.data(), sizeof(float) * ROWS * COLS, hipMemcpyHostToDevice), "cpy out"));

    void* dev_scales = nullptr;
    CHECK(rocminfer::rocmfp4_wmma_dot_resolve_scales(&dev_scales) == 0 && dev_scales != nullptr);
    const std::int64_t col_group = 0;
    CHECK(rocminfer::rocmfp4_wmma_dot(
              reinterpret_cast<float*>(dout), dp, reinterpret_cast<const ::half*>(dx),
              row_bytes, col_group, reinterpret_cast<const float*>(dev_scales),
              /*dual_scale=*/true, nullptr) == 0);
    CHECK(hip_ok(hipDeviceSynchronize(), "sync"));

    std::vector<float> got(ROWS * COLS);
    CHECK(hip_ok(hipMemcpy(got.data(), dout, sizeof(float) * ROWS * COLS,
                           hipMemcpyDeviceToHost), "cpy out"));

    double max_abs = 0.0, max_rel = 0.0;
    for (int i = 0; i < ROWS * COLS; ++i) {
        const double e = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        if (e > max_abs) max_abs = e;
        const double denom = std::max(std::fabs(static_cast<double>(ref[i])), 1e-300);
        if (e / denom > max_rel) max_rel = e / denom;
    }
    std::printf("  tile dequant+WMMA vs f32 oracle: max_abs=%.3g max_rel=%.3g\n", max_abs, max_rel);

    // f16 activations + f16 weights give one f16 round per product; with only 16
    // terms the WMMA path is expected to match the f32 oracle within a few ULPs.
    const bool ok = (max_rel < 1e-4);

    static_cast<void>(hipFree(dp));
    static_cast<void>(hipFree(dx));
    static_cast<void>(hipFree(dout));
    return ok;
}

bool test_tile_timing() {
    constexpr int ROWS = 16, COLS = 16;
    constexpr int iters = 500;
    std::uint64_t rng = 0xBEEF;
    std::vector<std::uint8_t> packed(ROWS * 18, 0);
    for (int r = 0; r < ROWS; ++r) {
        std::uint8_t codes[16];
        for (int i = 0; i < 16; ++i) codes[i] = static_cast<std::uint8_t>(splitmix64_next(&rng) & 0x0F);
        pack_block(packed.data() + r * 18, codes, static_cast<std::uint8_t>(1 + (r % 7)));
    }
    std::vector<::half> x(COLS, __float2half(0.125f));
    std::vector<float> z(ROWS * COLS, 0.0f);

    void* dp = nullptr; void* dx = nullptr; void* dout = nullptr;
    CHECK(hip_ok(hipMalloc(&dp, packed.size()), "hipMalloc(w)") &&
          hip_ok(hipMalloc(&dx, sizeof(::half) * COLS), "hipMalloc(x)") &&
          hip_ok(hipMalloc(&dout, sizeof(float) * ROWS * COLS), "hipMalloc(out)"));
    CHECK(hip_ok(hipMemcpy(dp, packed.data(), packed.size(), hipMemcpyHostToDevice), "cpy w") &&
          hip_ok(hipMemcpy(dx, x.data(), sizeof(::half) * COLS, hipMemcpyHostToDevice), "cpy x") &&
          hip_ok(hipMemcpy(dout, z.data(), sizeof(float) * ROWS * COLS, hipMemcpyHostToDevice), "cpy out"));

    void* dev_scales = nullptr;
    CHECK(rocminfer::rocmfp4_wmma_dot_resolve_scales(&dev_scales) == 0 && dev_scales != nullptr);
    const std::int64_t col_group = 0;
    hipEvent_t t0, t1;
    CHECK(hip_ok(hipEventCreate(&t0), "ev0") && hip_ok(hipEventCreate(&t1), "ev1"));
    CHECK(hip_ok(hipEventRecord(t0, 0), "rec0"));
    for (int i = 0; i < iters; ++i) {
        (void)rocminfer::rocmfp4_wmma_dot(
            reinterpret_cast<float*>(dout), dp, reinterpret_cast<const ::half*>(dx),
            18, col_group, reinterpret_cast<const float*>(dev_scales), true, nullptr);
    }
    CHECK(hip_ok(hipEventRecord(t1, 0), "rec1") && hip_ok(hipEventSynchronize(t1), "sync"));
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, t0, t1);
    std::printf("  ROCmFP4+WMMA tile x%d: %.3f us each (%.3f ms total)\n",
                iters, ms / iters * 1000.0f, ms);
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);

    static_cast<void>(hipFree(dp));
    static_cast<void>(hipFree(dx));
    static_cast<void>(hipFree(dout));
    return true;
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

    std::printf("ROCmFP4+WMMA tile correctness:\n");
    bool ok = test_tile_correctness();

    std::printf("\nROCmFP4+WMMA tile timing:\n");
    ok = test_tile_timing() && ok;

    std::printf("\n%s\n", ok && g_failures == 0
                             ? "PASSED: ROCmFP4+WMMA tile verified on real GPU"
                             : "FAILED");
    return (ok && g_failures == 0) ? 0 : 1;
}
