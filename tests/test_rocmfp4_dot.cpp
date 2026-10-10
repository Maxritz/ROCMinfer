/**
 * @file test_rocmfp4_dot.cpp
 * @brief Correctness tests for the fused ROCmFP4 dequantize-and-dot kernel.
 *
 * Self-contained: no test framework, no fixtures, no generated data. Exit code
 * is 0 when every check passes and 1 otherwise.
 *
 * ## What is actually under test
 *
 * The kernel's job is **decode**, so this test validates decode. It builds
 * packed ROCmFP4 blocks on the host, hands them to the GPU, and compares the
 * resulting dot products against `rocminfer::rocmfp4_dequantize_row` — the
 * engine's own CPU oracle, compiled from the same repository.
 *
 * Three families of input are used:
 *
 * 1. **Random packed blocks.** Nibble codes and scale bytes are drawn
 *    uniformly, with scale bytes covering the whole valid range 0x00..0x7E so
 *    subnormal scales and the degenerate 0x7E scale are both exercised. This is
 *    the strictest decode test because it does not depend on any quantizer.
 * 2. **Quantised FP32 matrices.** A deterministic pseudo-random FP32 matrix is
 *    compressed with a simple nearest-Codebook10 quantiser, exactly as the
 *    upstream format defines, and the decoded dot products are compared against
 *    the oracle. This test validates **decode, not encode**: the quantiser here
 *    picks the largest representable UE4M3 scale that keeps the maximum weight
 *    within code 10 and then rounds each weight to the nearest Codebook10
 *    magnitude. It deliberately does **not** reproduce upstream's MSE scale
 *    search, and its accuracy is irrelevant — the comparison never uses the
 *    original FP32 values, only the CPU decode of the packed bytes.
 * 3. **Exactly representable tensors.** Scale byte 64 decodes to 1.0f and the
 *    activations are signed powers of two small enough that every product and
 *    every partial sum is exact in FP32. Every summation order must therefore
 *    produce the *bit-identical* result, so this family asserts exact equality
 *    with the oracle and catches any error in the nibble or scale decode that
 *    the tolerance-based families could hide.
 *
 * ## Tolerance
 *
 * Every FP32 summation of `N = cols` terms is provably within
 * `(N + 1) * 2^-24 * sum_j |w_j * x_j|` of the exact value: each partial sum is
 * bounded by the sum of absolute term magnitudes, and the kernel rounds at most
 * once per term plus eight times in the reduction tree. The test measures that
 * bound per row instead of assuming it, and reports the worst observed fraction
 * of it. It additionally reports the CPU oracle's own error against a
 * double-precision reference, so the kernel can be judged against the accuracy
 * of the thing it replaces rather than against an arbitrary constant.
 */

#include <rocminfer/rocmfp4_dot.hpp>

#include "rocminfer/profiler.hpp"
#include "rocminfer/rocmfp4_reference.hpp"

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

/**
 * @brief Record a check result and report the failing source location.
 * @param[in] cond Result of the checked expression.
 * @param[in] expr Source text of the checked expression.
 * @param[in] line Source line of the checked expression.
 * @return Nothing; increments the global failure count when `cond` is false.
 * @note Diagnostics go to stdout rather than stderr so a single stream shows the
 *       full narrative of a run in order.
 */
void check(bool cond, const char* expr, int line) {
    if (cond) {
        return;
    }
    ++g_failures;
    std::printf("FAIL line %d: %s\n", line, expr);
}

#define CHECK(expr) check((expr), #expr, __LINE__)

/// Unit roundoff of IEEE-754 binary32, 2^-24.
constexpr double kFpEps = 1.1920928955078125e-07;

/**
 * @brief Empirical error cap, as a fraction of `|result| + sum|terms|`.
 *
 * This is a regression guard, not a proof. The rigorous criterion is
 * `(cols + 1) * kFpEps * sum|terms|`; this cap is tighter than that worst case
 * by a factor of 2.4 at `cols = 4096` and, as measured, still roughly 2500x
 * above the error the kernel actually produces.
 */
constexpr double kEmpiricalCap = 1e-4;

/**
 * @brief Report a HIP failure and count it as a test failure.
 * @param[in] status Status returned by the HIP call.
 * @param[in] what   Call description used in the diagnostic.
 * @return `true` when `status` indicates success.
 * @post Increments the global failure count when `status` is an error.
 */
bool hip_ok(hipError_t status, const char* what) {
    if (status == hipSuccess) {
        return true;
    }
    ++g_failures;
    std::printf("FAIL HIP %s: %s\n", what, hipGetErrorString(status));
    return false;
}

/// Shape of one test matrix.
struct Shape {
    std::int64_t rows;  ///< Output rows.
    std::int64_t cols;  ///< Weights per row; a multiple of 32.
};

/// Shapes exercised by the tolerance-based families, including a large one.
constexpr Shape kShapes[] = {
    {1, 32}, {1, 4096}, {3, 32}, {7, 64}, {33, 128}, {64, 512}, {512, 4096},
};

/// Shapes used by the exact-equality family; small enough that FP32 sums are exact.
constexpr Shape kExactShapes[] = {
    {4, 256}, {7, 512}, {16, 1024}, {64, 2048},
};

/**
 * @brief Deterministic 64-bit generator producing well-distributed doubles.
 *
 * SplitMix64. Chosen over `std::mt19937` because the state is a single word, so
 * the sequence is trivially reproducible across compilers and standard
 * libraries, and a failing row can be reproduced by index alone.
 */
class Rng {
public:
    /**
     * @brief Construct a generator with a fixed seed.
     * @param[in] seed Initial state; any value is accepted.
     */
    explicit Rng(std::uint64_t seed) : state_(seed) {}

    /**
     * @brief Draw the next 64 random bits.
     * @return Pseudo-random 64-bit value.
     */
    std::uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /**
     * @brief Draw a uniform value in `[lo, hi)`.
     * @param[in] lo Inclusive lower bound.
     * @param[in] hi Exclusive upper bound; must exceed `lo`.
     * @return Value in `[lo, hi)`.
     * @pre `hi > lo`.
     * @note The bound is exclusive and the domain is far smaller than 2^53 here,
     *       so the modulo reduction introduces no measurable bias.
     */
    std::uint32_t uniform(std::uint32_t lo, std::uint32_t hi) {
        return lo + static_cast<std::uint32_t>(next() % static_cast<std::uint64_t>(hi - lo));
    }

    /**
     * @brief Draw a uniform value in `[-1, 1)`.
     * @return Value with a fixed, reproducible distribution.
     */
    float signed_unit() {
        const double unit = static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
        return static_cast<float>(2.0 * unit - 1.0);
    }

private:
    std::uint64_t state_;  ///< SplitMix64 state.
};

/**
 * @brief Host copy of the UE4M3 half-scale table, taken from the CPU oracle.
 * @param[out] table Receives 127 floats; untouched on entry.
 * @return Nothing.
 * @note Sourcing the table from `rocmfp4_scale_to_float` rather than
 *       recomputing it means the quantiser in this test cannot accidentally
 *       agree with a kernel bug by sharing a second copy of the same mistake.
 */
void build_host_scale_table(float table[127]) {
    for (std::uint32_t i = 0; i < 127u; ++i) {
        table[i] = rocminfer::rocmfp4_scale_to_float(static_cast<std::uint8_t>(i));
    }
}

/// Codebook10 magnitudes indexed by the three magnitude bits.
constexpr float kMagnitude[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f};

/**
 * @brief Choose the packed code nearest to a magnitude.
 * @param[in] magnitude Absolute weight value, non-negative.
 * @param[in] scale     Block scale; values <= 0 select code 0.
 * @return Code in 0..7, without the sign bit.
 * @note Ties resolve to the smaller magnitude, which is deterministic and
 *       irrelevant to correctness because the test only decodes what it wrote.
 */
std::uint8_t nearest_code(float magnitude, float scale) {
    if (scale <= 0.0f) {
        return 0u;
    }
    const float target = magnitude / scale;
    std::uint8_t best = 0u;
    float best_delta = std::fabs(target - kMagnitude[0]);
    for (std::uint8_t code = 1u; code < 8u; ++code) {
        const float delta = std::fabs(target - kMagnitude[code]);
        if (delta < best_delta) {
            best_delta = delta;
            best = code;
        }
    }
    return best;
}

/**
 * @brief Pack an FP32 matrix into ROCmFP4 blocks with a nearest-code quantiser.
 * @param[out] packed     Destination for the packed tensor, resized to the exact
 *                        byte length for `dual_scale`.
 * @param[in]  values     Source FP32 matrix, `rows * cols` elements, row-major.
 * @param[in]  rows       Number of rows.
 * @param[in]  cols       Weights per row; a positive multiple of 32.
 * @param[in]  dual_scale Layout selector.
 * @param[in]  table      Host UE4M3 scale table from `build_host_scale_table`.
 * @return Nothing.
 * @note Not a faithful reproduction of the upstream quantiser: no MSE scale
 *       search, no per-row outlier handling. It only has to emit valid blocks;
 *       see the file header for why that is sufficient here.
 */
void quantize_rocmfp4(std::vector<std::uint8_t>& packed, const std::vector<float>& values,
                      std::int64_t rows, std::int64_t cols, bool dual_scale,
                      const float table[127]) {
    const std::size_t block_bytes =
        dual_scale ? rocminfer::QK_ROCMFP4_BYTES : rocminfer::QK_ROCMFP4_FAST_BYTES;
    packed.assign(static_cast<std::size_t>(rows) *
                      (static_cast<std::size_t>(cols / 32) * block_bytes),
                  0u);

    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t b = 0; b < cols / 32; ++b) {
            float largest = 0.0f;
            for (std::int32_t j = 0; j < 32; ++j) {
                const float magnitude =
                    std::fabs(values[static_cast<std::size_t>(r * cols + b * 32 + j)]);
                largest = magnitude > largest ? magnitude : largest;
            }

            const float target = largest / 10.0f;
            std::uint8_t index = 0u;
            for (std::uint32_t i = 1u; i < 126u; ++i) {
                if (table[i] > 0.0f && table[i] <= target) {
                    index = static_cast<std::uint8_t>(i);
                }
            }
            const float scale = table[index];

            std::uint8_t* block =
                packed.data() + static_cast<std::size_t>(r * (cols / 32) + b) * block_bytes;
            for (std::int32_t j = 0; j < 32; ++j) {
                const float value = values[static_cast<std::size_t>(r * cols + b * 32 + j)];
                const std::uint8_t magnitude_code =
                    nearest_code(std::fabs(value), scale);
                const std::uint8_t nibble =
                    (value < 0.0f) ? static_cast<std::uint8_t>(magnitude_code | 0x08u)
                                   : magnitude_code;
                const std::uint32_t packed_byte =
                    (j < 16) ? static_cast<std::uint32_t>(nibble)
                             : (static_cast<std::uint32_t>(nibble) << 4);
                block[j % 16] = static_cast<std::uint8_t>(block[j % 16] | packed_byte);
            }

            block[16] = index;
            if (dual_scale) {
                block[17] = index;
            }
        }
    }
}

/**
 * @brief Pack a matrix with random nibble codes and random valid scale bytes.
 * @param[out] packed     Destination for the packed tensor, resized to the exact
 *                        byte length for `dual_scale`.
 * @param[in]  rows       Number of rows.
 * @param[in]  cols       Weights per row; a positive multiple of 32.
 * @param[in]  dual_scale Layout selector.
 * @param[in]  seed       Generator seed.
 * @param[in]  scale_a    Scale byte for the low half of every block.
 * @param[in]  scale_b    Scale byte for the high half of every block; ignored
 *                        when `dual_scale` is false.
 * @return Nothing.
 * @note With `scale_a == scale_b == 0x7E` this produces the degenerate tensor
 *       whose blocks decode to all zeros, which the CPU oracle accepts.
 */
void pack_random(std::vector<std::uint8_t>& packed, std::int64_t rows, std::int64_t cols,
                 bool dual_scale, std::uint64_t seed, std::uint8_t scale_a, std::uint8_t scale_b) {
    const std::size_t block_bytes =
        dual_scale ? rocminfer::QK_ROCMFP4_BYTES : rocminfer::QK_ROCMFP4_FAST_BYTES;
    packed.assign(static_cast<std::size_t>(rows) *
                      (static_cast<std::size_t>(cols / 32) * block_bytes),
                  0u);

    Rng rng(seed);
    for (std::size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<std::uint8_t>(rng.uniform(0u, 256u));
    }
    for (std::size_t b = 0; b < static_cast<std::size_t>(rows * (cols / 32)); ++b) {
        std::uint8_t* block = packed.data() + b * block_bytes;
        block[16] = scale_a;
        if (dual_scale) {
            block[17] = scale_b;
        }
    }
}

/// Largest measured deviation, worst-bound fraction, and the row that produced them.
struct ErrorStats {
    double max_abs = 0.0;        ///< max |gpu - oracle_fp32|.
    double max_rel = 0.0;        ///< max |gpu - exact_f64| / max(|exact_f64|, tiny).
    double max_bound_fraction = 0.0;  ///< max |gpu - exact_f64| / rigorous_bound.
    double max_oracle_error = 0.0;     ///< max |oracle_fp32 - exact_f64|.
    double max_empirical_fraction = 0.0;  ///< max |gpu - exact| / empirical_cap.
};

/**
 * @brief Fold one case's statistics into the whole-run aggregate.
 * @param[in,out] stats Aggregate to update in place.
 * @param[in]      fresh Statistics from the case that just completed.
 * @return Nothing.
 */
void merge(ErrorStats& stats, const ErrorStats& fresh) noexcept {
    stats.max_abs = fresh.max_abs > stats.max_abs ? fresh.max_abs : stats.max_abs;
    stats.max_rel = fresh.max_rel > stats.max_rel ? fresh.max_rel : stats.max_rel;
    stats.max_bound_fraction =
        fresh.max_bound_fraction > stats.max_bound_fraction ? fresh.max_bound_fraction
                                                             : stats.max_bound_fraction;
    stats.max_oracle_error =
        fresh.max_oracle_error > stats.max_oracle_error ? fresh.max_oracle_error
                                                        : stats.max_oracle_error;
    stats.max_empirical_fraction =
        fresh.max_empirical_fraction > stats.max_empirical_fraction
            ? fresh.max_empirical_fraction
            : stats.max_empirical_fraction;
}

/// Whole-run aggregate across every case.
ErrorStats g_overall;

/**
 * @brief Run the kernel on host-built data and compare against the CPU oracle.
 * @param[in] packed      Packed ROCmFP4 tensor on the host.
 * @param[in] x           Activations on the host, `cols` floats.
 * @param[in] rows        Number of rows.
 * @param[in] cols        Weights per row.
 * @param[in] dual_scale  Layout selector.
 * @param[in] label       Short name used in printed output.
 * @param[in] exact       When true, demand bit-identical results.
 * @param[out] stats      Receives the measured errors; untouched on failure.
 * @return Nothing.
 * @post Increments the global failure count on any HIP error, on any disagreement
 *       with the oracle, or on any row exceeding its rigorous FP32 bound.
 * @note `exact` is justified only for inputs whose products and partial sums are
 *       exactly representable; see the file header.
 */
void run_case(const std::vector<std::uint8_t>& packed, const std::vector<float>& x,
              std::int64_t rows, std::int64_t cols, bool dual_scale, const char* label,
              bool exact, ErrorStats& stats) {
    const std::size_t weight_bytes = rocminfer::rocmfp4_dot_weight_bytes(rows, cols, dual_scale);
    if (weight_bytes != packed.size()) {
        ++g_failures;
        std::printf("FAIL %s: packed %zu bytes, kernel expects %zu\n", label, packed.size(),
                    weight_bytes);
        return;
    }

    const std::size_t weight_count = static_cast<std::size_t>(rows * cols);
    std::vector<float> weights(weight_count, 0.0f);
    std::vector<float> oracle_f32(static_cast<std::size_t>(rows), 0.0f);
    std::vector<double> oracle_f64(static_cast<std::size_t>(rows), 0.0);
    std::vector<double> sum_abs(static_cast<std::size_t>(rows), 0.0);

    for (std::int64_t r = 0; r < rows; ++r) {
        const std::uint8_t* row_bytes =
            packed.data() + static_cast<std::size_t>(r * (cols / 32)) *
                               (dual_scale ? rocminfer::QK_ROCMFP4_BYTES
                                           : rocminfer::QK_ROCMFP4_FAST_BYTES);
        float* row_weights = weights.data() + static_cast<std::size_t>(r * cols);
        const int status = rocminfer::rocmfp4_dequantize_row(
            row_weights, row_bytes, rocminfer::rocmfp4_dot_row_bytes(cols, dual_scale),
            static_cast<std::size_t>(cols), dual_scale);
        if (status != 0) {
            ++g_failures;
            std::printf("FAIL %s: oracle rejected its own input (status %d)\n", label, status);
            return;
        }

        float sum32 = 0.0f;
        double sum64 = 0.0;
        double magnitude = 0.0;
        for (std::int64_t j = 0; j < cols; ++j) {
            const float term = row_weights[j] * x[static_cast<std::size_t>(j)];
            sum32 += term;
            sum64 += static_cast<double>(term);
            magnitude += std::fabs(static_cast<double>(term));
        }
        oracle_f32[static_cast<std::size_t>(r)] = sum32;
        oracle_f64[static_cast<std::size_t>(r)] = sum64;
        sum_abs[static_cast<std::size_t>(r)] = magnitude;
    }

    void* device_weights = nullptr;
    void* device_x = nullptr;
    void* device_out = nullptr;
    if (!hip_ok(hipMalloc(&device_weights, weight_bytes), "hipMalloc(weights)") ||
        !hip_ok(hipMalloc(&device_x, sizeof(float) * static_cast<std::size_t>(cols)),
                "hipMalloc(x)") ||
        !hip_ok(hipMalloc(&device_out, sizeof(float) * static_cast<std::size_t>(rows)),
                "hipMalloc(out)")) {
        (void)hipFree(device_weights);
        (void)hipFree(device_x);
        (void)hipFree(device_out);
        return;
    }

    std::vector<float> gpu(static_cast<std::size_t>(rows), 0.0f);
    bool ok = hip_ok(hipMemcpy(device_weights, packed.data(), weight_bytes,
                               hipMemcpyHostToDevice),
                     "hipMemcpy(weights H2D)") &&
              hip_ok(hipMemcpy(device_x, x.data(), sizeof(float) * static_cast<std::size_t>(cols),
                               hipMemcpyHostToDevice),
                     "hipMemcpy(x H2D)") &&
              hip_ok(hipMemset(device_out, 0xFF, sizeof(float) * static_cast<std::size_t>(rows)),
                     "hipMemset(out)") &&
              hip_ok(rocminfer::rocmfp4_dot_f32(static_cast<float*>(device_out), device_weights,
                                                static_cast<const float*>(device_x), rows, cols,
                                                dual_scale, nullptr) == 0
                         ? hipSuccess
                         : hipErrorUnknown,
                     "rocmfp4_dot_f32") &&
              hip_ok(hipMemcpy(gpu.data(), device_out, gpu.size() * sizeof(float),
                               hipMemcpyDeviceToHost),
                     "hipMemcpy(out D2H)");

    (void)hipFree(device_weights);
    (void)hipFree(device_x);
    (void)hipFree(device_out);
    if (!ok) {
        return;
    }

    const double bound_scale = static_cast<double>(cols + 1) * kFpEps;
    for (std::int64_t r = 0; r < rows; ++r) {
        const std::size_t index = static_cast<std::size_t>(r);
        const double got = static_cast<double>(gpu[index]);
        const double expected32 = static_cast<double>(oracle_f32[index]);
        const double expected64 = oracle_f64[index];

        const double abs_vs_oracle = std::fabs(got - expected32);
        const double abs_vs_exact = std::fabs(got - expected64);
        const double bound = bound_scale * sum_abs[index] + 1e-300;
        const double oracle_error = std::fabs(expected32 - expected64);
        const double rel = abs_vs_exact / std::max(std::fabs(expected64), 1e-300);

        if (abs_vs_oracle > stats.max_abs) {
            stats.max_abs = abs_vs_oracle;
        }
        if (rel > stats.max_rel) {
            stats.max_rel = rel;
        }
        if (abs_vs_exact / bound > stats.max_bound_fraction) {
            stats.max_bound_fraction = abs_vs_exact / bound;
        }
        if (oracle_error > stats.max_oracle_error) {
            stats.max_oracle_error = oracle_error;
        }

        if (exact) {
            if (gpu[index] != oracle_f32[index]) {
                ++g_failures;
                std::printf("FAIL %s row %lld: not bit-exact, gpu=%.9g oracle=%.9g\n", label,
                            static_cast<long long>(r), static_cast<double>(gpu[index]),
                            oracle_f32[index]);
                return;
            }
        } else {
            if (abs_vs_exact > bound) {
                ++g_failures;
                std::printf("FAIL %s row %lld: |err|=%.6g exceeds bound %.6g\n", label,
                            static_cast<long long>(r), abs_vs_exact, bound);
                return;
            }
            // Empirical regression guard. The rigorous bound above is a
            // worst-case proof and is orders of magnitude looser than anything
            // this kernel actually produces, so it would not notice a single
            // mis-decoded weight. The cap below is 2.4x tighter than the
            // worst case at cols=4096 and 2500x looser than the measured error,
            // which puts it far below the perturbation a single wrong weight
            // would cause and far above ordinary summation noise.
            const double empirical_cap =
                kEmpiricalCap * (std::fabs(expected64) + sum_abs[index]) + 1e-300;
            const double empirical_fraction = abs_vs_exact / empirical_cap;
            if (empirical_fraction > stats.max_empirical_fraction) {
                stats.max_empirical_fraction = empirical_fraction;
            }
            if (abs_vs_exact > empirical_cap) {
                ++g_failures;
                std::printf("FAIL %s row %lld: |err|=%.6g exceeds empirical cap %.6g\n", label,
                            static_cast<long long>(r), abs_vs_exact, empirical_cap);
                return;
            }
        }
    }

    merge(g_overall, stats);
}

/**
 * @brief Exercise the documented EINVAL rejections of the launcher.
 * @return Nothing.
 * @post Increments the global failure count if any rejection is missing.
 * @note Every case here passes real device pointers except the null ones, so a
 *       check can only fail because the guard it targets was removed.
 */
void test_einval() {
    void* device_weights = nullptr;
    void* device_x = nullptr;
    void* device_out = nullptr;
    if (!hip_ok(hipMalloc(&device_weights, 64u), "hipMalloc(weights)")) {
        return;
    }
    if (!hip_ok(hipMalloc(&device_x, 256u), "hipMalloc(x)")) {
        (void)hipFree(device_weights);
        return;
    }
    if (!hip_ok(hipMalloc(&device_out, 64u), "hipMalloc(out)")) {
        (void)hipFree(device_weights);
        (void)hipFree(device_x);
        return;
    }

    float* out = static_cast<float*>(device_out);
    float* x = static_cast<float*>(device_x);
    auto* w = static_cast<const void*>(device_weights);

    CHECK(rocminfer::rocmfp4_dot_f32(nullptr, w, x, 1, 32, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, nullptr, x, 1, 32, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, nullptr, 1, 32, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, x, 0, 32, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, x, -1, 32, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, x, 1, 0, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, x, 1, -32, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, x, 1, 33, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, x, 1, 31, false, nullptr) == EINVAL);

auto* shifted_w = static_cast<const std::uint8_t*>(device_weights) + 1u;
auto* shifted_x = static_cast<std::uint8_t*>(device_x) + 1u;
auto* shifted_out = static_cast<std::uint8_t*>(device_out) + 1u;
    CHECK(rocminfer::rocmfp4_dot_f32(reinterpret_cast<float*>(shifted_out), shifted_w, x, 1, 32,
                                     true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, shifted_w, x, 1, 32, true, nullptr) == EINVAL);
    CHECK(rocminfer::rocmfp4_dot_f32(out, w, reinterpret_cast<float*>(shifted_x), 1, 32, true,
                                     nullptr) == EINVAL);

    CHECK(rocminfer::rocmfp4_dot_weight_bytes(0, 32, true) == 0u);
    CHECK(rocminfer::rocmfp4_dot_weight_bytes(1, 33, true) == 0u);
    CHECK(rocminfer::rocmfp4_dot_row_bytes(0, true) == 0u);
    CHECK(rocminfer::rocmfp4_dot_row_bytes(64, true) == 2u * rocminfer::QK_ROCMFP4_BYTES);
    CHECK(rocminfer::rocmfp4_dot_row_bytes(64, false) == 2u * rocminfer::QK_ROCMFP4_FAST_BYTES);

    (void)hipFree(device_weights);
    (void)hipFree(device_x);
    (void)hipFree(device_out);
}

/**
 * @brief Confirm the launcher honours the raw `void*` stream contract.
 * @return Nothing.
 * @post Increments the global failure count if the non-default stream path is
 *       not correct, or if the traced wrapper disagrees with the plain one.
 */
void test_stream_and_traced() {
    constexpr std::int64_t rows = 8;
    constexpr std::int64_t cols = 128;
    constexpr bool dual_scale = true;
    const std::size_t block_bytes =
        dual_scale ? rocminfer::QK_ROCMFP4_BYTES : rocminfer::QK_ROCMFP4_FAST_BYTES;
    const std::size_t weight_bytes = rocminfer::rocmfp4_dot_weight_bytes(rows, cols, dual_scale);

    Rng rng(0x5EED1234u);
    std::vector<std::uint8_t> packed(weight_bytes, 0u);
    for (std::size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<std::uint8_t>(rng.uniform(0u, 256u));
    }
    for (std::size_t b = 0; b < static_cast<std::size_t>(rows * (cols / 32)); ++b) {
        std::uint8_t* block = packed.data() + b * block_bytes;
        block[16] = static_cast<std::uint8_t>(rng.uniform(0u, 127u));
        block[17] = static_cast<std::uint8_t>(rng.uniform(0u, 127u));
    }

    std::vector<float> x(static_cast<std::size_t>(cols));
    for (std::int64_t j = 0; j < cols; ++j) {
        x[static_cast<std::size_t>(j)] = rng.signed_unit();
    }

    void* device_weights = nullptr;
    void* device_x = nullptr;
    void* device_out = nullptr;
    hipStream_t stream = nullptr;
    if (!hip_ok(hipMalloc(&device_weights, weight_bytes), "hipMalloc(weights)") ||
        !hip_ok(hipMalloc(&device_x, x.size() * sizeof(float)), "hipMalloc(x)") ||
        !hip_ok(hipMalloc(&device_out, sizeof(float) * static_cast<std::size_t>(rows)),
                "hipMalloc(out)") ||
        !hip_ok(hipStreamCreate(&stream), "hipStreamCreate")) {
        (void)hipFree(device_weights);
        (void)hipFree(device_x);
        (void)hipFree(device_out);
        return;
    }

    if (!hip_ok(hipMemcpy(device_weights, packed.data(), weight_bytes, hipMemcpyHostToDevice),
                "hipMemcpy(weights H2D)") ||
        !hip_ok(hipMemcpy(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy(x H2D)")) {
        (void)hipFree(device_weights);
        (void)hipFree(device_x);
        (void)hipFree(device_out);
        (void)hipStreamDestroy(stream);
        return;
    }

    rocminfer::TraceRecorder recorder;
    CHECK(rocminfer::rocmfp4_dot_f32(static_cast<float*>(device_out), device_weights,
                                     static_cast<const float*>(device_x), rows, cols, dual_scale,
                                     stream) == 0);
    CHECK(hip_ok(hipStreamSynchronize(stream), "hipStreamSynchronize"));

    std::vector<float> direct(static_cast<std::size_t>(rows), 0.0f);
    std::vector<float> traced(static_cast<std::size_t>(rows), 0.0f);
    CHECK(hip_ok(hipMemcpy(direct.data(), device_out,
                           direct.size() * sizeof(float), hipMemcpyDeviceToHost),
                 "hipMemcpy(direct D2H)"));

    CHECK(rocminfer::rocmfp4_dot_f32_traced(recorder, static_cast<float*>(device_out),
                                            device_weights,
                                            static_cast<const float*>(device_x), rows, cols,
                                            dual_scale, stream) == 0);
    CHECK(hip_ok(hipStreamSynchronize(stream), "hipStreamSynchronize(traced)"));
    CHECK(hip_ok(hipMemcpy(traced.data(), device_out,
                           traced.size() * sizeof(float), hipMemcpyDeviceToHost),
                 "hipMemcpy(traced D2H)"));

    for (std::size_t r = 0; r < direct.size(); ++r) {
        CHECK(direct[r] == traced[r]);
    }

    ErrorStats stats;
    run_case(packed, x, rows, cols, dual_scale, "stream", false, stats);

    (void)hipFree(device_weights);
    (void)hipFree(device_x);
    (void)hipFree(device_out);
    (void)hipStreamDestroy(stream);
}

/**
 * @brief Verify the two layouts agree with the oracle on random packed blocks.
 * @return Nothing.
 * @post Increments the global failure count on any disagreement.
 */
void test_random_packed(const float table[127]) {
    if (table[64] != 1.0f) {
        ++g_failures;
        std::printf("FAIL host scale table disagrees with the oracle at index 64\n");
        return;
    }

    for (const bool dual_scale : {true, false}) {
        for (const Shape& shape : kShapes) {
            Rng rng(0xA5A5ull + static_cast<std::uint64_t>(shape.rows) * 131u +
                    static_cast<std::uint64_t>(shape.cols) + (dual_scale ? 7u : 3u));
            const std::size_t block_bytes =
                dual_scale ? rocminfer::QK_ROCMFP4_BYTES : rocminfer::QK_ROCMFP4_FAST_BYTES;
            std::vector<std::uint8_t> packed(
                rocminfer::rocmfp4_dot_weight_bytes(shape.rows, shape.cols, dual_scale), 0u);
            for (std::size_t i = 0; i < packed.size(); ++i) {
                packed[i] = static_cast<std::uint8_t>(rng.uniform(0u, 256u));
            }
            for (std::size_t b = 0; b < static_cast<std::size_t>(shape.rows * (shape.cols / 32));
                 ++b) {
                std::uint8_t* block = packed.data() + b * block_bytes;
                block[16] = static_cast<std::uint8_t>(rng.uniform(0u, 127u));
                if (dual_scale) {
                    block[17] = static_cast<std::uint8_t>(rng.uniform(0u, 127u));
                }
            }

            std::vector<float> x(static_cast<std::size_t>(shape.cols));
            for (std::int64_t j = 0; j < shape.cols; ++j) {
                x[static_cast<std::size_t>(j)] = rng.signed_unit();
            }

            ErrorStats stats;
            const char* label = dual_scale ? "random-dual" : "random-fast";
            run_case(packed, x, shape.rows, shape.cols, dual_scale, label, false, stats);
            std::printf(
                "  %-12s rows=%4lld cols=%5lld  max_abs=%-12.6g max_rel=%-12.6g "
                "bound_used=%-9.3g oracle_err=%-12.6g\n",
                label, static_cast<long long>(shape.rows), static_cast<long long>(shape.cols),
                stats.max_abs, stats.max_rel, stats.max_bound_fraction, stats.max_oracle_error);
        }
    }
}

/**
 * @brief Verify the realistic path: an FP32 matrix compressed, then decoded.
 * @param[in] table Host UE4M3 scale table.
 * @return Nothing.
 * @post Increments the global failure count on any disagreement.
 */
void test_quantised(const float table[127]) {
    for (const bool dual_scale : {true, false}) {
        for (const Shape& shape : kShapes) {
            Rng rng(0x13579BDFull + static_cast<std::uint64_t>(shape.rows) * 17u +
                    static_cast<std::uint64_t>(shape.cols) + (dual_scale ? 11u : 5u));
            std::vector<float> values(static_cast<std::size_t>(shape.rows * shape.cols));
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = rng.signed_unit();
            }

            std::vector<float> x(static_cast<std::size_t>(shape.cols));
            for (std::int64_t j = 0; j < shape.cols; ++j) {
                x[static_cast<std::size_t>(j)] = rng.signed_unit();
            }

            std::vector<std::uint8_t> packed;
            quantize_rocmfp4(packed, values, shape.rows, shape.cols, dual_scale, table);

            ErrorStats stats;
            const char* label = dual_scale ? "quant-dual" : "quant-fast";
            run_case(packed, x, shape.rows, shape.cols, dual_scale, label, false, stats);
            std::printf(
                "  %-12s rows=%4lld cols=%5lld  max_abs=%-12.6g max_rel=%-12.6g "
                "bound_used=%-9.3g oracle_err=%-12.6g\n",
                label, static_cast<long long>(shape.rows), static_cast<long long>(shape.cols),
                stats.max_abs, stats.max_rel, stats.max_bound_fraction, stats.max_oracle_error);
        }
    }
}

/**
 * @brief Verify bit-exact agreement on exactly representable inputs.
 * @return Nothing.
 * @post Increments the global failure count on any disagreement.
 * @note Scale byte 64 decodes to 1.0f, so every weight is an integer in
 *       -10..10; the activations are signed powers of two down to 2^-7, so every
 *       product is exact and every partial sum of at most 2048 terms stays below
 *       2^22 in magnitude and therefore fits a binary32 significand. Every
 *       summation order must return the identical value.
 */
void test_bit_exact() {
    constexpr std::uint8_t kUnitScale = 64u;

    for (const bool dual_scale : {true, false}) {
        for (const Shape& shape : kExactShapes) {
            std::vector<std::uint8_t> packed;
            pack_random(packed, shape.rows, shape.cols, dual_scale,
                        0xC0FFEEull + static_cast<std::uint64_t>(shape.cols), kUnitScale,
                        kUnitScale);

            std::vector<float> x(static_cast<std::size_t>(shape.cols));
            for (std::int64_t j = 0; j < shape.cols; ++j) {
                const float magnitude = std::ldexp(1.0f, -static_cast<int>(j % 8));
                x[static_cast<std::size_t>(j)] = ((j % 3) == 0) ? -magnitude : magnitude;
            }

            ErrorStats stats;
            const char* label = dual_scale ? "exact-dual" : "exact-fast";
            run_case(packed, x, shape.rows, shape.cols, dual_scale, label, true, stats);
            std::printf("  %-12s rows=%4lld cols=%5lld  max_abs=%-12.6g (bit-exact)\n", label,
                        static_cast<long long>(shape.rows), static_cast<long long>(shape.cols),
                        stats.max_abs);
        }
    }
}

/**
 * @brief Verify the degenerate 0x7E scale zeroes a block on both layouts.
 * @return Nothing.
 * @post Increments the global failure count if any dot product is non-zero.
 */
void test_degenerate_scale() {
    constexpr std::uint8_t kDegenerate = rocminfer::ROCMFP4_SCALE_MAX;

    for (const bool dual_scale : {true, false}) {
        constexpr std::int64_t rows = 4;
        constexpr std::int64_t cols = 256;
        std::vector<std::uint8_t> packed;
        pack_random(packed, rows, cols, dual_scale, 0xFEEDull, kDegenerate, kDegenerate);

        std::vector<float> x(static_cast<std::size_t>(cols), 1.0f);
        ErrorStats stats;
        run_case(packed, x, rows, cols, dual_scale, "degenerate", true, stats);
        std::printf("  %-12s all blocks scaled by 0x7E decode to exactly zero\n",
                    dual_scale ? "degen-dual" : "degen-fast");
    }
}

}  // namespace

int main() {
    int device_count = 0;
    if (!hip_ok(hipGetDeviceCount(&device_count), "hipGetDeviceCount") || device_count < 1) {
        std::printf("FAIL: no HIP device available\n");
        return 1;
    }

    hipDeviceProp_t properties{};
    if (hip_ok(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties")) {
        std::printf("device: %s  arch: %s  warpSize: %d\n", properties.name,
                    properties.gcnArchName, properties.warpSize);
    }

    float table[127];
    build_host_scale_table(table);
    CHECK(table[0] == 0.0f);
    CHECK(table[64] == 1.0f);
    CHECK(table[126] == 0.0f);
    CHECK(rocminfer::rocmfp4_scale_to_float(0x7Fu) == 0.0f);
    CHECK(rocminfer::rocmfp4_scale_to_float(0x80u) == 0.0f);

    std::printf("\nrandom packed blocks\n");
    test_random_packed(table);

    std::printf("\nquantised FP32 matrices (decode is under test, not encode)\n");
    test_quantised(table);

    std::printf("\nbit-exact family (scale 1.0f, dyadic activations)\n");
    test_bit_exact();

    std::printf("\ndegenerate 0x7E scale\n");
    test_degenerate_scale();

    std::printf("\nargument rejection\n");
    test_einval();

    std::printf("\nnon-default stream and traced wrapper\n");
    test_stream_and_traced();

    std::printf("\ntolerance: |gpu - exact| <= (cols+1) * 2^-24 * sum|w_j * x_j|, measured per row\n");
    std::printf("summary over all cases:\n");
    std::printf("  max absolute error vs CPU oracle : %.6g\n", g_overall.max_abs);
    std::printf("  max relative error vs FP64 exact : %.6g\n", g_overall.max_rel);
    std::printf("  max fraction of rigorous bound   : %.6g\n", g_overall.max_bound_fraction);
    std::printf("  max fraction of empirical cap    : %.6g  (limit 1)\n",
                g_overall.max_empirical_fraction);
    std::printf("  max error of the CPU oracle      : %.6g\n", g_overall.max_oracle_error);

    if (g_failures != 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nPASSED: all checks green\n");
    return 0;
}