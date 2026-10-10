/**
 * @file test_int4_matvec.cpp
 * @brief Correctness tests for the integer 4-bit broadcast matvec.
 *
 * Self-contained: no test framework, no fixtures. The kernel computes a pure
 * integer matrix-vector product (`y[r] = sum_k s4(W[r,k]) * s4(x[k])` with a
 * single packed activation vector `x` shared by all rows), so the CPU oracle
 * is an exact int64 accumulation and the comparison is bit-exact — there is
 * no floating-point summation order to account for. Exit code is 0 when every
 * check passes and 1 otherwise.
 *
 * Coverage: randomised broadcast oracle over shapes {(1,8),(7,64),(64,256)},
 * plus EINVAL negatives (nulls, non-positive extents, `cols % 8 != 0`).
 *
 * @note This exercises genuine two's-complement signed int4 operands only. It
 *       is not a ROCmFP4 / Codebook10 test; raw FP4 nibbles must not reach
 *       `int4_matvec_i32` (see `int4_matvec.hpp` @warning).
 * @note Device output always lives in a `hipMalloc`'d buffer and is copied
 *       back with `hipMemcpy` DeviceToHost; a host pointer is never passed as
 *       the kernel's device `out` (cf. the RUN-006 defect in
 *       `tests/test_int4_dot.cpp`).
 */

#include <rocminfer/int4_matvec.hpp>
#include <rocminfer/profiler.hpp>

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (cond) {
        return;
    }
    ++g_failures;
    std::printf("FAIL line %d: %s\n", line, expr);
}

#define CHECK(expr) check((expr), #expr, __LINE__)

struct Shape {
    std::int64_t rows;  ///< Output rows (== grid width).
    std::int64_t cols;  ///< Weights per row; a multiple of 8.
};

constexpr Shape kShapes[] = {
    {1, 8}, {7, 64}, {64, 256},
};

class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    std::int8_t int4() { return static_cast<std::int8_t>((next() % 16) - 8); }
private:
    std::uint64_t state_;
};

std::int8_t sext4(std::uint8_t nibble) noexcept {
    const int v = static_cast<int>(nibble & 0x0Fu);
    return static_cast<std::int8_t>(v < 8 ? v : v - 16);
}

std::int32_t oracle_row(const std::int8_t* w, const std::int8_t* x, std::int64_t cols) {
    std::int64_t sum = 0;
    for (std::int64_t k = 0; k < cols; ++k) {
        sum += static_cast<std::int64_t>(w[static_cast<std::size_t>(k)]) *
               static_cast<std::int64_t>(x[static_cast<std::size_t>(k)]);
    }
    return static_cast<std::int32_t>(sum);
}

void pack_row(std::uint8_t* out_bytes, const std::int8_t* values, std::int64_t n) {
    for (std::int64_t k = 0; k < n; ++k) {
        const std::uint8_t nibble = static_cast<std::uint8_t>(values[static_cast<std::size_t>(k)] & 0x0Fu);
        if (k % 2 == 0) {
            out_bytes[static_cast<std::size_t>(k / 2)] |= nibble;
        } else {
            out_bytes[static_cast<std::size_t>(k / 2)] |= (nibble << 4);
        }
    }
}

void unpack_row(std::int8_t* out_vals, const std::uint8_t* bytes, std::int64_t n) {
    for (std::int64_t k = 0; k < n; ++k) {
        const std::uint8_t byte = bytes[static_cast<std::size_t>(k / 2)];
        const std::uint8_t nibble = (k % 2 == 0) ? (byte & 0x0Fu) : (byte >> 4);
        out_vals[static_cast<std::size_t>(k)] = sext4(nibble);
    }
}

void test_random_broadcast() {
    for (const Shape& shape : kShapes) {
        const std::size_t row_bytes = static_cast<std::size_t>(shape.cols) / 2u;
        const std::size_t w_bytes = static_cast<std::size_t>(shape.rows) * row_bytes;

        std::vector<std::uint8_t> W(w_bytes, 0u);
        std::vector<std::uint8_t> x(row_bytes, 0u);
        std::vector<std::int32_t> golden(static_cast<std::size_t>(shape.rows), 0);

        Rng rng(0x4A7CuLL + static_cast<std::uint64_t>(shape.rows) * 131u +
                static_cast<std::uint64_t>(shape.cols));
        std::vector<std::int8_t> vw(static_cast<std::size_t>(shape.cols));
        std::vector<std::int8_t> vx(static_cast<std::size_t>(shape.cols));
        std::vector<std::int8_t> uw(static_cast<std::size_t>(shape.cols));
        std::vector<std::int8_t> ux(static_cast<std::size_t>(shape.cols));

        for (std::int64_t k = 0; k < shape.cols; ++k) {
            vx[static_cast<std::size_t>(k)] = rng.int4();
        }
        std::memset(x.data(), 0, row_bytes);
        pack_row(x.data(), vx.data(), shape.cols);
        unpack_row(ux.data(), x.data(), shape.cols);
        CHECK(std::memcmp(vx.data(), ux.data(), static_cast<std::size_t>(shape.cols)) == 0);

        for (std::int64_t r = 0; r < shape.rows; ++r) {
            for (std::int64_t k = 0; k < shape.cols; ++k) {
                vw[static_cast<std::size_t>(k)] = rng.int4();
            }
            std::uint8_t* row = W.data() + static_cast<std::size_t>(r) * row_bytes;
            std::memset(row, 0, row_bytes);
            pack_row(row, vw.data(), shape.cols);

            unpack_row(uw.data(), row, shape.cols);
            golden[static_cast<std::size_t>(r)] = oracle_row(uw.data(), ux.data(), shape.cols);
            CHECK(oracle_row(vw.data(), vx.data(), shape.cols) == golden[static_cast<std::size_t>(r)]);
        }

        void* dW = nullptr;
        void* dx = nullptr;
        void* dy = nullptr;
        if (hipMalloc(&dW, W.size()) != hipSuccess ||
            hipMalloc(&dx, x.size()) != hipSuccess ||
            hipMalloc(&dy, sizeof(std::int32_t) * static_cast<std::size_t>(shape.rows)) != hipSuccess) {
            std::printf("FAIL int4_matvec random rows=%lld cols=%lld: hipMalloc\n",
                        static_cast<long long>(shape.rows), static_cast<long long>(shape.cols));
            (void)hipFree(dW);
            (void)hipFree(dx);
            (void)hipFree(dy);
            ++g_failures;
            continue;
        }
        if (hipMemcpy(dW, W.data(), W.size(), hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice) != hipSuccess) {
            ++g_failures;
            (void)hipFree(dW);
            (void)hipFree(dx);
            (void)hipFree(dy);
            continue;
        }

        std::vector<std::int32_t> got(static_cast<std::size_t>(shape.rows), 0xCDCDCDCD);
        const int rc = rocminfer::int4_matvec_i32(static_cast<std::int32_t*>(dy), dW, dx,
                                                 shape.rows, shape.cols, nullptr);
        CHECK(rc == 0);
        CHECK(hipMemcpy(got.data(), dy, got.size() * sizeof(std::int32_t),
                        hipMemcpyDeviceToHost) == hipSuccess);

        bool ok = (rc == 0);
        for (std::int64_t r = 0; r < shape.rows; ++r) {
            if (got[static_cast<std::size_t>(r)] != golden[static_cast<std::size_t>(r)]) {
                ok = false;
                std::printf("FAIL int4_matvec rows=%lld cols=%lld row %lld: got=%d want=%d\n",
                            static_cast<long long>(shape.rows),
                            static_cast<long long>(shape.cols), static_cast<long long>(r),
                            got[static_cast<std::size_t>(r)],
                            golden[static_cast<std::size_t>(r)]);
            }
        }
        std::printf("  random rows=%4lld cols=%5lld  %s\n", static_cast<long long>(shape.rows),
                    static_cast<long long>(shape.cols), ok ? "exact match" : "MISMATCH");

        (void)hipFree(dW);
        (void)hipFree(dx);
        (void)hipFree(dy);
        if (!ok) {
            ++g_failures;
        }
    }
}

void test_einval() {
    void* dW = nullptr;
    void* dx = nullptr;
    void* dy = nullptr;
    if (hipMalloc(&dW, 32u) != hipSuccess || hipMalloc(&dx, 32u) != hipSuccess ||
        hipMalloc(&dy, 16u) != hipSuccess) {
        (void)hipFree(dW);
        (void)hipFree(dx);
        (void)hipFree(dy);
        return;
    }
    std::int32_t* y = static_cast<std::int32_t*>(dy);
    const void* W = dW;
    const void* x = dx;

    CHECK(rocminfer::int4_matvec_i32(nullptr, W, x, 1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, nullptr, x, 1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, nullptr, 1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, x, 0, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, x, -1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, x, 1, 0, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, x, 1, -8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, x, 1, 7, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, x, 1, 9, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_i32(y, W, x, 1, 12, nullptr) == EINVAL);

    (void)hipFree(dW);
    (void)hipFree(dx);
    (void)hipFree(dy);
}

void test_traced_report() {
    const std::int64_t rows = 64;
    const std::int64_t cols = 256;
    const std::size_t row_bytes = static_cast<std::size_t>(cols) / 2u;
    const std::size_t w_bytes = static_cast<std::size_t>(rows) * row_bytes;

    void* dW = nullptr;
    void* dx = nullptr;
    void* dy = nullptr;
    if (hipMalloc(&dW, w_bytes) != hipSuccess ||
        hipMalloc(&dx, row_bytes) != hipSuccess ||
        hipMalloc(&dy, sizeof(std::int32_t) * static_cast<std::size_t>(rows)) != hipSuccess) {
        (void)hipFree(dW);
        (void)hipFree(dx);
        (void)hipFree(dy);
        return;
    }

    rocminfer::TraceConfig cfg;
    cfg.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(cfg);
    recorder.measure_floor();

    const int launches = 64;
    for (int i = 0; i < launches; ++i) {
        const int rc = rocminfer::int4_matvec_i32_traced(
            recorder, static_cast<std::int32_t*>(dy), dW, dx, rows, cols, nullptr);
        if (rc != 0) {
            std::printf("FAIL line %d: traced launch returned %d\n", __LINE__, rc);
            ++g_failures;
            (void)hipFree(dW);
            (void)hipFree(dx);
            (void)hipFree(dy);
            return;
        }
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL line %d: hipDeviceSynchronize\n", __LINE__);
        ++g_failures;
        (void)hipFree(dW);
        (void)hipFree(dx);
        (void)hipFree(dy);
        return;
    }

    const bool resolved = recorder.resolve_pending();
    std::printf("int4-matvec traced path, %d launches (resolve_ok=%d, dropped=%llu)\n",
                launches, resolved ? 1 : 0,
                static_cast<unsigned long long>(recorder.dropped_pending_scopes()));
    std::printf("  (drops are the documented gfx1201 non-monotonic GPU clock case in\n"
                "   docs/PERF_TRACE.md; a non-zero count means the trace is partial)\n");
    if (!recorder.report(std::cout)) {
        std::printf("FAIL line %d: report failed\n", __LINE__);
        ++g_failures;
    }
    std::printf("  NOTE: this shape launches in single-digit us, so its per-launch dev_ns sits\n"
                "  on the reported device floor. That figure is launch-gap dominated and is\n"
                "  NOT the kernel's cost; quote bench_matmul's amortised timing instead.\n");

    (void)hipFree(dW);
    (void)hipFree(dx);
    (void)hipFree(dy);
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("FAIL: no HIP device available\n");
        return 1;
    }

    std::printf("int4 broadcast matvec: packed W rows x shared x vs int4_matvec_i32\n");
    test_random_broadcast();
    test_einval();
    test_traced_report();

    if (g_failures != 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nPASSED: all checks green\n");
    return 0;
}
