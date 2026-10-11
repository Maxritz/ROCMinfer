/**
 * @file test_int4_wmma.cpp
 * @brief Correctness tests for the gfx1201 int4 x int4 WMMA tile.
 *
 * Self-contained: no test framework, no fixtures. The tile computes an exact
 * integer matmul (signed int4 x signed int4 -> int32), so the CPU oracle is an
 * exact int64 accumulation and the comparison is bit-exact --- there is no
 * floating-point summation order to account for. Exit code is 0 when every
 * check passes and 1 otherwise.
 *
 * ## What is under test
 *
 * Randomized 16x16 tiles packed with the engine int4 nibble order (element k
 * in the low nibble of byte k/2 for even k, high nibble otherwise), run
 * through `rocminfer::int4_wmma_16x16` and compared elementwise against the
 * exact integer oracle, plus EINVAL negatives for null pointers and a traced
 * launch parity check.
 *
 * The broadcast matvec `rocminfer::int4_matvec_iu4_i32` is covered the same
 * way: randomised `rows` x `cols` shapes (both extents multiples of 16, the
 * rule the launcher enforces) compared bit-exact against an int64 CPU oracle
 * computed from the *unpacked device-bound bytes*, plus EINVAL negatives for
 * every argument the boundary rejects.
 *
 * @note This exercises genuine two's-complement signed int4 operands only. It
 *       is not a ROCmFP4 / Codebook10 test; raw FP4 nibbles must not reach
 *       `int4_wmma_16x16` or `int4_matvec_iu4_i32` (see `int4_wmma.hpp`
 *       @warning).
 */

#include <rocminfer/int4_wmma.hpp>
#include <rocminfer/profiler.hpp>

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

/// Engine int4 packing: element k of a row sits in nibble k of the row bytes.
constexpr int kTile = 16;
constexpr std::size_t kTileBytes = 128u;
constexpr std::size_t kOutBytes = 1024u;

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

void pack_tile(std::uint8_t* out_bytes, const std::int8_t* values) {
    std::memset(out_bytes, 0, kTileBytes);
    for (int m = 0; m < kTile; ++m) {
        for (int k = 0; k < kTile; ++k) {
            const std::int8_t v = values[m * kTile + k];
            std::uint8_t* byte = out_bytes + static_cast<std::size_t>(m * 8 + k / 2);
            if (k % 2 == 0) {
                *byte |= static_cast<std::uint8_t>(v & 0x0F);
            } else {
                *byte |= static_cast<std::uint8_t>((v & 0x0F) << 4);
            }
        }
    }
}

void oracle_gemm(std::int32_t* out, const std::int8_t* a, const std::int8_t* b) {
    for (int m = 0; m < kTile; ++m) {
        for (int n = 0; n < kTile; ++n) {
            std::int64_t sum = 0;
            for (int k = 0; k < kTile; ++k) {
                sum += static_cast<std::int64_t>(a[m * kTile + k]) *
                       static_cast<std::int64_t>(b[k * kTile + n]);
            }
            out[m * kTile + n] = static_cast<std::int32_t>(sum);
        }
    }
}

void unpack_tile(std::int8_t* out_vals, const std::uint8_t* bytes) {
    for (int m = 0; m < kTile; ++m) {
        for (int k = 0; k < kTile; ++k) {
            const std::uint8_t byte = bytes[static_cast<std::size_t>(m * 8 + k / 2)];
            const std::uint8_t nibble =
                (k % 2 == 0) ? static_cast<std::uint8_t>(byte & 0x0F)
                             : static_cast<std::uint8_t>(byte >> 4);
            out_vals[m * kTile + k] = sext4(nibble);
        }
    }
}

bool hip_ok(hipError_t s, const char* what) {
    if (s == hipSuccess) {
        return true;
    }
    ++g_failures;
    std::printf("FAIL HIP %s: %s\n", what, hipGetErrorString(s));
    return false;
}

void test_random_tiles() {
    constexpr int kRounds = 8;
    int exact_rounds = 0;
    for (int round = 0; round < kRounds; ++round) {
        Rng rng(0x1AA41204ull + static_cast<std::uint64_t>(round) * 7919u);
        std::vector<std::int8_t> va(kTile * kTile);
        std::vector<std::int8_t> vb(kTile * kTile);
        for (int i = 0; i < kTile * kTile; ++i) {
            va[static_cast<std::size_t>(i)] = rng.int4();
            vb[static_cast<std::size_t>(i)] = rng.int4();
        }
        std::vector<std::uint8_t> a(kTileBytes, 0u);
        std::vector<std::uint8_t> b(kTileBytes, 0u);
        pack_tile(a.data(), va.data());
        pack_tile(b.data(), vb.data());

        std::vector<std::int8_t> ua(kTile * kTile);
        std::vector<std::int8_t> ub(kTile * kTile);
        unpack_tile(ua.data(), a.data());
        unpack_tile(ub.data(), b.data());
        std::vector<std::int32_t> golden(kTile * kTile, 0);
        oracle_gemm(golden.data(), ua.data(), ub.data());

        void* da = nullptr;
        void* db = nullptr;
        void* dout = nullptr;
        if (hipMalloc(&da, kTileBytes) != hipSuccess ||
            hipMalloc(&db, kTileBytes) != hipSuccess ||
            hipMalloc(&dout, kOutBytes) != hipSuccess) {
            std::printf("FAIL int4_wmma round %d: hipMalloc\n", round);
            (void)hipFree(da);
            (void)hipFree(db);
            (void)hipFree(dout);
            ++g_failures;
            continue;
        }
        if (hipMemcpy(da, a.data(), kTileBytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(db, b.data(), kTileBytes, hipMemcpyHostToDevice) != hipSuccess) {
            ++g_failures;
            (void)hipFree(da);
            (void)hipFree(db);
            (void)hipFree(dout);
            continue;
        }

        std::vector<std::int32_t> got(kTile * kTile, 0xCDCDCDCD);
        const int rc = rocminfer::int4_wmma_16x16(static_cast<std::int32_t*>(dout),
                                                 da, db, nullptr);
        CHECK(rc == 0);
        CHECK(hip_ok(hipDeviceSynchronize(), "sync"));
        CHECK(hip_ok(hipMemcpy(got.data(), dout, kOutBytes, hipMemcpyDeviceToHost),
                     "cpy out"));

        bool ok = (rc == 0);
        int bad = 0;
        for (int i = 0; i < kTile * kTile; ++i) {
            if (got[static_cast<std::size_t>(i)] != golden[static_cast<std::size_t>(i)]) {
                if (bad < 4) {
                    std::printf("FAIL int4_wmma round %d out[%d]: got=%d want=%d\n",
                                round, i, got[static_cast<std::size_t>(i)],
                                golden[static_cast<std::size_t>(i)]);
                }
                ++bad;
                ok = false;
            }
        }
        std::printf("  random tile %d  %s (%d/256 bad)\n", round,
                    ok ? "exact match" : "MISMATCH", bad);
        if (ok) {
            ++exact_rounds;
        } else {
            ++g_failures;
        }

        (void)hipFree(da);
        (void)hipFree(db);
        (void)hipFree(dout);
    }
    std::printf("  exact rounds: %d/%d\n", exact_rounds, kRounds);
}

void test_einval() {
    void* da = nullptr;
    void* db = nullptr;
    void* dout = nullptr;
    if (hipMalloc(&da, kTileBytes) != hipSuccess ||
        hipMalloc(&db, kTileBytes) != hipSuccess ||
        hipMalloc(&dout, kOutBytes) != hipSuccess) {
        (void)hipFree(da);
        (void)hipFree(db);
        (void)hipFree(dout);
        return;
    }
    std::int32_t* out = static_cast<std::int32_t*>(dout);

    CHECK(rocminfer::int4_wmma_16x16(nullptr, da, db, nullptr) == EINVAL);
    CHECK(rocminfer::int4_wmma_16x16(out, nullptr, db, nullptr) == EINVAL);
    CHECK(rocminfer::int4_wmma_16x16(out, da, nullptr, nullptr) == EINVAL);

    rocminfer::TraceRecorder recorder;
    CHECK(rocminfer::int4_wmma_16x16_traced(recorder, nullptr, da, db, nullptr) ==
          EINVAL);

    (void)hipFree(da);
    (void)hipFree(db);
    (void)hipFree(dout);
}

void test_traced_report() {
    void* da = nullptr;
    void* db = nullptr;
    void* dout = nullptr;
    if (hipMalloc(&da, kTileBytes) != hipSuccess || hipMalloc(&db, kTileBytes) != hipSuccess ||
        hipMalloc(&dout, kOutBytes) != hipSuccess) {
        (void)hipFree(da);
        (void)hipFree(db);
        (void)hipFree(dout);
        return;
    }

    rocminfer::TraceConfig cfg;
    cfg.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(cfg);
    recorder.measure_floor();

    const int launches = 64;
    for (int i = 0; i < launches; ++i) {
        const int rc = rocminfer::int4_wmma_16x16_traced(
            recorder, static_cast<std::int32_t*>(dout), da, db, nullptr);
        if (rc != 0) {
            std::printf("FAIL line %d: traced launch returned %d\n", __LINE__, rc);
            ++g_failures;
            (void)hipFree(da);
            (void)hipFree(db);
            (void)hipFree(dout);
            return;
        }
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        std::printf("FAIL line %d: hipDeviceSynchronize\n", __LINE__);
        ++g_failures;
        (void)hipFree(da);
        (void)hipFree(db);
        (void)hipFree(dout);
        return;
    }

    const bool resolved = recorder.resolve_pending();
    std::printf("int4-wmma traced path, %d launches (resolve_ok=%d, dropped=%llu)\n",
                launches, resolved ? 1 : 0,
                static_cast<unsigned long long>(recorder.dropped_pending_scopes()));
    std::printf("  (drops are the documented gfx1201 non-monotonic GPU clock case in\n"
                "   docs/PERF_TRACE.md; a non-zero count means the trace is partial)\n");
    if (!recorder.report(std::cout)) {
        std::printf("FAIL line %d: report failed\n", __LINE__);
        ++g_failures;
    }
    std::printf("  NOTE: a 16x16x16 tile launches in ~1.5 us, so its per-launch dev_ns sits\n"
                "  on the reported device floor. That figure is launch-gap dominated and is\n"
                "  NOT the kernel's cost; quote bench_matmul's amortised timing instead.\n");

    (void)hipFree(da);
    (void)hipFree(db);
    (void)hipFree(dout);
}

struct MvShape {
    std::int64_t rows;  ///< Output rows; a multiple of kInt4WmmaTile.
    std::int64_t cols;  ///< Weights per row; a multiple of kInt4WmmaTile.
};

constexpr MvShape kMvShapes[] = {
    {16, 16},  {16, 512}, {32, 32},  {32, 256}, {64, 64},
    {64, 512}, {128, 16}, {128, 128}, {128, 256}, {32, 64},
};

std::int32_t oracle_matvec_row(const std::int8_t* w, const std::int8_t* x,
                              std::int64_t cols) {
    std::int64_t sum = 0;
    for (std::int64_t k = 0; k < cols; ++k) {
        sum += static_cast<std::int64_t>(w[static_cast<std::size_t>(k)]) *
               static_cast<std::int64_t>(x[static_cast<std::size_t>(k)]);
    }
    return static_cast<std::int32_t>(sum);
}

void test_matvec_iu4_vs_oracle() {
    const std::size_t step_bytes = static_cast<std::size_t>(kTile) / 2u;
    for (const MvShape& shape : kMvShapes) {
        const std::size_t row_bytes = static_cast<std::size_t>(shape.cols) / 2u;
        const std::size_t w_bytes = static_cast<std::size_t>(shape.rows) * row_bytes;
        const std::int64_t tiles_per_row = shape.cols / kTile;
        const std::int64_t row_blocks = shape.rows / kTile;

        std::vector<std::uint8_t> W(w_bytes, 0u);
        std::vector<std::uint8_t> x(row_bytes, 0u);
        std::vector<std::int8_t> vw(static_cast<std::size_t>(shape.rows * shape.cols));
        std::vector<std::int8_t> vx(static_cast<std::size_t>(shape.cols));

        Rng rng(0x5B17C0DEull + static_cast<std::uint64_t>(shape.rows) * 257u +
                static_cast<std::uint64_t>(shape.cols) * 17u);
        for (std::int64_t k = 0; k < shape.cols; ++k) {
            vx[static_cast<std::size_t>(k)] = rng.int4();
        }
        for (std::int64_t r = 0; r < shape.rows; ++r) {
            for (std::int64_t k = 0; k < shape.cols; ++k) {
                vw[static_cast<std::size_t>(r) * static_cast<std::size_t>(shape.cols) +
                   static_cast<std::size_t>(k)] = rng.int4();
            }
        }

        std::vector<std::int8_t> tile(static_cast<std::size_t>(kTile * kTile), 0);
        std::vector<std::uint8_t> tmp(kTileBytes, 0u);
        for (std::int64_t rb = 0; rb < row_blocks; ++rb) {
            for (std::int64_t kb = 0; kb < tiles_per_row; ++kb) {
                for (int m = 0; m < kTile; ++m) {
                    for (int k = 0; k < kTile; ++k) {
                        tile[static_cast<std::size_t>(m * kTile + k)] =
                            vw[static_cast<std::size_t>((rb * kTile + m) * shape.cols +
                                                        kb * kTile + k)];
                    }
                }
                pack_tile(tmp.data(), tile.data());
                for (int m = 0; m < kTile; ++m) {
                    std::memcpy(W.data() + static_cast<std::size_t>(rb * kTile + m) *
                                                    row_bytes +
                                    static_cast<std::size_t>(kb) * step_bytes,
                                tmp.data() + static_cast<std::size_t>(m) * step_bytes,
                                step_bytes);
                }
            }
        }
        for (std::int64_t kb = 0; kb < tiles_per_row; ++kb) {
            for (int i = 0; i < kTile * kTile; ++i) {
                tile[static_cast<std::size_t>(i)] = 0;
            }
            for (int k = 0; k < kTile; ++k) {
                tile[static_cast<std::size_t>(k)] = vx[static_cast<std::size_t>(kb * kTile + k)];
            }
            for (std::uint8_t& b : tmp) {
                b = 0u;
            }
            pack_tile(tmp.data(), tile.data());
            std::memcpy(x.data() + static_cast<std::size_t>(kb) * step_bytes, tmp.data(),
                        step_bytes);
        }

        std::vector<std::int8_t> uw(static_cast<std::size_t>(shape.rows * shape.cols));
        std::vector<std::int8_t> ux(static_cast<std::size_t>(shape.cols));
        for (std::int64_t rb = 0; rb < row_blocks; ++rb) {
            for (std::int64_t kb = 0; kb < tiles_per_row; ++kb) {
                for (int m = 0; m < kTile; ++m) {
                    std::memcpy(tmp.data() + static_cast<std::size_t>(m) * step_bytes,
                                W.data() + static_cast<std::size_t>(rb * kTile + m) *
                                        row_bytes +
                                    static_cast<std::size_t>(kb) * step_bytes,
                                step_bytes);
                }
                unpack_tile(tile.data(), tmp.data());
                for (int m = 0; m < kTile; ++m) {
                    for (int k = 0; k < kTile; ++k) {
                        uw[static_cast<std::size_t>((rb * kTile + m) * shape.cols +
                                                    kb * kTile + k)] =
                            tile[static_cast<std::size_t>(m * kTile + k)];
                    }
                }
            }
        }
        for (std::int64_t kb = 0; kb < tiles_per_row; ++kb) {
            for (std::uint8_t& b : tmp) {
                b = 0u;
            }
            std::memcpy(tmp.data(), x.data() + static_cast<std::size_t>(kb) * step_bytes,
                        step_bytes);
            unpack_tile(tile.data(), tmp.data());
            for (int k = 0; k < kTile; ++k) {
                ux[static_cast<std::size_t>(kb * kTile + k)] = tile[static_cast<std::size_t>(k)];
            }
        }
        CHECK(std::memcmp(vw.data(), uw.data(), vw.size()) == 0);
        CHECK(std::memcmp(vx.data(), ux.data(), static_cast<std::size_t>(shape.cols)) == 0);

        std::vector<std::int32_t> golden(static_cast<std::size_t>(shape.rows), 0);
        for (std::int64_t r = 0; r < shape.rows; ++r) {
            golden[static_cast<std::size_t>(r)] = oracle_matvec_row(
                uw.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(shape.cols),
                ux.data(), shape.cols);
        }

        void* dW = nullptr;
        void* dx = nullptr;
        void* dy = nullptr;
        if (hipMalloc(&dW, W.size()) != hipSuccess ||
            hipMalloc(&dx, x.size()) != hipSuccess ||
            hipMalloc(&dy, sizeof(std::int32_t) * static_cast<std::size_t>(shape.rows)) !=
                hipSuccess) {
            std::printf("FAIL int4_matvec_iu4 random rows=%lld cols=%lld: hipMalloc\n",
                        static_cast<long long>(shape.rows),
                        static_cast<long long>(shape.cols));
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
        const int rc = rocminfer::int4_matvec_iu4_i32(
            static_cast<std::int32_t*>(dy), dW, dx, shape.rows, shape.cols, nullptr);
        CHECK(rc == 0);
        CHECK(hip_ok(hipMemcpy(got.data(), dy, got.size() * sizeof(std::int32_t),
                               hipMemcpyDeviceToHost),
                     "cpy matvec_iu4"));

        bool ok = (rc == 0);
        int bad = 0;
        for (std::int64_t r = 0; r < shape.rows; ++r) {
            if (got[static_cast<std::size_t>(r)] != golden[static_cast<std::size_t>(r)]) {
                if (bad < 4) {
                    std::printf("FAIL int4_matvec_iu4 rows=%lld cols=%lld row %lld: "
                                "got=%d want=%d\n",
                                static_cast<long long>(shape.rows),
                                static_cast<long long>(shape.cols),
                                static_cast<long long>(r), got[static_cast<std::size_t>(r)],
                                golden[static_cast<std::size_t>(r)]);
                }
                ++bad;
                ok = false;
            }
        }
        std::printf("  random rows=%4lld cols=%5lld  %s (%d/%d bad)\n",
                    static_cast<long long>(shape.rows),
                    static_cast<long long>(shape.cols), ok ? "exact match" : "MISMATCH",
                    bad, static_cast<int>(shape.rows));

        (void)hipFree(dW);
        (void)hipFree(dx);
        (void)hipFree(dy);
        if (!ok) {
            ++g_failures;
        }
    }
}

void test_matvec_iu4_einval() {
    const std::int64_t rows = 32;
    const std::int64_t cols = 64;
    const std::size_t row_bytes = static_cast<std::size_t>(cols) / 2u;
    const std::size_t w_bytes = static_cast<std::size_t>(rows) * row_bytes;
    const std::size_t slack = 8u;

    void* dW = nullptr;
    void* dx = nullptr;
    void* dy = nullptr;
    if (hipMalloc(&dW, w_bytes + slack) != hipSuccess ||
        hipMalloc(&dx, row_bytes + slack) != hipSuccess ||
        hipMalloc(&dy, sizeof(std::int32_t) * static_cast<std::size_t>(rows) + slack) !=
            hipSuccess) {
        (void)hipFree(dW);
        (void)hipFree(dx);
        (void)hipFree(dy);
        return;
    }
    std::int32_t* y = static_cast<std::int32_t*>(dy);
    const void* W = dW;
    const void* x = dx;
    std::uint8_t* const W_raw = static_cast<std::uint8_t*>(dW);
    std::uint8_t* const x_raw = static_cast<std::uint8_t*>(dx);
    std::uint8_t* const y_raw_base = static_cast<std::uint8_t*>(dy);

    CHECK(rocminfer::int4_matvec_iu4_i32(nullptr, W, x, rows, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, nullptr, x, rows, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, nullptr, rows, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, 0, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, -16, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, rows, 0, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, rows, -16, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, rows, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, rows, 24, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, rows, 520, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, 1, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, 17, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x, 33, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W_raw + 1, x, rows, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(y, W, x_raw + 1, rows, cols, nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32(reinterpret_cast<std::int32_t*>(y_raw_base + 1), W,
                                        x, rows, cols, nullptr) == EINVAL);

    rocminfer::TraceRecorder recorder;
    CHECK(rocminfer::int4_matvec_iu4_i32_traced(recorder, nullptr, W, x, rows, cols,
                                                nullptr) == EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32_traced(recorder, y, W, x, rows, 8, nullptr) ==
          EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32_traced(recorder, y, W, x, 17, cols, nullptr) ==
          EINVAL);
    CHECK(rocminfer::int4_matvec_iu4_i32_traced(recorder, y, W, x, rows, cols, nullptr) == 0);
    CHECK(hip_ok(hipDeviceSynchronize(), "sync"));

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
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
        std::printf("device: %s  arch: %s  warpSize: %d\n\n", props.name,
                    props.gcnArchName, props.warpSize);
    }

    std::printf("int4_wmma exact oracle vs int4_wmma_16x16\n");
    test_random_tiles();
    test_einval();
    test_traced_report();
    std::printf("\nint4 broadcast matvec: packed W rows x shared x vs "
                "int4_matvec_iu4_i32\n");
    test_matvec_iu4_vs_oracle();
    test_matvec_iu4_einval();

    if (g_failures != 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nPASSED: all checks green\n");
    return 0;
}
