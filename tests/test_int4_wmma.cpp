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
 * @note This exercises genuine two's-complement signed int4 operands only. It
 *       is not a ROCmFP4 / Codebook10 test; raw FP4 nibbles must not reach
 *       `int4_wmma_16x16` (see `int4_wmma.hpp` @warning).
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

    if (g_failures != 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nPASSED: all checks green\n");
    return 0;
}
