/**
 * @file test_int4_dot.cpp
 * @brief Correctness tests for the integer 4-bit dot primitive.
 *
 * Self-contained: no test framework, no fixtures. The kernel computes a pure
 * integer dot product (signed int4 x signed int4 -> int32), so the CPU oracle is
 * an exact int64 accumulation and the comparison is bit-exact — there is no
 * floating-point summation order to account for. Exit code is 0 when every check
 * passes and 1 otherwise.
 *
 * Per the ROCMinfer AGENTS.md, the kernel in `int4_dot.hip` is emit-verified
 * against `v_dot8_i32_i4` (see `docs/AMD_CAPABILITY_MATRIX.md` section 1). This
 * test exercises the host-side packing contract and the oracle; the device path
 * is validated by compile + ISA emission in CI on each target arch.
 *
 * ## What is under test
 *
 * The nibble packing order (weight k in the low nibble of byte k/2 when k is
 * even, high nibble otherwise), the strided group loop, and the tree
 * reduction, all pinned against an exact int64 reference.
 *
 * @note This exercises genuine two's-complement signed int4 operands only. It is
 *       not a ROCmFP4 / Codebook10 test; raw FP4 nibbles must not reach
 *       `int4_dot_i32` (see `int4_dot.hpp` @warning).
 */

#include <rocminfer/int4_dot.hpp>

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    std::int64_t count;  ///< Independent dot products.
    std::int64_t n;      ///< Weights per dot product; a multiple of 8.
};

constexpr Shape kShapes[] = {
    {1, 8}, {7, 4096}, {64, 256}, {512, 64}, {3, 1024}, {128, 104},
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

std::int32_t oracle_dot(const std::int8_t* a, const std::int8_t* b, std::int64_t n) {
    std::int64_t sum = 0;
    for (std::int64_t k = 0; k < n; ++k) {
        sum += static_cast<std::int64_t>(a[static_cast<std::size_t>(k)]) *
               static_cast<std::int64_t>(b[static_cast<std::size_t>(k)]);
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

void test_random_packed() {
    for (const Shape& shape : kShapes) {
        const std::size_t row_bytes = static_cast<std::size_t>(shape.n) / 2u;
        const std::size_t total = static_cast<std::size_t>(shape.count) * row_bytes;

        std::vector<std::uint8_t> a(total, 0u);
        std::vector<std::uint8_t> b(total, 0u);
        std::vector<std::int32_t> golden(static_cast<std::size_t>(shape.count), 0);

        Rng rng(0x1A47ull + static_cast<std::uint64_t>(shape.count) * 13u +
                static_cast<std::uint64_t>(shape.n));
        std::vector<std::int8_t> va(static_cast<std::size_t>(shape.n));
        std::vector<std::int8_t> vb(static_cast<std::size_t>(shape.n));
        for (std::int64_t r = 0; r < shape.count; ++r) {
            for (std::int64_t k = 0; k < shape.n; ++k) {
                va[static_cast<std::size_t>(k)] = rng.int4();
                vb[static_cast<std::size_t>(k)] = rng.int4();
            }
            std::uint8_t* ra = a.data() + static_cast<std::size_t>(r * row_bytes);
            std::uint8_t* rb = b.data() + static_cast<std::size_t>(r * row_bytes);
            std::memset(ra, 0, row_bytes);
            std::memset(rb, 0, row_bytes);
            pack_row(ra, va.data(), shape.n);
            pack_row(rb, vb.data(), shape.n);

            std::vector<std::int8_t> ua(static_cast<std::size_t>(shape.n));
            std::vector<std::int8_t> ub(static_cast<std::size_t>(shape.n));
            unpack_row(ua.data(), ra, shape.n);
            unpack_row(ub.data(), rb, shape.n);
            golden[static_cast<std::size_t>(r)] = oracle_dot(ua.data(), ub.data(), shape.n);
            CHECK(oracle_dot(va.data(), vb.data(), shape.n) == golden[static_cast<std::size_t>(r)]);
        }

        void* da = nullptr;
        void* db = nullptr;
        void* dout = nullptr;
        if (hipMalloc(&da, a.size()) != hipSuccess ||
            hipMalloc(&db, b.size()) != hipSuccess ||
            hipMalloc(&dout, sizeof(std::int32_t) * static_cast<std::size_t>(shape.count)) != hipSuccess) {
            std::printf("FAIL int4_dot random count=%lld n=%lld: hipMalloc\n",
                        static_cast<long long>(shape.count), static_cast<long long>(shape.n));
            (void)hipFree(da);
            (void)hipFree(db);
            (void)hipFree(dout);
            ++g_failures;
            continue;
        }
        if (hipMemcpy(da, a.data(), a.size(), hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(db, b.data(), b.size(), hipMemcpyHostToDevice) != hipSuccess) {
            ++g_failures;
            (void)hipFree(da);
            (void)hipFree(db);
            (void)hipFree(dout);
            continue;
        }

        std::vector<std::int32_t> got(static_cast<std::size_t>(shape.count), 0xCDCDCDCD);
        const int rc = rocminfer::int4_dot_i32(static_cast<std::int32_t*>(dout), da, db,
                                               shape.count, shape.n, nullptr);
        CHECK(rc == 0);
        CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(std::int32_t),
                        hipMemcpyDeviceToHost) == hipSuccess);

        bool ok = (rc == 0);
        for (std::int64_t r = 0; r < shape.count; ++r) {
            if (got[static_cast<std::size_t>(r)] != golden[static_cast<std::size_t>(r)]) {
                ok = false;
                std::printf("FAIL int4_dot count=%lld n=%lld row %lld: got=%d want=%d\n",
                            static_cast<long long>(shape.count),
                            static_cast<long long>(shape.n), static_cast<long long>(r),
                            got[static_cast<std::size_t>(r)],
                            golden[static_cast<std::size_t>(r)]);
            }
        }
        std::printf("  random count=%4lld n=%5lld  %s\n", static_cast<long long>(shape.count),
                    static_cast<long long>(shape.n), ok ? "exact match" : "MISMATCH");

        (void)hipFree(da);
        (void)hipFree(db);
        (void)hipFree(dout);
        if (!ok) {
            ++g_failures;
        }
    }
}

void test_einval() {
    void* da = nullptr;
    void* db = nullptr;
    void* dout = nullptr;
    if (hipMalloc(&da, 32u) != hipSuccess || hipMalloc(&db, 32u) != hipSuccess ||
        hipMalloc(&dout, 16u) != hipSuccess) {
        (void)hipFree(da);
        (void)hipFree(db);
        (void)hipFree(dout);
        return;
    }
    std::int32_t* out = static_cast<std::int32_t*>(dout);
    const void* a = da;
    const void* b = db;

    CHECK(rocminfer::int4_dot_i32(nullptr, a, b, 1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, nullptr, b, 1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, a, nullptr, 1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, a, b, 0, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, a, b, -1, 8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, a, b, 1, 0, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, a, b, 1, -8, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, a, b, 1, 7, nullptr) == EINVAL);
    CHECK(rocminfer::int4_dot_i32(out, a, b, 1, 9, nullptr) == EINVAL);

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

    std::printf("int4_dot packing + oracle vs int4_dot_i32 (sdot8)\n");
    test_random_packed();
    test_einval();

    if (g_failures != 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nPASSED: all checks green\n");
    return 0;
}
