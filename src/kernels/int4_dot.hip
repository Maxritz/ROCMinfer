/**
 * @file int4_dot.hip
 * @brief Device kernel + launcher for the integer 4-bit dot primitive.
 *
 * One translation unit compiled for every configured arch. The kernel uses only
 * `__builtin_amdgcn_sdot8` (`v_dot8_i32_i4`), which the AMD capability matrix
 * records as CONFIRMED-emitted on both `gfx1031` and `gfx1201`, so this source is
 * arch-neutral: it never includes rocWMMA and compiles identically for RDNA2 and
 * RDNA4. It is the "dp4a on 6700 XT" primitive gated by `__gfx1201__` on the
 * WMMA side and selected by the format registry on the integer side.
 */

#include <rocminfer/int4_dot.hpp>

#include <rocminfer/profiler.hpp>

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>

namespace rocminfer {

namespace {

/// True when `value` is a multiple of `alignment`, for a power-of-two alignment.
bool is_aligned(const void* pointer, std::uintptr_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

/// True when `a * b` is representable, for non-negative `a` and `b`.
bool product_fits(std::int64_t a, std::int64_t b) noexcept {
    if (a <= 0 || b <= 0) {
        return false;
    }
    return a <= (INT64_MAX / b);
}

/**
 * @brief Load one 4-byte group of eight int4 nibbles as a `uint32_t`.
 * @param[in] p Address of the first of four consecutive bytes, 4-byte aligned.
 * @return The four bytes as a little-endian `uint32_t`, byte `p+0` in bits 0..7.
 * @note In little-endian layout nibble `j` of the group occupies bits `4*j..4*j+3`
 *       with `j=0` in the low nibble of `p[0]`, which is exactly the operand
 *       ordering `__builtin_amdgcn_sdot8` sign-extends, so no byte shuffle is
 *       needed. Alignment is guaranteed by the launcher (`n` is a multiple of 8,
 *       hence `n/2` is a multiple of 4, hence every group offset is 4-aligned).
 */
__device__ inline std::uint32_t load_group(const std::uint8_t* p) noexcept {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

/**
 * @brief Integer 4-bit dot kernel: one workgroup computes one output row.
 * @param[out] out      Destination int32 sums, `count` elements.
 * @param[in]  a        Packed signed-int4 operand, `count` rows.
 * @param[in]  b        Packed signed-int4 operand, same layout as `a`.
 * @param[in]  count    Number of independent dot products (== grid width).
 * @param[in]  groups   `n / 8` eight-weight groups per row.
 * @param[in]  row_bytes `n / 2` bytes per row.
 * @return Nothing; writes `out[blockIdx.x]`.
 * @note Launched with `count` workgroups, so `blockIdx.x` needs no bounds test.
 *       Each thread owns a strided subset of the row's groups; the per-thread
 *       int64 partial is tree-reduced in shared memory, so the reduction width is
 *       independent of the wave size and the source is valid on Wave64 and Wave32.
 */
__global__ __launch_bounds__(kInt4DotThreads)
void int4_dot_i32_kernel(int32_t* __restrict__ out,
                         const std::uint8_t* __restrict__ a,
                         const std::uint8_t* __restrict__ b,
                         std::int64_t /*count*/, std::int64_t groups,
                         std::int64_t row_bytes) {
    static_assert(kInt4DotThreads == 256u, "kernel launch width changed without review");
    static_assert((kInt4DotThreads & (kInt4DotThreads - 1u)) == 0u,
                  "the tree reduction requires a power-of-two workgroup size");

    const unsigned int tid = threadIdx.x;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);

    const std::uint8_t* __restrict__ a_row = a + row * row_bytes;
    const std::uint8_t* __restrict__ b_row = b + row * row_bytes;

    // int64 per thread so the running partial cannot overflow before the
    // tree reduction, regardless of `n` or the operand magnitudes.
    std::int64_t acc = 0;
    for (std::int64_t g = static_cast<std::int64_t>(tid); g < groups;
         g += static_cast<std::int64_t>(kInt4DotThreads)) {
         // A group of kInt4DotGroup (8) int4 weights occupies 4 bytes (two nibbles
         // per byte), so the per-group byte stride is kInt4DotGroup/2. Striding by
         // kInt4DotGroup bytes would read every other group and run off the end of
         // the row for groups > 1 (an out-of-bounds read that faults the device).
         const std::int64_t group_bytes = kInt4DotGroup / 2;
         const std::uint32_t aw = load_group(a_row + g * group_bytes);
         const std::uint32_t bw = load_group(b_row + g * group_bytes);
        // Eight signed-int4 products. `__builtin_amdgcn_sdot8` lowers to
        // `v_dot8_i32_i4`, which the AMD capability matrix records as the
        // consumer RDNA2 integer-dot opcode. The local ROCm 7.16 compiler
        // accepts it on gfx1031 but refuses it on gfx1201 ("needs target
        // feature dot1-insts"), so the fast path is gated to gfx1031 and a
        // scalar fallback is used elsewhere (e.g. gfx1201, which routes
        // integer matmul through INT4 WMMA instead). `sdot8` sign-extends each
        // nibble, which is why only genuine two's-complement int4 may enter.
#if defined(__gfx1031__)
        acc += static_cast<std::int64_t>(__builtin_amdgcn_sdot8(aw, bw, 0, false));
#else
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const std::uint32_t na = (aw >> (4u * static_cast<std::uint32_t>(t))) & 0x0Fu;
            const std::uint32_t nb = (bw >> (4u * static_cast<std::uint32_t>(t))) & 0x0Fu;
            const int va = (na & 8u) ? (static_cast<int>(na) - 16) : static_cast<int>(na);
            const int vb = (nb & 8u) ? (static_cast<int>(nb) - 16) : static_cast<int>(nb);
            acc += static_cast<std::int64_t>(va) * static_cast<std::int64_t>(vb);
        }
#endif
    }

    __shared__ std::int64_t partial[kInt4DotThreads];
    partial[tid] = acc;
    __syncthreads();
    for (unsigned int stride = kInt4DotThreads / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        out[row] = static_cast<std::int32_t>(partial[0]);
    }
}

}  // namespace

std::size_t int4_dot_row_bytes(std::int64_t n) noexcept {
    if (n <= 0 || (n % static_cast<std::int64_t>(kInt4DotGroup)) != 0) {
        return 0u;
    }
    return static_cast<std::size_t>(n) / 2u;
}

std::size_t int4_dot_weight_bytes(std::int64_t count, std::int64_t n) noexcept {
    const std::size_t row_bytes = int4_dot_row_bytes(n);
    if (row_bytes == 0u || count <= 0) {
        return 0u;
    }
    if (count > static_cast<std::int64_t>(INT64_MAX / static_cast<std::int64_t>(row_bytes))) {
        return 0u;
    }
    return static_cast<std::size_t>(count) * row_bytes;
}

int int4_dot_i32(int32_t* out, const void* a, const void* b, std::int64_t count,
                 std::int64_t n, void* stream) {
    if (out == nullptr || a == nullptr || b == nullptr) {
        return EINVAL;
    }
    if (count <= 0 || n <= 0 || (n % static_cast<std::int64_t>(kInt4DotGroup)) != 0) {
        return EINVAL;
    }
    if (!is_aligned(out, 4u) || !is_aligned(a, 4u) || !is_aligned(b, 4u)) {
        return EINVAL;
    }
    if (count > static_cast<std::int64_t>(0xFFFFFFFFu)) {
        return EINVAL;
    }

    const std::int64_t row_bytes = static_cast<std::int64_t>(int4_dot_row_bytes(n));
    if (!product_fits(count, row_bytes)) {
        return EINVAL;
    }

    (void)hipGetLastError();

    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned int>(count), 1u, 1u);
    const dim3 threads(kInt4DotThreads, 1u, 1u);
    const std::int64_t groups = n / static_cast<std::int64_t>(kInt4DotGroup);

    hipLaunchKernelGGL(int4_dot_i32_kernel, grid, threads, 0u, hip_stream, out,
                       static_cast<const std::uint8_t*>(a),
                       static_cast<const std::uint8_t*>(b), count, groups, row_bytes);

    const hipError_t launch_status = hipGetLastError();
    return (launch_status == hipSuccess) ? 0 : EIO;
}

int int4_dot_i32_traced(TraceRecorder& recorder, int32_t* out, const void* a,
                        const void* b, std::int64_t count, std::int64_t n,
                        void* stream) {
    TraceScope scope(recorder, "int4-dot", TraceDomain::kDevice, stream);
    return int4_dot_i32(out, a, b, count, n, stream);
}

}  // namespace rocminfer
