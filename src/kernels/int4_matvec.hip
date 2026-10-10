/**
 * @file int4_matvec.hip
 * @brief Device kernel + launcher for the integer 4-bit broadcast matvec.
 *
 * One translation unit compiled for every configured arch. The kernel computes
 * `y[r] = sum_k s4(W[r,k]) * s4(x[k])` where `W` is a packed `rows` x `cols`
 * signed-int4 matrix and `x` is ONE packed `cols`-wide signed-int4 vector
 * shared by all rows (matrix-vector broadcast, not the paired-rows dot).
 *
 * Device path mirrors `int4_dot.hip`: one workgroup of `kInt4MatvecThreads`
 * threads computes one output row; each thread owns a strided subset of the
 * row's 8-weight groups; the per-thread int64 partial is tree-reduced in
 * shared memory, so the reduction width is independent of the wave size and
 * the source is valid on Wave64 and Wave32. The shared vector is read by
 * every workgroup through global loads; reuse is read-only and served by the
 * L2 broadcast, so no per-block staging is performed.
 *
 * The `__builtin_amdgcn_sdot8` fast path (`v_dot8_i32_i4`) is accepted by the
 * local ROCm 7.16 compiler on `gfx1031` only; on `gfx1201` it is rejected
 * ("needs target feature dot1-insts"), so the fast path is gated to
 * `__gfx1031__` and a scalar nibble fallback handles every other target.
 * `sdot8` sign-extends each nibble, which is why only genuine
 * two's-complement int4 may enter: ROCmFP4 `Codebook10` nibbles are rejected
 * upstream in `Model::matvec_int4x`, never here.
 */

#include <rocminfer/int4_matvec.hpp>

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
 *       needed. Alignment is guaranteed by the launcher (`cols` is a multiple
 *       of 8, hence `cols/2` is a multiple of 4, hence every group offset and
 *       the shared vector itself are 4-aligned).
 */
__device__ inline std::uint32_t load_group(const std::uint8_t* p) noexcept {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

/**
 * @brief Integer 4-bit broadcast matvec kernel: one workgroup computes one row.
 * @param[out] y          Destination int32 sums, `rows` elements.
 * @param[in]  W          Packed signed-int4 weight matrix, `rows` rows.
 * @param[in]  x          Packed signed-int4 activation vector, one row, shared.
 * @param[in]  rows       Number of output rows (== grid width).
 * @param[in]  groups     `cols / 8` eight-weight groups per row.
 * @param[in]  row_bytes  `cols / 2` bytes per weight row (== bytes of `x`).
 * @return Nothing; writes `y[blockIdx.x]`.
 * @note Launched with `rows` workgroups, so `blockIdx.x` needs no bounds test.
 *       Each thread owns a strided subset of the row's groups against the same
 *       shared-vector groups; the per-thread int64 partial is tree-reduced in
 *       shared memory, so the reduction width is independent of the wave size
 *       and the source is valid on Wave64 and Wave32.
 */
__global__ __launch_bounds__(kInt4MatvecThreads)
void int4_matvec_i32_kernel(std::int32_t* __restrict__ y,
                            const std::uint8_t* __restrict__ W,
                            const std::uint8_t* __restrict__ x,
                            std::int64_t /*rows*/, std::int64_t groups,
                            std::int64_t row_bytes) {
    static_assert(kInt4MatvecThreads == 256u, "kernel launch width changed without review");
    static_assert((kInt4MatvecThreads & (kInt4MatvecThreads - 1u)) == 0u,
                  "the tree reduction requires a power-of-two workgroup size");

    const unsigned int tid = threadIdx.x;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);

    const std::uint8_t* __restrict__ w_row = W + row * row_bytes;

    // int64 per thread so the running partial cannot overflow before the
    // tree reduction, regardless of `cols` or the operand magnitudes.
    std::int64_t acc = 0;
    for (std::int64_t g = static_cast<std::int64_t>(tid); g < groups;
         g += static_cast<std::int64_t>(kInt4MatvecThreads)) {
        // A group of kInt4MatvecGroup (8) int4 weights occupies 4 bytes (two
        // nibbles per byte), so the per-group byte stride is
        // kInt4MatvecGroup/2.
        const std::int64_t group_bytes = kInt4MatvecGroup / 2;
        const std::uint32_t ww = load_group(w_row + g * group_bytes);
        const std::uint32_t xw = load_group(x + g * group_bytes);
        // Eight signed-int4 products of this row's weights against the shared
        // activation. `__builtin_amdgcn_sdot8` lowers to `v_dot8_i32_i4`, which
        // the local ROCm 7.16 compiler accepts on gfx1031 but refuses on
        // gfx1201 ("needs target feature dot1-insts"), so the fast path is
        // gated to gfx1031 and a scalar fallback is used elsewhere. `sdot8`
        // sign-extends each nibble, which is why only genuine two's-complement
        // int4 may enter.
#if defined(__gfx1031__)
        acc += static_cast<std::int64_t>(__builtin_amdgcn_sdot8(ww, xw, 0, false));
#else
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const std::uint32_t nw = (ww >> (4u * static_cast<std::uint32_t>(t))) & 0x0Fu;
            const std::uint32_t nx = (xw >> (4u * static_cast<std::uint32_t>(t))) & 0x0Fu;
            const int vw = (nw & 8u) ? (static_cast<int>(nw) - 16) : static_cast<int>(nw);
            const int vx = (nx & 8u) ? (static_cast<int>(nx) - 16) : static_cast<int>(nx);
            acc += static_cast<std::int64_t>(vw) * static_cast<std::int64_t>(vx);
        }
#endif
    }

    __shared__ std::int64_t partial[kInt4MatvecThreads];
    partial[tid] = acc;
    __syncthreads();
    for (unsigned int stride = kInt4MatvecThreads / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        y[row] = static_cast<std::int32_t>(partial[0]);
    }
}

}  // namespace

std::size_t int4_matvec_row_bytes(std::int64_t cols) noexcept {
    if (cols <= 0 || (cols % static_cast<std::int64_t>(kInt4MatvecGroup)) != 0) {
        return 0u;
    }
    return static_cast<std::size_t>(cols) / 2u;
}

std::size_t int4_matvec_weight_bytes(std::int64_t rows, std::int64_t cols) noexcept {
    const std::size_t row_bytes = int4_matvec_row_bytes(cols);
    if (row_bytes == 0u || rows <= 0) {
        return 0u;
    }
    if (rows > static_cast<std::int64_t>(INT64_MAX / static_cast<std::int64_t>(row_bytes))) {
        return 0u;
    }
    return static_cast<std::size_t>(rows) * row_bytes;
}

int int4_matvec_i32(std::int32_t* y, const void* W, const void* x, std::int64_t rows,
                    std::int64_t cols, void* stream) {
#ifdef ROCMINFER_HAVE_HIP
    if (y == nullptr || W == nullptr || x == nullptr) {
        return EINVAL;
    }
    if (rows <= 0 || cols <= 0 || (cols % static_cast<std::int64_t>(kInt4MatvecGroup)) != 0) {
        return EINVAL;
    }
    if (!is_aligned(y, 4u) || !is_aligned(W, 4u) || !is_aligned(x, 4u)) {
        return EINVAL;
    }
    if (rows > static_cast<std::int64_t>(0xFFFFFFFFu)) {
        return EINVAL;
    }

    const std::int64_t row_bytes = static_cast<std::int64_t>(int4_matvec_row_bytes(cols));
    if (!product_fits(rows, row_bytes)) {
        return EINVAL;
    }

    (void)hipGetLastError();

    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned int>(rows), 1u, 1u);
    const dim3 threads(kInt4MatvecThreads, 1u, 1u);
    const std::int64_t groups = cols / static_cast<std::int64_t>(kInt4MatvecGroup);

    hipLaunchKernelGGL(int4_matvec_i32_kernel, grid, threads, 0u, hip_stream, y,
                       static_cast<const std::uint8_t*>(W),
                       static_cast<const std::uint8_t*>(x), rows, groups, row_bytes);

    const hipError_t launch_status = hipGetLastError();
    return (launch_status == hipSuccess) ? 0 : EIO;
#else
    (void)y;
    (void)W;
    (void)x;
    (void)rows;
    (void)cols;
    (void)stream;
    return ENOTSUP;
#endif
}

int int4_matvec_i32_traced(TraceRecorder& recorder, std::int32_t* y, const void* W,
                           const void* x, std::int64_t rows, std::int64_t cols,
                           void* stream) {
    TraceScope scope(recorder, "int4-matvec", TraceDomain::kDevice, stream);
    return int4_matvec_i32(y, W, x, rows, cols, stream);
}

}  // namespace rocminfer
