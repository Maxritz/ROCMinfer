/**
 * @file rocmfp4_wmma_dot.hip
 * @brief ROCmFP4 dequantize-to-f16 and WMMA matmul tile for RDNA4 (gfx1201).
 *
 * Phase 5 integration: reads one 16x16 window of packed 4-bit Codebook10 weights
 * (one 16-wide half-block per row), dequantizes 16 weights per row to `half` in
 * registers, stages the 16x16 f16 tile through shared memory in the row-major
 * layout the WMMA unit expects, loads it into a `matrix_a` fragment, broadcasts a
 * 16-wide activation slice across the 16 rows of a `matrix_b` fragment, and
 * accumulates `A * B` (16x16 f16 -> 16x16 f32) via `v_wmma_f32_16x16x16_f16`.
 *
 * The code-weight decode is bit-identical to `rocmfp4_dequantize_row` and the
 * scale bytes resolve through the same `rocmfp4_scale_to_float` oracle; the scale
 * table is supplied as a device pointer (uploaded once per process) so this
 * translation unit has no cross-TU `__constant__` dependency on `rocmfp4_dot.hip`.
 *
 * A full matvec tiles this 16-row kernel over `rows/16` row-groups and
 * `cols/32` weight-column groups (one 16-wide half-block per group), accumulating
 * the f32 partial across weight tiles within the WMMA accumulator.
 */

#include <rocminfer/rocmfp4_reference.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include <cerrno>
#include <cstdint>
#include <mutex>

// rocWMMA fragments and the tile kernel must live at global scope: HIP
// `__global__` functions cannot be members and rocWMMA expects the fragment
// types in the calling scope. rocWMMA is RDNA4-only; CMakeLists.txt restricts
// this source to `gfx1201` via the HIP_ARCHITECTURES source property so rocWMMA
// is never included while targeting gfx1031 (it static_asserts there).
using wmma_a_frag = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half,
                                      rocwmma::row_major>;
using wmma_b_frag = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half,
                                      rocwmma::row_major>;
using wmma_c_frag = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

namespace rocminfer {

namespace {

/// One-time host-side copy of the 127-entry scale table and its device mirror.
struct ScaleBuffer {
    float host[kScaleTableSize];
    void* dev = nullptr;
    std::once_flag once;
    int status = EIO;
};

/**
 * @brief Build the scale table on the host and upload it to the device, once.
 * @param[in,out] state Persistent state with a once-flag and the host table.
 * @return 0 on success, EIO if the upload fails.
 */
int ensure_scale_buffer(ScaleBuffer& state) noexcept {
    std::call_once(state.once, [&state]() noexcept {
        for (std::size_t i = 0; i < kScaleTableSize; ++i) {
            state.host[i] = rocmfp4_scale_to_float(static_cast<std::uint8_t>(i));
        }
        if (hipMalloc(&state.dev, sizeof(float) * kScaleTableSize) == hipSuccess &&
            state.dev != nullptr) {
            state.status = (hipMemcpy(state.dev, state.host, sizeof(float) * kScaleTableSize,
                                      hipMemcpyHostToDevice) == hipSuccess)
                               ? 0
                               : EIO;
        } else {
            state.status = EIO;
        }
    });
    return state.status;
}

}  // namespace

/**
 * @brief One-tile ROCmFP4 dequantize-and-WMMA matmul; C = A * B_tile.
 *
 * @tparam kDual true for the 18-byte dual-scale layout, false for FAST.
 * @param[out] out         16x16 f32 tile, row-major, device memory (>=1024 B).
 * @param[in]  row_weights Packed bytes for the 16 rows (16-row window), row-major,
 *                         `row_bytes` per row.
 * @param[in]  x           16-wide activation slice, f16, device memory.
 * @param[in]  row_bytes   Bytes per weight row (17 FAST or 18 dual-scale).
 * @param[in]  col_group   Half-block index; `block_idx = col_group/2`, even selects
 *                         the low 16 weight-columns, odd the high 16.
 * @param[in]  scales      Device pointer to the 127-entry scale table (f32).
 */
template <bool kDual>
__global__ __launch_bounds__(32)
void rocmfp4_wmma_dot_kernel(float* __restrict__ out,
                             const std::uint8_t* __restrict__ row_weights,
                             const half* __restrict__ x, std::int64_t row_bytes,
                             std::int64_t col_group, const float* __restrict__ scales) {
    const unsigned int tid = threadIdx.x;

    // ---- Dequantize the 16x16 A tile into shared memory (row-major). ----
    // Each of the 32 threads owns 8 of the 256 weight positions, flat-indexed
    // p = tid*8 + i (i=0..7). Position p maps to weight row r = p/16 and
    // tile-local column c = p%16. rocWMMA's cooperative load reads exactly
    // those 8 slots per thread from the row-major 16x16 buffer.
    __shared__ half sh_a[16 * 16];
    __shared__ half sh_b[16 * 16];

    const std::int64_t block_bytes = kDual ? static_cast<std::int64_t>(QK_ROCMFP4_BYTES)
                                           : static_cast<std::int64_t>(QK_ROCMFP4_FAST_BYTES);
    const std::int64_t block_idx = col_group / 2;     // full 32-weight block
    const bool high_half = (col_group & 1) != 0;     // 0..15 vs 16..31 within block
    const std::int64_t scale_lo_off = static_cast<std::int64_t>(QS_ROCMFP4);

    #pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
        const unsigned int p = tid * 8u + i;          // flat position in 16x16 tile
        const unsigned int r = p / 16u;               // weight row 0..15
        const unsigned int c = p % 16u;               // weight col (tile-local) 0..15

        const std::uint8_t* __restrict__ block =
            row_weights + r * row_bytes + block_idx * block_bytes;

        const std::uint8_t scale_lo_byte = block[scale_lo_off];
        const std::uint8_t scale_hi_byte =
            kDual ? block[static_cast<std::int64_t>(QS_ROCMFP4) + 1] : scale_lo_byte;
        const float scale_lo = scales[(scale_lo_byte <= ROCMFP4_SCALE_MAX) ? scale_lo_byte : 0u];
        const float scale_hi_val = scales[(scale_hi_byte <= ROCMFP4_SCALE_MAX) ? scale_hi_byte : 0u];

        const std::uint8_t nibble = high_half
                                   ? static_cast<std::uint8_t>(block[c] >> 4 & 0x0Fu)
                                   : static_cast<std::uint8_t>(block[c] & 0x0Fu);
        const std::uint32_t mag3 = nibble & 0x07u;
        const std::int32_t mag = (mag3 <= 4u) ? static_cast<std::int32_t>(mag3)
                                              : (2 * static_cast<std::int32_t>(mag3) - 4);
        const float weight = ((nibble & 0x08u) != 0u) ? -static_cast<float>(mag)
                                                       : static_cast<float>(mag);
        sh_a[p] = __float2half(weight * (high_half ? scale_hi_val : scale_lo));
    }
    __syncthreads();

    // ---- Build B as a row-broadcast activation: B(k,c) = x[k]. ----
    // Each row of the 16x16 B tile holds the same 16-wide activation slice, so
    // C(r,c) = sum_k A(r,k)*x[k] = the matvec dot for row r, replicated across
    // columns c. Threads 0..15 each own one row of B (== one activation lane).
    if (tid < 16u) {
        const half v = x[tid];
        #pragma unroll
        for (unsigned int col = 0; col < 16u; ++col) {
            sh_b[tid * 16u + col] = v;
        }
    }
    __syncthreads();

    wmma_a_frag a_frag{};
    wmma_b_frag b_frag{};
    wmma_c_frag c_frag{};
    rocwmma::fill_fragment(c_frag, 0.0f);
    rocwmma::load_matrix_sync(a_frag, sh_a, 16);
    rocwmma::load_matrix_sync(b_frag, sh_b, 16);
    rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    rocwmma::store_matrix_sync(out, c_frag, 16, rocwmma::mem_row_major);
}

/**
 * @brief Launch the WMMA ROCmFP4 matvec tile for a 16-row window.
 * @param[out] out        16x16 f32 tile, device memory.
 * @param[in]  row_weights Packed bytes for the 16 rows, row-major, device memory.
 * @param[in]  x          16-wide activation slice, f16, device memory.
 * @param[in]  row_bytes  17 (FAST) or 18 (dual-scale).
 * @param[in]  col_group  Half-block index (0 = low 16 cols of block 0).
 * @param[in]  scales     Device pointer to the 127-entry f32 scale table.
 * @param[in]  dual_scale Layout selector.
 * @param[in]  stream     HIP stream, or nullptr.
 * @return 0 on success; EINVAL if any pointer is null; EIO on launch failure;
 *         ENOTSUP on a host-only build.
 */
int rocmfp4_wmma_dot(float* out, const void* row_weights, const half* x,
                     std::int64_t row_bytes, std::int64_t col_group,
                     const float* scales, bool dual_scale, void* stream) {
    if (out == nullptr || row_weights == nullptr || x == nullptr || scales == nullptr) {
        return EINVAL;
    }
#ifdef ROCMINFER_HAVE_HIP
    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    if (dual_scale) {
        rocmfp4_wmma_dot_kernel<true><<<1, 32, 0, hip_stream>>>(
            out, static_cast<const std::uint8_t*>(row_weights), x, row_bytes,
            col_group, scales);
    } else {
        rocmfp4_wmma_dot_kernel<false><<<1, 32, 0, hip_stream>>>(
            out, static_cast<const std::uint8_t*>(row_weights), x, row_bytes,
            col_group, scales);
    }
    const hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? 0 : EIO;
#else
    // No matrix core on this target; the ROCmFP4 WMMA tile is not available here.
    (void)stream;
    return ENOTSUP;
#endif
}

/**
 * @brief Resolve (once, process-wide) a device pointer to the 127-entry scale table.
 * @param[out] dev_table Filled with the device pointer on success.
 * @return 0 on success; EIO if allocation or upload fails.
 * @note Mirrors the verified host-side table used by `rocmfp4_dot.hip`, rebuilt
 *       from `rocmfp4_scale_to_float` so the two kernels can never diverge.
 */
int rocmfp4_wmma_dot_resolve_scales(void** dev_table) {
    if (dev_table == nullptr) {
        return EINVAL;
    }
    static ScaleBuffer state{};
    const int status = ensure_scale_buffer(state);
    *dev_table = state.dev;
    return status;
}

}  // namespace rocminfer
