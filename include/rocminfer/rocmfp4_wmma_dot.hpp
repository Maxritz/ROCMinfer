/**
 * @file rocmfp4_wmma_dot.hpp
 * @brief Public API for the ROCmFP4-dequantize-and-WMMA matvec tile.
 *
 * Declares the launcher for the `rocmfp4_wmma_dot_kernel` device kernel, which
 * dequantizes 16x16 Codebook10 weight tiles to f16 and consumes them through an
 * RDNA4 WMMA unit (`v_wmma_f32_16x16x16_f16`) on gfx1201.
 */

#ifndef ROCMINFER_ROCMFP4_WMMA_DOT_HPP
#define ROCMINFER_ROCMFP4_WMMA_DOT_HPP

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

namespace rocminfer {

class TraceRecorder;

/**
 * @brief Resolve (once, process-wide) a device pointer to the 127-entry scale table.
 * @param[out] dev_table Filled with the device pointer on success.
 * @return 0 on success; EINVAL if `dev_table` is null; EIO if allocation or
 *         upload of the table fails.
 * @note Mirrors the verified host-side table used by the FMA kernel, rebuilt
 *       from `rocmfp4_scale_to_float` so the two kernels can never diverge.
 * @pre A HIP context exists on the calling thread.
 * @post On success `*dev_table` is a valid device pointer to `kScaleTableSize`
 *       floats, stable for the lifetime of the process.
 */
int rocmfp4_wmma_dot_resolve_scales(void** dev_table);

/**
 * @brief Launch one 16x16 WMMA ROCmFP4 dequantize-and-matmul tile.
 *
 * Dequantizes a 16-row window of packed Codebook10 weights (one 16-wide
 * half-block per row) into a 16x16 f16 tile, multiplies against a broadcast
 * 16-wide activation slice, and stores a 16x16 f32 output tile.
 *
 * @param[out] out         16x16 f32 tile, device memory, at least 1024 bytes.
 * @param[in]  row_weights Packed bytes for the 16 rows, row-major, device memory.
 * @param[in]  x           16-wide activation slice, f16, device memory.
 * @param[in]  row_bytes   Bytes per weight row (17 FAST or 18 dual-scale).
 * @param[in]  col_group   Half-block index; `block_idx = col_group/2`, even selects
 *                         the low 16 weight-columns, odd the high 16.
 * @param[in]  scales      Device pointer to the 127-entry f32 scale table
 *                         (obtained from `rocmfp4_wmma_dot_resolve_scales`).
 * @param[in]  dual_scale  Layout selector: true for 18-byte dual-scale, false for
 *                         17-byte FAST.
 * @param[in]  stream      HIP stream, or nullptr for the default stream.
 * @return 0 on success; EINVAL if any pointer is null; EIO on launch failure;
 *         ENOTSUP on a host-only build.
 * @pre `scales` points to a device buffer of at least `kScaleTableSize` floats,
 *      populated via `rocmfp4_wmma_dot_resolve_scales`.
 * @post On success the kernel is enqueued on `stream`; `out` holds the matmul
 *       result after synchronising the stream.
 */
int rocmfp4_wmma_dot(float* out, const void* row_weights, const ::half* x,
                     std::int64_t row_bytes, std::int64_t col_group,
                     const float* scales, bool dual_scale, void* stream);

}  // namespace rocminfer

#endif  // ROCMINFER_ROCMFP4_WMMA_DOT_HPP
