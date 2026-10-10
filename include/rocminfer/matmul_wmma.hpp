/**
 * @file matmul_wmma.hpp
 * @brief Public API for the RDNA4 WMMA FP16 matmul tile.
 *
 * Declares the launcher for the `matmul_wmma_16x16_kernel` device kernel. The
 * kernel emits a single `v_wmma_f32_16x16x16_f16` instruction per wave on gfx1201.
 *
 * @note Only available when `ROCMINFER_HAVE_HIP` is defined and the target
 *       supports WMMA (gfx1201). Host-only builds return `ENOTSUP` from the
 *       launcher without touching the device API.
 */

#ifndef ROCMINFER_MATMUL_WMMA_HPP
#define ROCMINFER_MATMUL_WMMA_HPP

#include <hip/hip_fp16.h>

namespace rocminfer {

class TraceRecorder;

/// Threads per wave on RDNA4 (gfx1201), Wave32.
inline constexpr unsigned int kWmmaWaveThreads = 32u;

/**
 * @brief Compute one 16x16 FP32 output tile via RDNA4 WMMA from two FP16 inputs.
 *
 * `C = A * B` with A, B each 16x16 row-major FP16 and C 16x16 row-major FP32.
 *
 * @param[out] c        Output tile base, device memory, at least 1024 bytes.
 * @param[in]  a        LHS tile, device memory, at least 256 bytes.
 * @param[in]  b        RHS tile, device memory, at least 256 bytes.
 * @param[in]  stream   HIP stream, or nullptr for the default stream.
 * @return 0 on success; EINVAL if any pointer is null; EIO on launch failure;
 *         ENOTSUP on a host-only build.
 * @pre `a`, `b`, `c` are valid device pointers, non-aliased, 16-byte aligned.
 * @post On success the kernel is enqueued on `stream`; `c` is updated after
 *       synchronising the stream.
 */
int matmul_wmma_16x16(float* c, const ::half* a, const ::half* b, void* stream);

/**
 * @brief Traced wrapper around `matmul_wmma_16x16`.
 * @param[in]  recorder Tracer; must outlive the launch.
 * @param[out] c        Output tile, device memory.
 * @param[in]  a        LHS tile, device memory.
 * @param[in]  b        RHS tile, device memory.
 * @param[in]  stream   HIP stream, or nullptr.
 * @return Exactly what `matmul_wmma_16x16` returns.
 * @pre The device kernel is compiled for a WMMA-capable target.
 * @post On success the launch is enqueued and the `wmma-matmul` scope closed.
 * @warning Do not nest two `wmma-matmul` scopes at the same depth.
 */
int matmul_wmma_16x16_traced(TraceRecorder& recorder, float* c, const ::half* a,
                             const ::half* b, void* stream);

}  // namespace rocminfer

#endif  // ROCMINFER_MATMUL_WMMA_HPP
