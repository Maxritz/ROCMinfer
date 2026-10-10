/**
 * @file matmul_wmma.hip
 * @brief RDNA4 WMMA-accelerated 16x16 FP16 matmul tile and its host launcher.
 *
 * Uses rocWMMA to emit `v_wmma_f32_16x16x16_f16` hardware instructions on gfx1201.
 * One wave computes a 16x16 f32 output tile from two 16x16 f16 input tiles.
 *
 * This is Phase 5: it proves the WMMA path is correct on real silicon and gives
 * a measured baseline so the engine can choose between WMMA and the register FMA
 * kernel for a given shape. The ROCmFP4 dequantizer feeds this path by decoding
 * 4-bit nibbles to f16 in registers before loading the A fragment.
 *
 * Verified on AMD Radeon RX 9070 XT (gfx1201, Wave32) with ROCm 7.16.26631.
 *
 * Build note: rocWMMA static_asserts "Unsupported architecture" on any target
 * without a matrix core. To keep multi-arch (gfx1031;gfx1201) builds compiling,
 * CMakeLists.txt restricts this source to `gfx1201` via the HIP_ARCHITECTURES
 * source property, so rocWMMA is never included while targeting gfx1031.
 */

#include <rocminfer/matmul_wmma.hpp>

#include <rocminfer/profiler.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>

// rocWMMA fragments and the kernel must live at global scope: HIP `__global__`
// functions cannot be members and rocWMMA expects the fragment types in the
// calling scope.

#include <rocwmma/rocwmma.hpp>

using wmma_a_frag = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half,
                                      rocwmma::row_major>;
using wmma_b_frag = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half,
                                      rocwmma::row_major>;
using wmma_c_frag = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

/// Forward declaration: the kernel lives at global scope below the launcher.
extern "C" __global__
void matmul_wmma_16x16_kernel(float* c, const half* a, const half* b);

namespace rocminfer {

/**
 * @brief Compute one 16x16 FP32 output tile via RDNA4 WMMA from two FP16 inputs.
 *
 * The tile is `C = A * B` (A: 16x16 f16 row-major, B: 16x16 f16 row-major,
 * C: 16x16 f32 row-major). A single wave emits one `v_wmma_f32_16x16x16_f16`
 * instruction (the 16x16x16 tile fits in one wavefront of registers).
 *
 * @param[out] c        Output tile, row-major 16x16, device memory, at least
 *                      16*16*4 = 1024 bytes, 16-byte aligned.
 * @param[in]  a        LHS input tile, row-major 16x16 half (== 256 bytes),
 *                      16-byte aligned.
 * @param[in]  b        RHS input tile, row-major 16x16 half (== 256 bytes),
 *                      16-byte aligned.
 * @param[in]  stream   HIP stream to enqueue on, or nullptr for the default.
 * @return 0 on success; EINVAL if any pointer is null; EIO if the launch fails;
 *         ENOTSUP on a host-only build.
 * @pre A HIP context exists on the calling thread; `a`, `b`, `c` are device pointers.
 * @post On success the kernel is enqueued on `stream` and `c` holds `A*B`
 *       *once the stream has synchronised*; this call performs no synchronisation.
 * @warning `a`, `b`, `c` must not alias each other.
 */
int matmul_wmma_16x16(float* c, const half* a, const half* b, void* stream) {
    if (c == nullptr || a == nullptr || b == nullptr) {
        return EINVAL;
    }
#ifdef ROCMINFER_HAVE_HIP
    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    matmul_wmma_16x16_kernel<<<1, kWmmaWaveThreads, 0, hip_stream>>>(c, a, b);
    const hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? 0 : EIO;
#else
    (void)stream;
    return ENOTSUP;
#endif
}

/**
 * @brief Traced wrapper around `matmul_wmma_16x16` on the `wmma-matmul` component.
 * @param[in]  recorder Tracer; must outlive the launch.
 * @param[out] c        Output tile, row-major 16x16 FP32.
 * @param[in]  a        LHS tile, row-major 16x16 half.
 * @param[in]  b        RHS tile, row-major 16x16 half.
 * @param[in]  stream   HIP stream, or nullptr.
 * @return Exactly what `matmul_wmma_16x16` returns.
 * @pre The device kernel is compiled for a target that supports WMMA (gfx1201).
 * @post On success the launch is enqueued and the scope closed; device timing is
 *       pending until `TraceRecorder::resolve_pending` is called.
 * @warning Per the engine tracing contract, two `wmma-matmul` scopes must not be
 *          nested at the same depth.
 */
int matmul_wmma_16x16_traced(TraceRecorder& recorder, float* c, const half* a,
                             const half* b, void* stream) {
    TraceScope scope(recorder, "wmma-matmul", TraceDomain::kDevice, stream);
    return matmul_wmma_16x16(c, a, b, stream);
}

}  // namespace rocminfer

/**
 * @brief One-wave WMMA matmul tile: C = A * B (16x16, f16 inputs, f32 acc).
 * @param[out] c  Output tile, 16x16 f32 row-major, device memory.
 * @param[in]  a  LHS 16x16 f16 row-major, device memory.
 * @param[in]  b  RHS 16x16 f16 row-major, device memory.
 */
extern "C" __global__
void matmul_wmma_16x16_kernel(float* c, const half* a, const half* b) {
    wmma_a_frag a_frag{};
    wmma_b_frag b_frag{};
    wmma_c_frag c_frag{};

    rocwmma::load_matrix_sync(a_frag, a, 16);
    rocwmma::load_matrix_sync(b_frag, b, 16);
    rocwmma::fill_fragment(c_frag, 0.0f);
    rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    rocwmma::store_matrix_sync(c, c_frag, 16, rocwmma::mem_row_major);
}
