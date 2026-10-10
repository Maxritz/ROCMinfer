/**
 * @file int4_wmma.hpp
 * @brief Public API for the gfx1201 int4 x int4 WMMA matmul tile.
 *
 * Declares the launcher for the `int4_wmma_16x16_kernel` device kernel, which
 * computes one 16x16 tile of signed int4 products accumulated in int32 via the
 * RDNA4 `v_wmma_i32_16x16x16_iu4` instruction on gfx1201 (raw builtin
 * `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` with both sign flags true
 * and clamping off).
 *
 * @note Only available when `ROCMINFER_HAVE_HIP` is defined. Host-only builds
 *       return `ENOTSUP` from the launcher without touching the device API.
 */

#ifndef ROCMINFER_INT4_WMMA_HPP
#define ROCMINFER_INT4_WMMA_HPP

#include <cstdint>

namespace rocminfer {

class TraceRecorder;

/// Threads per wave on RDNA4 (gfx1201), Wave32. One 16x16x16 tile per wave.
inline constexpr unsigned int kInt4WmmaWaveThreads = 32u;

/// Edge length of the square WMMA tile (16x16 outputs, 16-deep K).
inline constexpr int kInt4WmmaTile = 16;

/**
 * @brief Compute one 16x16 int32 output tile from two packed signed-int4 tiles.
 *
 * `C = A * B` with A, B each 16x16 row-major signed int4 and C 16x16 row-major
 * int32. Nibble packing matches the engine int4 contract: element `k` of a row
 * occupies the low nibble of byte `k/2` when `k` is even and the high nibble
 * when `k` is odd (identical to `int4_dot.hpp` packing). Each input tile is
 * therefore 16 rows x 8 bytes = 128 bytes; the output tile is 16x16x4 = 1024
 * bytes.
 *
 * @param[out] c      Output tile base, device memory, at least 1024 bytes.
 * @param[in]  a      LHS tile, packed signed int4, device memory, 128 bytes.
 * @param[in]  b      RHS tile, packed signed int4, device memory, 128 bytes.
 * @param[in]  stream HIP stream, or nullptr for the default stream.
 * @return 0 on success; EINVAL if any pointer is null or not 4-byte aligned;
 *         EIO on launch failure; ENOTSUP on a host-only build.
 * @pre `a`, `b`, `c` are valid device pointers, non-aliased, 4-byte aligned.
 * @post On success the kernel is enqueued on `stream`; `c` is updated after
 *       synchronising the stream.
 * @note int4 x int4 -> int32 is exact integer arithmetic; any mismatch against
 *       the CPU oracle is a mapping defect, never rounding.
 * @warning `a`, `b`, `c` must not alias each other. Operands are genuine
 *          two's-complement signed int4; codebook nibbles must not enter here.
 */
int int4_wmma_16x16(int32_t* c, const void* a, const void* b, void* stream);

/**
 * @brief Traced wrapper around `int4_wmma_16x16`.
 * @param[in]  recorder Tracer; must outlive the launch.
 * @param[out] c        Output tile, device memory.
 * @param[in]  a        LHS tile, device memory.
 * @param[in]  b        RHS tile, device memory.
 * @param[in]  stream   HIP stream, or nullptr.
 * @return Exactly what `int4_wmma_16x16` returns.
 * @pre The device kernel is compiled for a WMMA-capable target.
 * @post On success the launch is enqueued and the `int4-wmma` scope closed.
 * @warning Do not nest two `int4-wmma` scopes at the same depth.
 */
int int4_wmma_16x16_traced(TraceRecorder& recorder, int32_t* c, const void* a,
                           const void* b, void* stream);

}  // namespace rocminfer

#endif  // ROCMINFER_INT4_WMMA_HPP
