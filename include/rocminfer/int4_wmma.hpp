/**
 * @file int4_wmma.hpp
 * @brief Public API for the gfx1201 int4 x int4 WMMA matmul tile.
 *
 * Declares two launchers over one device source:
 *
 * - `int4_wmma_16x16_kernel`, which computes one 16x16 tile of signed int4
 *   products accumulated in int32 via the RDNA4 `v_wmma_i32_16x16x16_iu4`
 *   instruction on gfx1201 (raw builtin
 *   `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` with both sign flags
 *   true and clamping off).
 * - `int4_matvec_iu4_i32_kernel`, which drives that same instruction as a real
 *   engine matvec: `y[r] = sum_k s4(W[r,k]) * s4(x[k])` over a packed `rows`
 *   x `cols` weight matrix and one shared packed activation vector, with one
 *   wave per 16 output rows and the accumulator held in integer registers
 *   across every 16-channel K step.
 *
 * @note Only available when `ROCMINFER_HAVE_HIP` is defined. Host-only builds
 *       return `ENOTSUP` from the launchers without touching the device API.
 */
#ifndef ROCMINFER_INT4_WMMA_HPP
#define ROCMINFER_INT4_WMMA_HPP

#include <cstdint>

namespace rocminfer {

class TraceRecorder;

/// Threads per wave on RDNA4 (gfx1201), Wave32. One 16x16x16 tile per wave.
inline constexpr unsigned int kInt4WmmaWaveThreads = 32u;

/// Edge length of the square WMMA tile (16x16 outputs, 16-deep K).
///
/// Also the number of output rows one wave owns in the broadcast matvec
/// (`int4_matvec_iu4_i32`), so both `rows` and `cols` must be multiples of it.
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

/**
 * @brief Broadcast integer matvec driven by the RDNA4 IU4 WMMA instruction.
 *
 * Computes `y[r] = sum_{k<cols} s4(W[r,k]) * s4(x[k])` where `W` is a packed
 * `rows` x `cols` matrix of **two's-complement signed 4-bit** weights and `x`
 * is ONE packed `cols`-wide activation vector shared by every row (the same
 * broadcast-matvec contract as `int4_matvec_i32`; it is not the paired-rows
 * `int4_dot_i32` primitive). Each output is an exact signed 32-bit integer.
 *
 * One wave of `kInt4WmmaWaveThreads` threads owns 16 output rows, so the grid
 * is `rows / kInt4WmmaTile`. Inside the K loop the A operand of each
 * `v_wmma_i32_16x16x16_iu4` tile is the 16x16 weight slice and the B operand is
 * the corresponding 16-channel slice of `x` replicated across all 16 columns,
 * so every column of the output tile is the partial dot of one weight row
 * against that slice. The partial sums are accumulated in the WMMA's integer
 * registers across all `cols / 16` steps and never converted to floating
 * point, so the result is exact for any `cols` whose signed 32-bit sum is
 * representable.
 *
 * Packing is identical to `int4_wmma_16x16` and `int4_matvec_i32`: element `k`
 * of a row occupies the low nibble of byte `k/2` when `k` is even and the high
 * nibble when `k` is odd, and `W` holds `rows * (cols / 2)` bytes row-major
 * with no padding.
 *
 * @param[out] y      Destination for `rows` int32 sums, device memory,
 *                    4-byte aligned, caller-owned.
 * @param[in]  W      Packed signed-int4 weight matrix, device memory, 4-byte
 *                    aligned, exactly `int4_matvec_weight_bytes(rows, cols)`
 *                    bytes, row-major with no padding.
 * @param[in]  x      Packed signed-int4 activation vector, device memory,
 *                    4-byte aligned, exactly `int4_matvec_row_bytes(cols)`
 *                    bytes. This is a matrix-vector product: every row of `W`
 *                    is contracted against this one vector.
 * @param[in]  rows   Number of output rows; must be a positive multiple of
 *                    `kInt4WmmaTile`.
 * @param[in]  cols   Weights per row; must be a positive multiple of
 *                    `kInt4WmmaTile` (hence of `kInt4MatvecGroup`).
 * @param[in]  stream HIP stream to enqueue on, or nullptr for the default
 *                    stream.
 * @return 0 on success; `EINVAL` if any pointer is null, `rows`/`cols` is not
 *         positive, either is not a multiple of `kInt4WmmaTile`, alignment is
 *         wrong, or `rows * (cols / 2)` overflows; `EIO` if the kernel launch
 *         fails; `ENOTSUP` on a host-only build.
 * @pre `W`, `x`, and `y` are device pointers from the HIP allocator and a HIP
 *      context exists on the calling thread.
 * @post On success the launch is enqueued on `stream` and `y[0..rows)` holds
 *       the sums *once the stream has synchronised*; this call performs no
 *       synchronisation. On any failure the kernel is not enqueued and `y` is
 *       untouched.
 * @note Unaligned extents are rejected rather than padded: a wave owns exactly
 *       16 rows, so `rows % 16 != 0` would need a tail path whose partial
 *       output tile is silently wrong, and `cols % 16 != 0` would need a
 *       partial K step whose missing channels are undefined. Both are `EINVAL`
 *       at the boundary and the test asserts every such case.
 * @note The accumulator is the WMMA's int32 register set. A result wider than
 *       int32 cannot be produced by this kernel, exactly as for
 *       `int4_matvec_i32`; the bound is `|y[r]| <= cols * 64`.
 * @warning `W`, `x`, and `y` must not alias each other. Operands are genuine
 *          two's-complement signed int4; codebook nibbles must not enter here.
 *          Not thread-safe with respect to concurrent calls that share an
 *          output buffer; concurrent calls with distinct buffers are safe.
 */
int int4_matvec_iu4_i32(std::int32_t* y, const void* W, const void* x,
                        std::int64_t rows, std::int64_t cols, void* stream);

/**
 * @brief Traced wrapper around `int4_matvec_iu4_i32` on the `int4-matvec-iu4`
 *        component.
 *
 * Identical to `int4_matvec_iu4_i32` except the kernel launch is bracketed by
 * a `TraceScope` in the `kDevice` domain, so its device execution time is
 * attributed to the logical component `int4-matvec-iu4`.
 *
 * @param[in]  recorder Recorder; must outlive this call.
 * @param[out] y        Destination for `rows` int32 sums.
 * @param[in]  W        Packed signed-int4 weight matrix, same layout as
 *                      `int4_matvec_iu4_i32`.
 * @param[in]  x        Shared packed signed-int4 activation vector, same layout
 *                      as `int4_matvec_iu4_i32`.
 * @param[in]  rows     Number of output rows; positive multiple of 16.
 * @param[in]  cols     Weights per row; positive multiple of 16.
 * @param[in]  stream   HIP stream echoed into the scope and the launch.
 * @return Exactly what `int4_matvec_iu4_i32` returns; tracing cannot change the
 *         result.
 * @pre Every precondition of `int4_matvec_iu4_i32`, plus
 *      `recorder.enabled()` may be false, in which case the scope records
 *      nothing.
 * @post On success the launch is enqueued and the scope closed; device timing is
 *       pending until `TraceRecorder::resolve_pending` is called.
 * @warning Two `int4-matvec-iu4` scopes must not be nested at the same depth.
 */
int int4_matvec_iu4_i32_traced(TraceRecorder& recorder, std::int32_t* y,
                               const void* W, const void* x, std::int64_t rows,
                               std::int64_t cols, void* stream);

}  // namespace rocminfer

#endif  // ROCMINFER_INT4_WMMA_HPP
