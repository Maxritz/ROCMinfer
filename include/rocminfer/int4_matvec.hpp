/**
 * @file int4_matvec.hpp
 * @brief Broadcast integer 4-bit matrix-vector product for genuine signed int4.
 *
 * Computes, for a packed `rows` x `cols` matrix `W` of **two's-complement
 * signed 4-bit** weights and ONE packed activation vector `x` of length
 * `cols` shared by every row,
 *
 * ```
 *     y[r] = sum_{k<cols} s4(W[r,k]) * s4(x[k])
 * ```
 *
 * where `s4(nibble)` sign-extends the 4-bit value to the range `-8..7`. Each
 * output is an exact signed 32-bit integer (returned as `int32_t`).
 *
 * ## Packing
 *
 * Weights are stored two per byte, little-endian within the byte: weight `k`
 * occupies the low nibble when `k` is even and the high nibble when `k` is odd.
 * `W` holds `rows * (cols / 2)` bytes row-major with no padding; `x` holds one
 * `cols / 2`-byte row shared by all output rows. This is the same nibble order
 * `ggml` uses for its `Q4_0`/`Q4_1` integer types.
 *
 * ## Target ISA / verification
 *
 * On `gfx1031` (RDNA2 / RX 6700 XT) the fast path lowers to a single
 * `v_dot8_i32_i4` instruction per 8-weight group via `__builtin_amdgcn_sdot8`,
 * applied to the weight row against the shared activation vector.
 *
 * On `gfx1201` (RDNA4 / RX 9070 XT) that builtin is **REJECTED** by ROCm 7.16
 * clang (it errors with `needs target feature dot1-insts`), so this source
 * falls through to the scalar nibble fallback in `int4_matvec.hip`, verified
 * correct on real silicon by `tests/test_int4_matvec.cpp` (exact integer
 * match). The sdot8 hardware instruction is gfx1031-only and is compile+emit
 * verified on gfx1031 only.
 *
 * ## What this is NOT
 *
 * @warning This is the broadcast matvec (`W` rows x shared `x`), not the
 *          paired-rows primitive `int4_dot_i32` (`out[r] = dot(a[r], b[r])`
 *          from two `count` x `n` buffers). Arity, dtype contract, and reuse
 *          all differ: `Model::matvec` owns ONE packed `rows` x `cols` weight
 *          tensor plus ONE shared activation broadcast to all rows, so it must
 *          route through this entry point and never through `int4_dot_i32`.
 * @warning This kernel consumes **genuine signed two's-complement int4**
 *          operands. It must **not** be used for the ROCmFP4 `Codebook10`
 *          format: those nibbles encode magnitudes `{0,1,2,3,4,6,8,10}`, which
 *          are not two's-complement integers, so routing them through `sdot8`
 *          would produce silently wrong results. `Model::matvec_int4x` rejects
 *          `Codebook10` tensors with `EINVAL` before reaching this entry point.
 */

#ifndef ROCMINFER_INT4_MATVEC_HPP
#define ROCMINFER_INT4_MATVEC_HPP

#include <cstddef>
#include <cstdint>

namespace rocminfer {

/**
 * @brief Forward declaration of the engine tracing recorder.
 *
 * Only the traced overload needs the complete type, and only
 * `int4_matvec.hip` includes `rocminfer/profiler.hpp`.
 */
class TraceRecorder;

/**
 * @brief Threads per workgroup in the int4 broadcast matvec kernel.
 *
 * Fixed at 256 (a Wave64 on `gfx1031`, two waves of Wave32 on `gfx1201`); the
 * shared-memory reduction is statically sized to this.
 */
inline constexpr unsigned int kInt4MatvecThreads = 256u;

/**
 * @brief Weights represented by one int4 matvec group.
 *
 * `sdot8` consumes exactly eight signed int4 operands, so a group is eight
 * weights (four bytes) and `cols` must be a multiple of this.
 */
inline constexpr std::int64_t kInt4MatvecGroup = 8;

/**
 * @brief Packed byte length of one length-`cols` row of signed int4 weights.
 * @param[in] cols Number of weights in the row; must be a positive multiple of 8.
 * @return Exact byte length `cols / 2`, or 0 when `cols` is invalid or would overflow.
 * @pre None; invalid arguments are rejected by return value, not asserted.
 * @post The result is `(cols / 2)` whenever that value fits in `std::size_t`.
 */
std::size_t int4_matvec_row_bytes(std::int64_t cols) noexcept;

/**
 * @brief Packed byte length of the whole int4 weight matrix.
 * @param[in] rows Row count; must be positive.
 * @param[in] cols Weights per row; must be a positive multiple of 8.
 * @return `rows * (cols / 2)` bytes, or 0 when any argument is unusable or the
 *         product would overflow.
 * @pre None; invalid arguments are rejected by return value.
 * @post `0` is returned iff the caller's shape is malformed.
 * @note This sizes the `W` matrix only. The shared activation `x` occupies a
 *       further `int4_matvec_row_bytes(cols)` bytes.
 */
std::size_t int4_matvec_weight_bytes(std::int64_t rows, std::int64_t cols) noexcept;

/**
 * @brief Compute `y = W * x` over packed signed int4 on the GPU (broadcast).
 *
 * Each of the `rows` outputs is the signed 32-bit sum of `cols` products of
 * two's-complement signed int4 nibbles, where the second operand of every
 * product comes from the single shared vector `x`. The integer accumulator is
 * held in a 64-bit per-thread register and narrowed to int32 at the reduction
 * root, so the partial no wider than `cols * 7 * 7` never overflows int32 for
 * any realistic `cols`, and the 64-bit path cannot saturate for practical
 * tensor sizes.
 *
 * @param[out] y      Destination for `rows` int32 sums, device memory,
 *                    4-byte aligned, caller-owned.
 * @param[in]  W      Packed signed-int4 weight matrix, device memory, 4-byte
 *                    aligned, exactly `int4_matvec_weight_bytes(rows, cols)`
 *                    bytes, row-major with no padding.
 * @param[in]  x      Packed signed-int4 activation vector, device memory,
 *                    4-byte aligned, exactly `int4_matvec_row_bytes(cols)`
 *                    bytes. This is a matrix-vector product: every row of `W`
 *                    is contracted against this one vector, so the buffer is
 *                    `cols / 2` bytes in total and not `rows * cols / 2`.
 * @param[in]  rows   Number of output rows; must be positive.
 * @param[in]  cols   Weights per row; must be a positive multiple of 8.
 * @param[in]  stream HIP stream to enqueue on, or nullptr for the default
 *                    stream.
 * @return 0 on success; `EINVAL` if any pointer is null, `rows`/`cols` is not
 *         positive, `cols` is not a multiple of 8, alignment is wrong, or an
 *         extent overflows; `EIO` if the kernel launch fails; `ENOTSUP` on a
 *         host-only build.
 * @pre `W`, `x`, and `y` are device pointers from the HIP allocator and a HIP
 *      context exists on the calling thread.
 * @post On success the launch is enqueued on `stream` and `y[0..rows)` holds
 *       the sums *once the stream has synchronised*; this call performs no
 *       synchronisation. On any failure the kernel is not enqueued and `y` is
 *       untouched.
 * @note One workgroup of `kInt4MatvecThreads` threads computes one row; every
 *       row is independent, so `rows` is both the row count and the grid width
 *       with no bounds test inside the kernel. The shared `x` is read by every
 *       workgroup through global loads served by the L2 broadcast; it is
 *       deliberately not staged per-block because every thread needs the same
 *       groups and the reuse is read-only.
 * @warning Not thread-safe with respect to concurrent calls that share an
 *          output buffer; concurrent calls with distinct buffers are safe.
 */
int int4_matvec_i32(std::int32_t* y, const void* W, const void* x, std::int64_t rows,
                    std::int64_t cols, void* stream);

/**
 * @brief Traced wrapper around `int4_matvec_i32` on the `int4-matvec` component.
 *
 * Identical to `int4_matvec_i32` except the kernel launch is bracketed by a
 * `TraceScope` in the `kDevice` domain, so its device execution time is
 * attributed to the logical component `int4-matvec`.
 *
 * @param[in]  recorder Recorder; must outlive this call.
 * @param[out] y        Destination for `rows` int32 sums.
 * @param[in]  W        Packed signed-int4 weight matrix, same layout as
 *                      `int4_matvec_i32`.
 * @param[in]  x        Shared packed signed-int4 activation vector, same layout
 *                      as `int4_matvec_i32`.
 * @param[in]  rows     Number of output rows.
 * @param[in]  cols     Weights per row; positive multiple of 8.
 * @param[in]  stream   HIP stream echoed into the scope and the launch.
 * @return Exactly what `int4_matvec_i32` returns; tracing cannot change the result.
 * @pre Every precondition of `int4_matvec_i32`, plus `recorder.enabled()` may be
 *      false, in which case the scope records nothing.
 * @post On success the launch is enqueued and the scope closed; device timing is
 *       pending until `TraceRecorder::resolve_pending` is called.
 * @warning Two `int4-matvec` scopes must not be nested at the same depth.
 */
int int4_matvec_i32_traced(TraceRecorder& recorder, std::int32_t* y, const void* W,
                           const void* x, std::int64_t rows, std::int64_t cols,
                           void* stream);

}  // namespace rocminfer

#endif  // ROCMINFER_INT4_MATVEC_HPP
