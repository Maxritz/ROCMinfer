/**
 * @file int4_dot.hpp
 * @brief Integer 4-bit dot-product primitive backed by the AMD `sdot8` intrinsic.
 *
 * Computes, for `count` independent pairs of packed **two's-complement signed
 * 4-bit** weight buffers `a` and `b`, each of length `n` weights,
 *
 * ```
 *     out[i] = sum_{k<n} s4(a[i,k]) * s4(b[i,k])
 * ```
 *
 * where `s4(nibble)` sign-extends the 4-bit value to the range `-8..7`. Each
 * output is an exact signed 32-bit integer (returned as `int32_t`).
 *
 * ## Packing
 *
 * Weights are stored two per byte, little-endian within the byte: weight `k`
 * occupies the low nibble when `k` is even and the high nibble when `k` is odd.
 * This is identical to the nibble order `ggml` uses for its `Q4_0`/`Q4_1`
 * integer types, so a packed `Q4_0` block (ignoring its block scale) can be
 * fed here verbatim without a byte swap.
 *
 * ## Target ISA / verification
 *
 * On `gfx1031` (RDNA2 / RX 6700 XT) the fast path lowers to a single
 * `v_dot8_i32_i4` instruction per 8-weight group via `__builtin_amdgcn_sdot8`
 * (compiler-verified local probe, see `docs/AMD_CAPABILITY_MATRIX.md` §1 and §8).
 *
 * On `gfx1201` (RDNA4 / RX 9070 XT) that builtin is **REJECTED** by the local
 * ROCm 7.16 clang (it errors with `needs target feature dot1-insts`), so this
 * source falls through to the scalar nibble fallback in `int4_dot.hip`. That
 * fallback is verified correct on real silicon:
 * `rocminfer_int4_dot_tests` reports an exact integer match on the RX 9070 XT
 * for `n` in {8, 64, 104, 256, 1024, 4096}. The sdot8 hardware instruction is
 * gfx1031-only and has no local RDNA2 part to measure; it is compile+emit
 * verified on gfx1031 only.
 *
 * ## What this is NOT
 *
 * @warning This kernel consumes **genuine signed two's-complement int4**
 *          operands. It is the integer-dot primitive the AMD capability matrix
 *          refers to as the "dp4a on 6700 XT" path. It must **not** be used for
 *          the ROCmFP4 `Codebook10` format: those nibbles encode magnitudes
 *          `{0,1,2,3,4,6,8,10}`, which are not two's-complement integers, so
 *          routing them through `sdot8` would produce silently wrong results
 *          (see `docs/AMD_CAPABILITY_MATRIX.md` §6 and §8.6). The ROCmFP4 path
 *          stays on the FMA kernel in `rocmfp4_dot.hip`; the format registry
 *          owns the per-type dispatch and must not lower `kRocmFP4`/`kRocmFP4Fast`
 *          through this entry point.
 */

#ifndef ROCMINFER_INT4_DOT_HPP
#define ROCMINFER_INT4_DOT_HPP

#include <cstddef>
#include <cstdint>

namespace rocminfer {

/**
 * @brief Forward declaration of the engine tracing recorder.
 *
 * Only the traced overload needs the complete type, and only
 * `int4_dot.hip` includes `rocminfer/profiler.hpp`.
 */
class TraceRecorder;

/**
 * @brief Threads per workgroup in the int4 dot kernel.
 *
 * Fixed at 256 (a Wave64 on `gfx1031`, two waves of Wave32 on `gfx1201`); the
 * shared-memory reduction is statically sized to this.
 */
inline constexpr unsigned int kInt4DotThreads = 256u;

/**
 * @brief Weights represented by one int4 dot group.
 *
 * `sdot8` consumes exactly eight signed int4 operands, so a group is eight
 * weights (four bytes) and `n` must be a multiple of this.
 */
inline constexpr std::int64_t kInt4DotGroup = 8;

/**
 * @brief Packed byte length of one length-`n` row of signed int4 weights.
 * @param[in] n Number of weights in the row; must be a positive multiple of 8.
 * @return Exact byte length `n / 2`, or 0 when `n` is invalid or would overflow.
 * @pre None; invalid arguments are rejected by return value, not asserted.
 * @post The result is `(n / 2)` whenever that value fits in `std::size_t`.
 */
std::size_t int4_dot_row_bytes(std::int64_t n) noexcept;

/**
 * @brief Packed byte length of the whole int4 tensor.
 * @param[in] count Row count; must be positive.
 * @param[in] n     Weights per row; must be a positive multiple of 8.
 * @return `count * (n / 2)` bytes, or 0 when any argument is unusable or the
 *         product would overflow.
 * @pre None; invalid arguments are rejected by return value.
 * @post `0` is returned iff the caller's shape is malformed.
 */
std::size_t int4_dot_weight_bytes(std::int64_t count, std::int64_t n) noexcept;

/**
 * @brief Compute `count` integer 4-bit dot products on the GPU.
 *
 * Each of the `count` output rows is the signed 32-bit sum of `n` products of
 * two's-complement signed int4 nibbles. The integer accumulator is held in a
 * 64-bit per-thread register and narrowed to int32 at the reduction root, so
 * the partial no wider than `n * 7 * 7` never overflows int32 for any realistic
 * `n`, and the 64-bit path cannot saturate for practical tensor sizes.
 *
 * @param[out] out     Destination for `count` int32 sums, device memory,
 *                     4-byte aligned, caller-owned.
 * @param[in]  a       First packed signed-int4 operand, device memory, 4-byte
 *                     aligned, exactly `int4_dot_weight_bytes(count, n)` bytes.
 * @param[in]  b       Second packed signed-int4 operand, same layout as `a`.
 * @param[in]  count   Number of independent dot products; must be positive.
 * @param[in]  n       Weights per dot product; must be a positive multiple of 8.
 * @param[in]  stream  HIP stream to enqueue on, or nullptr for the default
 *                     stream.
 * @return 0 on success; `EINVAL` if any pointer is null, `count`/`n` is not
 *         positive, `n` is not a multiple of 8, alignment is wrong, or an
 *         extent overflows; `EIO` if the kernel launch fails.
 * @pre `a`, `b`, and `out` are device pointers from the HIP allocator and a HIP
 *      context exists on the calling thread.
 * @post On success the launch is enqueued on `stream` and `out[0..count)` holds
 *       the sums *once the stream has synchronised*; this call performs no
 *       synchronisation. On any failure the kernel is not enqueued and `out` is
 *       untouched.
 * @note One workgroup of `kInt4DotThreads` threads computes one row; every row
 *       is independent, so `count` is both the row count and the grid width with
 *       no bounds test inside the kernel.
 * @warning Not thread-safe with respect to concurrent calls that share an
 *          output buffer; concurrent calls with distinct buffers are safe.
 */
int int4_dot_i32(int32_t* out, const void* a, const void* b, std::int64_t count,
                 std::int64_t n, void* stream);

/**
 * @brief Traced wrapper around `int4_dot_i32` on the `int4-dot` component.
 *
 * Identical to `int4_dot_i32` except the kernel launch is bracketed by a
 * `TraceScope` in the `kDevice` domain, so its device execution time is
 * attributed to the logical component `int4-dot`.
 *
 * @param[in]  recorder Recorder; must outlive this call.
 * @param[out] out      Destination for `count` int32 sums.
 * @param[in]  a        First operand, same layout as `int4_dot_i32`.
 * @param[in]  b        Second operand, same layout as `int4_dot_i32`.
 * @param[in]  count    Number of independent dot products.
 * @param[in]  n        Weights per dot product; positive multiple of 8.
 * @param[in]  stream   HIP stream echoed into the scope and the launch.
 * @return Exactly what `int4_dot_i32` returns; tracing cannot change the result.
 * @pre Every precondition of `int4_dot_i32`, plus `recorder.enabled()` may be
 *      false, in which case the scope records nothing.
 * @post On success the launch is enqueued and the scope closed; device timing is
 *       pending until `TraceRecorder::resolve_pending` is called.
 * @warning Two `int4-dot` scopes must not be nested at the same depth.
 */
int int4_dot_i32_traced(TraceRecorder& recorder, int32_t* out, const void* a,
                        const void* b, std::int64_t count, std::int64_t n,
                        void* stream);

}  // namespace rocminfer

#endif  // ROCMINFER_INT4_DOT_HPP
