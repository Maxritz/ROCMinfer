/**
 * @file rocmfp4_dot.hpp
 * @brief Fused ROCmFP4 dequantize-and-dot kernel for ROCMinfer.
 *
 * Computes, for a packed ROCmFP4 weight matrix `W` and an FP32 activation
 * vector `x` of length `cols`,
 *
 * ```
 *     y[r] = sum_{j < cols} dequant(W_row_r, j) * x[j]
 * ```
 *
 * without ever materialising `dequant(W_row_r, j)` in memory. The packed
 * nibbles are read directly from device memory, the Codebook10 codes and the
 * UE4M3 block scales are decoded in registers, and the products are consumed
 * immediately by an FP32 FMA chain.
 *
 * ## Layout contract
 *
 * The engine's format registry (`rocminfer/format_registry.hpp`) owns the
 * authoritative layout. This header re-states only the two byte sizes the
 * launcher needs, and it asserts against `rocminfer_reference` at compile time
 * in `rocmfp4_dot.hip`:
 *
 * - `Q4_0_ROCMFP4`      — 18 bytes per 32 weights, dual scale
 *   (`e[0]` scales weights 0..15, `e[1]` scales weights 16..31).
 * - `Q4_0_ROCMFP4_FAST` — 17 bytes per 32 weights, one scale for the block.
 *
 * The two variants are selected by the runtime `dual_scale` flag, never by a
 * template parameter, so one kernel serves both. The flag is a uniform kernel
 * argument, so the branch is wave-uniform and free.
 *
 * ## Target ISA notes
 *
 * MFMA is CDNA-only and does not exist on RDNA2 (`gfx1031`) or RDNA4
 * (`gfx1201`); this kernel deliberately uses no matrix instruction at all, so
 * it compiles unchanged on both targets and on any future RDNA part. Wave size
 * differs between the targets (Wave64 on `gfx1031`, Wave32 on `gfx1201`), so
 * the workgroup reduction is a shared-memory tree rather than a
 * shuffle-based warp reduction.
 *
 * ## Numerical contract
 *
 * The *dequantized weight* produced by this kernel is bit-identical to
 * `rocminfer::rocmfp4_dequantize_row`: the Codebook10 code decode is exact in
 * integer arithmetic and the scale is looked up in a `__constant__` table that
 * is verified entry-by-entry against `rocminfer::rocmfp4_scale_to_float`
 * before the first launch. The *dot product* is not bit-identical to the CPU
 * oracle because the summations are associated differently and the compiler may
 * contract products into FMAs; the launcher therefore carries a hard bound
 * documented on `rocmfp4_dot_f32` and exercised by the test.
 */

#ifndef ROCMINFER_ROCMFP4_DOT_HPP
#define ROCMINFER_ROCMFP4_DOT_HPP

#include <cstddef>
#include <cstdint>

namespace rocminfer {

/**
 * @brief Forward declaration of the engine tracing recorder.
 *
 * Only the overload that accepts a recorder by reference needs the complete
 * type, and only `rocmfp4_dot.hip` includes `rocminfer/profiler.hpp`.
 */
class TraceRecorder;

/**
 * @brief Threads per workgroup in the fused ROCmFP4 dot kernel.
 *
 * One workgroup computes one output row. The value is a compile-time constant
 * because the shared-memory reduction buffer is statically sized.
 */
inline constexpr unsigned int kRocmFP4DotThreads = 256u;

/**
 * @brief Bytes of packed weight data that one row of a ROCmFP4 tensor occupies.
 * @param[in] cols       Weights per row; must be a non-zero multiple of 32.
 * @param[in] dual_scale true for the 18-byte dual-scale layout, false for the
 *                       17-byte FAST layout.
 * @return Exact packed byte length of the row, or 0 when `cols` is not a
 *         non-zero multiple of 32 or the product would overflow.
 * @pre None; invalid arguments are rejected by return value, not asserted.
 * @post The result equals `(cols / 32) * (dual_scale ? 18 : 17)` whenever that
 *       value is representable.
 * @note This is a host-side arithmetic helper. It performs no allocation and
 *       does not touch device memory, so it is safe to call from a boundary
 *       check.
 * @warning The returned length is the value the caller must use for
 *          `hipMalloc`; it is not validated against the tensor the caller
 *          actually owns, so a caller that lies about `cols` still gets an
 *          out-of-bounds read from the kernel.
 */
std::size_t rocmfp4_dot_row_bytes(std::int64_t cols, bool dual_scale) noexcept;

/**
 * @brief Bytes of packed weight data that a whole ROCmFP4 tensor occupies.
 * @param[in] rows       Output rows; must be positive.
 * @param[in] cols       Weights per row; must be a non-zero multiple of 32.
 * @param[in] dual_scale true for the 18-byte dual-scale layout, false for the
 *                       17-byte FAST layout.
 * @return Exact packed byte length of the tensor, or 0 when any argument is
 *         invalid or the product would overflow.
 * @pre None; invalid arguments are rejected by return value.
 * @post `0` is returned if and only if the caller's shape is unusable, so a
 *       caller can distinguish "no tensor" from "zero-length tensor".
 * @note Exists so callers size their device allocations from the same
 *       arithmetic the kernel uses, rather than from a second hand-written
 *       expression that can drift.
 * @warning `0` means the request is malformed; it never means "an empty but
 *          valid tensor", because the kernel rejects empty tensors too.
 */
std::size_t rocmfp4_dot_weight_bytes(std::int64_t rows, std::int64_t cols,
                                    bool dual_scale) noexcept;

/**
 * @brief Compute `y = dequant(W) * x` for every row of a packed ROCmFP4 tensor.
 *
 * The kernel is launched with one workgroup of `kRocmFP4DotThreads` threads per
 * row and a strided loop over that row's 32-weight blocks. Each thread reads its
 * block's sixteen packed nibble bytes with four 32-bit loads (funnel-shifted so
 * any 4-byte-aligned tensor address is accepted) and its sixteen activation
 * floats with four 128-bit loads, then reduces the workgroup partial sums
 * through a shared-memory tree.
 *
 * @param[out] out        Destination for `rows` FP32 dot products, device
 *                        memory, at least 4-byte aligned, caller-owned.
 * @param[in]  weights    Packed ROCmFP4 blocks, device memory, at least
 *                        4-byte aligned, exactly
 *                        `rocmfp4_dot_weight_bytes(rows, cols, dual_scale)`
 *                        bytes, laid out row-major with no padding.
 * @param[in]  x          Activation vector, device memory, 16-byte aligned,
 *                        exactly `cols` FP32 elements, read-only. This is a
 *                        matrix-vector product: every row of `weights` is
 *                        contracted against this one vector, so the buffer is
 *                        `cols` floats in total and not `rows * cols`.
 * @param[in]  rows       Number of output rows; must be positive.
 * @param[in]  cols       Weights per row; must be a positive multiple of 32.
 * @param[in]  dual_scale true for the 18-byte dual-scale layout (`Q4_0_ROCMFP4`),
 *                        false for the 17-byte FAST layout
 *                        (`Q4_0_ROCMFP4_FAST`).
 * @param[in]  stream     HIP stream to enqueue on, reinterpreted via
 *                        `static_cast<hipStream_t>`. Must be a pointer
 *                        obtained from the HIP runtime, or null for the default
 *                        stream.
 * @return 0 on success; `EINVAL` if any pointer is null, `rows` or `cols` is
 *         not positive, `cols` is not a multiple of 32, `out` is not 4-byte
 *         aligned, `weights` is not 4-byte aligned, `x` is not 16-byte aligned,
 *         or the byte extents overflow; `EILSEQ` if the device-side UE4M3 scale
 *         table disagrees with `rocminfer::rocmfp4_scale_to_float`; `EIO` if the
 *         scale-table upload or the kernel launch fails.
 * @pre `weights`, `x` and `out` are device pointers obtained from the HIP
 *      allocator, and a HIP context exists on the calling thread.
 * @post On success the launch is enqueued on `stream` and `out[0..rows)` holds
 *       the dot products *once the stream has synchronised*; this call performs
 *       no synchronisation. On any failure return the kernel is not enqueued and
 *       `out` is untouched.
 * @note The device-side scale table is uploaded exactly once per process, on
 *       first use, under `std::call_once`, and each of its 127 entries is
 *       compared against the CPU oracle before the upload. A mismatch yields
 *       `EILSEQ` on every subsequent call rather than silently decoding with a
 *       table that has drifted from the format definition.
 * @note The accumulated dot is subject to the usual FP32 summation error. With
 *       `N = cols` terms of magnitude bounded by `sumabs`, the result is
 *       provably within `(N + 1) * 2^-24 * sumabs` of the exact value, because
 *       every partial sum is bounded by `sumabs` and the kernel performs at most
 *       one rounding per term plus eight tree steps. `test_rocmfp4_dot.cpp`
 *       measures this bound rather than assuming it.
 * @warning `weights` is trusted. The kernel performs no per-block validation of
 *          scale bytes, because a validating pass would read the entire tensor
 *          twice. A scale byte above `ROCMFP4_SCALE_MAX` decodes to `0.0f`,
 *          matching `rocmfp4_scale_to_float`, but the CPU oracle rejects such a
 *          buffer with `EINVAL`; feed only buffers that already passed
 *          `rocminfer::rocmfp4_validate`.
 * @warning `stream` is cast with `static_cast<hipStream_t>` and is not checked.
 *          Passing a pointer that is not a `hipStream_t` is undefined behaviour.
 * @warning Not thread-safe with respect to concurrent calls that use the same
 *          `out` buffer; concurrent calls with distinct buffers are safe.
 */
int rocmfp4_dot_f32(float* out, const void* weights, const float* x,
                    std::int64_t rows, std::int64_t cols, bool dual_scale,
                    void* stream);

/**
 * @brief Traced wrapper around `rocmfp4_dot_f32`.
 *
 * Identical in every respect to the untraced entry point except that the
 * kernel launch is bracketed by a `TraceScope` in the `kDevice` domain on the
 * logical component `quant-dot`, so the launch's device execution time is
 * attributed separately from host work.
 *
 * @param[in]  recorder   Recorder to aggregate into; must outlive this call and
 *                        must not be destroyed while the scope is open.
 * @param[out] out        Destination for `rows` FP32 dot products, device memory.
 * @param[in]  weights    Packed ROCmFP4 blocks, device memory; see
 *                        `rocmfp4_dot_f32`.
 * @param[in]  x          Activation vector, device memory; see
 *                        `rocmfp4_dot_f32`.
 * @param[in]  rows       Number of output rows.
 * @param[in]  cols       Weights per row; positive multiple of 32.
 * @param[in]  dual_scale Layout selector; see `rocmfp4_dot_f32`.
 * @param[in]  stream     HIP stream echoed into the scope and the launch.
 * @return Exactly what `rocmfp4_dot_f32` returns; tracing cannot change the
 *         result.
 * @pre Every precondition of `rocmfp4_dot_f32`, plus `recorder.enabled()` may
 *      be false, in which case the scope records nothing.
 * @post On success the launch is enqueued and the scope has been closed. The
 *       device timing it records stays pending until the engine calls
 *       `TraceRecorder::resolve_pending`, exactly as for any other device scope.
 * @note The scope labels the *launch*, so `dev_ns` is GPU execution time
 *       measured with HIP events rather than host enqueue time.
 * @warning Per the engine tracing contract, two `quant-dot` scopes must not be
 *          nested at the same depth; the resulting aggregates would be
 *          ambiguous.
 */
int rocmfp4_dot_f32_traced(TraceRecorder& recorder, float* out, const void* weights,
                           const float* x, std::int64_t rows, std::int64_t cols,
                           bool dual_scale, void* stream);

/**
 * @brief Ensure the device UE4M3 scale table is loaded, exactly once per process.
 * @return 0 once the table is loaded and verified against the CPU oracle;
 *         otherwise the sticky status recorded by the first attempt, propagated
 *         to every later caller so a failure never silently retries-and-forgets.
 * @note Both the FMA kernel (`rocmfp4_dot_f32`) and the WMMA kernel
 *       (`rocmfp4_wmma_dot`) depend on this table; the WMMA launcher calls it
 *       directly because it may run before the FMA kernel has been invoked.
 * @pre A HIP context exists on the calling thread.
 * @post On success the `__constant__` symbol `c_rocmfp4_scale` holds 127 verified
 *       floats that all GPU kernels observe.
 */
int rocmfp4_ensure_scale_table() noexcept;

/**
 * @brief Decode a scale byte to its FP32 multiplier (host mirror of the device table).
 *
 * Exposed so launch paths that pass per-row scales as kernel arguments (rather
 * than reading the shared `__constant__` table) can resolve scale bytes without
 * duplicating the UE4M3 decode.
 * @param[in] e Scale byte.
 * @return Finite multiplier, or `0.0f` for the invalid/NaN byte `0x7F`.
 */
float rocmfp4_scale_to_float_host(std::uint8_t e) noexcept;

}  // namespace rocminfer

#endif  // ROCMINFER_ROCMFP4_DOT_HPP