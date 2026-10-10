/**
 * @file rocmfp4_reference.hpp
 * @brief Deterministic CPU reference decode for the ROCmFP4 weight family.
 *
 * Re-derived from the ROCmFP4 format definition pinned in
 * `THIRD_PARTY_SOURCES.md` (commit fb08d7c). This is the oracle the GPU kernels
 * in Phase 2 are measured against, so it favours obvious, auditable code over
 * speed and performs no allocation.
 *
 * ## Format summary
 *
 * A block packs 32 weights as 16 bytes of packed nibbles plus UE4M3 scale
 * bytes. Nibble `qs[j] & 0x0F` carries weight `j`; nibble `qs[j] >> 4` carries
 * weight `j + 16`.
 *
 * - `Q4_0_ROCMFP4`      : 18 bytes, scales `e[0]` for weights 0..15 and `e[1]`
 *                         for weights 16..31 (dual scale).
 * - `Q4_0_ROCMFP4_FAST` : 17 bytes, a single scale `e` for the whole block.
 *
 * ### Scale bytes
 *
 * Scale bytes are *unsigned finite* E4M3. Byte `0x7F` is the unsigned NaN
 * encoding and any byte with the sign bit set is not a valid scale, so validity
 * is `e <= 0x7E`.
 *
 * Valid bytes decode as: seven subnormal steps `M * 2^-10` for `M` in 0..7,
 * then for exponent `b` in 1..15 and mantissa `M` in 0..7, `(8 + M) * 2^(b-11)`
 * at table index `8 + (b - 1) * 8 + M`.
 *
 * @warning Byte `0x7E` is accepted by the validity rule but has no table entry
 *          and decodes to `0.0f`, which zeroes the block. See
 *          `rocmfp4_scale_is_degenerate` before treating a valid scale as usable.
 */

#ifndef ROCMINFER_ROCMFP4_REFERENCE_HPP
#define ROCMINFER_ROCMFP4_REFERENCE_HPP

#include <cstddef>
#include <cstdint>

namespace rocminfer {

/**
 * @brief Weights represented by one ROCmFP4 block.
 */
inline constexpr std::uint32_t QK_ROCMFP4 = 32;

/// Packed nibble bytes per block; each byte holds two weights.
inline constexpr std::uint32_t QS_ROCMFP4 = 16;

/// Packed block size in bytes for the dual-scale layout.
inline constexpr std::uint32_t QK_ROCMFP4_BYTES = 18;

/// Packed block size in bytes for the single-scale FAST layout.
inline constexpr std::uint32_t QK_ROCMFP4_FAST_BYTES = 17;

/// Largest valid scale byte; `0x7F` is the unsigned NaN encoding.
inline constexpr std::uint8_t ROCMFP4_SCALE_MAX = 0x7EU;

/**
 * @brief Test whether a scale byte is a valid ROCmFP4 scale.
 * @param[in] e Scale byte in any of 256 possible values.
 * @return true when `e <= 0x7E`, false for `0x7F` and for any set sign bit.
 * @note Valid does not imply usable: `0x7E` is valid yet decodes to zero.
 */
constexpr bool rocmfp4_scale_is_valid(std::uint8_t e) noexcept { return e <= ROCMFP4_SCALE_MAX; }

/**
 * @brief Test whether a valid scale byte decodes to zero and would zero a block.
 * @param[in] e Scale byte.
 * @return true only for `0x7E`, which is valid but has no table entry.
 * @note A block carrying this scale is readable but numerically empty. The
 *       reference decoder reproduces the behaviour rather than rejecting it, so
 *       it stays bit-exact with the pinned encoder; callers that need to reject
 *       degenerate blocks must call this predicate.
 */
constexpr bool rocmfp4_scale_is_degenerate(std::uint8_t e) noexcept {
    return e == ROCMFP4_SCALE_MAX;
}

/**
 * @brief Decode a packed 4-bit code to its integer weight.
 * @param[in] q Packed code; only the low nibble is used.
 * @return Weight in -10..10 drawn from the Codebook10 set
 *         `0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10`.
 * @note Bit 3 is the sign and bits 0..2 hold a three-bit magnitude code remapped
 *       as `M <= 4 ? M : 2*M - 4`.
 */
constexpr std::int8_t rocmfp4_decode_code(std::uint8_t q) noexcept {
    const std::uint8_t code = static_cast<std::uint8_t>(q & 0x0FU);
    const std::uint8_t mag3 = static_cast<std::uint8_t>(code & 0x07U);
    const std::int32_t mag = (mag3 <= 4U) ? static_cast<std::int32_t>(mag3)
                                           : (2 * static_cast<std::int32_t>(mag3) - 4);
    return (code & 0x08U) ? static_cast<std::int8_t>(-mag) : static_cast<std::int8_t>(mag);
}

/**
 * @brief Decode a scale byte to its FP32 multiplier.
 * @param[in] e Scale byte.
 * @return Finite multiplier, or `0.0f` for `0x7F` and any invalid byte.
 * @note Returns `0.0f` for invalid bytes rather than signalling; use
 *       `rocmfp4_scale_is_valid` to reject them explicitly.
 */
float rocmfp4_scale_to_float(std::uint8_t e) noexcept;

/**
 * @brief Validate a packed ROCmFP4 buffer without decoding it.
 * @param[in] data        Packed blocks, non-NULL.
 * @param[in] bytes       Byte length of `data`.
 * @param[in] dual_scale  true for the 18-byte dual-scale layout, false for the
 *                        17-byte FAST layout.
 * @return 0 on success, EINVAL when `data` is NULL, `bytes` is zero, `bytes` is
 *         not a whole number of blocks, or any scale byte is invalid.
 * @pre `bytes` must be the true length of the buffer; the caller cannot detect
 *      an over-reported length, so lengths originate from a trusted header.
 * @post On failure no writes occur and the reason is distinguishable by the
 *       return code alone; the first offending block index is not reported, so
 *       a large buffer may be expensive to diagnose and callers should bisect
 *       when a precise location is required.
 */
int rocmfp4_validate(const void* data, std::size_t bytes, bool dual_scale) noexcept;

/**
 * @brief Decode a packed ROCmFP4 row to FP32.
 * @param[out] out        Destination weights, caller-owned, at least `elements` floats.
 * @param[in]  data       Packed blocks, non-NULL.
 * @param[in]  bytes      Byte length of `data`, a whole number of blocks.
 * @param[in]  elements   Weight count, a non-zero multiple of `QK_ROCMFP4`.
 * @param[in]  dual_scale true for the dual-scale layout, false for FAST.
 * @return 0 on success, EINVAL on any violated precondition.
 * @pre `bytes` is exactly `(elements / 32) * (dual_scale ? 18 : 17)` bytes.
 * @post On success `out` holds `elements` finite weights; on failure `out` is
 *       left untouched so a rejected tensor cannot be partially consumed.
 * @note Invalid scale bytes are rejected outright rather than treated as zero,
 *       because a silent zero scale is indistinguishable from real data.
 */
int rocmfp4_dequantize_row(float* out, const void* data, std::size_t bytes,
                           std::size_t elements, bool dual_scale) noexcept;

/**
 * @brief Decode a single packed block to FP32.
 * @param[out] out        Destination for exactly `QK_ROCMFP4` floats, caller-owned.
 * @param[in]  block      Packed block bytes, non-NULL.
 * @param[in]  dual_scale Layout selector; see `rocmfp4_dequantize_row`.
 * @return 0 on success, EINVAL on any violated precondition.
 * @post On success `out` holds 32 finite weights; on failure `out` is untouched.
 */
int rocmfp4_dequantize_block(float* out, const void* block, bool dual_scale) noexcept;

}  // namespace rocminfer

#endif  // ROCMINFER_ROCMFP4_REFERENCE_HPP