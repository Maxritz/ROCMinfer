/**
 * @file rocmfp4_reference.cpp
 * @brief Deterministic CPU reference decode for the ROCmFP4 weight family.
 *
 * The UE4M3 scale table is generated from the format definition at compile time
 * rather than transcribed as a literal list, so a miscounted entry cannot hide
 * behind a copied table. `kDefinedScaleEntries` records how many indices carry a
 * defined value and the tests assert both that count and the values themselves.
 */

#include "rocminfer/rocmfp4_reference.hpp"

#include <array>
#include <cerrno>
#include <bit>

namespace rocminfer {

namespace {

/// Subnormal steps occupy the first seven table indices.
constexpr std::uint32_t kSubnormalEntries = 7;

/// Largest valid scale byte plus one, i.e. the table length including index 0x7E.
constexpr std::size_t kScaleTableSize = 127;

/// Indices below this carry a defined value; `0x7E` is valid yet undefined.
constexpr std::size_t kDefinedScaleEntries = 126;

/// Highest exponent in the finite unsigned E4M3 encoding.
constexpr std::uint32_t kMaxExponent = 15;

/**
 * @brief Compute `2^exponent` as an exact float.
 * @param[in] exponent Power of two, in -10..4 for this format.
 * @return Exactly representable power of two.
 * @note Uses the IEEE-754 bit pattern rather than `std::pow` so the result is
 *       constexpr and exactly representable with no rounding.
 */
constexpr float pow2_exact(int exponent) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(exponent + 127) << 23;
    return std::bit_cast<float>(bits);
}

/**
 * @brief Generate the UE4M3 scale table for the ROCmFP4 half-scale domain.
 * @return Table of 127 floats; indices 0..125 defined, index 126 left at zero.
 */
consteval std::array<float, kScaleTableSize> build_scale_table() noexcept {
    std::array<float, kScaleTableSize> table{};

    for (std::uint32_t mantissa = 0; mantissa <= kSubnormalEntries; ++mantissa) {
        table[mantissa] = static_cast<float>(mantissa) * pow2_exact(-10);
    }

    for (std::uint32_t exponent = 1; exponent <= kMaxExponent; ++exponent) {
        for (std::uint32_t mantissa = 0; mantissa < 8; ++mantissa) {
            const std::size_t index = 8 + (exponent - 1) * 8 + mantissa;
            if (index >= kDefinedScaleEntries) {
                continue;
            }
            table[index] = static_cast<float>(8 + mantissa) * pow2_exact(static_cast<int>(exponent) - 11);
        }
    }

    return table;
}

constexpr std::array<float, kScaleTableSize> kScaleTable = build_scale_table();

static_assert(kScaleTable[0] == 0.0f, "scale index 0 must be zero");
static_assert(kScaleTable[1] == 0.5f / 512.0f, "subnormal step must be 2^-10");
static_assert(kScaleTable[8] == 8.0f * pow2_exact(-10), "exponent 1 mantissa 0");
static_assert(kScaleTable[15] == 15.0f * pow2_exact(-10), "exponent 1 mantissa 7");
static_assert(kScaleTable[120] == 8.0f * pow2_exact(4), "exponent 15 mantissa 0");
static_assert(kScaleTable[125] == 13.0f * pow2_exact(4), "exponent 15 mantissa 5");
static_assert(kScaleTable[126] == 0.0f, "index 0x7E is valid but undefined");

/// Block size in bytes for a layout.
constexpr std::size_t block_bytes_for(bool dual_scale) noexcept {
    return dual_scale ? QK_ROCMFP4_BYTES : QK_ROCMFP4_FAST_BYTES;
}

/// Validate that a byte length is a whole number of blocks.
constexpr bool is_whole_blocks(std::size_t bytes, std::size_t block_bytes) noexcept {
    return bytes != 0 && (bytes % block_bytes) == 0;
}

/// Read the scale byte for weight half `half` of a block, dual-scale layout.
constexpr std::uint8_t dual_scale_byte(const std::uint8_t* block, std::uint32_t half) noexcept {
    return block[QS_ROCMFP4 + half];
}

}  // namespace

float rocmfp4_scale_to_float(std::uint8_t e) noexcept {
    return rocmfp4_scale_is_valid(e) ? kScaleTable[e] : 0.0f;
}

int rocmfp4_validate(const void* data, std::size_t bytes, bool dual_scale) noexcept {
    if (data == nullptr) {
        return EINVAL;
    }

    const std::size_t block_bytes = block_bytes_for(dual_scale);
    if (!is_whole_blocks(bytes, block_bytes)) {
        return EINVAL;
    }

    const std::size_t block_count = bytes / block_bytes;
    for (std::size_t i = 0; i < block_count; ++i) {
        const auto* block = static_cast<const std::uint8_t*>(data) + i * block_bytes;
        if (dual_scale) {
            if (!rocmfp4_scale_is_valid(dual_scale_byte(block, 0)) ||
                !rocmfp4_scale_is_valid(dual_scale_byte(block, 1))) {
                return EINVAL;
            }
        } else {
            if (!rocmfp4_scale_is_valid(block[QS_ROCMFP4])) {
                return EINVAL;
            }
        }
    }

    return 0;
}

int rocmfp4_dequantize_block(float* out, const void* block, bool dual_scale) noexcept {
    if (out == nullptr || block == nullptr) {
        return EINVAL;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(block);

    if (dual_scale) {
        const std::uint8_t e0 = dual_scale_byte(bytes, 0);
        const std::uint8_t e1 = dual_scale_byte(bytes, 1);
        if (!rocmfp4_scale_is_valid(e0) || !rocmfp4_scale_is_valid(e1)) {
            return EINVAL;
        }
        const float d0 = rocmfp4_scale_to_float(e0);
        const float d1 = rocmfp4_scale_to_float(e1);
        for (std::uint32_t j = 0; j < QS_ROCMFP4; ++j) {
            out[j] = static_cast<float>(rocmfp4_decode_code(bytes[j] & 0x0FU)) * d0;
            out[j + QS_ROCMFP4] = static_cast<float>(rocmfp4_decode_code(bytes[j] >> 4)) * d1;
        }
        return 0;
    }

    const std::uint8_t e = bytes[QS_ROCMFP4];
    if (!rocmfp4_scale_is_valid(e)) {
        return EINVAL;
    }
    const float d = rocmfp4_scale_to_float(e);
    for (std::uint32_t j = 0; j < QS_ROCMFP4; ++j) {
        out[j] = static_cast<float>(rocmfp4_decode_code(bytes[j] & 0x0FU)) * d;
        out[j + QS_ROCMFP4] = static_cast<float>(rocmfp4_decode_code(bytes[j] >> 4)) * d;
    }
    return 0;
}

int rocmfp4_dequantize_row(float* out, const void* data, std::size_t bytes,
                           std::size_t elements, bool dual_scale) noexcept {
    if (out == nullptr || data == nullptr) {
        return EINVAL;
    }
    if (elements == 0 || (elements % QK_ROCMFP4) != 0) {
        return EINVAL;
    }

    const std::size_t block_bytes = block_bytes_for(dual_scale);
    const std::size_t expected = (elements / QK_ROCMFP4) * block_bytes;
    if (bytes != expected) {
        return EINVAL;
    }

    const std::size_t block_count = elements / QK_ROCMFP4;
    for (std::size_t i = 0; i < block_count; ++i) {
        const int status = rocmfp4_dequantize_block(out + i * QK_ROCMFP4,
                                                    static_cast<const std::uint8_t*>(data) +
                                                        i * block_bytes,
                                                    dual_scale);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}

}  // namespace rocminfer