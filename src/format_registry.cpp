/**
 * @file format_registry.cpp
 * @brief Registry table mapping format identifiers to their exact layout.
 *
 * The numbers here are re-derived from the ROCmFP4 format definition pinned in
 * `THIRD_PARTY_SOURCES.md` (commit fb08d7c). Bits-per-weight is computed from
 * the block size rather than hard-coded, so a layout edit cannot leave a stale
 * figure behind.
 *
 * @note No ROCmFPX code is copied or transcribed here; only the format
 *       description is reproduced, and it is independently checked by the
 *       reference tests in `tests/test_rocmfp4.cpp`.
 */

#include "rocminfer/format_registry.hpp"

#include <array>

namespace rocminfer {

namespace {

constexpr std::uint32_t kQkRocmFP4 = 32;

/// Nominal bits per weight implied by a packed block layout.
constexpr float bits_per_weight(std::uint32_t block_bytes, std::uint32_t elements) noexcept {
    return (8.0f * static_cast<float>(block_bytes)) / static_cast<float>(elements);
}

/// Registry order defines the enumeration order of `format_at`.
constexpr std::array<FormatInfo, 2> kRegistry = {{
    {QuantFormat::kRocmFP4, "Q4_0_ROCMFP4", "Q4_0_ROCMFP4", 18, kQkRocmFP4,
     bits_per_weight(18, kQkRocmFP4), true},
    {QuantFormat::kRocmFP4Fast, "Q4_0_ROCMFP4_FAST", "Q4_0_ROCMFP4_FAST", 17, kQkRocmFP4,
     bits_per_weight(17, kQkRocmFP4), false},
}};

}  // namespace

const FormatInfo* format_info(QuantFormat id) noexcept {
    for (const FormatInfo& entry : kRegistry) {
        if (entry.id == id) {
            return &entry;
        }
    }
    return nullptr;
}

const FormatInfo* format_by_gguf_type(std::string_view gguf_type) noexcept {
    for (const FormatInfo& entry : kRegistry) {
        if (gguf_type == entry.gguf_type) {
            return &entry;
        }
    }
    return nullptr;
}

std::size_t format_registry_size() noexcept { return kRegistry.size(); }

const FormatInfo* format_at(std::size_t index) noexcept {
    return index < kRegistry.size() ? &kRegistry[index] : nullptr;
}

std::size_t format_row_bytes(const FormatInfo& info, std::size_t elements) noexcept {
    if (info.block_elements == 0) {
        return 0;
    }
    return (elements / info.block_elements) * info.block_bytes;
}

}  // namespace rocminfer