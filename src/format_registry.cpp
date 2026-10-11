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
 *
 * @note `kSignedInt4` is the one entry that is not a ROCmFPX format. ggml has
 *       no raw-int4 type, so this row is ROCMinfer's own engine-internal
 *       format with an engine-internal type id; see `GgmlType::kQ4_0Int4` for
 *       the id survey and for why 200 cannot collide with a ggml type. Its
 *       block layout is re-derived from the engine's int4 kernels rather than
 *       from any upstream source, so the "no porting" rule is not touched.
 */

#include "rocminfer/format_registry.hpp"

#include <array>

namespace rocminfer {

namespace {

constexpr std::uint32_t kQkRocmFP4 = 32;

/// Weights per packed int4 block: eight nibbles in four bytes.
constexpr std::uint32_t kQkInt4 = 8;

/// Bytes per packed int4 block: two four-bit weights per byte, no overhead.
constexpr std::uint32_t kInt4BlockBytes = 4;

/// Nominal bits per weight implied by a packed block layout.
constexpr float bits_per_weight(std::uint32_t block_bytes, std::uint32_t elements) noexcept {
    return (8.0f * static_cast<float>(block_bytes)) / static_cast<float>(elements);
}

/// Registry order defines the enumeration order of `format_at`.
constexpr std::array<FormatInfo, 3> kRegistry = {{
    {QuantFormat::kRocmFP4, "Q4_0_ROCMFP4", "Q4_0_ROCMFP4",
     static_cast<std::uint32_t>(GgmlType::kQ4_0RocmFP4), 18, kQkRocmFP4,
     bits_per_weight(18, kQkRocmFP4), true},
    {QuantFormat::kRocmFP4Fast, "Q4_0_ROCMFP4_FAST", "Q4_0_ROCMFP4_FAST",
     static_cast<std::uint32_t>(GgmlType::kQ4_0RocmFP4Fast), 17, kQkRocmFP4,
     bits_per_weight(17, kQkRocmFP4), false},
    {QuantFormat::kSignedInt4, "Q4_0_INT4", "Q4_0_INT4",
     static_cast<std::uint32_t>(GgmlType::kQ4_0Int4), kInt4BlockBytes, kQkInt4,
     bits_per_weight(kInt4BlockBytes, kQkInt4), false},
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

const FormatInfo* format_by_ggml_type(std::uint32_t type) noexcept {
    for (const FormatInfo& entry : kRegistry) {
        if (entry.ggml_type_id == type) {
            return &entry;
        }
    }
    return nullptr;
}

std::size_t format_row_bytes(const FormatInfo& info, std::size_t elements) noexcept {
    if (info.block_elements == 0) {
        return 0;
    }
    return (elements / info.block_elements) * info.block_bytes;
}

}  // namespace rocminfer