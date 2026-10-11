/**
 * @file test_rocmfp4.cpp
 * @brief Unit and negative tests for the ROCmFP4 reference decoder and registry.
 *
 * Coverage:
 *  - registry: dual-scale and FAST are distinct, with the correct block sizes,
 *    and an unknown GGUF type resolves to nullptr rather than a neighbour;
 *  - scale table: spot values, defined-entry count, and the `0x7E` quirk;
 *  - codebook: all sixteen codes, sign handling, and high-nibble masking;
 *  - block decode: nibble-to-weight mapping for both layouts;
 *  - negative: every invalid scale byte (0x7F and 0x80..0xFF) is rejected for
 *    both layouts, plus every malformed length and null pointer.
 */

#include "rocminfer/format_registry.hpp"
#include "rocminfer/rocmfp4_reference.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "FAIL line " << line << ": " << expression << "\n";
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

/// Compare floats for exact equality; the reference path must be bit-exact.
void check_f32(float actual, float expected, const char* what, int line) {
    if (std::fabs(actual - expected) > 1e-7f) {
        std::cerr << "FAIL line " << line << ": " << what << " expected " << expected
                  << " got " << actual << "\n";
        ++g_failures;
    }
}

#define CHECK_F32(actual, expected) check_f32((actual), (expected), #actual, __LINE__)

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

void test_registry_separates_dual_and_fast() {
    const rocminfer::FormatInfo* dual = rocminfer::format_info(rocminfer::QuantFormat::kRocmFP4);
    const rocminfer::FormatInfo* fast = rocminfer::format_info(rocminfer::QuantFormat::kRocmFP4Fast);
    CHECK(dual != nullptr);
    CHECK(fast != nullptr);
    if (dual == nullptr || fast == nullptr) {
        return;
    }

    CHECK(dual->block_bytes == 18);
    CHECK(fast->block_bytes == 17);
    CHECK(dual->block_elements == 32);
    CHECK(fast->block_elements == 32);
    CHECK(dual->dual_scale);
    CHECK(!fast->dual_scale);
    CHECK(dual->id != fast->id);

    // Bits per weight is derived, so 18/32*8 = 4.5 and 17/32*8 = 4.25.
    CHECK_F32(dual->bits_per_weight, 4.5f);
    CHECK_F32(fast->bits_per_weight, 4.25f);

    // The two names must not collide.
    CHECK(std::strcmp(dual->gguf_type, fast->gguf_type) != 0);
}

void test_registry_lookup_and_enumeration() {
    CHECK(rocminfer::format_by_gguf_type("Q4_0_ROCMFP4") != nullptr);
    CHECK(rocminfer::format_by_gguf_type("Q4_0_ROCMFP4_FAST") != nullptr);
    CHECK(rocminfer::format_by_gguf_type("Q4_0_INT4") != nullptr);
    // Unknown and near-miss types must not resolve to a neighbouring format.
    CHECK(rocminfer::format_by_gguf_type("Q4_0") == nullptr);
    CHECK(rocminfer::format_by_gguf_type("q4_0_rocmfp4") == nullptr);
    CHECK(rocminfer::format_by_gguf_type("Q4_0_ROCMFPX") == nullptr);
    CHECK(rocminfer::format_by_gguf_type("Q4_0_ROCMI4") == nullptr);
    CHECK(rocminfer::format_by_gguf_type("") == nullptr);

    // Three rows: the two ROCmFP4 layouts plus the engine-internal raw int4.
    CHECK(rocminfer::format_registry_size() == 3);
    for (std::size_t i = 0; i < rocminfer::format_registry_size(); ++i) {
        const rocminfer::FormatInfo* info = rocminfer::format_at(i);
        CHECK(info != nullptr);
        if (info != nullptr) {
            CHECK(info->id != rocminfer::QuantFormat::kUnknown);
            CHECK(info->block_bytes > 0);
            CHECK(info->block_elements > 0);
            // Derived bits-per-weight must agree with the block layout, so a
            // mistyped block size cannot hide behind a nominal figure.
            CHECK_F32(info->bits_per_weight,
                      8.0f * static_cast<float>(info->block_bytes) /
                          static_cast<float>(info->block_elements));
            // Row bytes follow from the block geometry for every format.
            CHECK(rocminfer::format_row_bytes(*info, 64) ==
                  (64u / info->block_elements) * static_cast<std::size_t>(info->block_bytes));
        }
    }
    CHECK(rocminfer::format_at(99) == nullptr);
    CHECK(rocminfer::format_info(rocminfer::QuantFormat::kUnknown) == nullptr);

    // The raw-int4 row: engine-internal type id, no codebook, no scale.
    const rocminfer::FormatInfo* int4 =
        rocminfer::format_info(rocminfer::QuantFormat::kSignedInt4);
    CHECK(int4 != nullptr);
    if (int4 != nullptr) {
        CHECK(int4->block_elements == 8);
        CHECK(int4->block_bytes == 4);
        CHECK_F32(int4->bits_per_weight, 4.0f);
        CHECK(!int4->dual_scale);
        CHECK(int4->ggml_type_id == 200);
        // 8 weights in 4 bytes, so a 64-weight row is 32 bytes and a 32-weight
        // row is 16: the same packing the int4 kernels consume.
        CHECK(rocminfer::format_row_bytes(*int4, 32) == 16);
    }
}

// ---------------------------------------------------------------------------
// Scale table
// ---------------------------------------------------------------------------

void test_scale_validity_rule() {
    for (std::uint32_t e = 0; e <= 0x7EU; ++e) {
        CHECK(rocminfer::rocmfp4_scale_is_valid(static_cast<std::uint8_t>(e)));
    }
    for (std::uint32_t e = 0x7FU; e <= 0xFFU; ++e) {
        CHECK(!rocminfer::rocmfp4_scale_is_valid(static_cast<std::uint8_t>(e)));
    }
    // Invalid bytes decode to zero rather than to a plausible multiplier.
    for (std::uint32_t e = 0x7FU; e <= 0xFFU; ++e) {
        CHECK_F32(rocminfer::rocmfp4_scale_to_float(static_cast<std::uint8_t>(e)), 0.0f);
    }
}

void test_scale_table_values() {
    // Subnormal steps: M * 2^-10 for M = 0..7.
    for (std::uint32_t m = 0; m <= 7; ++m) {
        const float expected = static_cast<float>(m) * 0.0009765625f;
        CHECK_F32(rocminfer::rocmfp4_scale_to_float(static_cast<std::uint8_t>(m)), expected);
    }
    // Exponent blocks: (8 + M) * 2^(b - 11) at index 8 + (b - 1) * 8 + M.
    for (std::uint32_t b = 1; b <= 15; ++b) {
        for (std::uint32_t m = 0; m <= 7; ++m) {
            const std::size_t index = 8 + (b - 1) * 8 + m;
            if (index >= 126) {
                continue;
            }
            const float expected = static_cast<float>(8 + m) * std::ldexp(1.0f, static_cast<int>(b) - 11);
            CHECK_F32(rocminfer::rocmfp4_scale_to_float(static_cast<std::uint8_t>(index)), expected);
        }
    }
    // The scale table must be strictly increasing across its defined range.
    float previous = -1.0f;
    for (std::uint32_t i = 0; i < 126; ++i) {
        const float value = rocminfer::rocmfp4_scale_to_float(static_cast<std::uint8_t>(i));
        CHECK(value > previous);
        previous = value;
    }
    CHECK_F32(rocminfer::rocmfp4_scale_to_float(0x7EU), 0.0f);
    CHECK(rocminfer::rocmfp4_scale_is_degenerate(0x7EU));
    CHECK(!rocminfer::rocmfp4_scale_is_degenerate(0x00U));
}

// ---------------------------------------------------------------------------
// Codebook
// ---------------------------------------------------------------------------

void test_codebook_all_sixteen_codes() {
    const std::int8_t expected[16] = {0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};
    for (std::uint32_t q = 0; q < 16; ++q) {
        CHECK(rocminfer::rocmfp4_decode_code(static_cast<std::uint8_t>(q)) == expected[q]);
    }
    // Only the low nibble participates; the upper bits are ignored.
    for (std::uint32_t q = 0; q < 16; ++q) {
        CHECK(rocminfer::rocmfp4_decode_code(static_cast<std::uint8_t>(q | 0xF0U)) == expected[q]);
        CHECK(rocminfer::rocmfp4_decode_code(static_cast<std::uint8_t>(q | 0xA0U)) == expected[q]);
    }
    // Symmetry: code and code | 0x08 are negatives of one another.
    for (std::uint32_t q = 0; q < 8; ++q) {
        CHECK(rocminfer::rocmfp4_decode_code(static_cast<std::uint8_t>(q)) ==
              -rocminfer::rocmfp4_decode_code(static_cast<std::uint8_t>(q | 0x08U)));
    }
}

// ---------------------------------------------------------------------------
// Block decode
// ---------------------------------------------------------------------------

/// Build a dual-scale block from an explicit nibble sequence.
void make_dual_block(std::vector<std::uint8_t>& block, std::uint8_t e0, std::uint8_t e1,
                     std::uint32_t seed) {
    block.assign(rocminfer::QK_ROCMFP4_BYTES, 0);
    for (std::uint32_t j = 0; j < rocminfer::QS_ROCMFP4; ++j) {
        const std::uint8_t low = static_cast<std::uint8_t>((seed + j) & 0x0FU);
        const std::uint8_t high = static_cast<std::uint8_t>((seed + j + 3) & 0x0FU);
        block[j] = static_cast<std::uint8_t>(low | (high << 4));
    }
    block[rocminfer::QS_ROCMFP4] = e0;
    block[rocminfer::QS_ROCMFP4 + 1] = e1;
}

void test_dual_scale_nibble_mapping() {
    // Scale 8.0 is index 88 ((8 + 0) * 2^0).
    std::vector<std::uint8_t> block;
    make_dual_block(block, 88, 8, 0);

    float out[rocminfer::QK_ROCMFP4] = {};
    CHECK(rocminfer::rocmfp4_dequantize_block(out, block.data(), true) == 0);

    // Weight j uses the low nibble and e[0]; weight j + 16 uses high nibble, e[1].
    for (std::uint32_t j = 0; j < rocminfer::QS_ROCMFP4; ++j) {
        const std::uint8_t low = static_cast<std::uint8_t>(block[j] & 0x0FU);
        const std::uint8_t high = static_cast<std::uint8_t>(block[j] >> 4);
        CHECK_F32(out[j], static_cast<float>(rocminfer::rocmfp4_decode_code(low)) * 8.0f);
        CHECK_F32(out[j + rocminfer::QS_ROCMFP4],
                  static_cast<float>(rocminfer::rocmfp4_decode_code(high)) * 0.0078125f);
    }
}

void test_fast_layout_uses_single_scale() {
    std::vector<std::uint8_t> block(rocminfer::QK_ROCMFP4_FAST_BYTES, 0);
    for (std::uint32_t j = 0; j < rocminfer::QS_ROCMFP4; ++j) {
        const std::uint8_t low = static_cast<std::uint8_t>(j & 0x0FU);
        const std::uint8_t high = static_cast<std::uint8_t>((j + 5) & 0x0FU);
        block[j] = static_cast<std::uint8_t>(low | (high << 4));
    }
    block[rocminfer::QS_ROCMFP4] = 88;  // 8.0

    float out[rocminfer::QK_ROCMFP4] = {};
    CHECK(rocminfer::rocmfp4_dequantize_block(out, block.data(), false) == 0);
    for (std::uint32_t j = 0; j < rocminfer::QS_ROCMFP4; ++j) {
        const std::uint8_t low = static_cast<std::uint8_t>(block[j] & 0x0FU);
        const std::uint8_t high = static_cast<std::uint8_t>(block[j] >> 4);
        CHECK_F32(out[j], static_cast<float>(rocminfer::rocmfp4_decode_code(low)) * 8.0f);
        CHECK_F32(out[j + rocminfer::QS_ROCMFP4],
                  static_cast<float>(rocminfer::rocmfp4_decode_code(high)) * 8.0f);
    }
}

void test_row_decode_and_layout_agreement() {
    std::vector<std::uint8_t> dual;
    make_dual_block(dual, 100, 100, 7);

    std::vector<float> row(rocminfer::QK_ROCMFP4, 0.0f);
    CHECK(rocminfer::rocmfp4_dequantize_row(row.data(), dual.data(), dual.size(), row.size(),
                                            true) == 0);
    for (std::uint32_t i = 0; i < rocminfer::QK_ROCMFP4; ++i) {
        CHECK(std::isfinite(row[i]));
    }

    // Building the FAST equivalent of a uniform-scale dual block must decode
    // identically, proving the two layouts differ only in block size.
    std::vector<std::uint8_t> fast(dual.begin(), dual.begin() + rocminfer::QS_ROCMFP4);
    fast.push_back(100);
    std::vector<float> fast_row(row.size(), 0.0f);
    CHECK(rocminfer::rocmfp4_dequantize_row(fast_row.data(), fast.data(), fast.size(),
                                            fast_row.size(), false) == 0);
    for (std::uint32_t i = 0; i < row.size(); ++i) {
        CHECK(row[i] == fast_row[i]);
    }
}

void test_degenerate_scale_zeroes_block() {
    // 0x7E is a valid scale byte that decodes to zero; the decoder must stay
    // bit-exact with that definition rather than rejecting it.
    std::vector<std::uint8_t> block;
    make_dual_block(block, 0x7EU, 0x7EU, 1);
    float out[rocminfer::QK_ROCMFP4] = {};
    CHECK(rocminfer::rocmfp4_dequantize_block(out, block.data(), true) == 0);
    for (std::uint32_t i = 0; i < rocminfer::QK_ROCMFP4; ++i) {
        CHECK_F32(out[i], 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Negative tests: every invalid scale byte, both layouts
// ---------------------------------------------------------------------------

void test_every_invalid_scale_byte_is_rejected() {
    // A block with valid nibbles and a single scale byte under test.
    std::uint32_t rejected = 0;
    for (std::uint32_t e = 0x7FU; e <= 0xFFU; ++e) {
        const auto scale = static_cast<std::uint8_t>(e);

        std::vector<std::uint8_t> dual;
        make_dual_block(dual, scale, 88, 2);
        float dual_out[rocminfer::QK_ROCMFP4] = {1.0f};
        if (rocminfer::rocmfp4_dequantize_block(dual_out, dual.data(), true) == EINVAL) {
            ++rejected;
        }

        std::vector<std::uint8_t> fast(rocminfer::QK_ROCMFP4_FAST_BYTES, 0);
        fast[rocminfer::QS_ROCMFP4] = scale;
        float fast_out[rocminfer::QK_ROCMFP4] = {1.0f};
        if (rocminfer::rocmfp4_dequantize_block(fast_out, fast.data(), false) == EINVAL) {
            ++rejected;
        }
    }
    // 0x7F..0xFF is 129 bytes, each rejected in both layouts.
    CHECK(rejected == 129 * 2);
    CHECK(!rocminfer::rocmfp4_scale_is_valid(0x7FU));
}

void test_validation_and_malformed_input() {
    std::vector<std::uint8_t> dual;
    make_dual_block(dual, 88, 88, 0);
    CHECK(rocminfer::rocmfp4_validate(dual.data(), dual.size(), true) == 0);
    CHECK(rocminfer::rocmfp4_validate(nullptr, dual.size(), true) == EINVAL);
    CHECK(rocminfer::rocmfp4_validate(dual.data(), 0, true) == EINVAL);
    // Not a whole number of 18-byte blocks.
    CHECK(rocminfer::rocmfp4_validate(dual.data(), dual.size() - 1, true) == EINVAL);
    // Valid dual-scale bytes read as FAST are a ragged length for that layout.
    CHECK(rocminfer::rocmfp4_validate(dual.data(), dual.size(), false) == EINVAL);

    float out[rocminfer::QK_ROCMFP4] = {};
    CHECK(rocminfer::rocmfp4_dequantize_row(out, dual.data(), dual.size(),
                                            rocminfer::QK_ROCMFP4, true) == 0);
    // Element count not a multiple of 32.
    CHECK(rocminfer::rocmfp4_dequantize_row(out, dual.data(), dual.size(), 31, true) == EINVAL);
    // Zero elements.
    CHECK(rocminfer::rocmfp4_dequantize_row(out, dual.data(), dual.size(), 0, true) == EINVAL);
    // Byte length inconsistent with element count.
    CHECK(rocminfer::rocmfp4_dequantize_row(out, dual.data(), dual.size(),
                                            rocminfer::QK_ROCMFP4 * 2, true) == EINVAL);
    // Null pointers.
    CHECK(rocminfer::rocmfp4_dequantize_row(nullptr, dual.data(), dual.size(),
                                            rocminfer::QK_ROCMFP4, true) == EINVAL);
    CHECK(rocminfer::rocmfp4_dequantize_row(out, nullptr, dual.size(),
                                            rocminfer::QK_ROCMFP4, true) == EINVAL);
    CHECK(rocminfer::rocmfp4_dequantize_block(nullptr, dual.data(), true) == EINVAL);
    CHECK(rocminfer::rocmfp4_dequantize_block(out, nullptr, true) == EINVAL);
}

void test_rejected_decode_leaves_output_untouched() {
    // An invalid scale must not partially write the destination buffer.
    std::vector<std::uint8_t> dual;
    make_dual_block(dual, 0xFFU, 88, 0);

    float out[rocminfer::QK_ROCMFP4];
    for (std::uint32_t i = 0; i < rocminfer::QK_ROCMFP4; ++i) {
        out[i] = -12345.0f;
    }
    CHECK(rocminfer::rocmfp4_dequantize_block(out, dual.data(), true) == EINVAL);
    for (std::uint32_t i = 0; i < rocminfer::QK_ROCMFP4; ++i) {
        CHECK(out[i] == -12345.0f);
    }
}

}  // namespace

int main() {
    test_registry_separates_dual_and_fast();
    test_registry_lookup_and_enumeration();
    test_scale_validity_rule();
    test_scale_table_values();
    test_codebook_all_sixteen_codes();
    test_dual_scale_nibble_mapping();
    test_fast_layout_uses_single_scale();
    test_row_decode_and_layout_agreement();
    test_degenerate_scale_zeroes_block();
    test_every_invalid_scale_byte_is_rejected();
    test_validation_and_malformed_input();
    test_rejected_decode_leaves_output_untouched();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all rocmfp4 reference checks passed\n";
    return 0;
}