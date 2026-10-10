/**
 * @file test_rocmfp4_payload.cpp
 * @brief Decodes real ROCmFP4 tensor payloads and proves the layouts are distinct.
 *
 * These tests run against actual model artifacts, so they prove the reference
 * decoder against upstream-produced bytes rather than against synthetic input
 * this project generated itself. The negative tests are the point: they show
 * that selecting the wrong layout, or a single invalid scale byte, is detected
 * rather than silently producing plausible wrong weights.
 *
 * Model paths come from ROCMINFER_TEST_MODEL_A / _B, defaulting to known-good
 * files on the development machine. When absent the suite SKIPS and still exits
 * zero, so CI without model files stays green.
 */

#include "rocminfer/format_registry.hpp"
#include "rocminfer/gguf_reader.hpp"
#include "rocminfer/rocmfp4_reference.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Portable C library calls are deprecated on the MSVC ABI but required for one
// source to build under both the MSVC-ABI clang toolchain and MinGW.
#if defined(_MSC_VER)
#define ROCMINFER_DISABLE_CRT_WARNINGS_BEGIN \
    _Pragma("warning(push)") _Pragma("warning(disable : 4996)")
#define ROCMINFER_DISABLE_CRT_WARNINGS_END _Pragma("warning(pop)")
#else
#define ROCMINFER_DISABLE_CRT_WARNINGS_BEGIN
#define ROCMINFER_DISABLE_CRT_WARNINGS_END
#endif

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "FAIL line " << line << ": " << expression << "\n";
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

/// Read an environment variable, falling back to a compiled-in default.
std::string model_path(char which) {
    const char* variable = (which == 'A') ? "ROCMINFER_TEST_MODEL_A" : "ROCMINFER_TEST_MODEL_B";
    ROCMINFER_DISABLE_CRT_WARNINGS_BEGIN
    const char* value = std::getenv(variable);
    ROCMINFER_DISABLE_CRT_WARNINGS_END
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    if (which == 'A') {
        return "G:\\More-models\\gemma-4-E2B-it-Q4_0_ROCMFP4_COHERENT.gguf";
    }
    return "G:\\More-models\\Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf";
}

/// Largest scale multiplier across the scale bytes of `count` blocks.
float max_scale_over(const std::vector<std::uint8_t>& payload, std::size_t count,
                     const rocminfer::FormatInfo& format) {
    const std::uint32_t block_bytes = format.block_bytes;
    float best = 0.0f;
    for (std::size_t b = 0; b < count; ++b) {
        const std::uint8_t* block = payload.data() + b * block_bytes;
        // ROCmFP4 packs nibbles first; scale bytes follow at QS_ROCMFP4.
        const float s0 = rocminfer::rocmfp4_scale_to_float(block[rocminfer::QS_ROCMFP4]);
        best = std::max(best, s0);
        if (format.dual_scale) {
            const float s1 =
                rocminfer::rocmfp4_scale_to_float(block[rocminfer::QS_ROCMFP4 + 1]);
            best = std::max(best, s1);
        }
    }
    return best;
}

/// Find the first tensor of a registered ROCmFP4 type.
const rocminfer::GgufTensorInfo* first_rocmfp4_tensor(const rocminfer::GgufFile& file,
                                                      const rocminfer::FormatInfo** out_format) {
    for (const rocminfer::GgufTensorInfo& info : file.tensors) {
        if (const rocminfer::FormatInfo* format =
                rocminfer::format_by_ggml_type(info.type_id)) {
            *out_format = format;
            return &info;
        }
    }
    return nullptr;
}

/// Decode a bounded prefix of a real tensor and sanity-check the values.
void decode_prefix(const rocminfer::FormatInfo& format,
                   const std::vector<std::uint8_t>& payload) {
    const std::size_t block_bytes = format.block_bytes;
    const std::size_t blocks_total = payload.size() / block_bytes;
    const std::size_t blocks = std::min<std::size_t>(blocks_total, 4096);
    CHECK(blocks > 0);

    std::vector<float> out(blocks * rocminfer::QK_ROCMFP4, 0.0f);
    CHECK(rocminfer::rocmfp4_dequantize_row(out.data(), payload.data(),
                                            blocks * block_bytes,
                                            blocks * rocminfer::QK_ROCMFP4,
                                            format.dual_scale) == 0);

    const float max_scale = max_scale_over(payload, blocks, format);
    CHECK(max_scale > 0.0f);

    bool any_nonzero = false;
    for (float value : out) {
        CHECK(std::isfinite(value));
        // Codebook10 max magnitude is 10, so no weight can exceed 10*scale.
        CHECK(std::fabs(value) <= 10.0f * max_scale * 1.001f);
        if (value != 0.0f) {
            any_nonzero = true;
        }
    }
    CHECK(any_nonzero);
}

/// Decoding real bytes with the wrong layout must not silently succeed.
void wrong_layout_is_detectable(const rocminfer::FormatInfo& format,
                                const std::vector<std::uint8_t>& payload) {
    const std::size_t block_bytes = format.block_bytes;
    const std::size_t blocks = std::min<std::size_t>(payload.size() / block_bytes, 512);
    if (blocks == 0) {
        return;
    }
    const std::size_t bytes = blocks * block_bytes;
    const std::size_t elements = blocks * rocminfer::QK_ROCMFP4;

    std::vector<float> correct(elements, 0.0f);
    std::vector<float> wrong(elements, 0.0f);

    const int rc_correct =
        rocminfer::rocmfp4_dequantize_row(correct.data(), payload.data(), bytes, elements,
                                          format.dual_scale);
    // Flip the layout: dual bytes read as FAST, or vice versa.
    const int rc_wrong =
        rocminfer::rocmfp4_dequantize_row(wrong.data(), payload.data(), bytes, elements,
                                          !format.dual_scale);

    bool detectable = false;
    if (rc_wrong != 0) {
        // Byte length no longer matches the layout, so it refused outright.
        detectable = true;
    } else if (rc_correct == 0) {
        detectable = (correct != wrong);
    }
    CHECK(detectable);

    // The correct decode must always succeed for a well-formed tensor.
    CHECK(rc_correct == 0);
}

/// Corrupting a real scale byte must be rejected, not decoded.
void corrupt_scale_is_rejected(const rocminfer::FormatInfo& format,
                               const std::vector<std::uint8_t>& payload) {
    if (payload.size() < format.block_bytes) {
        return;
    }
    std::vector<std::uint8_t> block(payload.begin(),
                                    payload.begin() + format.block_bytes);

    for (std::uint32_t bad : {0x7FU, 0x80U, 0xFFU}) {
        // Scale bytes live after the 16 packed-nibble bytes.
        block[rocminfer::QS_ROCMFP4] = static_cast<std::uint8_t>(bad);
        CHECK(rocminfer::rocmfp4_validate(block.data(), block.size(), format.dual_scale) ==
              EINVAL);

        std::vector<float> out(rocminfer::QK_ROCMFP4, 0.0f);
        CHECK(rocminfer::rocmfp4_dequantize_row(out.data(), block.data(), block.size(),
                                                rocminfer::QK_ROCMFP4, format.dual_scale) ==
              EINVAL);
        // The rejection must not have written anything.
        CHECK(out[0] == 0.0f);
    }
}

void test_model(const char* label, const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::cout << label << ": SKIPPED (not present: " << path << ")\n";
        return;
    }

    rocminfer::GgufReader reader;
    const int open_status = reader.open(path.c_str());
    if (open_status != 0) {
        std::cerr << "FAIL open '" << path << "' status=" << open_status << "\n";
        ++g_failures;
        return;
    }

    // Tensor extents must all fit inside the real file size.
    CHECK(rocminfer::gguf_validate_tensors(reader.manifest_mutable(), 0, reader.file_size()) == 0);

    std::map<std::string, std::size_t> census;
    for (const rocminfer::GgufTensorInfo& info : reader.manifest().tensors) {
        const rocminfer::FormatInfo* format = rocminfer::format_by_ggml_type(info.type_id);
        census[format != nullptr ? format->name : "other"] += 1;
    }
    std::cout << label << ": " << reader.manifest().tensors.size() << " tensors, file "
              << reader.file_size() << " bytes\n";
    for (const auto& entry : census) {
        std::cout << "    " << entry.first << ": " << entry.second << "\n";
    }

    const rocminfer::FormatInfo* format = nullptr;
    const rocminfer::GgufTensorInfo* info = first_rocmfp4_tensor(reader.manifest(), &format);
    if (info == nullptr || format == nullptr) {
        std::cout << label << ": no ROCmFP4 tensor to decode\n";
        return;
    }

    std::cout << "    first ROCmFP4 tensor: " << info->name << " (" << format->name
              << ", " << format->block_bytes << " B/block)\n";

    std::vector<std::uint8_t> payload;
    const int read_status = reader.read_tensor(*info, &payload);
    CHECK(read_status == 0);
    if (read_status != 0 || payload.empty()) {
        return;
    }
    CHECK(payload.size() == info->byte_size);

    decode_prefix(*format, payload);
    wrong_layout_is_detectable(*format, payload);
    corrupt_scale_is_rejected(*format, payload);

    // A ragged length must be refused for the format's block size.
    CHECK(rocminfer::rocmfp4_validate(payload.data(), payload.size() - 1, format->dual_scale) ==
          EINVAL);
}

}  // namespace

int main() {
    test_model("model A", model_path('A'));
    test_model("model B", model_path('B'));

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all rocmfp4 payload checks passed\n";
    return 0;
}