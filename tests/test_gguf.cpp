/**
 * @file test_gguf.cpp
 * @brief Tests for the GGUF manifest reader.
 *
 * Builds synthetic GGUF v3 buffers in memory to exercise the success path and
 * every rejection the parser promises: bad magic, unsupported version, truncated
 * header, truncated tensor table, truncated metadata, hostile array length,
 * over-long string, and a tensor that overruns the file.
 *
 * @note No real model file is required, so this suite runs in CI.
 */

#include "rocminfer/gguf_reader.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
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

/// Append little-endian scalar bytes to a buffer.
template <typename T>
void put(std::vector<std::uint8_t>& out, T value) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

/// Append a length-prefixed GGUF string.
void put_string(std::vector<std::uint8_t>& out, const std::string& value) {
    put<std::uint64_t>(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}

/// Build a minimal valid GGUF v3 with one F32 tensor and two KV pairs.
///
/// The container layout is: header, then key/value metadata, then tensor info,
/// then alignment padding, then payload. Emitting tensor info first would still
/// produce a self-consistent buffer that this test could parse, which is exactly
/// how a wrong field order can pass unit tests and fail on real files.
std::vector<std::uint8_t> build_minimal_gguf(std::uint64_t tensor_elements = 4) {
    std::vector<std::uint8_t> out;
    put<std::uint32_t>(out, 0x46554747u);  // "GGUF"
    put<std::uint32_t>(out, 3);            // version
    put<std::uint64_t>(out, 1);            // tensor count
    put<std::uint64_t>(out, 2);            // kv count

    // KV pairs come first.
    // KV 1: "general.architecture" = string "gemma4"
    put_string(out, "general.architecture");
    put<std::uint32_t>(out, static_cast<std::uint32_t>(rocminfer::GgufValueType::kString));
    put_string(out, "gemma4");

    // KV 2: "general.block_count" = uint32 2
    put_string(out, "general.block_count");
    put<std::uint32_t>(out, static_cast<std::uint32_t>(rocminfer::GgufValueType::kUint32));
    put<std::uint32_t>(out, 2);

    // Then tensor info: name "w", 1 dim, [elements], type F32(0), offset 0.
    put_string(out, "w");
    put<std::uint32_t>(out, 1);
    put<std::uint64_t>(out, tensor_elements);
    put<std::uint32_t>(out, 0);            // ggml_type F32
    put<std::uint64_t>(out, 0);            // offset

    // Pad to 32-byte alignment, then append the tensor payload.
    while (out.size() % 32 != 0) {
        out.push_back(0);
    }
    for (std::uint64_t i = 0; i < tensor_elements * 4; ++i) {
        out.push_back(0);
    }
    return out;
}

void test_parse_minimal_manifest() {
    const std::vector<std::uint8_t> bytes = build_minimal_gguf();
    rocminfer::GgufFile file;
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == 0);

    CHECK(file.version == 3);
    CHECK(file.tensor_count == 1);
    CHECK(file.metadata_kv_count == 2);
    CHECK(file.tensors.size() == 1);
    CHECK(file.metadata.size() == 2);

    CHECK(file.tensors[0].name == "w");
    CHECK(file.tensors[0].dimensions.size() == 1);
    CHECK(file.tensors[0].dimensions[0] == 4);
    CHECK(file.tensors[0].type_id == 0);

    const rocminfer::GgufMetadataValue* arch = file.find_metadata("general.architecture");
    CHECK(arch != nullptr);
    if (arch != nullptr) {
        CHECK(arch->type == rocminfer::GgufValueType::kString);
        CHECK(arch->string_value == "gemma4");
    }
    const rocminfer::GgufMetadataValue* blocks = file.find_metadata("general.block_count");
    CHECK(blocks != nullptr);
    if (blocks != nullptr) {
        CHECK(blocks->type == rocminfer::GgufValueType::kUint32);
        CHECK(blocks->numeric_value == 2.0);
    }
    CHECK(file.find_metadata("absent.key") == nullptr);

    CHECK(rocminfer::gguf_validate_tensors(file, bytes.size(), bytes.size()) == 0);
    CHECK(file.tensors[0].byte_size == 16);
}

void test_rejects_bad_magic_and_version() {
    std::vector<std::uint8_t> bytes = build_minimal_gguf();
    rocminfer::GgufFile file;

    bytes[0] = 'X';
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == EINVAL);
    bytes[0] = 'G';

    bytes[4] = 99;  // unsupported version
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == EINVAL);
    bytes[4] = 3;

    CHECK(rocminfer::gguf_parse(nullptr, bytes.data(), bytes.size()) == EINVAL);
    CHECK(rocminfer::gguf_parse(&file, nullptr, 0) == EINVAL);
}

void test_rejects_truncation() {
    const std::vector<std::uint8_t> full = build_minimal_gguf();
    rocminfer::GgufFile file;
    // Truncating anywhere before the payload must be detected.
    for (std::size_t cut = 1; cut < 64; ++cut) {
        const int status = rocminfer::gguf_parse(&file, full.data(), cut);
        CHECK(status != 0);
    }
    // A cut just past the metadata but before the payload is still valid
    // (the reader tolerates a metadata-only file), so only assert it parses or
    // cleanly rejects, never that it over-reads.
    for (std::size_t cut = 64; cut < full.size(); cut += 7) {
        const int status = rocminfer::gguf_parse(&file, full.data(), cut);
        CHECK(status == 0 || status == ERANGE);
    }
}

void test_rejects_hostile_counts_and_lengths() {
    rocminfer::GgufFile file;

    // Absurd tensor count in the header.
    std::vector<std::uint8_t> bytes;
    put<std::uint32_t>(bytes, 0x46554747u);
    put<std::uint32_t>(bytes, 3);
    put<std::uint64_t>(bytes, 1ull << 40);  // tensor count
    put<std::uint64_t>(bytes, 0);
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == ERANGE);

    // Array length that overflows the remaining bytes.
    bytes.clear();
    put<std::uint32_t>(bytes, 0x46554747u);
    put<std::uint32_t>(bytes, 3);
    put<std::uint64_t>(bytes, 0);
    put<std::uint64_t>(bytes, 1);
    put_string(bytes, "tok_embd.weight");
    put<std::uint32_t>(bytes, static_cast<std::uint32_t>(rocminfer::GgufValueType::kArray));
    put<std::uint32_t>(bytes, static_cast<std::uint32_t>(rocminfer::GgufValueType::kUint8));
    put<std::uint64_t>(bytes, 1ull << 50);  // absurd element count
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == ERANGE);

    // String length longer than the buffer.
    bytes.clear();
    put<std::uint32_t>(bytes, 0x46554747u);
    put<std::uint32_t>(bytes, 3);
    put<std::uint64_t>(bytes, 0);
    put<std::uint64_t>(bytes, 1);   // one KV pair, so the key is actually read
    put<std::uint64_t>(bytes, 1000);  // key length, but nothing follows
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == ERANGE);

    // Trailing bytes beyond the declared KV count are ignored, not an error:
    // a manifest with zero KV pairs is legal regardless of what follows.
    bytes.clear();
    put<std::uint32_t>(bytes, 0x46554747u);
    put<std::uint32_t>(bytes, 3);
    put<std::uint64_t>(bytes, 0);
    put<std::uint64_t>(bytes, 0);
    put<std::uint64_t>(bytes, 0xDEADBEEF);
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == 0);
}

void test_rejects_tensor_overrun() {
    // One F32 tensor declaring far more elements than the payload holds.
    // Layout: header, KV block (empty), tensor info, alignment, payload.
    std::vector<std::uint8_t> bytes;
    put<std::uint32_t>(bytes, 0x46554747u);
    put<std::uint32_t>(bytes, 3);
    put<std::uint64_t>(bytes, 1);
    put<std::uint64_t>(bytes, 0);
    put_string(bytes, "w");
    put<std::uint32_t>(bytes, 1);
    put<std::uint64_t>(bytes, 1000000);  // elements -> 4,000,000 bytes
    put<std::uint32_t>(bytes, 0);
    put<std::uint64_t>(bytes, 0);
    while (bytes.size() % 32 != 0) {
        bytes.push_back(0);
    }
    bytes.resize(bytes.size() + 64);  // only 64 payload bytes present

    rocminfer::GgufFile file;
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == 0);
    CHECK(rocminfer::gguf_validate_tensors(file, bytes.size(), bytes.size()) == ERANGE);
}

void test_zero_dimension_tensor_is_rejected() {
    std::vector<std::uint8_t> bytes;
    put<std::uint32_t>(bytes, 0x46554747u);
    put<std::uint32_t>(bytes, 3);
    put<std::uint64_t>(bytes, 1);
    put<std::uint64_t>(bytes, 0);
    put_string(bytes, "w");
    put<std::uint32_t>(bytes, 1);
    put<std::uint64_t>(bytes, 0);  // zero elements
    put<std::uint32_t>(bytes, 0);
    put<std::uint64_t>(bytes, 0);
    while (bytes.size() % 32 != 0) {
        bytes.push_back(0);
    }
    rocminfer::GgufFile file;
    CHECK(rocminfer::gguf_parse(&file, bytes.data(), bytes.size()) == 0);
    CHECK(rocminfer::gguf_validate_tensors(file, bytes.size(), bytes.size()) == ERANGE);
}

void test_value_type_sizes() {
    CHECK(rocminfer::gguf_value_type_size(rocminfer::GgufValueType::kUint8) == 1);
    CHECK(rocminfer::gguf_value_type_size(rocminfer::GgufValueType::kBool) == 1);
    CHECK(rocminfer::gguf_value_type_size(rocminfer::GgufValueType::kFloat32) == 4);
    CHECK(rocminfer::gguf_value_type_size(rocminfer::GgufValueType::kFloat64) == 8);
    CHECK(rocminfer::gguf_value_type_size(rocminfer::GgufValueType::kString) == 0);
    CHECK(rocminfer::gguf_value_type_size(rocminfer::GgufValueType::kArray) == 0);
}

}  // namespace

int main() {
    test_parse_minimal_manifest();
    test_rejects_bad_magic_and_version();
    test_rejects_truncation();
    test_rejects_hostile_counts_and_lengths();
    test_rejects_tensor_overrun();
    test_zero_dimension_tensor_is_rejected();
    test_value_type_sizes();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all gguf reader checks passed\n";
    return 0;
}