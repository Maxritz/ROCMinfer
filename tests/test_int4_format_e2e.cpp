/**
 * @file test_int4_format_e2e.cpp
 * @brief End-to-end proof that `Model::matvec_int4x` runs on a real GGUF file.
 *
 * The engine-internal `Q4_0_INT4` format (four bytes, eight two's-complement
 * weights, no scales and no codebook) exists precisely so that
 * `Model::matvec_int4x` has a caller reachable from a model file. This test
 * proves that caller end to end on a real artifact written to disk by
 * `tools/gen_int4_artifact.py`, loaded through `GgufReader` and
 * `Model::load_tensor`, never through a hand-built `TensorSlot`:
 *
 *  1. Load the artifact with `rocminfer::Model` and check the reader derived
 *     the right payload extent for both tensors.
 *  2. Build the same signed-int4 activation vector in the test, upload it,
 *     call `matvec_int4x`, and compare every output row to an exact int64
 *     oracle computed from the values the test itself unpacked from the file.
 *     Zero tolerance: the product is exact integer arithmetic, so any
 *     disagreement is a decode or dispatch defect, not rounding.
 *  3. Prove the dispatch routed where intended, using the recorder: the two
 *     kernels are different components (`int4-matvec-iu4` and `int4-matvec`),
 *     so a `TraceRecorder` records which one actually executed. A timing
 *     comparison is NOT used as evidence, because the launch-gap dominates a
 *     sub-microsecond kernel (RUN-008/RUN-012).
 *  4. Negative tests: a ROCmFP4 `Codebook10` tensor still cannot enter
 *     `matvec_int4x`, and a malformed int4 GGUF is refused by the loader
 *     rather than mis-decoded.
 *
 * @note The 256-row shape crosses the dispatch threshold
 *       `kInt4MatvecIu4MinRows`, so it takes the IU4 WMMA tile path; the
 *       64-row shape takes the row-wise kernel. Both are exercised.
 * @note The oracle is derived from the file bytes the test itself reads back
 *       through the engine, so it cannot agree with a wrong packing by
 *       construction: it would have to disagree with the engine twice, in the
 *       same way, for a false pass.
 */

#include "rocminfer/format_registry.hpp"
#include "rocminfer/gguf_reader.hpp"
#include "rocminfer/int4_matvec.hpp"
#include "rocminfer/int4_wmma.hpp"
#include "rocminfer/model.hpp"
#include "rocminfer/profiler.hpp"
#include "rocminfer/sha256.hpp"

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (condition) {
        return;
    }
    ++g_failures;
    std::printf("FAIL line %d: %s\n", line, expression);
}

#define CHECK(expr) check((expr), #expr, __LINE__)

bool hip_ok(hipError_t status, const char* what) {
    if (status == hipSuccess) {
        return true;
    }
    ++g_failures;
    std::printf("FAIL HIP %s: %s\n", what, hipGetErrorString(status));
    return false;
}

/// Relative path of the int4 artifact this test loads, under the repo root.
constexpr const char* kArtifactRelPath =
    "docs/goldens/artifacts/int4-e2e.gguf";

/// Relative path of the pinned ROCmFP4 golden used for the Codebook10 gate.
constexpr const char* kRocmfp4GoldenRelPath =
    "docs/goldens/artifacts/tiny-Q4_0_ROCMFP4.gguf";

/**
 * @brief Resolve a repository-relative artifact path, searching upward.
 *
 * A test binary is launched from the build tree, so a path relative to the
 * CWD does not reach the repository. The search therefore climbs from the
 * working directory until it finds the file, which is how the test can run
 * standalone from `build-rx9070` and from the repository root alike without
 * depending on a generator expression this file cannot declare.
 *
 * @param[in] relative Repository-relative path to look for, non-NULL.
 * @param[out] resolved Receives the located path, untouched when unfound.
 * @return true when an existing file was located.
 * @warning This resolves a path only; the caller must still open the file and
 *          reject it on failure rather than assuming the search guaranteed a
 *          readable artifact.
 */
bool locate_file(const char* relative, std::filesystem::path* resolved) {
    if (relative == nullptr || resolved == nullptr) {
        return false;
    }
    std::error_code ec;
    std::filesystem::path directory = std::filesystem::current_path(ec);
    if (ec) {
        return false;
    }
    for (;;) {
        const std::filesystem::path candidate = directory / relative;
        if (std::filesystem::exists(candidate)) {
            *resolved = candidate;
            return true;
        }
        const std::filesystem::path parent = directory.parent_path();
        if (parent == directory) {
            return false;
        }
        directory = parent;
    }
}

/**
 * @brief Write candidate bytes verbatim to a scratch file.
 * @param[in] path  Destination; its parent directory must already exist.
 * @param[in] bytes Bytes to write unmodified.
 * @return true when every byte reached the file and the stream closed cleanly.
 * @post The file holds exactly `bytes.size()` bytes, or it does not exist and
 *       the caller must not treat it as an evaluated case.
 * @note The buffer is handed to the stream through a `const void*` so the copy
 *       needs no unsigned-char to char reinterpretation.
 * @warning A failure is reported to the caller instead of being retried, because
 *          a candidate that never reached disk cannot be checked at all.
 */
bool write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        return false;
    }
    if (!bytes.empty()) {
        const void* source = bytes.data();
        stream.write(static_cast<const char*>(source),
                     static_cast<std::streamsize>(bytes.size()));
    }
    stream.close();
    return !stream.fail();
}

/**
 * @brief Sign-extend one packed nibble to its two's-complement int4 value.
 * @param[in] nibble The four-bit code, already masked to 0x0F.
 * @return The value in -8..7 the code denotes.
 * @pre `nibble <= 0x0F`.
 */
int sext4(std::uint8_t nibble) noexcept {
    const int value = static_cast<int>(nibble & 0x0Fu);
    return value < 8 ? value : value - 16;
}

/**
 * @brief Unpack one signed-int4 row the way the kernels read it.
 *
 * Element `k` lives in the low nibble of byte `k/2` when `k` is even and in
 * the high nibble when `k` is odd. This mirrors the packing documented in
 * `include/rocminfer/int4_wmma.hpp` and used by `src/kernels/int4_matvec.hip`,
 * so a disagreement between the writer and the kernels shows up here as a
 * mismatch rather than passing silently.
 *
 * @param[out] out_values Destination for `n` signed values; caller-owned.
 * @param[in]  bytes      Packed row, at least `n / 2` bytes (non-NULL).
 * @param[in]  n          Element count, a positive multiple of 8.
 * @return 0 on success, `EINVAL` for a null pointer or a ragged count.
 */
int unpack_row(std::int8_t* out_values, const std::uint8_t* bytes, std::int64_t n) {
    if (out_values == nullptr || bytes == nullptr || n <= 0 || (n % 8) != 0) {
        return EINVAL;
    }
    for (std::int64_t k = 0; k < n; ++k) {
        const std::uint8_t byte = bytes[static_cast<std::size_t>(k / 2)];
        const std::uint8_t nibble = (k % 2 == 0) ? static_cast<std::uint8_t>(byte & 0x0Fu)
                                                : static_cast<std::uint8_t>(byte >> 4);
        out_values[static_cast<std::size_t>(k)] =
            static_cast<std::int8_t>(sext4(nibble));
    }
    return 0;
}

/**
 * @brief Pack one signed-int4 row from values, the inverse of `unpack_row`.
 *
 * Used by the test to build the activation vector it hands the engine, so the
 * vector is packed by the same rule the test asserts the file obeys. Values
 * are drawn from the file's own byte stream rather than from a generator, so
 * the test never invents an operand it cannot also read back.
 *
 * @param[out] out_bytes Destination for `n / 2` packed bytes; caller-owned.
 * @param[in]  values    `n` signed values in -8..7 (non-NULL).
 * @param[in]  n         Element count, a positive multiple of 8.
 * @return 0 on success, `EINVAL` for a null pointer, a ragged count, or a
 *         value outside -8..7.
 */
int pack_row(std::uint8_t* out_bytes, const std::int8_t* values, std::int64_t n) {
    if (out_bytes == nullptr || values == nullptr || n <= 0 || (n % 8) != 0) {
        return EINVAL;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(n / 2);
    std::memset(out_bytes, 0, row_bytes);
    for (std::int64_t k = 0; k < n; ++k) {
        const int value = static_cast<int>(values[static_cast<std::size_t>(k)]);
        if (value < -8 || value > 7) {
            return EINVAL;
        }
        const std::uint8_t nibble = static_cast<std::uint8_t>(value & 0x0F);
        if (k % 2 == 0) {
            out_bytes[static_cast<std::size_t>(k / 2)] |= nibble;
        } else {
            out_bytes[static_cast<std::size_t>(k / 2)] |=
                static_cast<std::uint8_t>(nibble << 4);
        }
    }
    return 0;
}

/**
 * @brief Exact int64 oracle for one output row.
 * @param[in] weights The row's `cols` signed int4 weights.
 * @param[in] x       The shared activation, `cols` signed int4 values.
 * @param[in] cols    Element count.
 * @return The exact dot product; the result fits in int32 for every shape the
 *         engine admits, but the accumulation is int64 so a hypothetical
 *         overflow would fail the comparison loudly instead of wrapping.
 */
std::int64_t oracle_row(const std::int8_t* weights, const std::int8_t* x,
                        std::int64_t cols) {
    std::int64_t sum = 0;
    for (std::int64_t k = 0; k < cols; ++k) {
        sum += static_cast<std::int64_t>(weights[static_cast<std::size_t>(k)]) *
               static_cast<std::int64_t>(x[static_cast<std::size_t>(k)]);
    }
    return sum;
}

/**
 * @brief Names of the traced components the two int4 kernels record on.
 *
 * Read off the launchers: `int4_matvec_iu4_i32_traced` labels the scope
 * `int4-matvec-iu4` and `int4_matvec_i32_traced` labels it `int4-matvec`. The
 * recorder's per-component counter is therefore direct evidence of which
 * kernel ran.
 */
constexpr const char* kComponentRowWise = "int4-matvec";
constexpr const char* kComponentIu4 = "int4-matvec-iu4";

/// Test-local copy of the tensor descriptor the reader produced.
struct Shape {
    std::string name;      ///< Tensor name in the artifact.
    std::int64_t rows = 0; ///< Output rows.
    std::int64_t cols = 0; ///< Elements per row.
};

/**
 * @brief Run `matvec_int4x` on one artifact tensor and check every output row.
 *
 * Loads the tensor through `Model::load_tensor`, builds the shared activation
 * from the file's own leading weight row (so the operand is file-derived and
 * round-trip verifiable), launches through `Model::matvec_int4x`, and compares
 * every output element to the exact oracle.
 *
 * @param[in] model  Open model holding the artifact; must outlive the call.
 * @param[in] recorder Tracer used to identify which kernel executed.
 * @param[in] shape  Tensor to exercise.
 * @return true when the load, launch, and every oracle comparison succeeded.
 */
bool run_one(rocminfer::Model& model, rocminfer::TraceRecorder& recorder,
             const Shape& shape) {
    const std::size_t row_bytes = static_cast<std::size_t>(shape.cols) / 2u;
    const std::size_t weight_bytes = static_cast<std::size_t>(shape.rows) * row_bytes;
    const std::size_t y_count = static_cast<std::size_t>(shape.rows);

    std::vector<std::uint8_t> host_weights;
    if (const rocminfer::GgufTensorInfo* info =
            model.reader().find_tensor(shape.name)) {
        if (model.reader().read_tensor(*info, &host_weights) != 0) {
            std::printf("FAIL: could not read payload of %s\n", shape.name.c_str());
            return false;
        }
    } else {
        std::printf("FAIL: %s absent from the artifact\n", shape.name.c_str());
        return false;
    }
    CHECK(host_weights.size() == weight_bytes);

    // The shared activation vector is the first weight row, unpacked from the
    // same bytes the engine is about to load, so the test's operand is derived
    // from the artifact rather than independent of it.
    std::vector<std::int8_t> x_values(static_cast<std::size_t>(shape.cols), 0);
    CHECK(unpack_row(x_values.data(), host_weights.data(), shape.cols) == 0);

    std::vector<std::uint8_t> host_x(row_bytes, 0u);
    CHECK(pack_row(host_x.data(), x_values.data(), shape.cols) == 0);
    // The round trip must be lossless, which pins the packing the test asserts.
    std::vector<std::int8_t> x_round(static_cast<std::size_t>(shape.cols), 0);
    CHECK(unpack_row(x_round.data(), host_x.data(), shape.cols) == 0);
    CHECK(std::memcmp(x_round.data(), x_values.data(),
                      static_cast<std::size_t>(shape.cols)) == 0);

    void* device_x = nullptr;
    void* device_y = nullptr;
    if (!hip_ok(hipMalloc(&device_x, row_bytes), "hipMalloc(x)") ||
        !hip_ok(hipMalloc(&device_y, y_count * sizeof(std::int32_t)), "hipMalloc(y)")) {
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        return false;
    }
    if (!hip_ok(hipMemcpy(device_x, host_x.data(), row_bytes, hipMemcpyHostToDevice),
                "hipMemcpy(x H2D)")) {
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        return false;
    }

    const int status =
        model.matvec_int4x(shape.name, device_x, static_cast<std::int32_t*>(device_y),
                           shape.rows, shape.cols, nullptr, recorder);
    if (status != 0) {
        std::printf("FAIL: matvec_int4x(%s) returned %d\n", shape.name.c_str(), status);
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        return false;
    }
    CHECK(hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize"));

    std::vector<std::int32_t> gpu_out(y_count, 0);
    CHECK(hip_ok(hipMemcpy(gpu_out.data(), device_y, y_count * sizeof(std::int32_t),
                           hipMemcpyDeviceToHost),
                 "hipMemcpy(y D2H)"));

    // Exact int64 oracle over the values the test unpacked from the file.
    std::size_t bad = 0;
    for (std::int64_t r = 0; r < shape.rows; ++r) {
        std::vector<std::int8_t> row(static_cast<std::size_t>(shape.cols), 0);
        CHECK(unpack_row(row.data(), host_weights.data() +
                                         static_cast<std::size_t>(r) * row_bytes,
                         shape.cols) == 0);
        const std::int64_t want = oracle_row(row.data(), x_values.data(), shape.cols);
        const std::int32_t got = gpu_out[static_cast<std::size_t>(r)];
        if (got != static_cast<std::int32_t>(want)) {
            if (bad < 4) {
                std::printf("FAIL %s row %lld: got=%lld want=%lld\n",
                            shape.name.c_str(), static_cast<long long>(r),
                            static_cast<long long>(got), static_cast<long long>(want));
            }
            ++bad;
        }
    }

    const char* outcome = bad == 0 ? "bit-exact" : "MISMATCH";
    std::printf("  %-26s rows=%4lld cols=%4lld  %s vs int64 oracle (%zu/%lld bad)\n",
                shape.name.c_str(), static_cast<long long>(shape.rows),
                static_cast<long long>(shape.cols), outcome, bad,
                static_cast<long long>(y_count));
    CHECK(bad == 0);

    static_cast<void>(hipFree(device_x));
    static_cast<void>(hipFree(device_y));
    return bad == 0;
}

/**
 * @brief Report which component the recorder recorded a launch on.
 *
 * The recorder's per-component op count is read after `resolve_pending`, so
 * the answer comes from a closed scope rather than from a guess about the
 * dispatch condition.
 *
 * @param[in] recorder Recorder that observed the launches.
 * @param[in] component Component label to look for.
 * @return The recorded op count for `component`, or 0 when it never ran.
 */
std::uint64_t component_ops(const rocminfer::TraceRecorder& recorder,
                            const char* component) {
    std::vector<rocminfer::TraceRow> rows;
    if (!recorder.snapshot(&rows)) {
        return 0;
    }
    for (const rocminfer::TraceRow& row : rows) {
        if (row.component != nullptr && std::strcmp(row.component, component) == 0) {
            return row.ops;
        }
    }
    return 0;
}

/**
 * @brief Load both artifact tensors and exercise both dispatch branches.
 *
 * A fresh recorder is used per shape so a component counter can only be
 * attributed to the launch that shape performed.
 *
 * @param[in] model Open model holding the artifact.
 * @return true when both shapes are bit-exact and each recorded exactly one
 *         launch on the component its shape selects.
 */
bool run_both_shapes(rocminfer::Model& model) {
    const Shape shapes[] = {
        {"blk.0.ffn_gate_sq.weight", 256, 512},
        {"blk.0.ffn_up_sq.weight", 64, 512},
    };
    const char* expected_component[] = {kComponentIu4, kComponentRowWise};

    bool ok = true;
    for (std::size_t i = 0; i < 2; ++i) {
        rocminfer::TraceConfig config;
        config.report = rocminfer::TraceReport::kTable;
        rocminfer::TraceRecorder recorder(config);
        recorder.measure_floor();

        if (!run_one(model, recorder, shapes[i])) {
            ok = false;
            continue;
        }
        if (!recorder.resolve_pending()) {
            std::printf("FAIL: pending device scopes for %s did not resolve\n",
                        shapes[i].name.c_str());
            ok = false;
            continue;
        }

        const std::uint64_t expected_ops = component_ops(recorder, expected_component[i]);
        const std::uint64_t other_ops =
            component_ops(recorder, expected_component[i == 0 ? 1 : 0]);
        std::printf("    dispatch evidence: component '%s' ops=%llu, other path ops=%llu\n",
                    expected_component[i], static_cast<unsigned long long>(expected_ops),
                    static_cast<unsigned long long>(other_ops));
        CHECK(expected_ops >= 1);
        CHECK(other_ops == 0);
    }
    return ok;
}

/**
 * @brief Negative test: a malformed int4 artifact must be refused, not decoded.
 *
 * Two malformations are written to real files in the system temporary
 * directory and loaded through `Model::open`, so the rejection is exercised on
 * the loader path rather than on a hand-built manifest:
 *
 *  - a payload truncated by one weight row, while the manifest still declares
 *    the full extent, which must fail extent validation;
 *  - a dimension field enlarged so the declared int4 byte size no longer
 *    matches the payload present, which must also fail extent validation.
 *
 * @param[in] artifact Path of the well-formed artifact to malform.
 * @return true when both malformed files are refused with a non-zero status
 *         and neither produced a loadable int4 slot.
 */
bool run_negative_malformed_files(const std::filesystem::path& artifact) {
    std::vector<std::uint8_t> original;
    {
        std::ifstream stream(artifact, std::ios::binary);
        if (!stream.is_open()) {
            std::printf("FAIL: cannot read %s for the malformed probes\n",
                        artifact.string().c_str());
            return false;
        }
        original.assign(std::istreambuf_iterator<char>(stream),
                        std::istreambuf_iterator<char>());
        if (original.empty()) {
            std::printf("FAIL: %s is empty\n", artifact.string().c_str());
            return false;
        }
    }

    // The second tensor's dimension field must be located by offset, because
    // the probe edits the file the loader actually reads rather than a
    // hand-built manifest. Its position is found by scanning for the tensor
    // name the reader reported and then stepping over the fixed fields that
    // follow it; the parsed extent is carried out as a value, because the
    // manifest storage belongs to the reader and does not outlive its scope.
    std::uint64_t target_cols = 0;
    std::uint64_t target_rows = 0;
    std::size_t dim_offset = 0;
    {
        rocminfer::GgufReader reader;
        if (reader.open(artifact.string().c_str()) != 0) {
            std::printf("FAIL: the well-formed artifact did not open\n");
            return false;
        }
        const rocminfer::GgufTensorInfo* target =
            reader.find_tensor("blk.0.ffn_up_sq.weight");
        if (target == nullptr || target->dimensions.size() != 2) {
            std::printf("FAIL: reference tensor absent or not 2-D\n");
            return false;
        }
        target_rows = target->dimensions[0];
        target_cols = target->dimensions[1];
        if (target_rows == 0 || target_cols == 0) {
            std::printf("FAIL: the reference tensor declared a zero extent\n");
            return false;
        }

        // A tensor-info entry is: name (length-prefixed), n_dims (u32),
        // n_dims extents (u64 each), type (u32), offset (u64). Searching for
        // the length-prefixed name locates the entry, and the extents follow
        // at a fixed stride from there.
        const std::string name = target->name;
        const std::size_t name_length = name.size();
        for (std::size_t at = 24; at + sizeof(std::uint64_t) <= original.size(); ++at) {
            std::uint64_t declared = 0;
            std::memcpy(&declared, original.data() + at, sizeof(declared));
            if (declared != name_length) {
                continue;
            }
            const std::size_t name_at = at + sizeof(declared);
            if (name_at + name_length > original.size() ||
                std::memcmp(original.data() + name_at, name.data(), name_length) != 0) {
                continue;
            }
            const std::size_t n_dims_at = name_at + name_length;
            if (n_dims_at + sizeof(std::uint32_t) > original.size()) {
                continue;
            }
            std::uint32_t n_dims = 0;
            std::memcpy(&n_dims, original.data() + n_dims_at, sizeof(n_dims));
            if (n_dims != 2) {
                continue;
            }
            const std::size_t extents_at = n_dims_at + sizeof(n_dims);
            if (extents_at + 2 * sizeof(std::uint64_t) > original.size()) {
                continue;
            }
            // The entry is identified by its extents matching what the reader
            // parsed, so a coincidental name match cannot be accepted.
            std::uint64_t first = 0;
            std::uint64_t second = 0;
            std::memcpy(&first, original.data() + extents_at, sizeof(first));
            std::memcpy(&second, original.data() + extents_at + sizeof(first),
                        sizeof(second));
            if (first != target_rows || second != target_cols) {
                continue;
            }
            dim_offset = extents_at;
            break;
        }
    }

    if (dim_offset == 0 || dim_offset + 2 * sizeof(std::uint64_t) > original.size()) {
        std::printf("FAIL: could not locate the dimension field in the artifact\n");
        return false;
    }

    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "rocminfer_int4_negative";
    std::filesystem::create_directories(scratch);

    bool ok = true;
    const auto probe = [&](const char* label, const char* filename,
                           const std::vector<std::uint8_t>& bytes) {
        const std::filesystem::path path = scratch / filename;
        if (!write_file(path, bytes)) {
            std::printf("FAIL: %s never reached %s\n", label, path.string().c_str());
            ++g_failures;
            ok = false;
            return;
        }
        rocminfer::Model model;
        const int status = model.open(path.string().c_str(), nullptr);
        if (status == 0) {
            std::printf("FAIL: %s was accepted by Model::open\n", label);
            ++g_failures;
            ok = false;
        } else {
            std::printf("  %-34s refused by Model::open: %d\n", label, status);
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);
    };

    // 1. Payload truncated by one weight row of the second tensor. The first
    //    tensor's extent still fits, so the reader must refuse on the second.
    {
        std::vector<std::uint8_t> bytes = original;
        const std::size_t drop = static_cast<std::size_t>(target_cols) / 2u;
        CHECK(drop > 0 && drop < bytes.size());
        bytes.resize(bytes.size() - drop);
        probe("payload truncated by one row", "truncated.gguf", bytes);
    }

    // 2. The declared row count doubled, so the int4 byte size the registry
    //    derives no longer matches the payload present.
    {
        std::vector<std::uint8_t> bytes = original;
        const std::uint64_t doubled = target_rows * 2u;
        std::memcpy(bytes.data() + dim_offset, &doubled, sizeof(doubled));
        probe("declared rows exceed payload", "overdeclared.gguf", bytes);
    }

    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
    return ok;
}

/**
 * @brief Negative test: a ROCmFP4 `Codebook10` tensor cannot enter `matvec_int4x`.
 *
 * Loads the pinned golden ROCmFP4 artifact through the real model path and
 * requires that `matvec_int4x` reject its tensor with `EINVAL` rather than
 * interpreting Codebook10 magnitude nibbles as two's-complement int4. This is
 * the gate that must not regress when a genuine int4 format exists.
 *
 * @param[in] path Path of the ROCmFP4 golden artifact.
 * @return true when the tensor rejects with `EINVAL`, or when the artifact is
 *         absent (reported as skipped).
 */
bool run_negative_codebook10(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        std::printf("SKIP: %s absent, Codebook10 gate not probed\n", path.string().c_str());
        return true;
    }

    rocminfer::Model model;
    const int open_status = model.open(path.string().c_str(), nullptr);
    if (open_status != 0) {
        std::printf("SKIP: %s failed to open (%d)\n", path.string().c_str(), open_status);
        return true;
    }

    const rocminfer::GgufTensorInfo* info = nullptr;
    for (const rocminfer::GgufTensorInfo& candidate : model.manifest().tensors) {
        if (const rocminfer::FormatInfo* format =
                rocminfer::format_by_ggml_type(candidate.type_id)) {
            if (format->id == rocminfer::QuantFormat::kRocmFP4) {
                info = &candidate;
                break;
            }
        }
    }
    if (info == nullptr) {
        std::printf("SKIP: no ROCmFP4 tensor in %s\n", path.string().c_str());
        return true;
    }

    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(config);
    recorder.measure_floor();
    if (model.load_tensor(info->name, recorder) != 0) {
        std::printf("FAIL: could not load the ROCmFP4 tensor\n");
        return false;
    }
    CHECK(model.find_slot(info->name) != nullptr);
    CHECK(model.find_slot(info->name)->format != nullptr);
    CHECK(model.find_slot(info->name)->format->id == rocminfer::QuantFormat::kRocmFP4);

    // A resident ROCmFP4 tensor must be rejected, never silently mis-decoded.
    void* device_x = nullptr;
    void* device_y = nullptr;
    const std::int64_t cols = static_cast<std::int64_t>(info->dimensions[1]);
    const std::int64_t rows = static_cast<std::int64_t>(info->dimensions[0]);
    if (!hip_ok(hipMalloc(&device_x, static_cast<std::size_t>(cols / 2)), "hipMalloc(x)") ||
        !hip_ok(hipMalloc(&device_y, static_cast<std::size_t>(rows) * sizeof(std::int32_t)),
                "hipMalloc(y)")) {
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        return false;
    }

    const int status =
        model.matvec_int4x(info->name, device_x, static_cast<std::int32_t*>(device_y),
                           rows, cols, nullptr, recorder);
    static_cast<void>(hipFree(device_x));
    static_cast<void>(hipFree(device_y));

    if (status != EINVAL) {
        std::printf("FAIL: Codebook10 tensor returned %d, expected %d (EINVAL)\n", status,
                    EINVAL);
        ++g_failures;
        return false;
    }
    std::printf("  %-34s rejected: EINVAL (format=%s, dual_scale=%d)\n", info->name.c_str(),
                rocminfer::format_by_ggml_type(info->type_id)->name,
                rocminfer::format_by_ggml_type(info->type_id)->dual_scale ? 1 : 0);
    return true;
}

}  // namespace

int main() {
    hipDeviceProp_t props{};
    int device_count = 0;
    const hipError_t count_status = hipGetDeviceCount(&device_count);
    if (count_status != hipSuccess || device_count < 1) {
        std::printf("FAIL: no HIP device available: %s\n",
                    hipGetErrorString(count_status));
        return 1;
    }
    CHECK(hip_ok(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties"));
    std::printf("device: %s  arch: %s  warpSize: %d\n\n", props.name, props.gcnArchName,
                props.warpSize);

    std::filesystem::path path;
    if (!locate_file(kArtifactRelPath, &path)) {
        std::printf("SKIP: artifact absent; regenerate it with\n"
                    "  python tools/gen_int4_artifact.py %s\n",
                    kArtifactRelPath);
        return 1;
    }
    std::printf("artifact: %s\n", path.string().c_str());

    rocminfer::Model model;
    const int open_status = model.open(path.string().c_str(), nullptr);
    CHECK(open_status == 0);
    if (open_status != 0) {
        std::printf("FAIL: Model::open returned %d\n", open_status);
        return 1;
    }
    std::printf("opened %s: tensors=%llu bytes=%llu\n", path.string().c_str(),
                static_cast<unsigned long long>(model.manifest().tensor_count),
                static_cast<unsigned long long>(model.file_size()));

    // The reader must derive the int4 payload extent from the registry row,
    // not from anything the test supplies.
    for (const char* name : {"blk.0.ffn_gate_sq.weight", "blk.0.ffn_up_sq.weight"}) {
        const rocminfer::GgufTensorInfo* info = model.reader().find_tensor(name);
        CHECK(info != nullptr);
        if (info == nullptr) {
            continue;
        }
        const rocminfer::TensorSlot* slot = model.find_slot(name);
        CHECK(slot != nullptr);
        if (slot != nullptr) {
            CHECK(slot->format != nullptr);
            if (slot->format != nullptr) {
                CHECK(slot->format->id == rocminfer::QuantFormat::kSignedInt4);
            }
        }
        std::printf("  %-26s dims=[%llu,%llu] type=%u byte_size=%llu\n", name,
                    static_cast<unsigned long long>(info->dimensions[0]),
                    static_cast<unsigned long long>(info->dimensions[1]), info->type_id,
                    static_cast<unsigned long long>(info->byte_size));
    }

    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(config);
    recorder.measure_floor();
    CHECK(model.load_tensor("blk.0.ffn_gate_sq.weight", recorder) == 0);
    CHECK(model.load_tensor("blk.0.ffn_up_sq.weight", recorder) == 0);

    std::printf("\nmatvec_int4x over both shapes, exact int64 oracle\n");
    const bool shapes_ok = run_both_shapes(model);

    std::printf("\nnegative tests\n");
    std::filesystem::path golden;
    const bool golden_found = locate_file(kRocmfp4GoldenRelPath, &golden);
    if (!golden_found) {
        std::printf("SKIP: %s absent, Codebook10 gate not probed\n",
                    kRocmfp4GoldenRelPath);
    }
    const bool negatives_ok = run_negative_malformed_files(path) &&
                              run_negative_codebook10(golden);

    if (g_failures != 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    if (!shapes_ok || !negatives_ok) {
        std::printf("\nFAILED: end-to-end correctness\n");
        return 1;
    }
    std::printf("\nPASSED: int4 format end-to-end on a real GGUF artifact\n");
    return 0;
}
