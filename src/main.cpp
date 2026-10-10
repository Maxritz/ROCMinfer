/**
 * @file main.cpp
 * @brief Command-line front end for ROCMinfer.
 *
 * Parses the global switches, configures the trace recorder, and dispatches to
 * the requested subcommand. `floor`, `inspect`, and `validate` are functional
 * now; inference subcommands are registered by later phases.
 *
 * @note This is the only translation unit permitted to throw to the terminal.
 *       Everything else reports failures through status codes.
 */

#include "rocminfer/format_registry.hpp"
#include "rocminfer/gguf_reader.hpp"
#include "rocminfer/profiler.hpp"
#include "rocminfer/rocmfp4_reference.hpp"
#include "rocminfer/sha256.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// We use the portable C file and error APIs deliberately so one source builds
// under both the MSVC-ABI clang toolchain and MinGW; the *_s variants are not
// available on every target.
#if defined(_MSC_VER)
#define ROCMINFER_DISABLE_CRT_WARNINGS_BEGIN \
    _Pragma("warning(push)") _Pragma("warning(disable : 4996)")
#define ROCMINFER_DISABLE_CRT_WARNINGS_END _Pragma("warning(pop)")
#else
#define ROCMINFER_DISABLE_CRT_WARNINGS_BEGIN
#define ROCMINFER_DISABLE_CRT_WARNINGS_END
#endif

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 2;
constexpr int kExitError = 1;

/// Human-readable description of an errno value, portable across toolchains.
std::string status_text(int status) {
    ROCMINFER_DISABLE_CRT_WARNINGS_BEGIN
    return std::string(std::strerror(status));
    ROCMINFER_DISABLE_CRT_WARNINGS_END
}

void print_usage(std::ostream& os, std::string_view program) {
    os << "usage: " << program << " [options] [command] [args]\n"
       << "\n"
       << "Commands:\n"
       << "  floor                     Measure and print the tracing instrumentation floor.\n"
       << "  inspect <file.gguf>       Print the GGUF manifest: architecture, tensor table, formats.\n"
       << "  validate <file.gguf>      Parse the manifest and verify every tensor fits in the file.\n"
       << "  decode <file.gguf> [name] Decode one ROCmFP4 tensor; print stats and a SHA-256 golden.\n"
       << "  digest <file.gguf>        Print the SHA-256 of the whole artifact, for pinning.\n"
       << "\n"
       << "Global options:\n"
       << "  --profileperf[=table|json|off]\n"
       << "                           Record per-component timings and print a report at exit.\n"
       << "                           'table' (default) is human-readable; 'json' is machine-readable.\n"
       << "                           Tracing is disabled unless this switch is given.\n"
       << "  --profile-floor          Measure the instrumentation floor and exit.\n"
       << "  -h, --help               Show this message.\n"
       << "  --version                Show the engine version.\n";
}

/**
 * @brief Parse `--profileperf[=value]` into a report mode.
 * @param[in] argument The full switch text including the leading dashes.
 * @param[out] out_mode Receives the parsed mode; untouched when `found` is false.
 * @param[out] out_found Receives true when the switch was present.
 * @return true when the switch was present and its value was valid.
 * @pre `argument`, `out_mode`, and `out_found` are non-NULL.
 * @post `out_found` is set before any error return so callers can report the
 *       argument name alongside the value error.
 */
bool parse_profileperf(std::string_view argument, rocminfer::TraceReport* out_mode,
                       bool* out_found) {
    constexpr std::string_view kSwitch = "--profileperf";
    *out_found = argument == kSwitch || argument.rfind("--profileperf=", 0) == 0;
    if (!*out_found) {
        return true;
    }

    if (argument == kSwitch) {
        *out_mode = rocminfer::TraceReport::kTable;
        return true;
    }

    const std::string_view value = argument.substr(std::strlen("--profileperf="));
    if (value == "table") {
        *out_mode = rocminfer::TraceReport::kTable;
        return true;
    }
    if (value == "json") {
        *out_mode = rocminfer::TraceReport::kJson;
        return true;
    }
    if (value == "off") {
        *out_mode = rocminfer::TraceReport::kOff;
        return true;
    }

    std::cerr << "rocminfer: invalid --profileperf value '" << value
              << "'; expected table, json, or off\n";
    return false;
}

/**
 * @brief Read a bounded prefix of a file into memory.
 * @param[in]  path        Path to read, non-NULL and NUL-terminated.
 * @param[out] out_bytes   Receives the leading bytes of the file.
 * @param[out] out_total   Receives the true total file size in bytes.
 * @param[in]  max_bytes   Upper bound on bytes to read into memory.
 * @return true on success; false if the file is missing, empty, or unreadable.
 * @pre `out_bytes` and `out_total` are non-NULL; `max_bytes >= 1`.
 * @post On success `out_bytes->size()` equals `min(file size, max_bytes)` and
 *       `*out_total` is the true file size, which may exceed the buffer.
 * @note Only the manifest is needed for `inspect` and `validate`, so this reads
 *       a prefix rather than a multi-gigabyte payload. Sizes come from the
 *       filesystem, which is 64-bit on Windows; `ftell` would truncate at 2 GiB.
 * @warning A file whose manifest exceeds `max_bytes` will fail to parse with
 *          ERANGE. That is reported, never silently truncated.
 */
bool read_file_prefix(const char* path, std::vector<std::uint8_t>* out_bytes,
                      std::uint64_t* out_total, std::size_t max_bytes) {
    std::error_code ec;
    const std::filesystem::path fs_path(path);
    const std::uintmax_t total = std::filesystem::file_size(fs_path, ec);
    if (ec || total == 0) {
        return false;
    }

    ROCMINFER_DISABLE_CRT_WARNINGS_BEGIN
    std::FILE* file = std::fopen(path, "rb");
    ROCMINFER_DISABLE_CRT_WARNINGS_END
    if (file == nullptr) {
        return false;
    }

    const std::size_t want =
        static_cast<std::size_t>(std::min<std::uintmax_t>(total, max_bytes));
    out_bytes->resize(want);
    const std::size_t read = std::fread(out_bytes->data(), 1, want, file);
    std::fclose(file);

    if (read != want) {
        out_bytes->clear();
        return false;
    }
    *out_total = static_cast<std::uint64_t>(total);
    return true;
}

/// Human-readable name for a `ggml_type` id from the file.
std::string describe_type(std::uint32_t type_id) {
    if (const rocminfer::FormatInfo* info = rocminfer::format_by_ggml_type(type_id)) {
        return info->name;
    }
    switch (static_cast<rocminfer::GgmlType>(type_id)) {
        case rocminfer::GgmlType::kF32:  return "F32";
        case rocminfer::GgmlType::kF16:  return "F16";
        case rocminfer::GgmlType::kQ4_0: return "Q4_0";
        default: break;
    }
    return "type_" + std::to_string(type_id);
}

/// Count tensors per format and per raw type id for the summary.
struct TypeSummary {
    std::map<std::string, std::uint64_t> counts;  ///< format/type name -> tensor count
    std::map<std::uint32_t, std::uint64_t> raw;   ///< raw type id -> tensor count
};

/**
 * @brief Print a GGUF manifest: header, metadata, and per-format tensor census.
 * @param[in] file Parsed manifest.
 * @param[in] os   Destination stream.
 */
void print_manifest(const rocminfer::GgufFile& file, std::ostream& os) {
    os << "GGUF v" << file.version << "  tensors=" << file.tensor_count
       << "  metadata=" << file.metadata_kv_count << "\n";
    os << "data section offset = " << file.data_offset << "\n\n";

    os << "metadata:\n";
    for (const auto& entry : file.metadata) {
        const rocminfer::GgufMetadataValue& value = entry.second;
        if (value.type == rocminfer::GgufValueType::kString) {
            os << "  " << entry.first << " = \"" << value.string_value << "\"\n";
        } else if (value.type == rocminfer::GgufValueType::kArray) {
            os << "  " << entry.first << " = ["
               << rocminfer::gguf_value_type_size(value.array_element_type) << "-byte x "
               << value.array_length << "]\n";
        } else if (value.type == rocminfer::GgufValueType::kBool) {
            os << "  " << entry.first << " = " << (value.bool_value ? "true" : "false") << "\n";
        } else {
            os << "  " << entry.first << " = " << value.numeric_value << "\n";
        }
    }

    TypeSummary summary;
    for (const rocminfer::GgufTensorInfo& tensor : file.tensors) {
        summary.counts[describe_type(tensor.type_id)] += 1;
        summary.raw[tensor.type_id] += 1;
    }

    os << "\ntensor types:\n";
    for (const auto& entry : summary.counts) {
        os << "  " << entry.first << ": " << entry.second << "\n";
    }

    // Flag which of the present types this engine can actually decode.
    os << "\ndecode support:\n";
    bool any_supported = false;
    for (std::size_t i = 0; i < rocminfer::format_registry_size(); ++i) {
        const rocminfer::FormatInfo* info = rocminfer::format_at(i);
        const auto it = summary.raw.find(info->ggml_type_id);
        if (it != summary.raw.end() && it->second > 0) {
            any_supported = true;
            os << "  " << info->name << ": " << it->second
               << " tensors (dual_scale=" << (info->dual_scale ? "yes" : "no")
               << ", " << info->block_bytes << " B/block, " << info->bits_per_weight
               << " bpw)\n";
        }
    }
    if (!any_supported) {
        os << "  none of the tensor types present are registered for decode\n";
    }
}

/// Print the SHA-256 of an entire file's bytes.
void print_file_digest(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::cerr << "rocminfer: cannot open '" << path << "' for digest\n";
        return;
    }
    rocminfer::Sha256 hash;
    std::vector<char> chunk(1u << 20);
    while (stream) {
        stream.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = stream.gcount();
        if (got > 0) {
            hash.update(chunk.data(), static_cast<std::size_t>(got));
        }
    }
    std::cout << "sha256 " << path << "\n" << hash.hex_digest() << "\n";
}

/// Decode one ROCmFP4 tensor and report statistics plus a payload digest.
int decode_tensor(const char* path, const std::string& tensor_name) {
    rocminfer::GgufReader reader;
    const int open_status = reader.open(path);
    if (open_status != 0) {
        std::cerr << "rocminfer: cannot open '" << path << "': " << status_text(open_status)
                  << "\n";
        return kExitError;
    }

    int status = rocminfer::gguf_validate_tensors(reader.manifest_mutable(), 0, reader.file_size());
    if (status != 0) {
        std::cerr << "rocminfer: tensor bounds invalid: " << status_text(status) << "\n";
        return kExitError;
    }

    const rocminfer::GgufTensorInfo* info = nullptr;
    const rocminfer::FormatInfo* format = nullptr;
    if (tensor_name.empty()) {
        for (const rocminfer::GgufTensorInfo& candidate : reader.manifest().tensors) {
            if (const rocminfer::FormatInfo* f =
                    rocminfer::format_by_ggml_type(candidate.type_id)) {
                info = &candidate;
                format = f;
                break;
            }
        }
        if (info == nullptr) {
            std::cerr << "rocminfer: no ROCmFP4 tensor found in '" << path << "'\n";
            return kExitError;
        }
    } else {
        info = reader.find_tensor(tensor_name);
        if (info == nullptr) {
            std::cerr << "rocminfer: no tensor named '" << tensor_name << "'\n";
            return kExitError;
        }
        format = rocminfer::format_by_ggml_type(info->type_id);
        if (format == nullptr) {
            std::cerr << "rocminfer: tensor '" << tensor_name
                      << "' is not a registered ROCmFPX format (type "
                      << info->type_id << ")\n";
            return kExitError;
        }
    }

    std::vector<std::uint8_t> payload;
    status = reader.read_tensor(*info, &payload);
    if (status != 0) {
        std::cerr << "rocminfer: cannot read tensor payload: " << status_text(status) << "\n";
        return kExitError;
    }

    std::cout << "tensor: " << info->name << "\n"
              << "  format: " << format->name << " (" << format->block_bytes
              << " B/block, " << format->bits_per_weight << " bpw, dual_scale="
              << (format->dual_scale ? "yes" : "no") << ")\n"
              << "  dims:";
    for (std::uint64_t dim : info->dimensions) {
        std::cout << " " << dim;
    }
    std::cout << "\n  payload bytes: " << payload.size() << "\n"
              << "  payload sha256: " << rocminfer::Sha256::hex(payload.data(), payload.size())
              << "\n";

    // Decode a bounded prefix for statistics.
    const std::size_t block_bytes = format->block_bytes;
    const std::size_t blocks = std::min<std::size_t>(payload.size() / block_bytes, 1u << 20);
    if (blocks == 0) {
        std::cerr << "rocminfer: tensor smaller than one block\n";
        return kExitError;
    }
    std::vector<float> out(blocks * rocminfer::QK_ROCMFP4, 0.0f);
    status = rocminfer::rocmfp4_dequantize_row(out.data(), payload.data(),
                                               blocks * block_bytes,
                                               blocks * rocminfer::QK_ROCMFP4,
                                               format->dual_scale);
    if (status != 0) {
        std::cerr << "rocminfer: decode failed: " << status_text(status) << "\n";
        return kExitError;
    }

    float minimum = out[0];
    float maximum = out[0];
    double sum = 0.0;
    for (float value : out) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        sum += value;
    }
    std::cout << "  decoded " << (blocks * rocminfer::QK_ROCMFP4) << " weights from "
              << blocks << " blocks\n"
              << "    min: " << minimum << "\n"
              << "    max: " << maximum << "\n"
              << "    mean: " << (sum / static_cast<double>(out.size())) << "\n";
    return kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view program = (argc > 0 && argv[0] != nullptr) ? argv[0] : "rocminfer";

    rocminfer::TraceReport report = rocminfer::TraceReport::kOff;
    bool want_floor = false;
    bool want_help = false;
    bool want_version = false;
    std::string command;
    std::string positional;
    std::string extra;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            want_help = true;
            continue;
        }
        if (arg == "--version") {
            want_version = true;
            continue;
        }
        if (arg == "--profile-floor") {
            want_floor = true;
            continue;
        }
        if (arg == "--") {
            continue;
        }

        bool found = false;
        rocminfer::TraceReport parsed = rocminfer::TraceReport::kOff;
        if (!parse_profileperf(arg, &parsed, &found)) {
            return kExitUsage;
        }
        if (found) {
            report = parsed;
            continue;
        }

        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "rocminfer: unknown option '" << arg << "'\n";
            return kExitUsage;
        }
        if (command.empty()) {
            command = std::string(arg);
        } else if (positional.empty()) {
            positional = std::string(arg);
        } else if (extra.empty()) {
            extra = std::string(arg);
        } else {
            std::cerr << "rocminfer: unexpected extra argument '" << arg << "'\n";
            return kExitUsage;
        }
    }

    if (want_help) {
        print_usage(std::cout, program);
        return kExitOk;
    }
    if (want_version) {
        std::cout << "rocminfer 0.1.0 (Phase 0)\n";
        return kExitOk;
    }

    rocminfer::TraceConfig config;
    config.report = report;
    config.measure_floor = report != rocminfer::TraceReport::kOff;

    rocminfer::TraceRecorder recorder(config);

    if (want_floor || command == "floor") {
        // Measuring the floor requires an armed recorder, otherwise there is no
        // instrumentation to calibrate.
        rocminfer::TraceConfig floor_config;
        floor_config.report = rocminfer::TraceReport::kTable;
        floor_config.measure_floor = true;
        rocminfer::TraceRecorder floor_recorder(floor_config);
        floor_recorder.measure_floor();
        floor_recorder.report(std::cout);
        return kExitOk;
    }

    if (command == "digest") {
        if (positional.empty()) {
            std::cerr << "rocminfer: digest requires a .gguf path\n";
            return kExitUsage;
        }
        rocminfer::TraceScope scope(recorder, "artifact-digest",
                                    rocminfer::TraceDomain::kHost);
        print_file_digest(positional.c_str());
        if (recorder.enabled()) {
            recorder.measure_floor();
            if (recorder.report_mode() == rocminfer::TraceReport::kJson) {
                recorder.report_json(std::cout);
            } else {
                recorder.report(std::cout);
            }
        }
        return kExitOk;
    }

    if (command == "decode") {
        if (positional.empty()) {
            std::cerr << "rocminfer: decode requires a .gguf path\n";
            return kExitUsage;
        }
        // The second positional (already captured) names an optional tensor.
        const int decode_status = [&]() {
            rocminfer::TraceScope scope(recorder, "tensor-decode",
                                        rocminfer::TraceDomain::kHost);
            return decode_tensor(positional.c_str(), extra);
        }();
        if (recorder.enabled()) {
            recorder.measure_floor();
            if (recorder.report_mode() == rocminfer::TraceReport::kJson) {
                recorder.report_json(std::cout);
            } else {
                recorder.report(std::cout);
            }
        }
        return decode_status;
    }

    if (command == "inspect" || command == "validate") {
        if (positional.empty()) {
            std::cerr << "rocminfer: " << command << " requires a .gguf path\n";
            return kExitUsage;
        }

        // Only the manifest is read for inspect/validate. The dominant consumer is the
// tokenizer metadata array, which can run to tens of megabytes for a large
// vocabulary; 512 MiB leaves ample headroom without reading multi-GB payloads.
constexpr std::size_t kManifestProbeBytes = 512u * 1024u * 1024u;

    std::vector<std::uint8_t> bytes;
        std::uint64_t file_size = 0;
        {
            rocminfer::TraceScope scope(recorder, "manifest-read",
                                        rocminfer::TraceDomain::kAlloc);
            if (!read_file_prefix(positional.c_str(), &bytes, &file_size,
                                  kManifestProbeBytes)) {
                std::cerr << "rocminfer: cannot read '" << positional << "'\n";
                return kExitError;
            }
        }
        if (bytes.size() < file_size) {
            std::cerr << "rocminfer: note: read " << bytes.size() << " of " << file_size
                      << " bytes (manifest prefix only)\n";
        }

        rocminfer::GgufFile file;
        int status = 0;
        {
            rocminfer::TraceScope scope(recorder, "manifest-parse",
                                        rocminfer::TraceDomain::kHost);
            status = rocminfer::gguf_parse(&file, bytes.data(), bytes.size());
        }
        if (status != 0) {
            std::cerr << "rocminfer: cannot parse '" << positional << "' ("
                      << status_text(status) << ")\n";
            return kExitError;
        }

        if (command == "validate") {
            {
                rocminfer::TraceScope scope(recorder, "manifest-validate",
                                            rocminfer::TraceDomain::kHost);
                status = rocminfer::gguf_validate_tensors(file, bytes.size(), file_size);
            }
            if (status != 0) {
                std::cerr << "rocminfer: '" << positional << "' failed validation: "
                          << status_text(status) << "\n";
                return kExitError;
            }
            std::cout << positional << ": manifest valid, " << file.tensor_count
                      << " tensors, data at offset " << file.data_offset << "\n";
        } else {
            {
                rocminfer::TraceScope scope(recorder, "manifest-validate",
                                            rocminfer::TraceDomain::kHost);
                status = rocminfer::gguf_validate_tensors(file, bytes.size(), file_size);
            }
            print_manifest(file, std::cout);
            if (status != 0) {
                std::cout << "\nWARNING: tensor bounds check failed ("
                          << status_text(status) << ")\n";
            }
        }

        if (recorder.enabled()) {
            recorder.measure_floor();
            if (recorder.report_mode() == rocminfer::TraceReport::kJson) {
                recorder.report_json(std::cout);
            } else {
                recorder.report(std::cout);
            }
        }
        return kExitOk;
    }

    if (!command.empty()) {
        std::cerr << "rocminfer: unknown command '" << command << "'\n";
        print_usage(std::cerr, program);
        return kExitUsage;
    }

    if (recorder.enabled()) {
        recorder.measure_floor();
        std::vector<rocminfer::TraceRow> rows;
        recorder.snapshot(&rows);
        if (recorder.report_mode() == rocminfer::TraceReport::kJson) {
            recorder.report_json(std::cout);
        } else {
            recorder.report(std::cout);
        }
    } else {
        print_usage(std::cout, program);
    }

    return kExitOk;
}