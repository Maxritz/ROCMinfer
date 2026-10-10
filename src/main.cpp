/**
 * @file main.cpp
 * @brief Command-line front end for ROCMinfer.
 *
 * Parses the global switches, configures the trace recorder, and dispatches to
 * the requested subcommand. `floor`, `inspect`, `validate`, `decode`, `digest`,
 * `info`, and `bench` are functional; `pull` is declared with a clear
 * not-implemented path. Inference subcommands are registered by later phases.
 *
 * @note This is the only translation unit permitted to throw to the terminal.
 *       Everything else reports failures through status codes.
 */

#include "rocminfer/format_registry.hpp"
#include "rocminfer/gguf_reader.hpp"
#include "rocminfer/model.hpp"
#include "rocminfer/profiler.hpp"
#include "rocminfer/rocmfp4_reference.hpp"
#include "rocminfer/sha256.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
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

#ifdef ROCMINFER_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

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
        << "  info <file.gguf>          Report architecture, quant census, memory estimates, and hash.\n"
        << "  bench <file.gguf> [tensor] [--iters N] [--profileperf=table]\n"
        << "                           Load a tensor to GPU, run a matvec, and report timing + output norm.\n"
        << "  pull <model-id>           Fetch model artifacts (not yet implemented; fails with an error).\n"
       << "  decode <file.gguf> [name] [--json]\n"
       << "                           Decode one ROCmFP4 tensor. Default prints stats and a payload\n"
       << "                           digest; --json emits decoded weights for cross-checking.\n"
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

/**
 * @brief Decode one ROCmFP4 tensor and report statistics plus a payload digest.
 * @param[in] path         Path to the GGUF artifact, NUL-terminated.
 * @param[in] tensor_name  Tensor to decode, or empty for the first ROCmFPX tensor.
 * @param[in] json_output  Emit decoded weights as JSON instead of a summary.
 * @return Process exit status: 0 on success, 1 on failure.
 */
int decode_tensor(const char* path, const std::string& tensor_name, bool json_output) {
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

    // The human-readable header goes to stderr in JSON mode so stdout is a
    // single parseable document.
    std::ostream& report = json_output ? std::cerr : std::cout;
    report << "tensor: " << info->name << "\n"
           << "  format: " << format->name << " (" << format->block_bytes
           << " B/block, " << format->bits_per_weight << " bpw, dual_scale="
           << (format->dual_scale ? "yes" : "no") << ")\n"
           << "  dims:";
    for (std::uint64_t dim : info->dimensions) {
        report << " " << dim;
    }
    report << "\n  payload bytes: " << payload.size() << "\n"
           << "  payload sha256: "
           << rocminfer::Sha256::hex(payload.data(), payload.size()) << "\n";
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

    if (json_output) {
        // Machine-readable dump for cross-checking against an independent
        // implementation. Weights are emitted as their raw IEEE-754 bit
        // patterns rather than decimal text: JSON has no hex-float literal and
        // decimal formatting is not guaranteed to round-trip, whereas the bit
        // pattern is exact and unambiguous.
        std::uint32_t bits = 0;
        std::memcpy(&bits, &out[0], sizeof(bits));
        std::cout << "{\"tensor\":\"" << info->name << "\",\"format\":\""
                  << format->name << "\",\"encoding\":\"ieee754_bits\","
                  << "\"count\":" << out.size() << ",\"values\":[";
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::memcpy(&bits, &out[i], sizeof(bits));
            std::cout << bits;
        }
        std::cout << "]}\n";
        return kExitOk;
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

namespace rocminfer {

/**
 * @brief Run an end-to-end GPU matvec benchmark on a real model tensor.
 * @param[in] path      Path to the GGUF model file.
 * @param[in] tensor_name  Tensor to benchmark; empty to auto-select.
 * @param[in] iters     Number of kernel iterations to time.
 * @param[in] want_json  Emit machine-readable output.
 * @param[in] recorder   Trace recorder for instrumentation.
 * @param[in] report     Report mode for the trace.
 * @return Process exit status: 0 on success, 1 on runtime failure.
 */
int run_bench(const char* path, const char* tensor_name, int iters,
              bool want_json, TraceRecorder& recorder, TraceReport report) {
    if (path == nullptr) {
        std::cerr << "rocminfer: bench requires a .gguf path\n";
        return kExitUsage;
    }

    Model model;
    int status = model.open(path, want_json ? &recorder : nullptr);
    if (status != 0) {
        std::cerr << "rocminfer: cannot open '" << path << "': "
                  << status_text(status) << "\n";
        return kExitError;
    }

    const GgufTensorInfo* info = nullptr;
    const FormatInfo* format = nullptr;
    if (tensor_name[0] != '\0') {
        const TensorSlot* slot = model.find_slot(tensor_name);
        if (slot == nullptr) {
            std::cerr << "rocminfer: no ROCmFP4 tensor named '" << tensor_name << "'\n";
            return kExitError;
        }
        info = slot->info;
        format = slot->format;
    } else {
        for (const GgufTensorInfo& candidate : model.manifest().tensors) {
            if (const FormatInfo* f = format_by_ggml_type(candidate.type_id)) {
                if (candidate.dimensions.size() >= 2 &&
                    candidate.dimensions[1] % static_cast<std::uint64_t>(QK_ROCMFP4) == 0 &&
                    candidate.dimensions[1] / QK_ROCMFP4 > 0) {
                    info = &candidate;
                    format = f;
                    break;
                }
            }
        }
        if (info == nullptr) {
            std::cerr << "rocminfer: no suitable ROCmFP4 2-D tensor found\n";
            return kExitError;
        }
    }

    const std::int64_t rows = static_cast<std::int64_t>(info->dimensions[0]);
    const std::int64_t cols = static_cast<std::int64_t>(info->dimensions[1]);
    std::cout << "model: " << std::filesystem::path(path).filename() << "\n"
              << "tensor: " << info->name << "\n"
              << "  format: " << format->name << " (" << format->block_bytes
              << " B/block, dual_scale=" << (format->dual_scale ? "yes" : "no") << ")\n"
              << "  rows=" << rows << " cols=" << cols << "\n";

#ifndef ROCMINFER_HAVE_HIP
    std::cerr << "rocminfer: bench requires a HIP-enabled build\n";
    return kExitError;
#else
    {
        TraceScope scope(recorder, "weight-load", TraceDomain::kTransfer);
        status = model.load_tensor(info->name, recorder);
    }
    if (status != 0) {
        std::cerr << "rocminfer: failed to load tensor: " << status_text(status) << "\n";
        return kExitError;
    }

    std::vector<float> activations(static_cast<std::size_t>(cols));
    std::uint64_t seed = 0x4242424242424242ull;
    for (std::size_t i = 0; i < activations.size(); ++i) {
        seed += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        const double unit = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
        activations[i] = static_cast<float>(2.0 * unit - 1.0);
    }

    void* device_x = nullptr;
    void* device_y = nullptr;
    void* device_out = nullptr;
    hipStream_t stream = nullptr;
    const std::size_t x_bytes = sizeof(float) * static_cast<std::size_t>(cols);
    const std::size_t y_bytes = sizeof(float) * static_cast<std::size_t>(rows);

    hipError_t alloc_err = hipSuccess;
    if ((alloc_err = hipMalloc(&device_x, x_bytes)) != hipSuccess ||
        (alloc_err = hipMalloc(&device_y, y_bytes)) != hipSuccess ||
        (alloc_err = hipMalloc(&device_out, y_bytes)) != hipSuccess ||
        (alloc_err = hipStreamCreate(&stream)) != hipSuccess) {
        std::cerr << "rocminfer: GPU allocation or stream creation failed\n";
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        static_cast<void>(hipFree(device_out));
        return kExitError;
    }

    if (hipMemcpy(device_x, activations.data(), x_bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
        std::cerr << "rocminfer: activation upload failed\n";
        static_cast<void>(hipStreamDestroy(stream));
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        static_cast<void>(hipFree(device_out));
        return kExitError;
    }

    if (iters == 0) iters = 1;

    TraceScope scope_kit(recorder, "matvec-bench", TraceDomain::kDevice, stream);
    int launch_status = 0;
    for (int i = 0; i < iters; ++i) {
        launch_status = model.matvec(info->name,
                                     static_cast<const float*>(device_x),
                                     static_cast<float*>(device_y),
                                     rows, cols, stream, recorder);
        if (launch_status != 0) break;
    }

    if (launch_status != 0) {
        std::cerr << "rocminfer: matvec failed: " << status_text(launch_status) << "\n";
        static_cast<void>(hipStreamDestroy(stream));
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        static_cast<void>(hipFree(device_out));
        return kExitError;
    }

    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::cerr << "rocminfer: stream sync failed\n";
        static_cast<void>(hipStreamDestroy(stream));
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        static_cast<void>(hipFree(device_out));
        return kExitError;
    }

    std::vector<float> output(static_cast<std::size_t>(rows));
    if (hipMemcpy(output.data(), device_y, y_bytes, hipMemcpyDeviceToHost) !=
        hipSuccess) {
        std::cerr << "rocminfer: output download failed\n";
        static_cast<void>(hipStreamDestroy(stream));
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        static_cast<void>(hipFree(device_out));
        return kExitError;
    }

    double sum_sq = 0.0;
    float min_val = output[0];
    float max_val = output[0];
    double sum = 0.0;
    for (float v : output) {
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        sum += v;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    const double norm = std::sqrt(sum_sq);
    const double mean = sum / static_cast<double>(output.size());

    if (want_json) {
        std::cout << "," << std::filesystem::path(path).filename() << ","
                  << info->name << "," << format->name << ","
                  << rows << "," << cols << "," << iters << ","
                  << norm << "," << min_val << "," << max_val << "," << mean
                  << "\n";
    } else {
        std::cout << "  output rows: " << rows << "\n"
                  << "  l2 norm: " << norm << "\n"
                  << "  min: " << min_val << "\n"
                  << "  max: " << max_val << "\n"
                  << "  mean: " << mean << "\n"
                  << "  iters: " << iters << "\n";
    }

    if (recorder.enabled()) {
        recorder.resolve_pending();
        recorder.measure_floor();
        if (report == TraceReport::kJson) {
            recorder.report_json(std::cout);
        } else {
            recorder.report(std::cout);
        }
    }

    static_cast<void>(hipStreamDestroy(stream));
    static_cast<void>(hipFree(device_x));
    static_cast<void>(hipFree(device_y));
    static_cast<void>(hipFree(device_out));
    model.evict_tensor(info->name);
    model.close();
    return kExitOk;
#endif
}

}  // namespace rocminfer

int main(int argc, char** argv) {
    const std::string_view program = (argc > 0 && argv[0] != nullptr) ? argv[0] : "rocminfer";

    rocminfer::TraceReport report = rocminfer::TraceReport::kOff;
    bool want_floor = false;
    bool want_help = false;
    bool want_version = false;
    bool want_json = false;
    int bench_iters = 10;
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
        if (arg == "--json") {
            want_json = true;
            continue;
        }
        if (arg == "--") {
            continue;
        }
        if (arg.rfind("--iters=", 0) == 0) {
            bench_iters = std::atoi(arg.data() + 8);
            if (bench_iters <= 0) {
                bench_iters = 1;
            }
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
            return decode_tensor(positional.c_str(), extra, want_json);
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

// inspect and validate share GgufReader with decode, so the manifest
        // probe size and the file-size handling live in exactly one place.
        rocminfer::GgufReader reader;
        int status = 0;
        {
            rocminfer::TraceScope scope(recorder, "manifest-open",
                                        rocminfer::TraceDomain::kAlloc);
            status = reader.open(positional.c_str());
        }
        if (status != 0) {
            std::cerr << "rocminfer: cannot parse '" << positional << "' ("
                      << status_text(status) << ")\n";
            return kExitError;
        }

        {
            rocminfer::TraceScope scope(recorder, "manifest-validate",
                                        rocminfer::TraceDomain::kHost);
            status = rocminfer::gguf_validate_tensors(reader.manifest_mutable(), 0,
                                                      reader.file_size());
        }

        if (command == "validate") {
            if (status != 0) {
                std::cerr << "rocminfer: '" << positional << "' failed validation: "
                          << status_text(status) << "\n";
                return kExitError;
            }
            std::cout << positional << ": manifest valid, "
                      << reader.manifest().tensor_count << " tensors, data at offset "
                      << reader.manifest().data_offset << "\n";
        } else {
            print_manifest(reader.manifest(), std::cout);
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

    if (command == "info") {
        if (positional.empty()) {
            std::cerr << "rocminfer: info requires a .gguf path\n";
            return kExitUsage;
        }

        rocminfer::GgufReader reader;
        int status = 0;
        {
            rocminfer::TraceScope scope(recorder, "manifest-open",
                                        rocminfer::TraceDomain::kAlloc);
            status = reader.open(positional.c_str());
        }
        if (status != 0) {
            std::cerr << "rocminfer: cannot open '" << positional << "': "
                      << status_text(status) << "\n";
            return kExitError;
        }

        {
            rocminfer::TraceScope scope(recorder, "manifest-validate",
                                        rocminfer::TraceDomain::kHost);
            status = rocminfer::gguf_validate_tensors(reader.manifest_mutable(), 0,
                                                      reader.file_size());
        }

        const rocminfer::GgufFile& file = reader.manifest();

        std::cout << "model: " << std::filesystem::path(positional).filename() << "\n";
        std::cout << "file:  " << positional << "\n";
        std::cout << "sha256: " << [&]() {
            std::ifstream stream(positional, std::ios::binary);
            rocminfer::Sha256 hash;
            std::vector<char> chunk(1u << 20);
            while (stream) {
                stream.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                const std::streamsize got = stream.gcount();
                if (got > 0) hash.update(chunk.data(), static_cast<std::size_t>(got));
            }
            return hash.hex_digest();
        }() << "\n";
        std::cout << "size:  " << reader.file_size() << " bytes\n";

        const rocminfer::GgufMetadataValue* arch =
            file.find_metadata("general.architecture");
        const std::string arch_name = arch ? arch->string_value : std::string();
        const rocminfer::GgufMetadataValue* vocab_len =
            file.find_metadata("tokenizer.length");
        std::cout << "architecture: " << (arch ? arch_name : "(unknown)")
                  << "\n"
                  << "tokenizer_length: "
                  << (vocab_len ? static_cast<std::int64_t>(vocab_len->numeric_value) : 0)
                  << "\n";

        const auto arch_field = [&](const char* suffix) -> std::string {
            if (!arch_name.empty()) {
                return arch_name + "." + suffix;
            }
            return std::string();
        };

        const rocminfer::GgufMetadataValue* embed_len = nullptr;
        for (const std::string& key : {arch_field("embedding_length"), std::string("llama.embedding_length"),
                                       std::string("qwen2.embedding_length"), std::string("qwen3.embedding_length"),
                                       std::string("qwen35.embedding_length")}) {
            if ((embed_len = file.find_metadata(key)) != nullptr) break;
        }
        const rocminfer::GgufMetadataValue* ffn_len = nullptr;
        for (const std::string& key : {arch_field("feed_forward_length"),
                                       std::string("llama.feed_forward_length"),
                                       std::string("qwen2.feed_forward_length")}) {
            if ((ffn_len = file.find_metadata(key)) != nullptr) break;
        }
        const rocminfer::GgufMetadataValue* ctx_len = nullptr;
        for (const std::string& key : {arch_field("context_length"),
                                       std::string("llama.context_length"),
                                       std::string("qwen35.context_length")}) {
            if ((ctx_len = file.find_metadata(key)) != nullptr) break;
        }
        std::cout << "embedding_length: "
                  << (embed_len ? static_cast<std::int64_t>(embed_len->numeric_value) : 0)
                  << "\n";
        std::cout << "feed_forward_length: "
                  << (ffn_len ? static_cast<std::int64_t>(ffn_len->numeric_value) : 0)
                  << "\n";
        std::cout << "context_length: "
                  << (ctx_len ? static_cast<std::int64_t>(ctx_len->numeric_value) : 0)
                  << "\n";

        std::map<std::string, std::uint64_t> type_counts;
        std::uint64_t rocmfp4_bytes = 0;
        std::uint64_t f32_bytes = 0;
        for (const rocminfer::GgufTensorInfo& t : file.tensors) {
            if (const rocminfer::FormatInfo* f =
                    rocminfer::format_by_ggml_type(t.type_id)) {
                type_counts[f->name] += 1;
                rocmfp4_bytes += t.byte_size;
            } else if (t.type_id == 0) {
                type_counts["F32"] += 1;
                f32_bytes += t.byte_size;
            } else {
                type_counts["type_" + std::to_string(t.type_id)] += 1;
            }
        }

        std::cout << "\ntensor census ("
                  << file.tensors.size() << " total):\n";
        for (const auto& [name, count] : type_counts) {
            std::cout << "  " << name << ": " << count << "\n";
        }
        std::cout << "packed weight bytes (ROCmFP4): " << rocmfp4_bytes << "\n";
        std::cout << "f32 weight bytes:              " << f32_bytes << "\n";
        std::cout << "validation: " << (status == 0 ? "PASS" : "FAIL") << " ("
                  << (status == 0 ? "all tensors fit" : status_text(status)) << ")\n";

        std::cout << "\nsupported formats:\n";
        for (std::size_t i = 0; i < rocminfer::format_registry_size(); ++i) {
            const rocminfer::FormatInfo* info = rocminfer::format_at(i);
            std::cout << "  " << info->name << " (ggml_type " << info->ggml_type_id
                      << ", " << info->block_bytes << " B/block, "
                      << info->bits_per_weight << " bpw, dual_scale="
                      << (info->dual_scale ? "yes" : "no") << ")\n";
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

    if (command == "bench") {
        if (positional.empty()) {
            std::cerr << "rocminfer: bench requires a .gguf path\n";
            return kExitUsage;
        }
        const std::string tensor_name = extra;
        return run_bench(positional.c_str(), tensor_name.c_str(), bench_iters,
                         want_json, recorder, report);
    }

    if (command == "pull") {
        std::cerr << "rocminfer: 'pull' is not yet implemented; this engine reads "
                  << "model files directly from disk by path. Use a separate "
                  << "download tool to obtain .gguf artifacts, then run this "
                  << "engine on the local file.\n";
        return kExitError;
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