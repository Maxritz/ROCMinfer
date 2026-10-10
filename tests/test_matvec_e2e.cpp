/**
 * @file test_matvec_e2e.cpp
 * @brief End-to-end GPU matvec on a real model tensor, validated against the CPU oracle.
 *
 * Loads a quantized ROCmFP4 tensor from a real GGUF model file, uploads its packed
 * bytes to the GPU, creates a random activation vector, and runs the fused
 * dequantize-and-dot kernel to produce one output row. The result is compared
 * against the engine's CPU reference decoder on the same packed bytes.
 *
 * This is the Phase 3/4 integration gate: it proves the GGUF reader, the GPU
 * upload, the kernel launcher, and the tracing path all work together on
 * actual upstream-produced artifacts. It skips cleanly when no model file is
 * present, so CI without large artifacts stays green.
 */

#include "rocminfer/format_registry.hpp"
#include "rocminfer/gguf_reader.hpp"
#include "rocminfer/model.hpp"
#include "rocminfer/profiler.hpp"
#include "rocminfer/rocmfp4_dot.hpp"
#include "rocminfer/rocmfp4_reference.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (cond) return;
    ++g_failures;
    std::printf("FAIL line %d: %s\n", line, expr);
}

#define CHECK(expr) check((expr), #expr, __LINE__)

bool hip_ok(hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    ++g_failures;
    std::printf("FAIL HIP %s: %s\n", what, hipGetErrorString(status));
    return false;
}

/// One row of SplitMix64-generated FP32 activations in [-1, 1).
std::uint64_t splitmix64_next(std::uint64_t* state) {
    std::uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

float random_activation(std::uint64_t* state) {
    const std::uint64_t bits = splitmix64_next(state) >> 11;
    const double unit = static_cast<double>(bits) * (1.0 / 9007199254740992.0);
    return static_cast<float>(2.0 * unit - 1.0);
}

/// Find the first ROCmFP4 tensor whose column count is a multiple of 32.
const rocminfer::GgufTensorInfo* find_matvec_tensor(
    const rocminfer::GgufFile& file, const rocminfer::FormatInfo** out_format) {
    for (const rocminfer::GgufTensorInfo& info : file.tensors) {
        if (const rocminfer::FormatInfo* f =
                rocminfer::format_by_ggml_type(info.type_id)) {
            if (info.dimensions.size() >= 2 &&
                info.dimensions[1] % rocminfer::QK_ROCMFP4 == 0) {
                *out_format = f;
                return &info;
            }
        }
    }
    return nullptr;
}

/**
 * @brief Load a tensor from the GGUF, run the GPU matvec, and verify against CPU.
 */
bool run_matvec_e2e(const char* path) {
    rocminfer::Model model;
    if (model.open(path, nullptr) != 0) {
        std::printf("SKIP: cannot open '%s'\n", path);
        return true;
    }

    const rocminfer::FormatInfo* format = nullptr;
    const rocminfer::GgufTensorInfo* info =
        find_matvec_tensor(model.manifest(), &format);
    if (info == nullptr || format == nullptr) {
        std::printf("SKIP: no matvec-eligible ROCmFP4 tensor in '%s'\n", path);
        return true;
    }

    const std::int64_t rows = static_cast<std::int64_t>(info->dimensions[0]);
    const std::int64_t cols = static_cast<std::int64_t>(info->dimensions[1]);
    std::printf("  tensor: %s  format=%s  rows=%lld cols=%lld\n",
                info->name.c_str(), format->name,
                static_cast<long long>(rows), static_cast<long long>(cols));

    rocminfer::TraceConfig cfg;
    cfg.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(cfg);
    recorder.measure_floor();

    if (model.load_tensor(info->name, recorder) != 0) {
        std::printf("FAIL: load_tensor failed\n");
        return false;
    }

    const rocminfer::TensorSlot* slot = model.find_slot(info->name);
    CHECK(slot != nullptr);
    CHECK(slot->resident);
    CHECK(slot->device.ptr() != nullptr);
    CHECK(slot->device.size() == info->byte_size);

    const std::size_t x_count = static_cast<std::size_t>(cols);
    const std::size_t y_count = static_cast<std::size_t>(rows);
    std::vector<float> host_x(x_count);
    std::uint64_t seed = 0xBEEFCAFEBEEFCAFEull;
    for (std::size_t i = 0; i < x_count; ++i) {
        host_x[i] = random_activation(&seed);
    }

    void* device_x = nullptr;
    void* device_y = nullptr;
    if (!hip_ok(hipMalloc(&device_x, x_count * sizeof(float)), "hipMalloc(x)") ||
        !hip_ok(hipMalloc(&device_y, y_count * sizeof(float)), "hipMalloc(y)")) {
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        return false;
    }

    if (!hip_ok(hipMemcpy(device_x, host_x.data(), x_count * sizeof(float),
                          hipMemcpyHostToDevice), "hipMemcpy(x H2D)")) {
        static_cast<void>(hipFree(device_x));
        static_cast<void>(hipFree(device_y));
        return false;
    }

    CHECK(model.matvec(info->name, static_cast<const float*>(device_x),
                       static_cast<float*>(device_y), rows, cols, nullptr, recorder) == 0);

    CHECK(hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize"));

    std::vector<float> gpu_out(y_count);
    CHECK(hip_ok(hipMemcpy(gpu_out.data(), device_y, y_count * sizeof(float),
                           hipMemcpyDeviceToHost), "hipMemcpy(y D2H)"));

    static_cast<void>(hipFree(device_x));
    static_cast<void>(hipFree(device_y));

    // CPU oracle: dequantise every row, then dot each row against x.
    std::vector<float> cpu_weights(static_cast<std::size_t>(rows * cols));
    const std::size_t row_block_bytes =
        rocminfer::rocmfp4_dot_row_bytes(cols, format->dual_scale);
    std::vector<std::uint8_t> host_bytes;
    CHECK(model.reader().read_tensor(*info, &host_bytes) == 0);
    CHECK(host_bytes.size() == info->byte_size);

    for (std::int64_t r = 0; r < rows; ++r) {
        const std::size_t row_offset =
            static_cast<std::size_t>(r) * row_block_bytes;
        const std::uint8_t* row_ptr = host_bytes.data() + row_offset;
        CHECK(rocminfer::rocmfp4_dequantize_row(
                  cpu_weights.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols),
                  row_ptr, row_block_bytes, static_cast<std::size_t>(cols),
                  format->dual_scale) == 0);
    }

    // Compare: GPU matvec vs CPU dot product. Use the rigorous FP32 bound.
    const double kFpEps = 1.1920928955078125e-07;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double worst_bound_frac = 0.0;
    for (std::int64_t r = 0; r < rows; ++r) {
        double cpu_dot = 0.0;
        double term_sum = 0.0;
        for (std::int64_t j = 0; j < cols; ++j) {
            const std::size_t idx = static_cast<std::size_t>(r * cols + j);
            const float w = cpu_weights[idx];
            const float xv = host_x[static_cast<std::size_t>(j)];
            cpu_dot += static_cast<double>(w) * static_cast<double>(xv);
            term_sum += std::fabs(static_cast<double>(w) * static_cast<double>(xv));
        }
        const double bound = (static_cast<double>(cols) + 1.0) * kFpEps * term_sum + 1e-300;
        const double gpu_val = static_cast<double>(gpu_out[static_cast<std::size_t>(r)]);
        const double abs_err = std::fabs(gpu_val - cpu_dot);
        if (abs_err > max_abs) max_abs = abs_err;
        const double rel = abs_err / std::max(std::fabs(cpu_dot), 1e-300);
        if (rel > max_rel) max_rel = rel;
        const double bound_frac = abs_err / bound;
        if (bound_frac > worst_bound_frac) worst_bound_frac = bound_frac;
    }

    std::printf("  cpu oracle vs gpu: max_abs=%.6g max_rel=%.6g bound_frac=%.4f\n",
                max_abs, max_rel, worst_bound_frac);

    if (worst_bound_frac > 1.0) {
        std::printf("FAIL: GPU output exceeds the rigorous FP32 bound\n");
        return false;
    }

    double norm = 0.0;
    for (float v : gpu_out) {
        norm += static_cast<double>(v) * static_cast<double>(v);
    }
    norm = std::sqrt(norm);
    std::printf("  output l2 norm: %.6g\n", norm);

    recorder.resolve_pending();
    recorder.report(std::cout);
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
        std::printf("FAIL: no HIP device available\n");
        return 1;
    }

    hipDeviceProp_t props{};
    CHECK(hip_ok(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties"));
    std::printf("device: %s  arch: %s  warpSize: %d\n",
                props.name, props.gcnArchName, props.warpSize);

    const char* models[] = {
        "G:\\More-models\\gemma-4-E2B-it-Q4_0_ROCMFP4_COHERENT.gguf",
        "G:\\More-models\\Ornith-1.0-9b-ROCmFPX-STRIX_LEAN.gguf",
    };

    bool all_ok = true;
    for (const char* path : models) {
        if (!std::filesystem::exists(path)) {
            std::printf("SKIP: '%s' not present\n", path);
            continue;
        }
        std::printf("\n=== %s ===\n", path);
        if (!run_matvec_e2e(path)) {
            all_ok = false;
        }
    }

    if (g_failures != 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    if (!all_ok) {
        std::printf("\nFAILED: end-to-end correctness\n");
        return 1;
    }
    std::printf("\nPASSED: end-to-end GPU matvec on real model tensors\n");
    return 0;
}
