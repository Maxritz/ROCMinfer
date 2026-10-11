/**
 * @file model.cpp
 * @brief Implementation of `Model`: GGUF-backed tensor residency and traced matvec.
 *
 * Tensors are loaded to device memory on demand. Each `TensorSlot` caches the
 * packed bytes in a `GpuBuffer` and records the payload SHA-256 so the graph
 * engine can prove a resident tensor came from the right file offset.
 */

#include "rocminfer/model.hpp"

#include "rocminfer/int4_matvec.hpp"
#include "rocminfer/int4_wmma.hpp"
#include "rocminfer/profiler.hpp"
#include "rocminfer/rocmfp4_dot.hpp"
#include "rocminfer/sha256.hpp"

#ifdef ROCMINFER_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

#include <cerrno>
#include <cstring>
#include <fstream>
#include <vector>

namespace rocminfer {

namespace {

#ifdef ROCMINFER_HAVE_HIP
/**
 * @brief Report whether this device can run the IU4 WCMA int4 tile.
 *
 * The device arch is queried once and cached, because `__gfx1201__` is a device
 * compilation macro and is not reliably visible from a host translation unit:
 * testing it from host code could silently answer false on a machine that does
 * have the instruction, and a dispatch that quietly never selects its fast path
 * is the failure mode `AGENTS.md` forbids.
 *
 * @return true when the arch is GFX12 (RDNA4 and successors), which is the only
 *         family whose int4 matrix path is `v_wmma_i32_16x16x16_iu4`.
 * @pre `hipSetDevice` has selected the intended device, or device 0 is correct.
 * @post The first call queries the runtime; later calls read the cached answer.
 * @note The `gfx12` prefix covers gfx1200, gfx1201 and gfx1250 without naming
 *       each one, so a newly released RDNA4 derivative does not silently fall
 *       back to the slower path.
 */
bool int4_matvec_iu4_is_available() noexcept {
    static const bool available = [] {
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
            return false;
        }
        return std::strncmp(props.gcnArchName, "gfx12", 5) == 0;
    }();
    return available;
}
#endif

}  // namespace

Model::~Model() { close(); }

Model::Model(Model&& other) noexcept
    : reader_(std::move(other.reader_)),
      slots_(std::move(other.slots_)) {
    other.slots_.clear();
}

Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        close();
        reader_ = std::move(other.reader_);
        slots_ = std::move(other.slots_);
        other.slots_.clear();
    }
    return *this;
}

int Model::open(const char* path, TraceRecorder* recorder) {
    if (path == nullptr) {
        return EINVAL;
    }
    int status = reader_.open(path);
    if (status != 0) {
        return status;
    }

    status = gguf_validate_tensors(reader_.manifest_mutable(), 0, reader_.file_size());
    if (status != 0) {
        reader_.close();
        return status;
    }

    slots_.clear();
    for (const GgufTensorInfo& info : reader_.manifest().tensors) {
        TensorSlot slot;
        slot.info = &info;
        slot.format = format_by_ggml_type(info.type_id);
        if (slot.format != nullptr) {
            slots_.emplace(info.name, std::move(slot));
        }
    }

    if (recorder != nullptr && recorder->enabled()) {
        TraceScope scope(*recorder, "model-open", TraceDomain::kHost);
        (void)scope;
    }

    return 0;
}

void Model::close() noexcept {
    slots_.clear();
    reader_.close();
}

const TensorSlot* Model::find_slot(std::string_view name) const noexcept {
    auto it = slots_.find(std::string(name));
    if (it == slots_.end()) {
        return nullptr;
    }
    return &it->second;
}

int Model::load_tensor(std::string_view name, TraceRecorder& recorder) {
    auto it = slots_.find(std::string(name));
    if (it == slots_.end()) {
        return ENOENT;
    }
    TensorSlot& slot = it->second;
    if (slot.resident) {
        return 0;
    }
    if (slot.format == nullptr) {
        return EINVAL;
    }

    std::vector<std::uint8_t> host_bytes;
    int read_status = reader_.read_tensor(*slot.info, &host_bytes);
    if (read_status != 0) {
        return read_status;
    }

    slot.payload_digest = Sha256::hex(host_bytes.data(), host_bytes.size());

    {
        TraceScope scope(recorder, "transfer-weights", TraceDomain::kTransfer);
        int alloc_status = slot.device.allocate(host_bytes.size());
        if (alloc_status != 0) {
            return alloc_status;
        }
    }

#ifdef ROCMINFER_HAVE_HIP
    const hipError_t copy_err = hipMemcpy(slot.device.ptr(), host_bytes.data(),
                                          host_bytes.size(), hipMemcpyHostToDevice);
    if (copy_err != hipSuccess) {
        slot.device.reset();
        return EIO;
    }
#endif

    slot.resident = true;
    return 0;
}

void Model::evict_tensor(std::string_view name) noexcept {
    auto it = slots_.find(std::string(name));
    if (it != slots_.end()) {
        it->second.device.reset();
        it->second.resident = false;
    }
}

int Model::matvec(std::string_view name, const float* x, float* y,
                  std::int64_t rows, std::int64_t cols, void* stream,
                  TraceRecorder& recorder) {
    if (x == nullptr || y == nullptr) {
        return EINVAL;
    }
    const TensorSlot* slot = find_slot(name);
    if (slot == nullptr || !slot->resident) {
        return ENOENT;
    }
    if (slot->format == nullptr) {
        return EINVAL;
    }
    // The fused dequantize-and-dot kernel only understands a ROCmFP4 block.
    // A registered format that is not one of them (the engine-internal raw
    // int4 row) carries no scales and no Codebook10 magnitudes, so routing it
    // here would mis-decode every weight instead of failing loudly.
    if (slot->format->id != QuantFormat::kRocmFP4 &&
        slot->format->id != QuantFormat::kRocmFP4Fast) {
        return EINVAL;
    }

    const std::uint64_t rows_u = static_cast<std::uint64_t>(rows);
    const std::uint64_t cols_u = static_cast<std::uint64_t>(cols);
    if (slot->info->dimensions.size() < 2 ||
        static_cast<std::uint64_t>(slot->info->dimensions[0]) != rows_u ||
        static_cast<std::uint64_t>(slot->info->dimensions[1]) != cols_u) {
        return EINVAL;
    }

#ifdef ROCMINFER_HAVE_HIP
    return rocmfp4_dot_f32_traced(recorder, y, slot->device.ptr(), x,
                                  rows, cols, slot->format->dual_scale, stream);
#else
    (void)stream;
    return ENOTSUP;
#endif
}

int Model::matvec_int4x(std::string_view name, const void* x_int4, std::int32_t* y,
                        std::int64_t rows, std::int64_t cols, void* stream,
                        TraceRecorder& recorder) {
    if (x_int4 == nullptr || y == nullptr) {
        return EINVAL;
    }
    const TensorSlot* slot = find_slot(name);
    if (slot == nullptr || !slot->resident) {
        return ENOENT;
    }
    if (slot->format == nullptr) {
        return EINVAL;
    }
    if (slot->format->id == QuantFormat::kRocmFP4 ||
        slot->format->id == QuantFormat::kRocmFP4Fast ||
        slot->format->dual_scale) {
        return EINVAL;
    }

    const std::uint64_t rows_u = static_cast<std::uint64_t>(rows);
    const std::uint64_t cols_u = static_cast<std::uint64_t>(cols);
    if (slot->info->dimensions.size() < 2 ||
        static_cast<std::uint64_t>(slot->info->dimensions[0]) != rows_u ||
        static_cast<std::uint64_t>(slot->info->dimensions[1]) != cols_u) {
        return EINVAL;
    }
    const std::size_t want_bytes = int4_matvec_weight_bytes(rows, cols);
    if (want_bytes == 0u || slot->device.size() != want_bytes) {
        return EINVAL;
    }

#ifdef ROCMINFER_HAVE_HIP
    if (int4_matvec_iu4_is_available() && rows >= kInt4MatvecIu4MinRows &&
        (rows % 16) == 0 && (cols % 16) == 0) {
        return int4_matvec_iu4_i32_traced(recorder, y, slot->device.ptr(), x_int4,
                                         rows, cols, stream);
    }
    return int4_matvec_i32_traced(recorder, y, slot->device.ptr(), x_int4,
                                 rows, cols, stream);
#else
    (void)stream;
    return ENOTSUP;
#endif
}

}  // namespace rocminfer
