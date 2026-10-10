/**
 * @file model.hpp
 * @brief GGUF model handle with lazy GPU residency and traced matvec.
 *
 * A `Model` binds a parsed GGUF manifest to GPU-resident weight buffers. Tensors
 * are loaded to device memory on first access and retained until the model is
 * closed or evicted, which is the residency contract Phase 4 needs: hot weights
 * stay resident while the graph engine decides what to page in or out.
 *
 * The matvec entry point (`matvec`) routes through the fused ROCmFP4
 * dequantize-and-dot kernel, so a dense `W*x` over a real model tensor runs
 * end-to-end on the GPU without ever materialising the dequantised weights in
 * host or device memory.
 *
 * @note Only functional when `ROCMINFER_HAVE_HIP` is defined. On a host-only
 *       build, `matvec` returns `ENOTSUP` rather than calling any device API.
 */

#ifndef ROCMINFER_MODEL_HPP
#define ROCMINFER_MODEL_HPP

#include "rocminfer/format_registry.hpp"
#include "rocminfer/gguf_reader.hpp"
#include "rocminfer/gpu_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace rocminfer {

class TraceRecorder;

/**
 * @brief One tensor's GPU residency state inside a `Model`.
 *
 * A tensor is resident when `host_bytes` has been uploaded into `device`. The
 * `payload_digest` is computed as a content tag so the graph engine can verify
 * a tensor was loaded from the expected file region.
 */
struct TensorSlot {
    const GgufTensorInfo* info = nullptr;     ///< Borrowed manifest entry.
    const FormatInfo* format = nullptr;        ///< Resolved format, or nullptr for F32/F16.
    GpuBuffer device;                          ///< Device-side copy of packed bytes.
    std::string payload_digest;               ///< SHA-256 of the packed payload, for provenance.
    bool resident = false;                     ///< True when `device` holds live bytes.
};

/**
 * @brief A loaded GGUF model with GPU-backed tensor residency.
 *
 * Non-copyable: it owns file handles and device allocations. Construct, use,
 * and destroy in one scope. The engine's scheduler (Phase 4) will wrap this in
 * a paging layer that calls `load_tensor` / `evict_tensor` as experts are
 * selected.
 */
class Model {
public:
    /**
     * @brief Construct an empty, closed model.
     */
    Model() = default;

    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    /**
     * @brief Open and parse a GGUF file from disk, then validate every tensor.
     * @param[in] path NUL-terminated path to the model file, non-NULL.
     * @param[in] recorder Optional tracer; may be nullptr when tracing is off.
     * @return 0 on success; `ENOENT` if the file is missing; `EINVAL` / `ERANGE`
     *         if the manifest or tensor bounds fail to parse.
     * @pre `path` names an existing GGUF v2/v3 file.
     * @post `is_open()` is true and `manifest()` reflects the parsed header.
     */
    int open(const char* path, TraceRecorder* recorder);

    /**
     * @brief Close the model and free all GPU buffers.
     * @post `is_open()` is false; all `TensorSlot`s are cleared.
     */
    void close() noexcept;

    /**
     * @brief Report whether a model file is currently open.
     * @return true when a file is open.
     */
    bool is_open() const noexcept { return reader_.is_open(); }

    /**
     * @brief Access the parsed manifest.
     * @return Const reference to the GGUF manifest; empty when closed.
     * @pre `is_open()` is true.
     */
    const GgufFile& manifest() const noexcept { return reader_.manifest(); }

    /**
     * @brief Total on-disk size of the model file in bytes.
     * @return File size, or 0 when closed.
     */
    std::uint64_t file_size() const noexcept { return reader_.file_size(); }

    /**
     * @brief Find a tensor by name and resolve its format.
     * @param[in] name Tensor name, e.g. "blk.0.ffn_down.weight".
     * @return Pointer to the slot, or nullptr when the tensor is absent or its
     *         type is not a registered ROCmFPX format.
     * @pre `is_open()` is true.
     */
    const TensorSlot* find_slot(std::string_view name) const noexcept;

    /**
     * @brief Load one tensor's packed bytes into GPU memory.
     * @param[in] name Tensor name to load.
     * @param[in] recorder Tracer for the transfer; must outlive the call.
     * @return 0 on success; `ENOENT` if the tensor is absent; `EINVAL` if it is
     *         not a registered quant format; `ENOMEM` / `EIO` on GPU failure.
     * @pre `is_open()` is true.
     * @post On success `slot(name)->resident == true` and `device.ptr()` is valid.
     */
    int load_tensor(std::string_view name, TraceRecorder& recorder);

    /**
     * @brief Evict a tensor from GPU memory, freeing its device buffer.
     * @param[in] name Tensor name to evict.
     * @post `slot(name)->resident == false` and `device` is empty.
     * @note Evicting a tensor that is not resident is a no-op.
     */
    void evict_tensor(std::string_view name) noexcept;

    /**
     * @brief Run a fused dequantize-and-dot matvec on a loaded tensor.
     *
     * Computes `y = dequant(W) * x` on the GPU, where `W` is the packed tensor
     * `name` (already loaded by `load_tensor`) and `x` is a device pointer to
     * `cols` FP32 activations. The result `y` holds `rows` FP32 values.
     *
     * @param[in] name    Tensor name; must have been loaded with `load_tensor`.
     * @param[in] x       Activation vector base pointer in device memory.
     * @param[out] y      Output vector base pointer in device memory.
     * @param[in] rows    Output rows; must equal the tensor's row count.
     * @param[in] cols    Activation length; must equal the tensor's column count.
     * @param[in] stream  HIP stream (nullptr for default).
     * @param[in] recorder Tracer for the kernel launch.
     * @return 0 on success; `ENOENT` if not loaded; `EINVAL` on shape mismatch.
     * @pre The tensor is resident, `x` has `cols` floats, `y` has `rows` floats.
     * @post On success the kernel launch is enqueued on `stream`.
     */
    int matvec(std::string_view name, const float* x, float* y,
               std::int64_t rows, std::int64_t cols, void* stream,
               TraceRecorder& recorder);

    /**
     * @brief Run an integer 4-bit broadcast matvec on a loaded tensor.
     *
     * Computes `y[r] = sum_{k<cols} s4(W[r,k]) * s4(x[k])` on the GPU, where
     * `W` is the packed `rows` x `cols` signed-int4 tensor `name` (already
     * loaded by `load_tensor`) and `x` is ONE packed `cols`-wide signed-int4
     * activation vector in device memory broadcast to every row. Each output
     * is an exact signed 32-bit integer.
     *
     * This is the broadcast matvec, not the paired-rows primitive
     * `int4_dot_i32` (`out[r] = dot(a[r], b[r])` from two `count` x `n`
     * buffers): arity, dtype contract, and reuse all differ, so this path
     * routes through `int4_matvec_i32_traced` and never through
     * `int4_dot_i32`.
     *
     * @param[in] name     Tensor name; must have been loaded with `load_tensor`.
     * @param[in] x_int4   Packed signed-int4 activation vector base pointer in
     *                     device memory, exactly `cols / 2` bytes shared by all
     *                     rows.
     * @param[out] y       Output vector base pointer in device memory, `rows`
     *                     int32 values.
     * @param[in] rows     Output rows; must equal the tensor's row count.
     * @param[in] cols     Activation length; must equal the tensor's column
     *                     count and be a positive multiple of 8.
     * @param[in] stream   HIP stream (nullptr for default).
     * @param[in] recorder Tracer for the kernel launch.
     * @return 0 on success; `ENOENT` if not loaded; `EINVAL` on shape mismatch,
     *         on a non-integer format (ROCmFP4 `Codebook10`/dual-scale tensors
     *         carry magnitudes, not two's-complement int4, so they are rejected
     *         rather than silently mis-decoded), or on null pointers;
     *         `ENOTSUP` on a host-only build.
     * @pre The tensor is resident, `x_int4` holds `cols / 2` packed bytes,
     *      `y` holds `rows` int32 values.
     * @post On success the kernel launch is enqueued on `stream`.
     * @warning The tensor's nibbles must be genuine two's-complement signed
     *          int4; this entry point performs no per-nibble validation.
     */
    int matvec_int4x(std::string_view name, const void* x_int4, std::int32_t* y,
                     std::int64_t rows, std::int64_t cols, void* stream,
                     TraceRecorder& recorder);

    /**
     * @brief Access the underlying GGUF reader for direct payload reads.
     * @return Const reference to the reader; usable when the model is open.
     * @pre `is_open()` is true.
     */
    const GgufReader& reader() const noexcept { return reader_; }

private:
    GgufReader reader_;
    std::map<std::string, TensorSlot> slots_;  ///< Keyed by tensor name.
};

}  // namespace rocminfer

#endif  // ROCMINFER_MODEL_HPP
