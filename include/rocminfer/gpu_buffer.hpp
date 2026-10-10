/**
 * @file gpu_buffer.hpp
 * @brief RAII wrapper around HIP device memory allocations.
 *
 * Acquires device memory with `hipMalloc` in the constructor and releases it
 * with `hipFree` in the destructor, so a single return path or an exception
 * cannot leak a buffer. The size and raw pointer are queryable; the pointer
 * moves only on explicit `reset`.
 *
 * @note Only available when `ROCMINFER_HAVE_HIP` is defined. Host-only builds
 *       see a type whose methods return `EINVAL` rather than calling into the
 *       HIP runtime, so callers that do not depend on a GPU can still compile
 *       and link against the core library.
 * @warning The buffer is not thread-safe; distinct buffers are safe to use
 *          concurrently, but two threads must not operate on the same buffer
 *          without external synchronisation.
 */

#ifndef ROCMINFER_GPU_BUFFER_HPP
#define ROCMINFER_GPU_BUFFER_HPP

#include <cstdint>
#include <cstddef>

namespace rocminfer {

/**
 * @brief Owned GPU memory region.
 *
 * Wraps a single `hipMalloc` / `hipFree` pair. The buffer resizes only on
 * explicit calls to `resize` or `reset`; construction and destruction are
 * the only implicit size changes.
 */
class GpuBuffer {
public:
    /**
     * @brief Construct an empty buffer (no allocation).
     * @post `ptr() == nullptr` and `size() == 0`.
     */
    GpuBuffer() noexcept = default;

    /**
     * @brief Allocate `bytes` on the device.
     * @param[in] bytes Requested allocation size in bytes.
     * @return 0 on success, `ENOMEM` if the HIP runtime rejects the request or
     *         HIP is not compiled in, `EINVAL` if `bytes` is zero.
     * @pre The calling thread has a HIP context.
     * @post On success `ptr() != nullptr` and `size() == bytes`. On failure
     *       the object is left empty and no allocation leaks.
     */
    int allocate(std::size_t bytes);

    /**
     * @brief Release the allocation and reset to the empty state.
     * @post `ptr() == nullptr` and `size() == 0`.
     */
    void reset() noexcept;

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;

    ~GpuBuffer();

    /**
     * @brief Access the raw device pointer.
     * @return The pointer passed to device APIs, or `nullptr` when empty.
     */
    void* ptr() const noexcept { return ptr_; }

    /**
     * @brief Report the allocation size in bytes.
     * @return Size at the last successful `allocate`, or 0 when empty.
     */
    std::size_t size() const noexcept { return size_; }

    bool empty() const noexcept { return ptr_ == nullptr; }

private:
    void* ptr_ = nullptr;       ///< Device pointer; nullptr when unallocated.
    std::size_t size_ = 0;      ///< Bytes in the current allocation.
};

}  // namespace rocminfer

#endif  // ROCMINFER_GPU_BUFFER_HPP
