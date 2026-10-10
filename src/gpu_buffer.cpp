/**
 * @file gpu_buffer.cpp
 * @brief Implementation of `GpuBuffer`, the RAII HIP allocation wrapper.
 *
 * Only compiled when `ROCMINFER_HAVE_HIP` is defined; otherwise a stub that
 * returns `ENOMEM` is emitted so the rest of the engine links without a GPU.
 */

#include "rocminfer/gpu_buffer.hpp"

#ifdef ROCMINFER_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

#include <cerrno>

namespace rocminfer {

int GpuBuffer::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return EINVAL;
    }
    reset();
#ifdef ROCMINFER_HAVE_HIP
    void* ptr = nullptr;
    const hipError_t err = hipMalloc(&ptr, bytes);
    if (err != hipSuccess) {
        return ENOMEM;
    }
    ptr_ = ptr;
    size_ = bytes;
    return 0;
#else
    (void)bytes;
    return ENOMEM;
#endif
}

void GpuBuffer::reset() noexcept {
#ifdef ROCMINFER_HAVE_HIP
    if (ptr_ != nullptr) {
        static_cast<void>(hipFree(ptr_));
    }
#endif
    ptr_ = nullptr;
    size_ = 0;
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

GpuBuffer::~GpuBuffer() { reset(); }

}  // namespace rocminfer
