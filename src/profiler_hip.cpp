/**
 * @file profiler_hip.cpp
 * @brief HIP event pool backing device-side trace timing.
 *
 * Timing is recorded into a pool of reusable `hipEvent_t` pairs. A device scope
 * records a begin event at construction and an end event at destruction, then
 * defers the `hipEventElapsedTime` read to a single batch in `snapshot()`, so no
 * stream synchronisation happens on the hot path.
 *
 * @note Compiled only when `ROCMINFER_HAVE_HIP` is defined. The host-only
 *       configuration uses `profiler_device_null.cpp` instead.
 * @warning HIP event objects are not documented as safe to record from multiple
 *          host threads concurrently. This pool guards acquire/release and record
 *          with a mutex to stay within that contract.
 */

#if defined(ROCMINFER_HAVE_HIP)

#include "rocminfer/profiler.hpp"

#include <hip/hip_runtime.h>

#include <mutex>
#include <vector>

namespace rocminfer::detail {

namespace {

/// Clear any sticky HIP error so a failed probe does not poison later calls.
inline void clear_hip_error() noexcept {
    static_cast<void>(hipGetLastError());
}

/// Destroy an event, tolerating a partially constructed pool.
inline void destroy_event(hipEvent_t event) noexcept {
    static_cast<void>(hipEventDestroy(event));
}

}  // namespace

/**
 * @brief Reusable pool of HIP timing events.
 */
class HipEventPool final : public DeviceEventPool {
public:
    ~HipEventPool() override { shutdown(); }

    bool init(std::uint32_t capacity) override {
        if (available_) {
            return true;
        }
        if (capacity < 2) {
            return false;
        }
        const std::uint32_t pair_count = capacity / 2;
        events_.resize(pair_count * 2);

        std::lock_guard<std::mutex> guard(mutex_);
        for (std::uint32_t i = 0; i < events_.size(); ++i) {
            const hipError_t err = hipEventCreate(&events_[i]);
            if (err != hipSuccess) {
                clear_hip_error();
                for (std::uint32_t j = 0; j < i; ++j) {
                    destroy_event(events_[j]);
                }
                events_.clear();
                free_list_.clear();
                return false;
            }
        }
        free_list_.reserve(pair_count);
        for (std::uint32_t p = 0; p < pair_count; ++p) {
            free_list_.push_back(p * 2);
        }
        available_ = true;
        return true;
    }

    void shutdown() override {
        if (!available_) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        for (hipEvent_t& ev : events_) {
            destroy_event(ev);
        }
        events_.clear();
        free_list_.clear();
        available_ = false;
    }

    bool acquire_pair(std::uint32_t* out_begin, std::uint32_t* out_end) override {
        if (!available_) {
            return false;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        if (free_list_.empty()) {
            return false;
        }
        const std::uint32_t begin = free_list_.back();
        free_list_.pop_back();
        *out_begin = begin;
        *out_end = begin + 1;
        return true;
    }

    void release_pair(std::uint32_t begin, std::uint32_t end) override {
        if (!available_ || end != begin + 1) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        free_list_.push_back(begin);
    }

    bool record(std::uint32_t slot, void* stream) override {
        if (!available_ || slot >= events_.size()) {
            return false;
        }
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const hipError_t err = hipEventRecord(events_[slot], hip_stream);
        if (err != hipSuccess) {
            clear_hip_error();
            return false;
        }
        return true;
    }

    bool synchronize(std::uint32_t slot) override {
        if (!available_ || slot >= events_.size()) {
            return false;
        }
        const hipError_t err = hipEventSynchronize(events_[slot]);
        if (err != hipSuccess) {
            clear_hip_error();
            return false;
        }
        return true;
    }

    bool elapsed_ns(std::uint32_t begin, std::uint32_t end, std::uint64_t* out_ns) override {
        if (!available_ || begin >= events_.size() || end >= events_.size()) {
            return false;
        }
        float ms = 0.0f;
        const hipError_t err = hipEventElapsedTime(&ms, events_[begin], events_[end]);
        if (err != hipSuccess) {
            clear_hip_error();
            return false;
        }
        if (ms < 0.0f) {
            // Observed on ROCm 7.16 / gfx1201: hipEventElapsedTime returns
            // hipSuccess with a negative span when the two events bracket an
            // idle gap and the GPU clock is not monotonic between them. The
            // timestamps are already fixed, so re-reading cannot fix it. The
            // measurement is discarded and the caller counts it as unmeasured
            // rather than reporting a negative or clamped duration as real.
            return false;
        }
        // float milliseconds carry ~7 significant digits; scaling into nanoseconds
        // preserves sub-microsecond resolution, which is what the floor needs.
        *out_ns = static_cast<std::uint64_t>(static_cast<double>(ms) * 1.0e6);
        return true;
    }

    bool available() const noexcept override { return available_; }

private:
    std::vector<hipEvent_t> events_;
    std::vector<std::uint32_t> free_list_;  ///< Stores the begin index of each free pair.
    std::mutex mutex_;
    bool available_ = false;
};

}  // namespace rocminfer::detail

namespace rocminfer::detail {

/**
 * @brief Construct the HIP-backed event pool.
 * @return A pool that reports `available() == true` before `init` is called.
 */
std::unique_ptr<DeviceEventPool> make_device_event_pool() {
    return std::make_unique<HipEventPool>();
}

}  // namespace rocminfer::detail

#endif  // ROCMINFER_HAVE_HIP