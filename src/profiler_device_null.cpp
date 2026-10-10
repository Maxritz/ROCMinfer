/**
 * @file profiler_device_null.cpp
 * @brief Device event pool for builds without a HIP runtime.
 *
 * This is the complete, correct implementation of `detail::DeviceEventPool` for
 * configurations where no GPU timing backend is linked. Every device query
 * reports unavailability, so `TraceRecorder` records host time only and reports
 * `device_timing_available() == false`. Host-only unit tests use this to exercise
 * the interning, aggregation, and reporting paths without a GPU present.
 *
 * @note This file is not compiled when `ROCMINFER_HAVE_HIP` is defined.
 * @warning A report produced in this configuration has no `dev us` column
 *          content. Do not read it as a device bottleneck measurement.
 */

#if !defined(ROCMINFER_HAVE_HIP)

#include "device_pool_factory.hpp"

namespace rocminfer::detail {

namespace {

/**
 * @brief Event pool that reports every capability as absent.
 */
class NullEventPool final : public DeviceEventPool {
public:
    bool init(std::uint32_t /*capacity*/) override { return false; }

    void shutdown() override {}

    bool acquire_pair(std::uint32_t* /*out_begin*/, std::uint32_t* /*out_end*/) override {
        return false;
    }

    void release_pair(std::uint32_t /*begin*/, std::uint32_t /*end*/) override {}

    bool record(std::uint32_t /*slot*/, void* /*stream*/) override { return false; }

    bool synchronize(std::uint32_t /*slot*/) override { return false; }

    bool elapsed_ns(std::uint32_t /*begin*/, std::uint32_t /*end*/,
                    std::uint64_t* /*out_ns*/) override {
        return false;
    }

    bool available() const noexcept override { return false; }
};

}  // namespace

std::unique_ptr<DeviceEventPool> make_device_event_pool() {
    return std::make_unique<NullEventPool>();
}

}  // namespace rocminfer::detail

#endif  // !ROCMINFER_HAVE_HIP