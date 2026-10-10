/**
 * @file device_pool_factory.hpp
 * @brief Internal factory selecting the device-timing backend for a build.
 * @warning This header is private to the implementation; it is not installed and
 *          carries no stability guarantee.
 */

#ifndef ROCMINFER_DEVICE_POOL_FACTORY_HPP
#define ROCMINFER_DEVICE_POOL_FACTORY_HPP

#include "rocminfer/profiler.hpp"

namespace rocminfer::detail {

/**
 * @brief Construct the device event pool appropriate for this build.
 * @return A pool that yields device timings when `ROCMINFER_HAVE_HIP` is defined,
 *         otherwise a pool that reports `available() == false`.
 * @post The caller owns the returned pool and must destroy it.
 */
std::unique_ptr<DeviceEventPool> make_device_event_pool();

}  // namespace rocminfer::detail

#endif  // ROCMINFER_DEVICE_POOL_FACTORY_HPP