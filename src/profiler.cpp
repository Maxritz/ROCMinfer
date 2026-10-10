/**
 * @file profiler.cpp
 * @brief Aggregation, reporting, and floor calibration for the trace recorder.
 *
 * The recorder maintains a fixed-size intern table of component labels and a
 * per-slot set of atomic counters. Device scopes reserve a HIP event pair at
 * open time, record an end timestamp at close time, and defer the elapsed-time
 * read until `resolve_pending`, so instrumentation never introduces a stream
 * synchronisation the engine did not already perform.
 *
 * @note The intern table is bounded. When it fills, further labels aggregate into
 *       a dedicated overflow bucket whose size is reported, because a truncated
 *       trace that looks complete is worse than no trace at all.
 * @warning Counters are relaxed atomics updated from every thread. This is
 *          correct for aggregation but does not give a consistent snapshot while
 *          other threads are still recording; call `snapshot` at a quiescent point.
 */

#include "rocminfer/profiler.hpp"

#include "device_pool_factory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <thread>

namespace rocminfer {

namespace {

constexpr std::uint32_t kOverflowSlot = 0xFFFFFFFFu;
constexpr std::size_t kMaxNameLen = 48;  ///< 47 characters plus the NUL terminator.
constexpr std::size_t kDomainCount = static_cast<std::size_t>(TraceDomain::kCount);

/// Domains whose cost is only meaningful when measured on the device.
constexpr bool domain_uses_device_events(TraceDomain domain) noexcept {
    return domain == TraceDomain::kDevice || domain == TraceDomain::kTransfer ||
           domain == TraceDomain::kAlloc || domain == TraceDomain::kSync;
}

/// FNV-1a over a component label, used to pick the first intern slot to probe.
std::uint32_t fnv1a(const char* text) noexcept {
    std::uint32_t hash = 2166136261u;
    for (; *text != '\0'; ++text) {
        hash ^= static_cast<std::uint32_t>(static_cast<unsigned char>(*text));
        hash *= 16777619u;
    }
    return hash;
}

/// Escape a component label for embedding in a JSON string literal.
std::string json_escape(const char* text) {
    std::string out;
    out.reserve(std::strlen(text) + 8);
    for (const char* p = text; *p != '\0'; ++p) {
        switch (*p) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(*p) < 0x20u) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(*p)));
                    out += buf;
                } else {
                    out += *p;
                }
                break;
        }
    }
    return out;
}

}  // namespace

const char* trace_domain_name(TraceDomain domain) noexcept {
    switch (domain) {
        case TraceDomain::kHost:     return "host";
        case TraceDomain::kDevice:   return "device";
        case TraceDomain::kTransfer: return "transfer";
        case TraceDomain::kAlloc:    return "alloc";
        case TraceDomain::kSync:     return "sync";
        case TraceDomain::kDispatch: return "dispatch";
        case TraceDomain::kCount:    break;
    }
    return "invalid";
}

/**
 * @brief One interned component label with its accumulated timings.
 *
 * `state` drives a three-state installation protocol: 0 = unclaimed, 1 = a thread
 * owns the slot and is writing `name`, 2 = `name` is published. Readers spin on
 * state 1, which lasts only for a bounded copy of at most 47 bytes.
 */
struct TracedSlot {
    std::atomic<std::uint8_t> state{0};  ///< 0 free, 1 installing, 2 ready.
    char name[kMaxNameLen] = {};
    std::atomic<std::uint64_t> ops{0};
    std::atomic<std::uint64_t> host_ns{0};
    std::atomic<std::uint64_t> dev_ns{0};
    std::atomic<std::uint64_t> domain_ops[kDomainCount]{};
};

/// A device scope awaiting a batched elapsed-time read.
struct PendingDeviceSample {
    std::uint32_t slot = kOverflowSlot;
    std::uint32_t begin_event = 0;
    std::uint32_t end_event = 0;
};

struct TraceRecorder::Impl {
    explicit Impl(const TraceConfig& cfg_in) : cfg(cfg_in) {
        enabled = cfg.report != TraceReport::kOff;
        if (enabled) {
            slots = std::make_unique<TracedSlot[]>(cfg.max_components);
        }
    }

    /// Find or install `component`, returning its slot index.
    std::uint32_t intern(const char* component) noexcept {
        const std::uint32_t count = cfg.max_components;
        const std::uint32_t start = fnv1a(component) % count;

        for (std::uint32_t probe = 0; probe < count; ++probe) {
            TracedSlot& slot = slots[(start + probe) % count];

            std::uint8_t state = slot.state.load(std::memory_order_acquire);
            if (state == 0) {
                std::uint8_t expected = 0;
                if (slot.state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
                    std::size_t length = 0;
                    while (component[length] != '\0' && length + 1 < kMaxNameLen) {
                        slot.name[length] = component[length];
                        ++length;
                    }
                    slot.name[length] = '\0';
                    slot.state.store(2, std::memory_order_release);
                    return static_cast<std::uint32_t>(&slot - slots.get());
                }
                state = 1;
            }

            if (state == 1) {
                int spins = 0;
                while (slot.state.load(std::memory_order_acquire) == 1) {
                    if (++spins > 64) {
                        std::this_thread::yield();
                    }
                }
                state = slot.state.load(std::memory_order_acquire);
            }

            if (state == 2 && std::strcmp(slot.name, component) == 0) {
                return static_cast<std::uint32_t>(&slot - slots.get());
            }
        }
        return kOverflowSlot;
    }

    /// Ensure the device pool exists before first use.
    bool ensure_pool() {
        if (!pool) {
            pool = detail::make_device_event_pool();
            if (!pool->init(cfg.device_event_pool)) {
                pool.reset();
                return false;
            }
        }
        return true;
    }

    TraceConfig cfg;
    bool enabled = false;
    std::unique_ptr<TracedSlot[]> slots;
    std::atomic<std::uint64_t> overflow_ops{0};
    std::atomic<std::uint64_t> overflow_host_ns{0};
    std::atomic<std::uint64_t> overflow_dev_ns{0};
    std::atomic<std::uint64_t> dropped_pending{0};
    /// Scopes whose device time was successfully measured; the floor divisor.
    std::atomic<std::uint64_t> device_measured{0};

    mutable std::mutex pending_mutex;
    std::vector<PendingDeviceSample> pending;

    std::unique_ptr<detail::DeviceEventPool> pool;

    mutable std::mutex floor_mutex;
    std::uint64_t floor_host_ns = 0;
    std::uint64_t floor_device_ns = 0;
    std::uint32_t floor_samples = 0;
};

TraceRecorder::TraceRecorder(const TraceConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
    if (config.max_components < 1) {
        throw std::invalid_argument("TraceRecorder: max_components must be >= 1");
    }
    if (config.floor_samples < 1) {
        throw std::invalid_argument("TraceRecorder: floor_samples must be >= 1");
    }
    if (config.device_event_pool < 2 || (config.device_event_pool % 2) != 0) {
        throw std::invalid_argument("TraceRecorder: device_event_pool must be even and >= 2");
    }
}

TraceRecorder::~TraceRecorder() {
    if (impl_ && impl_->pool) {
        impl_->pool->shutdown();
    }
}

bool TraceRecorder::enabled() const noexcept { return impl_->enabled; }

TraceReport TraceRecorder::report_mode() const noexcept { return impl_->cfg.report; }

std::uint64_t TraceRecorder::now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

TraceScopeToken TraceRecorder::begin_scope(const char* component, TraceDomain domain, void* stream,
                                           std::uint64_t /*now_ns*/) noexcept {
    TraceScopeToken token;
    if (!impl_->enabled || component == nullptr) {
        return token;
    }

    token.slot = impl_->intern(component);
    token.active = true;

    if (domain_uses_device_events(domain) && impl_->ensure_pool()) {
        std::uint32_t begin_event = 0;
        std::uint32_t end_event = 0;
        if (impl_->pool->acquire_pair(&begin_event, &end_event)) {
            if (impl_->pool->record(begin_event, stream)) {
                token.begin_event = begin_event;
                token.end_event = end_event;
                token.has_device = true;
            } else {
                impl_->pool->release_pair(begin_event, end_event);
            }
        }
    }

    return token;
}

void TraceRecorder::end_scope(const TraceScopeToken& token, TraceDomain domain, void* stream,
                              std::uint64_t host_start_ns, std::uint64_t host_end_ns) noexcept {
    if (!token.active) {
        return;
    }

    const std::uint64_t host_ns = (host_end_ns >= host_start_ns) ? (host_end_ns - host_start_ns) : 0;

    if (token.slot == kOverflowSlot) {
        impl_->overflow_ops.fetch_add(1, std::memory_order_relaxed);
        impl_->overflow_host_ns.fetch_add(host_ns, std::memory_order_relaxed);
        return;
    }

    TracedSlot& slot = impl_->slots[token.slot];
    slot.ops.fetch_add(1, std::memory_order_relaxed);
    slot.host_ns.fetch_add(host_ns, std::memory_order_relaxed);
    slot.domain_ops[static_cast<std::size_t>(domain)].fetch_add(1, std::memory_order_relaxed);

    if (token.has_device && impl_->pool) {
        if (impl_->pool->record(token.end_event, stream)) {
            std::lock_guard<std::mutex> guard(impl_->pending_mutex);
            PendingDeviceSample sample;
            sample.slot = token.slot;
            sample.begin_event = token.begin_event;
            sample.end_event = token.end_event;
            impl_->pending.push_back(sample);
        } else {
            impl_->pool->release_pair(token.begin_event, token.end_event);
            impl_->dropped_pending.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

bool TraceRecorder::resolve_pending() const {
    if (!impl_->enabled || !impl_->pool) {
        return true;
    }

    std::vector<PendingDeviceSample> batch;
    {
        std::lock_guard<std::mutex> guard(impl_->pending_mutex);
        if (impl_->pending.empty()) {
            return true;
        }
        batch.swap(impl_->pending);
    }

    bool ok = true;
    if (!batch.empty() && !impl_->pool->synchronize(batch.back().end_event)) {
        ok = false;
    }

    for (const PendingDeviceSample& sample : batch) {
        std::uint64_t dev_ns = 0;
        if (!impl_->pool->elapsed_ns(sample.begin_event, sample.end_event, &dev_ns)) {
            impl_->dropped_pending.fetch_add(1, std::memory_order_relaxed);
            ok = false;
        } else if (sample.slot == kOverflowSlot) {
            impl_->overflow_dev_ns.fetch_add(dev_ns, std::memory_order_relaxed);
            impl_->device_measured.fetch_add(1, std::memory_order_relaxed);
        } else {
            impl_->slots[sample.slot].dev_ns.fetch_add(dev_ns, std::memory_order_relaxed);
            impl_->device_measured.fetch_add(1, std::memory_order_relaxed);
        }
        impl_->pool->release_pair(sample.begin_event, sample.end_event);
    }

    return ok;
}

void TraceRecorder::measure_floor() {
    if (!impl_->enabled) {
        return;
    }

    const std::uint32_t samples = impl_->cfg.floor_samples;

    // Host floor: two clock reads plus an empty aggregation step, matching the
    // cost of an empty kHost scope.
    std::uint64_t host_total = 0;
    for (std::uint32_t i = 0; i < samples; ++i) {
        const std::uint64_t start = now_ns();
        host_total += (now_ns() - start);
    }

    // Device floor: an empty kDevice scope recorded through the real event path.
    // The pool holds device_event_pool/2 pairs and pairs stay reserved until
    // resolve_pending, so the queue is flushed periodically to keep every sample
    // measurable. The divisor is the number of scopes that actually obtained a
    // device event, not `samples`: dividing by a larger count would understate
    // the floor and make sub-floor measurements look significant.
    std::uint64_t device_total = 0;
    const std::uint64_t measured_before = impl_->device_measured.load(std::memory_order_relaxed);
    if (impl_->ensure_pool()) {
        const std::uint32_t flush_every = (impl_->cfg.device_event_pool / 2) > 0
                                            ? (impl_->cfg.device_event_pool / 2)
                                            : 1;
        for (std::uint32_t i = 0; i < samples; ++i) {
            TraceScopeToken token = begin_scope("__floor__", TraceDomain::kDevice, nullptr, 0);
            end_scope(token, TraceDomain::kDevice, nullptr, 0, 0);
            if ((i + 1) % flush_every == 0) {
                resolve_pending();
            }
        }
        resolve_pending();
        // The floor scopes aggregated under "__floor__"; recover and zero them.
        const std::uint32_t floor_slot = impl_->intern("__floor__");
        if (floor_slot != kOverflowSlot) {
            TracedSlot& slot = impl_->slots[floor_slot];
            device_total = slot.dev_ns.load(std::memory_order_relaxed);
            slot.dev_ns.store(0, std::memory_order_relaxed);
            slot.ops.store(0, std::memory_order_relaxed);
            slot.host_ns.store(0, std::memory_order_relaxed);
            for (std::size_t d = 0; d < kDomainCount; ++d) {
                slot.domain_ops[d].store(0, std::memory_order_relaxed);
            }
        }
    }
    // Divide by the number of scopes that actually produced a measurement. A
    // HIP event pair can be recorded successfully yet return a non-monotonic
    // elapsed time; dividing by the attempt count would understate the floor and
    // make sub-floor measurements look significant.
    const std::uint64_t device_measured_count =
        impl_->device_measured.load(std::memory_order_relaxed) - measured_before;

    std::lock_guard<std::mutex> guard(impl_->floor_mutex);
    impl_->floor_host_ns = host_total / (samples == 0 ? 1 : samples);
    impl_->floor_device_ns =
        (device_measured_count == 0) ? 0 : device_total / device_measured_count;
    impl_->floor_samples = samples;
}

std::uint64_t TraceRecorder::floor_host_ns() const noexcept {
    std::lock_guard<std::mutex> guard(impl_->floor_mutex);
    return impl_->floor_host_ns;
}

std::uint64_t TraceRecorder::floor_device_ns() const noexcept {
    std::lock_guard<std::mutex> guard(impl_->floor_mutex);
    return impl_->floor_device_ns;
}

std::uint32_t TraceRecorder::floor_sample_count() const noexcept {
    std::lock_guard<std::mutex> guard(impl_->floor_mutex);
    return impl_->floor_samples;
}

bool TraceRecorder::snapshot(std::vector<TraceRow>* out_rows) const {
    if (out_rows == nullptr) {
        return false;
    }
    if (!impl_->enabled) {
        return false;
    }

    resolve_pending();
    out_rows->clear();

    const std::uint32_t count = impl_->cfg.max_components;
    for (std::uint32_t i = 0; i < count; ++i) {
        const TracedSlot& slot = impl_->slots[i];
        const std::uint64_t ops = slot.ops.load(std::memory_order_relaxed);
        if (ops == 0) {
            continue;
        }
        TraceRow row;
        row.component = slot.name;
        row.ops = ops;
        row.host_ns = slot.host_ns.load(std::memory_order_relaxed);
        row.dev_ns = slot.dev_ns.load(std::memory_order_relaxed);
        row.idle_ns = (row.host_ns > row.dev_ns) ? (row.host_ns - row.dev_ns) : 0;
        TraceDomain best = TraceDomain::kHost;
        std::uint64_t best_count = 0;
        for (std::size_t d = 0; d < kDomainCount; ++d) {
            const std::uint64_t value = slot.domain_ops[d].load(std::memory_order_relaxed);
            if (value > best_count) {
                best_count = value;
                best = static_cast<TraceDomain>(d);
            }
        }
        row.domain = best;
        out_rows->push_back(row);
    }

    const std::uint64_t overflow_ops = impl_->overflow_ops.load(std::memory_order_relaxed);
    if (overflow_ops > 0) {
        TraceRow row;
        row.component = "__overflow__";
        row.ops = overflow_ops;
        row.host_ns = impl_->overflow_host_ns.load(std::memory_order_relaxed);
        row.dev_ns = impl_->overflow_dev_ns.load(std::memory_order_relaxed);
        row.idle_ns = (row.host_ns > row.dev_ns) ? (row.host_ns - row.dev_ns) : 0;
        row.domain = TraceDomain::kHost;
        out_rows->push_back(row);
    }

    std::sort(out_rows->begin(), out_rows->end(),
              [](const TraceRow& lhs, const TraceRow& rhs) {
                  if (lhs.dev_ns != rhs.dev_ns) {
                      return lhs.dev_ns > rhs.dev_ns;
                  }
                  return lhs.host_ns > rhs.host_ns;
              });
    return true;
}

bool TraceRecorder::report(std::ostream& os) const {
    if (!impl_->enabled) {
        return false;
    }

    std::vector<TraceRow> rows;
    snapshot(&rows);

    std::uint64_t total_dev = 0;
    for (const TraceRow& row : rows) {
        total_dev += row.dev_ns;
    }

    os << "     component           ops   %dev      dev us     idle us     host us\n";
    os << std::fixed;
    for (const TraceRow& row : rows) {
        const double dev_us = static_cast<double>(row.dev_ns) / 1000.0;
        const double idle_us = static_cast<double>(row.idle_ns) / 1000.0;
        const double host_us = static_cast<double>(row.host_ns) / 1000.0;
        const double pct = (total_dev > 0) ? (100.0 * static_cast<double>(row.dev_ns) /
                                                 static_cast<double>(total_dev))
                                           : 0.0;
        os << "     " << std::left << std::setw(20) << row.component << std::right
           << std::setw(5) << row.ops << std::setw(6) << std::setprecision(1) << pct << "%"
           << std::setw(12) << std::setprecision(2) << dev_us
           << std::setw(12) << idle_us << std::setw(12) << host_us << "\n";
    }

    std::lock_guard<std::mutex> guard(impl_->floor_mutex);
    if (impl_->floor_samples > 0) {
        const double host_us = static_cast<double>(impl_->floor_host_ns) / 1000.0;
        const double device_us = static_cast<double>(impl_->floor_device_ns) / 1000.0;
        os << "   instrumentation floor, " << impl_->floor_samples
           << " empty ops timed through the same begin/end path: "
           << std::setprecision(2) << host_us << " us host, " << device_us << " us device each.\n";
    }

    return static_cast<bool>(os);
}

bool TraceRecorder::report_json(std::ostream& os) const {
    if (!impl_->enabled) {
        return false;
    }

    std::vector<TraceRow> rows;
    snapshot(&rows);

    std::uint64_t total_dev = 0;
    for (const TraceRow& row : rows) {
        total_dev += row.dev_ns;
    }

    os << "{\n";
    os << "  \"schema\": \"rocminfer.perftrace/1\",\n";
    os << "  \"floor\": { \"samples\": " << floor_sample_count()
       << ", \"host_ns\": " << floor_host_ns()
       << ", \"device_ns\": " << floor_device_ns() << " },\n";
    os << "  \"components\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const TraceRow& row = rows[i];
        os << "    { \"component\": \"" << json_escape(row.component)
           << "\", \"domain\": \"" << trace_domain_name(row.domain)
           << "\", \"ops\": " << row.ops
           << ", \"host_ns\": " << row.host_ns
           << ", \"dev_ns\": " << row.dev_ns
           << ", \"idle_ns\": " << row.idle_ns
           << ", \"pct_dev\": " << std::fixed << std::setprecision(4)
           << ((total_dev > 0) ? (100.0 * static_cast<double>(row.dev_ns) /
                                   static_cast<double>(total_dev))
                               : 0.0)
           << " }" << (i + 1 < rows.size() ? "," : "") << "\n";
        os.unsetf(std::ios::floatfield);
    }
    os << "  ],\n";
    os << "  \"dropped\": { \"component_scopes\": " << dropped_component_scopes()
       << ", \"pending_scopes\": " << dropped_pending_scopes() << " }\n";
    os << "}\n";
    return static_cast<bool>(os);
}

std::uint64_t TraceRecorder::dropped_component_scopes() const noexcept {
    return impl_->overflow_ops.load(std::memory_order_relaxed);
}

std::uint64_t TraceRecorder::dropped_pending_scopes() const noexcept {
    return impl_->dropped_pending.load(std::memory_order_relaxed);
}

bool TraceRecorder::device_timing_available() const noexcept {
    return impl_->pool != nullptr && impl_->pool->available();
}

}  // namespace rocminfer