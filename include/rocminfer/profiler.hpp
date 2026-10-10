/**
 * @file profiler.hpp
 * @brief Per-component performance tracing for ROCMinfer.
 *
 * Every engine component that does measurable work opens a `TraceScope`. The
 * recorder separates device execution time from host work from idle wait, so a
 * bottleneck can be attributed to a logical component instead of guessed at.
 *
 * The contract, including column semantics and the instrumentation floor, is
 * specified in `docs/PERF_TRACE.md`. This header deliberately includes no HIP
 * type; device resources are held behind `detail::DeviceEventPool`.
 *
 * @note Tracing is disabled unless a `TraceConfig` with a non-`kOff` report mode
 *       is supplied to the recorder. In that case every scope degenerates to a
 *       single flag test with no allocation, clock call, or event creation.
 * @warning A `TraceRecorder` must not be destroyed while any `TraceScope`
 *          referencing it is still open.
 */

#ifndef ROCMINFER_PROFILER_HPP
#define ROCMINFER_PROFILER_HPP

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rocminfer {

/**
 * @brief Category of work a traced scope performs.
 *
 * The domain distinguishes device compute from staging copies, synchronisation
 * waits, allocation, and dispatch, so that identical-looking idle time can be
 * attributed to the right cause.
 */
enum class TraceDomain : std::uint8_t {
    kHost = 0,      ///< Pure host computation; no GPU events are recorded.
    kDevice = 1,    ///< GPU kernel execution; `dev_ns` is HIP event time.
    kTransfer = 2,  ///< Host/device or device/device data movement.
    kAlloc = 3,     ///< Device or host allocation and deallocation.
    kSync = 4,      ///< Stream/event synchronisation and queue waits.
    kDispatch = 5,  ///< Graph or kernel launch bookkeeping.
    kCount = 6,     ///< Number of domains; not a valid domain value.
};

/**
 * @brief Output format selected by `--profileperf`.
 */
enum class TraceReport : std::uint8_t {
    kOff = 0,   ///< Tracing disabled. This is the default.
    kTable = 1, ///< Human-readable table on stdout.
    kJson = 2,  ///< Machine-readable JSON document on stdout.
};

/**
 * @brief Human-readable name of a trace domain.
 * @param[in] domain Domain to name; must not be `TraceDomain::kCount`.
 * @return Stable NUL-terminated string owned by the library; never NULL.
 * @pre `domain != TraceDomain::kCount`.
 * @post The returned pointer remains valid for the lifetime of the process.
 */
const char* trace_domain_name(TraceDomain domain) noexcept;

/**
 * @brief Construction-time configuration for a `TraceRecorder`.
 */
struct TraceConfig {
    TraceReport report = TraceReport::kOff;  ///< kOff disables all tracing.
    std::uint32_t max_components = 256;      ///< Intern table capacity; see dropped count.
    std::uint32_t floor_samples = 256;       ///< Empty scopes timed to calibrate the floor.
    std::uint32_t device_event_pool = 256;   ///< Reusable HIP event pairs.
    std::uint32_t max_pending = 65536;       ///< Device scopes awaiting deferred resolution.
    bool measure_floor = true;               ///< Calibrate and print the floor at report time.
};

/**
 * @brief Handle identifying one open scope.
 *
 * Opaque bookkeeping produced by `TraceRecorder::begin_scope` and consumed
 * exactly once by `TraceRecorder::end_scope`.
 */
struct TraceScopeToken {
    std::uint32_t slot = 0xFFFFFFFFu;      ///< Intern table index, or the overflow slot.
    std::uint32_t begin_event = 0;         ///< Device event reserved at scope start.
    std::uint32_t end_event = 0;           ///< Device event reserved at scope start.
    bool has_device = false;               ///< True when a device event pair was reserved.
    bool active = false;                   ///< False when tracing is disabled.
};

namespace detail {

/**
 * @brief Backend-specific pool of reusable HIP event pairs.
 *
 * Declarations live here so the public header stays free of HIP types.
 * `profiler_hip.cpp` supplies the HIP implementation; `profiler_device_null.cpp`
 * supplies the host-only configuration, in which every device query reports
 * unavailability and device scopes record no device time.
 */
class DeviceEventPool {
public:
    virtual ~DeviceEventPool() = default;

    /**
     * @brief Create the backing event objects.
     * @param[in] capacity Total number of event slots; rounded down to pairs.
     * @return true if at least one event pair is usable.
     * @pre `capacity >= 2`.
     * @post On success, `capacity/2` pairs may be acquired concurrently.
     */
    virtual bool init(std::uint32_t capacity) = 0;

    /**
     * @brief Release all event objects and any pool bookkeeping.
     * @pre No pair acquired from this pool is still outstanding.
     */
    virtual void shutdown() = 0;

    /**
     * @brief Take an unused event pair from the pool.
     * @param[out] out_begin Receives the begin-event index; untouched on failure.
     * @param[out] out_end Receives the end-event index; untouched on failure.
     * @return true if a pair was acquired; false if the pool is exhausted.
     * @pre `out_begin` and `out_end` are non-NULL.
     * @post On success the caller owns the pair until `release_pair`.
     */
    virtual bool acquire_pair(std::uint32_t* out_begin, std::uint32_t* out_end) = 0;

    /**
     * @brief Return a previously acquired pair to the pool.
     * @param[in] begin Begin-event index from `acquire_pair`.
     * @param[in] end   End-event index from `acquire_pair`.
     * @pre The pair came from `acquire_pair` on this pool.
     */
    virtual void release_pair(std::uint32_t begin, std::uint32_t end) = 0;

    /**
     * @brief Record a device timestamp on a stream.
     * @param[in] slot   Event index.
     * @param[in] stream HIP stream; NULL selects the default stream.
     * @return true if the event was recorded.
     * @pre `slot` was obtained from `acquire_pair`.
     */
    virtual bool record(std::uint32_t slot, void* stream) = 0;

    /**
     * @brief Wait until a previously recorded event has completed.
     * @param[in] slot Event index.
     * @return true if the event has completed.
     * @pre `record` was called on `slot` beforehand.
     */
    virtual bool synchronize(std::uint32_t slot) = 0;

    /**
     * @brief Measure elapsed device time between two completed events.
     * @param[in]  begin  Begin-event index.
     * @param[in]  end    End-event index.
     * @param[out] out_ns Receives elapsed nanoseconds; untouched on failure.
     * @return true if the measurement was taken.
     * @pre Both events were recorded and have completed.
     */
    virtual bool elapsed_ns(std::uint32_t begin, std::uint32_t end, std::uint64_t* out_ns) = 0;

    /**
     * @brief Report whether this backend can measure device time at all.
     * @return true when HIP event timing is available in this build.
     */
    virtual bool available() const noexcept = 0;
};

}  // namespace detail

/**
 * @brief Aggregated timing row for one logical component.
 */
struct TraceRow {
    const char* component = nullptr;                ///< Interned label, owned by the recorder.
    std::uint64_t ops = 0;                          ///< Number of closed scopes.
    std::uint64_t host_ns = 0;                      ///< Total wall-clock host time inside scopes.
    std::uint64_t dev_ns = 0;                       ///< Total device event execution time.
    std::uint64_t idle_ns = 0;                      ///< `host_ns - dev_ns`, clamped at zero.
    TraceDomain domain = TraceDomain::kCount;       ///< Domain contributing the most ops.
};

/**
 * @brief Owns component interning, timing aggregation, and the deferred device
 *        resolution queue.
 *
 * A recorder is created explicitly by the engine and passed by reference to
 * every `TraceScope`. There is no global recorder and no static mutable state,
 * so tests and concurrent engines can each hold an independent one.
 *
 * Device scopes do not synchronise when they close. They enqueue an event pair,
 * and elapsed times are resolved in one batch by `resolve_pending`, which should
 * be called where the engine already synchronises. Synchronising per scope
 * would serialise the engine and make the instrument change what is measured.
 */
class TraceRecorder {
public:
    /**
     * @brief Construct a recorder and allocate its tables.
     * @param[in] config Configuration; the default disables tracing.
     * @pre `config.max_components >= 1`, `config.floor_samples >= 1`,
     *      and `config.device_event_pool` is even and at least 2.
     * @post No device events exist yet; the pool initialises on first need.
     */
    explicit TraceRecorder(const TraceConfig& config = TraceConfig{});

    /**
     * @brief Destroy the recorder, releasing event objects and buffers.
     * @pre No `TraceScope` referencing this recorder is still open.
     */
    ~TraceRecorder();

    TraceRecorder(const TraceRecorder&) = delete;
    TraceRecorder& operator=(const TraceRecorder&) = delete;
    TraceRecorder(TraceRecorder&&) = delete;
    TraceRecorder& operator=(TraceRecorder&&) = delete;

    /**
     * @brief Report whether tracing is active.
     * @return true when a non-`kOff` report mode was configured.
     */
    bool enabled() const noexcept;

    /**
     * @brief Read the monotonic host clock that supplies `host_ns`.
     * @return Nanoseconds since an unspecified fixed origin.
     * @note This is the only sanctioned clock read in the engine. Instrumentation
     *       outside `TraceScope` must not call it.
     */
    static std::uint64_t now_ns() noexcept;

    /**
     * @brief Report the configured output format.
     * @return The report mode supplied at construction.
     */
    TraceReport report_mode() const noexcept;

    /**
     * @brief Open a scope, interning the component and reserving device events.
     * @param[in]  component Stable NUL-terminated label, non-NULL, at most 47 characters.
     * @param[in]  domain    Domain recorded when the scope closes.
     * @param[in]  stream    HIP stream for device domains; NULL selects the default stream.
     * @param[in]  now_ns    Host timestamp captured by `now_ns()` at scope entry.
     * @return A token to pass to `end_scope`; `active == false` when tracing is off.
     * @pre `component != NULL` and outlives the recorder.
     * @post Device-capable domains reserve one event pair that stays owned until
     *       `resolve_pending` runs.
     */
    TraceScopeToken begin_scope(const char* component, TraceDomain domain, void* stream,
                                std::uint64_t now_ns) noexcept;

    /**
     * @brief Close a scope and record its host time.
     * @param[in] token         Token returned by the matching `begin_scope`.
     * @param[in] domain        Domain passed to the matching `begin_scope`.
     * @param[in] stream        Stream passed to the matching `begin_scope`.
     * @param[in] host_start_ns Timestamp passed to the matching `begin_scope`.
     * @param[in] host_end_ns   Timestamp captured by `now_ns()` at scope exit.
     * @pre `token.active == true`; the token must not be used again.
     * @post Component counters include this scope exactly once.
     */
    void end_scope(const TraceScopeToken& token, TraceDomain domain, void* stream,
                   std::uint64_t host_start_ns, std::uint64_t host_end_ns) noexcept;

    /**
     * @brief Resolve every outstanding device scope and fold it into the totals.
     * @return true if all outstanding scopes resolved; false if any were dropped.
     * @post The pending queue is empty. Calling twice in a row is harmless and
     *       does not double count.
     * @warning Blocks until the device queue reaches the last recorded end event.
     *          Call it where the engine already synchronises.
     */
    bool resolve_pending() const;

    /**
     * @brief Calibrate the instrumentation floor through the identical code path.
     *
     * Pushes `floor_samples()` empty host scopes and, when the device backend is
     * available, the same number of empty device scopes through the same
     * acquire/record/enqueue path used by real scopes.
     *
     * @pre `enabled()` is true.
     * @post `floor_host_ns()` and `floor_sample_count()` are non-zero.
     * @warning This measures the instrument, so never call it while timing a workload.
     */
    void measure_floor();

    /**
     * @brief Per-op host cost of an empty traced scope.
     * @return Nanoseconds, or 0 if the floor was never measured.
     */
    std::uint64_t floor_host_ns() const noexcept;

    /**
     * @brief Per-op device cost of an empty traced device scope.
     * @return Nanoseconds, or 0 if unmeasured or no HIP backend is present.
     */
    std::uint64_t floor_device_ns() const noexcept;

    /**
     * @brief Number of empty scopes timed to produce the floor.
     * @return The sample count used by the last `measure_floor` call.
     */
    std::uint32_t floor_sample_count() const noexcept;

    /**
     * @brief Resolve pending device scopes and aggregate every component.
     * @param[out] out_rows Cleared and refilled with one row per component with non-zero ops.
     * @return true on success; false if a pending scope could not be resolved.
     * @pre `out_rows` is non-NULL.
     * @post Rows are sorted by `dev_ns` descending, then `host_ns` descending.
     */
    bool snapshot(std::vector<TraceRow>* out_rows) const;

    /**
     * @brief Print the human-readable table followed by the instrumentation floor.
     * @param[in,out] os Destination stream.
     * @return true on success.
     * @pre `enabled()` is true.
     * @post Exactly one floor line is emitted when the floor was measured.
     */
    bool report(std::ostream& os) const;

    /**
     * @brief Print the JSON report for the `rocminfer.perftrace/1` schema.
     * @param[in,out] os Destination stream.
     * @return true on success.
     * @pre `enabled()` is true.
     */
    bool report_json(std::ostream& os) const;

    /**
     * @brief Scopes that could not be attributed to a named component.
     * @return Count aggregated into the `__overflow__` row.
     * @note Non-zero means the component table filled; their time was still recorded.
     */
    std::uint64_t dropped_component_scopes() const noexcept;

    /**
     * @brief Device scopes discarded because their event pair could not be resolved.
     * @return Count of discarded device samples.
     */
    std::uint64_t dropped_pending_scopes() const noexcept;

    /**
     * @brief Whether device timing is available in this build and configuration.
     * @return true when HIP event timing is compiled in and the pool initialised.
     */
    bool device_timing_available() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief RAII scope covering one component's measurable work.
 *
 * Construction opens the scope, destruction closes it. The destructor is
 * `noexcept` and performs no allocation, so it is safe on unwind paths.
 *
 * @note A disabled recorder makes the scope a no-op beyond one flag test.
 * @warning Do not nest two scopes on the same component at the same depth; the
 *          resulting aggregates are ambiguous. Use distinct labels for nested work.
 */
class TraceScope final {
public:
    /**
     * @brief Open a scope on `recorder`.
     * @param[in] recorder  Recorder to aggregate into; must outlive this scope.
     * @param[in] component Stable label naming the logical role, non-NULL.
     * @param[in] domain    Category of work performed.
     * @param[in] stream    HIP stream for device domains; NULL selects the default stream.
     * @pre `recorder.enabled()` may be false; the scope then records nothing.
     */
    TraceScope(TraceRecorder& recorder, const char* component, TraceDomain domain,
               void* stream = nullptr) noexcept
        : recorder_(&recorder), domain_(domain), stream_(stream), host_start_ns_(0) {
        if (!recorder_->enabled()) {
            return;
        }
        host_start_ns_ = TraceRecorder::now_ns();
        token_ = recorder_->begin_scope(component, domain_, stream_, host_start_ns_);
    }

    /**
     * @brief Close the scope and aggregate its timings.
     * @pre The scope was opened successfully and is not being closed twice.
     */
    ~TraceScope() noexcept {
        if (!token_.active) {
            return;
        }
        recorder_->end_scope(token_, domain_, stream_, host_start_ns_,
                             TraceRecorder::now_ns());
    }

    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;
    TraceScope(TraceScope&&) = delete;
    TraceScope& operator=(TraceScope&&) = delete;

    /**
     * @brief Report whether this scope is recording.
     * @return false when the recorder was disabled at construction.
     */
    bool active() const noexcept { return token_.active; }

private:
    TraceRecorder* recorder_;   ///< Never null; the caller always supplies a recorder.
    TraceScopeToken token_;     ///< Valid only while `active` is true.
    TraceDomain domain_;        ///< Echoed back to `end_scope`.
    void* stream_;              ///< Opaque HIP stream echoed back to `end_scope`.
    std::uint64_t host_start_ns_;  ///< Entry timestamp; valid only while `active`.
};

}  // namespace rocminfer

#endif  // ROCMINFER_PROFILER_HPP