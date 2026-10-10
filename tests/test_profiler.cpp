/**
 * @file test_profiler.cpp
 * @brief Unit tests for the trace recorder.
 *
 * Covers interning (including label reuse and overflow), host-time aggregation,
 * idle clamping, deferred device resolution, the off path, and report emission.
 * The device-specific assertions run only when a HIP backend is compiled in and
 * reports itself available; host aggregation is verified unconditionally.
 *
 * @note Self-contained harness: no external test framework dependency, so the
 *       suite builds anywhere the engine builds.
 */

#include "rocminfer/profiler.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "FAIL line " << line << ": " << expression << "\n";
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

/// Find a row by component label; returns nullptr when absent.
const rocminfer::TraceRow* find_row(const std::vector<rocminfer::TraceRow>& rows,
                                    const std::string& name) {
    for (const rocminfer::TraceRow& row : rows) {
        if (name == row.component) {
            return &row;
        }
    }
    return nullptr;
}

void test_disabled_recorder_records_nothing() {
    rocminfer::TraceConfig config;  // report defaults to kOff
    rocminfer::TraceRecorder recorder(config);
    CHECK(!recorder.enabled());
    CHECK(recorder.report_mode() == rocminfer::TraceReport::kOff);

    rocminfer::TraceScope scope(recorder, "attention", rocminfer::TraceDomain::kHost);
    CHECK(!scope.active());

    std::vector<rocminfer::TraceRow> rows;
    CHECK(!recorder.snapshot(&rows));
    CHECK(rows.empty());
    CHECK(recorder.floor_host_ns() == 0);
}

void test_scope_aggregates_ops_and_host_time() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(config);
    CHECK(recorder.enabled());

    for (int i = 0; i < 5; ++i) {
        rocminfer::TraceScope scope(recorder, "norms", rocminfer::TraceDomain::kHost);
        CHECK(scope.active());
    }

    std::vector<rocminfer::TraceRow> rows;
    CHECK(recorder.snapshot(&rows));
    CHECK(rows.size() == 1);

    const rocminfer::TraceRow* norms = find_row(rows, "norms");
    CHECK(norms != nullptr);
    if (norms != nullptr) {
        CHECK(norms->ops == 5);
        CHECK(norms->host_ns > 0);
        CHECK(norms->domain == rocminfer::TraceDomain::kHost);
        // Host-only scopes have no device time, so idle equals host time.
        CHECK(norms->idle_ns == norms->host_ns);
    }
}

void test_distinct_labels_share_one_intern_table() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(config);

    {
        rocminfer::TraceScope scope(recorder, "attention", rocminfer::TraceDomain::kHost);
    }
    {
        rocminfer::TraceScope scope(recorder, "projections", rocminfer::TraceDomain::kHost);
    }
    {
        rocminfer::TraceScope scope(recorder, "attention", rocminfer::TraceDomain::kHost);
    }

    std::vector<rocminfer::TraceRow> rows;
    CHECK(recorder.snapshot(&rows));
    CHECK(rows.size() == 2);

    const rocminfer::TraceRow* attention = find_row(rows, "attention");
    const rocminfer::TraceRow* projections = find_row(rows, "projections");
    CHECK(attention != nullptr);
    CHECK(projections != nullptr);
    if (attention != nullptr) {
        // The repeated label merged rather than creating a third row.
        CHECK(attention->ops == 2);
    }
    if (projections != nullptr) {
        CHECK(projections->ops == 1);
    }
}

void test_component_table_overflow_is_reported_not_hidden() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    config.max_components = 2;
    rocminfer::TraceRecorder recorder(config);

    for (int i = 0; i < 4; ++i) {
        rocminfer::TraceScope scope(recorder, i == 0 ? "alpha" : (i == 1 ? "beta" : "gamma"),
                                    rocminfer::TraceDomain::kHost);
    }

    std::vector<rocminfer::TraceRow> rows;
    CHECK(recorder.snapshot(&rows));
    CHECK(recorder.dropped_component_scopes() == 2);

    // Two interned labels plus the overflow bucket.
    CHECK(rows.size() == 3);
    const rocminfer::TraceRow* overflow = find_row(rows, "__overflow__");
    CHECK(overflow != nullptr);
    if (overflow != nullptr) {
        CHECK(overflow->ops == 2);
    }
}

void test_snapshot_is_idempotent() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(config);

    {
        rocminfer::TraceScope scope(recorder, "residual", rocminfer::TraceDomain::kHost);
    }

    std::vector<rocminfer::TraceRow> first;
    std::vector<rocminfer::TraceRow> second;
    CHECK(recorder.snapshot(&first));
    CHECK(recorder.snapshot(&second));
    CHECK(first.size() == second.size());
    if (!first.empty() && !second.empty()) {
        CHECK(first[0].ops == second[0].ops);
        CHECK(first[0].host_ns == second[0].host_ns);
    }
}

void test_concurrent_scopes_are_counted_exactly() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    config.max_components = 8;
    rocminfer::TraceRecorder recorder(config);

    constexpr int kThreads = 4;
    constexpr int kIterations = 500;

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&recorder]() {
            for (int i = 0; i < kIterations; ++i) {
                rocminfer::TraceScope scope(recorder, "projections",
                                            rocminfer::TraceDomain::kHost);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    std::vector<rocminfer::TraceRow> rows;
    CHECK(recorder.snapshot(&rows));
    CHECK(recorder.dropped_component_scopes() == 0);

    const rocminfer::TraceRow* projections = find_row(rows, "projections");
    CHECK(projections != nullptr);
    if (projections != nullptr) {
        CHECK(projections->ops == static_cast<std::uint64_t>(kThreads) *
                                     static_cast<std::uint64_t>(kIterations));
    }
}

void test_floor_measurement_is_positive_and_reported() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    config.floor_samples = 64;
    rocminfer::TraceRecorder recorder(config);

    recorder.measure_floor();
    CHECK(recorder.floor_sample_count() == 64);
    CHECK(recorder.floor_host_ns() > 0);

    // Floor scopes must not appear as a reportable component.
    std::vector<rocminfer::TraceRow> rows;
    CHECK(recorder.snapshot(&rows));
    CHECK(find_row(rows, "__floor__") == nullptr);

    std::ostringstream stream;
    CHECK(recorder.report(stream));
    const std::string text = stream.str();
    CHECK(text.find("instrumentation floor") != std::string::npos);
    CHECK(text.find("empty ops timed through the same begin/end path") != std::string::npos);
}

void test_device_scopes_resolve_or_degrade_honestly() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(config);

    for (int i = 0; i < 4; ++i) {
        rocminfer::TraceScope scope(recorder, "attention", rocminfer::TraceDomain::kDevice);
    }

    std::vector<rocminfer::TraceRow> rows;
    CHECK(recorder.snapshot(&rows));

    const rocminfer::TraceRow* attention = find_row(rows, "attention");
    CHECK(attention != nullptr);
    if (attention != nullptr) {
        CHECK(attention->ops == 4);
        // Host time is always recorded, even with no device backend.
        CHECK(attention->host_ns > 0);
        CHECK(attention->domain == rocminfer::TraceDomain::kDevice);
    }

    if (recorder.device_timing_available()) {
        // A real HIP event pair must have produced a real elapsed measurement.
        // An empty scope measures near the device instrumentation floor rather
        // than zero, which is what distinguishes "measured" from "not measured".
        CHECK(attention->dev_ns > 0);
        // Drops are permitted: on ROCm 7.16 / gfx1201, hipEventElapsedTime can
        // return hipSuccess with a non-monotonic span, which the backend rejects.
        // What must hold is that drops never exceed the scopes actually opened.
        CHECK(recorder.dropped_pending_scopes() <= attention->ops);
        std::cout << "device timing active: " << attention->dev_ns << " ns over "
                  << attention->ops << " scopes, " << recorder.dropped_pending_scopes()
                  << " unmeasurable\n";
    } else {
        // Without a backend, device time must be absent rather than fabricated.
        CHECK(attention->dev_ns == 0);
        std::cout << "device timing unavailable in this build; host path verified only\n";
    }
}

void test_report_emits_table_header_and_rows() {
    rocminfer::TraceConfig config;
    config.report = rocminfer::TraceReport::kTable;
    rocminfer::TraceRecorder recorder(config);

    {
        rocminfer::TraceScope scope(recorder, "attention", rocminfer::TraceDomain::kHost);
    }
    {
        rocminfer::TraceScope scope(recorder, "embed", rocminfer::TraceDomain::kHost);
    }
    recorder.measure_floor();

    std::ostringstream table;
    CHECK(recorder.report(table));
    const std::string text = table.str();
    CHECK(text.find("component") != std::string::npos);
    CHECK(text.find("dev us") != std::string::npos);
    CHECK(text.find("idle us") != std::string::npos);
    CHECK(text.find("host us") != std::string::npos);
    CHECK(text.find("attention") != std::string::npos);
    CHECK(text.find("embed") != std::string::npos);

    std::ostringstream json;
    CHECK(recorder.report_json(json));
    const std::string json_text = json.str();
    CHECK(json_text.find("rocminfer.perftrace/1") != std::string::npos);
    CHECK(json_text.find("\"components\"") != std::string::npos);
    CHECK(json_text.find("\"attention\"") != std::string::npos);
    CHECK(json_text.find("\"dropped\"") != std::string::npos);
}

void test_domain_names_are_distinct() {
    const rocminfer::TraceDomain domains[] = {
        rocminfer::TraceDomain::kHost,     rocminfer::TraceDomain::kDevice,
        rocminfer::TraceDomain::kTransfer, rocminfer::TraceDomain::kAlloc,
        rocminfer::TraceDomain::kSync,     rocminfer::TraceDomain::kDispatch,
    };
    for (std::size_t i = 0; i < std::size(domains); ++i) {
        for (std::size_t j = i + 1; j < std::size(domains); ++j) {
            CHECK(std::strcmp(rocminfer::trace_domain_name(domains[i]),
                              rocminfer::trace_domain_name(domains[j])) != 0);
        }
    }
}

}  // namespace

int main() {
    test_disabled_recorder_records_nothing();
    test_scope_aggregates_ops_and_host_time();
    test_distinct_labels_share_one_intern_table();
    test_component_table_overflow_is_reported_not_hidden();
    test_snapshot_is_idempotent();
    test_concurrent_scopes_are_counted_exactly();
    test_floor_measurement_is_positive_and_reported();
    test_device_scopes_resolve_or_degrade_honestly();
    test_report_emits_table_header_and_rows();
    test_domain_names_are_distinct();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all profiler checks passed\n";
    return 0;
}