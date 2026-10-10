/**
 * @file main.cpp
 * @brief Command-line front end for ROCMinfer.
 *
 * Parses the global switches, configures the trace recorder, and dispatches to
 * the requested subcommand. Inference subcommands are registered by later phases;
 * `--profile-floor` and `--help` are functional now.
 *
 * @note This is the only translation unit permitted to throw to the terminal.
 *       Everything else reports failures through status codes.
 */

#include "rocminfer/profiler.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 2;

void print_usage(std::ostream& os, std::string_view program) {
    os << "usage: " << program << " [options] [command]\n"
       << "\n"
       << "Commands:\n"
       << "  floor           Measure and print the tracing instrumentation floor.\n"
       << "\n"
       << "Global options:\n"
       << "  --profileperf[=table|json|off]\n"
       << "                 Record per-component timings and print a report at exit.\n"
       << "                 'table' (default) is human-readable; 'json' is machine-readable.\n"
       << "                 Tracing is disabled unless this switch is given.\n"
       << "  --profile-floor Measure the instrumentation floor and exit.\n"
       << "  -h, --help      Show this message.\n"
       << "  --version       Show the engine version.\n";
}

/**
 * @brief Parse `--profileperf[=value]` into a report mode.
 * @param[in] argument The full switch text including the leading dashes.
 * @param[out] out_mode Receives the parsed mode; untouched when `found` is false.
 * @param[out] out_found Receives true when the switch was present.
 * @return true when the switch was present and its value was valid.
 * @pre `argument`, `out_mode`, and `out_found` are non-NULL.
 * @post `out_found` is set before any error return so callers can report the
 *       argument name alongside the value error.
 */
bool parse_profileperf(std::string_view argument, rocminfer::TraceReport* out_mode,
                       bool* out_found) {
    constexpr std::string_view kSwitch = "--profileperf";
    *out_found = argument == kSwitch || argument.rfind("--profileperf=", 0) == 0;
    if (!*out_found) {
        return true;
    }

    if (argument == kSwitch) {
        *out_mode = rocminfer::TraceReport::kTable;
        return true;
    }

    const std::string_view value = argument.substr(std::strlen("--profileperf="));
    if (value == "table") {
        *out_mode = rocminfer::TraceReport::kTable;
        return true;
    }
    if (value == "json") {
        *out_mode = rocminfer::TraceReport::kJson;
        return true;
    }
    if (value == "off") {
        *out_mode = rocminfer::TraceReport::kOff;
        return true;
    }

    std::cerr << "rocminfer: invalid --profileperf value '" << value
              << "'; expected table, json, or off\n";
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view program = (argc > 0 && argv[0] != nullptr) ? argv[0] : "rocminfer";

    rocminfer::TraceReport report = rocminfer::TraceReport::kOff;
    bool want_floor = false;
    bool want_help = false;
    bool want_version = false;
    std::string command;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            want_help = true;
            continue;
        }
        if (arg == "--version") {
            want_version = true;
            continue;
        }
        if (arg == "--profile-floor") {
            want_floor = true;
            continue;
        }
        if (arg == "--") {
            continue;
        }

        bool found = false;
        rocminfer::TraceReport parsed = rocminfer::TraceReport::kOff;
        if (!parse_profileperf(arg, &parsed, &found)) {
            return kExitUsage;
        }
        if (found) {
            report = parsed;
            continue;
        }

        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "rocminfer: unknown option '" << arg << "'\n";
            return kExitUsage;
        }
        if (!command.empty()) {
            std::cerr << "rocminfer: unexpected extra argument '" << arg << "'\n";
            return kExitUsage;
        }
        command = std::string(arg);
    }

    if (want_help) {
        print_usage(std::cout, program);
        return kExitOk;
    }
    if (want_version) {
        std::cout << "rocminfer 0.1.0 (Phase 0)\n";
        return kExitOk;
    }

    rocminfer::TraceConfig config;
    config.report = report;
    config.measure_floor = report != rocminfer::TraceReport::kOff;

    rocminfer::TraceRecorder recorder(config);

    if (want_floor || command == "floor") {
        // Measuring the floor requires an armed recorder, otherwise there is no
        // instrumentation to calibrate.
        rocminfer::TraceConfig floor_config;
        floor_config.report = rocminfer::TraceReport::kTable;
        floor_config.measure_floor = true;
        rocminfer::TraceRecorder floor_recorder(floor_config);
        floor_recorder.measure_floor();
        floor_recorder.report(std::cout);
        return kExitOk;
    }

    if (!command.empty()) {
        std::cerr << "rocminfer: unknown command '" << command << "'\n";
        print_usage(std::cerr, program);
        return kExitUsage;
    }

    if (recorder.enabled()) {
        recorder.measure_floor();
        std::vector<rocminfer::TraceRow> rows;
        recorder.snapshot(&rows);
        if (recorder.report_mode() == rocminfer::TraceReport::kJson) {
            recorder.report_json(std::cout);
        } else {
            recorder.report(std::cout);
        }
    } else {
        print_usage(std::cout, program);
    }

    return kExitOk;
}