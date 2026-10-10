# PERF_TRACE — performance tracing contract

Authoritative for ROCMinfer instrumentation. Enforced by `AGENTS.md` §4.

## Why

Sherlock analysis (`AGENTS.md` §6) forbids optimising by guesswork. That needs
per-component timing that separates **device execution** from **host work** from
**idle wait**, because a component that is 95% of device time and a component
that is 95% idle wait need completely different fixes.

## Switch

```
--profileperf            # table report to stdout at exit  (default mode)
--profileperf=table      # explicit table
--profileperf=json       # machine-readable JSON report
--profileperf=off        # disable (default; the switch is a no-op)
--profile-floor          # measure and print the instrumentation floor, then exit
```

Tracing is **off by default**. With `--profileperf=off`, and when the recorder
was never armed, every `TraceScope` compiles to a single pointer/flag test: no
allocation, no HIP event creation, no clock call beyond the flag test.

## Scopes

```cpp
rocminfer::TraceScope t(tracer, "attention", rocminfer::TraceDomain::kDevice, stream);
```

RAII. Begin in the constructor, end in the destructor. The destructor must not
throw and must not allocate.

| Parameter | Meaning |
|---|---|
| `TraceRecorder&` | Explicitly injected. No hidden global, no static singleton. |
| `component` | Stable `const char*` label naming the logical role. Interned on first use. |
| `domain` | `kHost`, `kDevice`, `kTransfer`, `kAlloc`, `kSync`, `kDispatch` |
| `stream` | `void*` holding a `hipStream_t`, or `nullptr` for the default stream. Only read under `ROCMINFER_HAVE_HIP`. |

Domains exist so that the report can distinguish, for the same component, work
that ran on the GPU from staging copies and from synchronisation waits.

## Column semantics

Each closed scope produces one record:

| Field | Meaning |
|---|---|
| `host_ns` | Wall-clock host time inside the scope: launch setup, argument marshalling, any host work the component does. |
| `dev_ns` | GPU execution time, from two HIP events (`hipEventRecord` before and after the launch), `hipEventElapsedTime`. Zero for `kHost` scopes. |
| `idle_ns` | `host_ns - dev_ns` when a device event pair was recorded, clamped at 0. This is enqueue + queue-wait + host-work time. |
| `kind` | The domain, used for grouping and for the JSON schema. |
| `component` | The interned label. |

`dev_ns` is **GPU execution time, not enqueue time**. A scope that only calls a
launch without device events would report its host cost as device time and make
the report actively wrong; that is why `kDevice` scopes record events.

A component that interleaves launches and host work must use **separate nested
scopes** for the launch (`kDevice`) and the host portion (`kHost`) so that
`idle_ns` is measured rather than inferred.

## Report

```
     component           ops   %dev      dev us     idle us     host us
     attention            24  95.4%   535880.80      377.62       49.20
     projections         187   3.2%    17723.32     2026.19      460.80
     ...
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.70 us host, 19.54 us device each.
```

- Rows sorted by `dev us` descending.
- `%dev` is the share of total device time across all components.
- The final line is **mandatory** whenever tracing was enabled. Without it, a
  small `dev us` cannot be distinguished from a measurement below the
  instrumentation floor.

## Instrumentation floor

Calibrated by timing `N` (default 256) empty scopes pushed through the *identical*
code path — same begin/end calls, same event pair, same intern lookup — and
reporting the per-op cost of each. The floor is the resolution limit of the
instrumentation: any measured value at or below it is noise, not signal.

The device floor is typically much larger than the host floor because HIP event
record and elapsed-time queries cost more than a `steady_clock` read. That is
precisely why the floor is reported separately per domain instead of folded into
one number.

The device floor divides by the number of scopes that produced a **usable**
measurement, not by the number attempted. See "Unmeasurable device samples" below.

If a component's total is close to the floor, the run is reported as
**instrumented** and the uninstrumented number must be quoted alongside it
before any conclusion is drawn.

### Measured on the development machine

RX 9070 XT (gfx1201), ROCm 7.16.26331, ROCm LLVM clang 23, Windows 11:

```
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.02 us host, 23.61 us device each.
```

The device floor is roughly three orders of magnitude above the host floor. Any
component whose `dev us` sits in the tens of microseconds is at the floor and
carries no information; meaningful component cost starts in the hundreds of
microseconds and above. Re-measure on any new host rather than reusing this
number.

### Unmeasurable device samples

`hipEventElapsedTime` on ROCm 7.16 / gfx1201 can return `hipSuccess` with a
**negative** elapsed span when two events bracket an idle gap and the GPU clock
is not monotonic between them. Measured magnitudes were -0.016 ms to -0.039 ms.

The timestamps are fixed once the events complete, so re-reading cannot repair
them. The backend therefore rejects such a sample and the recorder counts it in
`dropped.pending_scopes` (JSON) / `dropped_pending_scopes()`.

This is deliberate. Clamping a negative span to zero, or silently dropping it
without counting, would both produce a trace that looks complete while quietly
under-reporting device time. A non-zero drop count means the trace is partial and
must be read as such.

## JSON schema (`--profileperf=json`)

```json
{
  "schema": "rocminfer.perftrace/1",
  "floor": { "samples": 256, "host_ns": 700, "device_ns": 19540 },
  "components": [
    { "component": "attention", "domain": "device",
      "ops": 24, "host_ns": 49200, "dev_ns": 535880800, "idle_ns": 377620 }
  ],
  "overflow": { "samples_dropped": 0, "unlabelled_components": 0 }
}
```

`overflow` is always present. A full ring or an intern table that hit its cap
reports a non-zero count rather than silently discarding samples, because a
truncated trace that looks complete is worse than no trace.

`dropped.component_scopes` counts scopes aggregated into the `__overflow__` row
because the intern table was full. `dropped.pending_scopes` counts device scopes
whose device time could not be measured, including the non-monotonic-span case
above. Both are zero in a healthy run.

## Rules

- Never call `steady_clock`, `QueryPerformanceCounter`, `hipEventElapsedTime` or
  any other clock directly outside this module. Add a scope, don't add timing.
- Never add a scope without a stable component label. Labels are interned and
  matched by content; unbounded per-caller string construction defeats the
  table.
- Never nest two scopes on the same component at the same depth.
- Never report a percentage without the denominator's absolute value beside it.