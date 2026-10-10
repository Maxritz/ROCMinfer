# Sherlock Analysis Log

Append-only. One entry per investigation.

## [RUN-001] 2026-10-10 — MODE: FULL — TARGET: SHA-256 golden digest

### BASELINE
- Purpose: content-address model artifacts and tensor payloads for format goldens.
- Host: Windows 11, RX 9070 XT (gfx1201), ROCm 7.16.26331, ROCm LLVM clang 23.
- Oracle: Python `hashlib` (CPython, OpenSSL-backed), cross-checked against
  .NET `System.Security.Cryptography.SHA256`. Both agree.

### FINDINGS

| Rank | Symptom | Evidence | Status |
|---|---|---|---|
| 1 | Every digest wrong, including the empty string | `sha256("")` = `b841e9ac...`, truth `e3b0c442...` | CONFIRMED |
| 2 | Schedule and round intermediates verified correct | `w[16]=61626380`, `w[17]=000f0000`, round-0 `t1=54da50e8`, `a=5d6aebcd` all match the published SHA-256("abc") trace | CONFIRMED |
| 3 | `rotr` verified correct | three `static_assert` cases pass | CONFIRMED |
| 4 | K table had three transposed hex digits | see root cause | CONFIRMED |

### ROOT CAUSE
Three FIPS 180-4 round constants were mistranscribed:

| Index | Written | Correct |
|---|---|---|
| K[5]  | `59f11142` | `59f111f1` |
| K[21] | `4a74843a` | `4a7484aa` |
| K[37] | `766a0aab` | `766a0abb` |

### WHY IT WAS HARD TO SEE
A wrong SHA-256 constant does not fail loudly. It produces a well-formed 32-byte
digest that differs from the truth but is internally consistent, so every
self-consistency check passes: one-shot equals chunked, streaming equals
one-shot, and a second hand-written reference agrees. Two implementations
reproducing the same transcription error would agree with each other and both
be wrong, which is exactly what happened here.

Compounding it, two *expected* values in the test were also recalled from memory
and were wrong (a 64-zero-byte digest and a 448-bit vector digest). Memory is not
an oracle. Every expected digest in `tests/test_sha256.cpp` is now produced by
an external implementation and re-verified, not recalled.

### ACTIONS
- [x] Corrected the three constants against FIPS 180-4.
- [x] Added `static_assert` guards on K[5], K[21], K[37], K[63] so a repeat
      transcription slip is a compile error rather than a silent wrong digest.
- [x] Replaced every recalled expected digest with a value confirmed against
      Python `hashlib` and .NET.
- [x] Added a 145-byte three-block vector and a million-`a` vector so both the
      single-block and multi-block paths are pinned.

### RESIDUAL RISK
A mistranscribed constant elsewhere in the table would still pass, because the
vectors above exercise all 64 rounds. Mitigation is the external oracle: any new
expected value must be produced by a trusted implementation and re-run, never
written from memory.

### VALIDATION GATES
- [x] Baseline recorded with full config
- [x] At least three measurements per hypothesis
- [x] Hypothesis status updated to CONFIRMED
- [x] Actions have checkboxes and are complete