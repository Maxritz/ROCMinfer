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

### STRUCTURAL FIX (RUN-002)
Correcting three literals treats the symptom. The defect is that the constants
were *transcribed* when FIPS 180-4 *defines* them:

    K[i] = first 32 bits of frac(cbrt(p_i)),  i = 0..63
    H[i] = first 32 bits of frac(sqrt(p_i)),  i = 0..7

`include/rocminfer/detail/sha256_constants.hpp` now evaluates that definition
with exact integer arithmetic: scaling by a power of two moves the fractional
part into the integer domain, so `floor(cbrt(p << 96)) & 0xFFFFFFFF` is exactly
the wanted word, and the roots come from integer binary search rather than
floating point. There is no longer a table to mistype.

Verification is deliberately independent of that derivation: the test
recomputes every constant in `long double` via its own prime generator, so a
shared mistake would have to occur in two unrelated implementations to survive.

### ACTIONS
- [x] Corrected the three constants against FIPS 180-4.
- [x] Replaced the transcribed table with a derivation from the FIPS definition
      using exact integer arithmetic; the transcription step no longer exists.
- [x] Verification uses a different arithmetic path (long double) and a
      separate prime generator from the production one.
- [x] Replaced every recalled expected digest with a value confirmed against
      Python `hashlib` and .NET.
- [x] Added a 145-byte three-block vector and a million-`a` vector so both the
      single-block and multi-block paths are pinned.
- [x] Added `tools/gen_goldens.py`, which binds the digests committed under
      `docs/goldens` to CPython's OpenSSL-backed `hashlib`, removing the last
      place where a digest was self-certified.

### RESIDUAL RISK
The constant table can no longer be wrong, because it is computed rather than
typed. The remaining exposure is the *expected digests* in the test, which are
still literals; `tools/gen_goldens.py --verify` now covers the artifact digests,
but the in-test vectors have no such binding. Treat any newly added vector the
same way: produce it with a trusted implementation, never from memory.

### VALIDATION GATES
- [x] Baseline recorded with full config
- [x] At least three measurements per hypothesis
- [x] Hypothesis status updated to CONFIRMED
- [x] Actions have checkboxes and are complete

## [RUN-002] 2026-10-10 — MODE: HIGH — TARGET: SHA-256 constant provenance

### DELTA from RUN-001
- Transcribed K table replaced by derivation from the FIPS 180-4 definition.
- Verification moved to an independent arithmetic path (long double vs integer).
- Artifact digests bound to an external oracle via tools/gen_goldens.py.
- Duplicated manifest probe constant in main.cpp removed by routing
  inspect/validate through GgufReader.

### UPDATED FINDINGS
| Rank | Component | Cost | Delta | Status |
|---|---|---|---|---|
| 1 | SHA-256 constants | 3 wrong entries | transcription removed entirely | FIXED |
| 2 | Test vectors | 2 wrong values | now oracle-produced | FIXED |
| 3 | Golden digests | self-certified | bound to CPython hashlib | FIXED |
| 4 | Manifest probe size | duplicated in 2 files | single definition | FIXED |