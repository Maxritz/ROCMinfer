# Phase 6 — KV cache quantization: intended data model

Written before any code, per `AGENTS.md` (data model first). Every field below
names its unit, valid range, and the invariant that must hold. When the
implementation or a bug disagrees with this model, name the exact divergence
against it; do not argue the model loosely.

## Scope

Phase 6 in `AGENTS.md` lists "KV-cache quant". This document covers only that.
The specs referenced are pinned in `THIRD_PARTY_SOURCES.md` at commit
`2cba9e42` (DeepSeek-V4.1-Flash, MIT). **No entity here is implemented yet.**

## 0. The three KV stores are distinct, and must never be merged

Upstream runs three separate quantized stores. They differ in payload type,
group size, and scale type. `AGENTS.md` §8 requires dual-scale and single-scale
variants of a family to be registered as *distinct* formats; the same rule
applies here, with more force, because these three are not variants of one
family at all.

| id | payload | group | scale | scale nota bene | serves |
|---|---|---|---|---|---|
| `ROCKVFP4_E4M3_16` | FP4 E2M1 | 16 channels | FP8 E4M3, one per group | value scale `amax/6.0`, floored at `6*2^-9` so an all-zero group never gets a zero scale | main compressed KV |
| `ROCKVFP4_E8M0_32` | FP4 E2M1 | 32 channels | UE8M0, power of two | exponent only, floored at `6*2^-126` | indexer K |
| `ROCKVFP8_E4M3_UE8M0_32` | FP8 E4M3 | 32 channels | UE8M0, power of two | value store, not FP4 | per-layer sliding-window KV |

Invariant: a registered format carries its own block size, bits per weight,
scale encoding, and a CPU reference decode. Registering one of these three must
not change the decode of either other one. The existing ROCmFP4 `Codebook10`
formats are a *weight* family with a codebook and dual-scale halves; they are
unrelated to all three and must not be aliased to them.

## 1. Entities and fields

### `KvFormat` (registry entry, one per row above)

| field | type | unit | invariant |
|---|---|---|---|
| `id` | enum | — | unique; one registry row per id |
| `payload_bits` | 4 or 8 | bits | 4 iff the row's payload is E2M1 |
| `group` | 16 or 32 | channels | divides `head_dim`; `head_dim % group == 0` |
| `scale_type` | enum {E4M3, UE8M0} | — | E4M3 carries a value; UE8M0 carries an exponent |
| `scale_bytes` | 1 | bytes per scale | exactly one scale element per group of `group` channels |
| `block_bytes` | derived | bytes | `payload_bytes + (channels/group)*scale_bytes` |

`payload_bytes` is `channels*payload_bits/8`, i.e. `channels/2` for the FP4 rows.
A single scale tensor of `channels/group` elements is stored *beside* the
payload, not interleaved. That is a different geometry from ROCmFP4, where the
scale lives inside the block; the registry must therefore carry its own
`scale_layout` field rather than assuming the ROCmFP4 arrangement.

Invariant that currently **fails**: the intra-byte nibble order is not published
by the pinned source. Until it is derived from the pinned quantizer rather than
assumed, no `ROCKVFP4_*` row may be written to the registry, and no golden may
be generated. Record this as blocked rather than picking an order.

### `KvSlot` (one position's worth of KV, resident on device)

| field | type | unit | invariant |
|---|---|---|---|
| `format` | `KvFormat`* | — | the owning tensor's format |
| `payload` | device ptr | bytes | `channels/2` for FP4, `channels` for FP8 |
| `scales` | device ptr | bytes | `channels/group` |
| `position` | int64 | tokens | `>= 0`, monotonically increasing within a slot |
| `resident` | bool | — | false ⇒ `payload` and `scales` are null |

`KvSlot` is immutable once written. A position is never rewritten in place, which
is what makes the layer-sharing in §2 safe without a per-slot lock.

### `KvOwnershipMap` (config-derived, static)

Fields taken verbatim from the pinned `inference/config.json`:

| field | type | invariant |
|---|---|---|
| `kv_source_layers` | vector&lt;int&gt; | strictly increasing; every entry `< n_layers` |
| `index_source_layers` | vector&lt;int&gt; | strictly increasing; every entry `< n_layers` |
| `compress_ratios` | vector&lt;uint8&gt; | length == `n_layers + n_mtp_layers`; entry `>= 1` for every kv source, `0` for a layer with no compressed path |
| `owner_of(layer)` | map | every non-source layer maps to the greatest source `< layer` |

The `owner_of` relation is the part upstream does **not** publish as a field; it
follows sequential execution, with each source overwriting the shared slot. That
is a design choice to re-derive, not a fact to copy, and ROCMinfer must make it
explicit rather than inheriting it from an ordering side effect. Per `AGENTS.md`
§4 the map is passed in explicitly; no module-global shared slot, which is what
upstream uses.

Cross-layer sharing invariant: for a given position, a consumer layer reads byte
for byte the KV its owner wrote, because `KvSlot` is immutable and the owner is a
pure function of the layer index. If `owner_of` ever becomes dynamic, this
invariant is void and the sharing must be re-verified from scratch.

## 2. Truth table over the states that actually conflict

The interesting failures here are not arithmetic, they are ownership and
residency. Flags: `slot_written` (the owner has produced it for this position)
and `slot_resident` (it is still in its tier).

| `slot_written` | `slot_resident` | state | correct behaviour |
|---|---|---|---|
| F | F | cold | owner must produce it before any consumer reads; a consumer that arrives first is an error, not a zero read |
| F | T | **invalid** | cannot happen; a slot exists only because it was written. If observed, the ownership map is wrong |
| T | F | evicted | re-derive from the owner's tier, or replay; must not silently return zeros |
| T | T | normal | consumer reads it directly |

The F/T row is the one a naive implementation gets wrong, because "not resident"
and "not written" are the same observable if the code holds a null pointer for
both. They must be distinguished in the type, which is why `KvSlot` carries
`resident` separately from a null `payload`.

## 3. What is explicitly out of scope, and why

- **CED / prefill-decode parameter split.** The paper's "8B activated in prefill
  vs 16B in decode" has no field in either `config.json` and no implementation in
  the released `model.py`, which is a single flat 40-layer stack. Supporting it
  would mean inventing a design. Unresolved; implement nothing.
- **SWA Bounded Replay.** Prose only. The published code allocates a 128-slot
  ring and never replays or offloads. `window_size: 128` is the only usable
  number, and it is not even confirmed to be the replay window `n_win`.
- **"Full / Reindex / Reuse" mode labels.** Named in the README, absent from
  both configs and from the code. The machine-readable equivalent is
  `kv_source_layers` / `index_source_layers`, which is what this model uses.
- **Serving this model.** 552B backbone, ~763B stored parameters, ~196B engram
  tables. Not on 16 GB consumer hardware; no claim to the contrary is permitted.

## 4. Verification this model must pass before it is called done

1. **Registry separation** — each of the three `ROCKV*` formats decodes its own
   block correctly and is unaffected by the other two, verified by a per-format
   CPU reference and a negative test for every invalid scale byte.
2. **Nibble order derived, not assumed** — obtained from the pinned quantizer,
   with the derivation recorded. A packing order chosen from convention is a
   layout the test invented, which `AGENTS.md` §8a forbids.
3. **Footprint re-derivation** — any bytes-per-token claim recomputed the way
   `THIRD_PARTY_SOURCES.md` does it (720 + 170 = 890), never copied from prose.
4. **Ownership truth table** — all four rows exercised, with the F/T row
   asserted to be unreachable rather than merely untested.
5. **No hidden global** — the ownership map is an injected parameter; no
   module-global shared KV slot.

Until (2) is satisfied, the registry rows in §1 are a design, not a format.
