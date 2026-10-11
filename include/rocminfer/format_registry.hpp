/**
 * @file format_registry.hpp
 * @brief Single registry of every quantized weight format ROCMinfer understands.
 *
 * Formats live here and nowhere else. There is exactly one definition of each
 * block layout, bits-per-weight, and scale policy, so the parser, the reference
 * decoders, and the future GPU kernels cannot drift apart.
 *
 * @note Dual-scale and single-scale variants of a family are registered as
 *       distinct entries and must never be merged behind one name. `kRocmFP4`
 *       and `kRocmFP4Fast` differ in block size, scale policy, and bits per
 *       weight, so treating them as one format would silently misread tensors.
 * @warning This header is the engine's only source of truth for format layout.
 *          Adding a format anywhere else is a defect.
 */

#ifndef ROCMINFER_FORMAT_REGISTRY_HPP
#define ROCMINFER_FORMAT_REGISTRY_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rocminfer {

/**
 * @brief Identifier for a registered quantized weight format.
 *
 * @note Values are stable and safe to persist; new formats append.
 */
enum class QuantFormat : std::uint8_t {
    kUnknown = 0,      ///< Unrecognised type. Never accepted from a file.
    kRocmFP4,          ///< Q4_0_ROCMFP4, dual-scale, 18 bytes per 32 weights.
    kRocmFP4Fast,      ///< Q4_0_ROCMFP4_FAST, single-scale, 17 bytes per 32 weights.
    kSignedInt4,       ///< Q4_0_INT4, raw two's-complement int4, 4 bytes per 8 weights.
};

/**
 * @brief GGML type identifier, as stored in GGUF tensor metadata.
 *
 * Values 0..41 mirror the `ggml_type` enum of the pinned ROCmFPX revision so a
 * file's on-disk type field maps to a registry entry without a translation
 * table. Values 100..101 are the two ROCmFP4 types that revision assigns.
 *
 * @note The dense low band (0..41) is still growing upstream: ggml added
 *       `MXFP4` (39), `NVFP4` (40) and `Q1_0` (41) only recently, so no id in
 *       0..41 can ever be considered permanently free. The range 42..99 is
 *       currently unassigned but sits inside that band, so it is not safe
 *       either.
 * @warning `kQ4_0Int4` is an ROCMinfer-internal id, deliberately placed far
 *          above every id any ggml revision can assign: it is **not** a ggml
 *          type, and a file carrying it is not loadable by llama.cpp,
 *          ggml-py, or any other ggml-based tool. See the enumeration's
 *          per-value `@note` for the full survey.
 */
enum class GgmlType : std::uint32_t {
    kF32 = 0,
    kF16 = 1,
    kQ4_0 = 2,
    kQ4_0RocmFP4 = 100,      ///< Dual-scale ROCmFP4.
    kQ4_0RocmFP4Fast = 101,  ///< Single-scale ROCmFP4 FAST.

    /**
     * @brief Raw two's-complement signed int4, packed two weights per byte.
     *
     * ggml defines **no** raw-int4 type: every 4-bit ggml type is a block type
     * carrying scales (Q4_0, Q4_1, MXFP4, NVFP4, ...) or a codebook (IQ4_*),
     * and the two nearest neighbours are both taken: `Q4_0` (2) is block +
     * FP16 scale and `IQ4_NL` (20) is a non-linear codebook. The pinned
     * ROCmFPX revision adds `Q4_0_ROCMI4` (108), "native signed-nibble 4-bit +
     * UE4M3 scale", i.e. signed nibbles *with a scale byte* — still not a raw
     * int4, so it is not this format and must not be merged with it.
     *
     * @note Id survey, against the pinned revision
     *       `H:\ports\ROCmFPX\ggml\include\ggml.h` (`enum ggml_type`,
     *       `GGML_TYPE_COUNT = 109`):
     *       - 0..41 are assigned by ggml (with 4, 5, 31-33 and 36-38 removed);
     *       - 42..99 are unassigned but inside the dense band ggml keeps
     *         extending, so they are not stable enough to claim;
     *       - 100..108 are the ROCmFPX experimental types (`Q4_0_ROCMFP4`,
     *         `_FAST`, `Q6_0_ROCMFPX`, `Q8_0_ROCMFPX`, `Q3_0_ROCMFPX`,
     *         `TURBO3_0`, `TURBO4_0`, `Q2_0_ROCMFPX`, `Q4_0_ROCMI4`);
     *       - 109 is `GGML_TYPE_COUNT`, i.e. the first id ggml has not issued.
     *       200 is chosen because it is strictly above the largest id that
     *       revision issues, so no ggml consumer can ever mistake it for a
     *       type it knows, and no future ggml type can be issued into the
     *       low band and collide with it.
     * @note Engine-internal: ROCMinfer owns this id. It exists so a
     *       two's-complement int4 weight tensor can be declared in a real GGUF
     *       and reach the int4 kernels through the ordinary file path. A file
     *       using it is rejected by every other ggml reader, which is the
     *       intended outcome rather than a defect.
     * @warning Never reuse this id for a format that is not raw two's-
     *          complement int4 with no scales and no codebook: the byte-size
     *          derivation and the int4 kernels both assume exactly 8 weights
     *          in 4 bytes with no per-block overhead.
     */
    kQ4_0Int4 = 200,
};

/**
 * @brief Complete description of one registered format.
 *
 * Every field is a compile-time constant for the lifetime of the process. Two
 * formats are registered as distinct entries whenever they differ in block
 * size, scale policy, or bits per weight, so no field of this struct is ever
 * a "default" that could silently describe the wrong layout.
 *
 * @note `dual_scale` is the ROCmFP4 two-UE4M3-scales flag. It is false for a
 *       single-scale format and also false for a format that carries **no**
 *       scale at all (`kSignedInt4`); it must never be read as "this format
 *       has exactly one scale".
 */
struct FormatInfo {
    QuantFormat id = QuantFormat::kUnknown;   ///< Registry identifier.
    const char* name = "";                    ///< Canonical ROCMinfer name.
    const char* gguf_type = "";               ///< Type string as it appears in GGUF metadata.
    std::uint32_t ggml_type_id = 0;           ///< Numeric `ggml_type` stored in the file.
    std::uint32_t block_bytes = 0;            ///< Bytes per packed block.
    std::uint32_t block_elements = 0;         ///< Weights represented per block.
    float bits_per_weight = 0.0f;             ///< Nominal BPW including scale overhead.
    bool dual_scale = false;                  ///< True when a block carries two UE4M3 scales.
};

/**
 * @brief Look up a format by its registry identifier.
 * @param[in] id Format identifier to resolve.
 * @return Pointer to static registry storage, or nullptr for `kUnknown`.
 * @post The returned pointer and all its `const char*` members outlive the process.
 */
const FormatInfo* format_info(QuantFormat id) noexcept;

/**
 * @brief Look up a format by the type string used in GGUF tensor metadata.
 * @param[in] gguf_type NUL-terminated or `string_view` type name, case-sensitive.
 * @return Pointer to static registry storage, or nullptr when unregistered.
 * @note Only formats this engine can decode are matched. Unregistered types
 *       return nullptr rather than being approximated by a neighbouring format.
 */
const FormatInfo* format_by_gguf_type(std::string_view gguf_type) noexcept;

/**
 * @brief Number of formats registered in this build.
 * @return Count of registry entries excluding the unknown sentinel.
 */
std::size_t format_registry_size() noexcept;

/**
 * @brief Retrieve a registry entry by index for enumeration.
 * @param[in] index Zero-based position, below `format_registry_size()`.
 * @return Pointer to static registry storage.
 * @pre `index < format_registry_size()`.
 */
const FormatInfo* format_at(std::size_t index) noexcept;

/**
 * @brief Resolve a GGML type id from tensor metadata to a registered format.
 * @param[in] type Type id read verbatim from the file.
 * @return Pointer to static registry storage, or nullptr when unsupported.
 * @note Only types with a working reference decoder resolve. F32, F16 and Q4_0
 *       deliberately return nullptr here: they are readable but not ROCmFPX
 *       formats, so `validate` reports them as unsupported rather than passing
 *       them off as something this engine decodes.
 */
const FormatInfo* format_by_ggml_type(std::uint32_t type) noexcept;

/**
 * @brief Bytes a packed row occupies for a format.
 * @param[in] info       Registered format; must not be null.
 * @param[in] elements   Weight count in the row.
 * @return Exact byte length of the packed row.
 * @pre `info != nullptr` and `info->block_elements > 0`.
 * @post `elements` must be a multiple of `block_elements`; this function does
 *       not validate that, so callers must reject ragged rows beforehand.
 */
std::size_t format_row_bytes(const FormatInfo& info, std::size_t elements) noexcept;

}  // namespace rocminfer

#endif  // ROCMINFER_FORMAT_REGISTRY_HPP