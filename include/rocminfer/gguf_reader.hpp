/**
 * @file gguf_reader.hpp
 * @brief Bounds-checked reader for GGUF v3 model manifests.
 *
 * Parses the magic, version, tensor table, and key/value metadata without
 * reading tensor payload bytes. Every field is validated against the file
 * length before use, so a truncated, mis-reported, or malformed file is rejected
 * rather than trusted.
 *
 * @note The parser never allocates from a length field in the file. Array lengths
 *       are checked against the remaining byte count, and every read goes through
 *       `Cursor::take`, which refuses to advance past the end of the buffer.
 * @warning This reader accepts a whole file in memory. That is fine for metadata
 *          inspection and small artifacts; Phase 3 replaces the payload read with
 *          a streaming view so multi-gigabyte models never need to be resident.
 */

#ifndef ROCMINFER_GGUF_READER_HPP
#define ROCMINFER_GGUF_READER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rocminfer {

/// GGUF magic bytes, "GGUF" little-endian.
inline constexpr std::uint32_t GGUF_MAGIC = 0x46554747u;

/// GGUF container versions this reader understands.
inline constexpr std::uint32_t GGUF_VERSION_V2 = 2;
inline constexpr std::uint32_t GGUF_VERSION_V3 = 3;

/// Default tensor-data alignment in the pinned GGUF revision.
inline constexpr std::uint64_t GGUF_DEFAULT_ALIGNMENT = 32;

/**
 * @brief Type tag of a GGUF metadata value.
 *
 * Mirrors the `gguf_metadata_value_type` enum of the pinned GGUF revision.
 */
enum class GgufValueType : std::uint32_t {
    kUint8 = 0,
    kInt8 = 1,
    kUint16 = 2,
    kInt16 = 3,
    kUint32 = 4,
    kInt32 = 5,
    kFloat32 = 6,
    kBool = 7,
    kString = 8,
    kArray = 9,
    kUint64 = 10,
    kInt64 = 11,
    kFloat64 = 12,
};

/**
 * @brief Byte width of a fixed-size metadata value type.
 * @param[in] type Value type tag.
 * @return Width in bytes, or 0 for variable-size and invalid types.
 */
std::uint32_t gguf_value_type_size(GgufValueType type) noexcept;

/**
 * @brief One entry in a GGUF tensor-info table.
 */
struct GgufTensorInfo {
    std::string name;                            ///< Tensor identifier, e.g. "blk.0.attn_q.weight".
    std::vector<std::uint64_t> dimensions;       ///< Shape, outermost dimension first.
    std::uint32_t type_id = 0;                   ///< Raw `ggml_type` from the file.
    std::uint64_t offset = 0;                    ///< Byte offset within the tensor-data section.
    std::uint64_t byte_size = 0;                 ///< Computed payload size, filled by the parser.
};

/**
 * @brief One metadata key/value pair, flattened for inspection.
 *
 * Arrays and strings are kept as their raw scalar elements because manifest
 * inspection reports shapes and counts, not decoded content.
 */
struct GgufMetadataValue {
    GgufValueType type = GgufValueType::kUint32;  ///< Value type tag.
    GgufValueType array_element_type = GgufValueType::kUint32;  ///< Element type when `type` is kArray.
    std::uint64_t array_length = 0;               ///< Element count when `type` is kArray.
    std::string string_value;                     ///< Value when `type` is kString.
    double numeric_value = 0.0;                   ///< Value for scalar numeric types.
    bool bool_value = false;                      ///< Value when `type` is kBool.
};

/**
 * @brief A parsed GGUF manifest: header, tensor table, and metadata.
 */
struct GgufFile {
    std::uint32_t version = 0;                        ///< Container version from the header.
    std::uint64_t tensor_count = 0;                   ///< Tensors declared in the header.
    std::uint64_t metadata_kv_count = 0;              ///< Key/value pairs declared in the header.
    std::uint64_t data_offset = 0;                    ///< Absolute file offset of tensor payload start.
    std::vector<GgufTensorInfo> tensors;              ///< Tensor table, in file order.
    std::vector<std::pair<std::string, GgufMetadataValue>> metadata;  ///< Metadata, in file order.

    /**
     * @brief Find a metadata value by key.
     * @param[in] key Exact key to search for, e.g. "general.architecture".
     * @return Pointer to the stored value, or nullptr when the key is absent.
     * @post The pointer is invalidated by any mutation of this object.
     */
    const GgufMetadataValue* find_metadata(std::string_view key) const noexcept;
};

/**
 * @brief Parse a GGUF v2/v3 manifest from an in-memory buffer.
 * @param[out] out     Destination manifest, overwritten on success.
 * @param[in]  data    File bytes beginning at offset zero, non-NULL.
 * @param[in]  size    Length of `data` in bytes.
 * @return 0 on success; EINVAL for a null or empty buffer or bad magic/version;
 *         ERANGE when the declared counts or lengths exceed the buffer.
 * @pre `data` holds the complete file; truncation must be simulated by passing
 *      a shorter `size`, which this function detects.
 * @post On failure `out` is left in a valid but unspecified state; callers must
 *       not read tensor offsets from it.
 * @warning Does not verify tensor payload bytes; only manifest structure. Use
 *          `gguf_validate_tensors` to check that every tensor fits in the file.
 */
int gguf_parse(GgufFile* out, const void* data, std::size_t size) noexcept;

/**
 * @brief Check that every declared tensor lies inside the file and is whole.
 * @param[in,out] file        Parsed manifest; each tensor's `byte_size` is filled in.
 * @param[in]     buffer_size Bytes actually held in memory.
 * @param[in]     file_size   Total size of the file on disk.
 * @return 0 when every tensor fits, ERANGE when one overruns the file or has a
 *         ragged element count for its type.
 * @pre `file` came from `gguf_parse` over the leading bytes of this file.
 * @note `file_size` may exceed `buffer_size` when only the manifest was read.
 *       Offsets are then checked against `file_size`, which still catches a
 *       truncated download, without requiring the payload to be resident.
 * @note Tensors whose type has no registry entry and no known fixed width are
 *       skipped rather than rejected, so a file mixing ROCmFP4 with F32 still
 *       validates. `byte_size` is left zero for those.
 * @warning Mutates `file`, so do not call concurrently with readers of the same
 *          manifest.
 */
int gguf_validate_tensors(GgufFile& file, std::size_t buffer_size,
                          std::uint64_t file_size) noexcept;

/**
 * @brief On-disk GGUF file opened for random-access tensor payload reads.
 *
 * Owns a file handle and the parsed manifest. Lets a caller decode a single
 * tensor without mapping the multi-gigabyte payload section, which is the whole
 * point: validating a 3 GB model must not require 3 GB of host memory.
 *
 * @note Non-copyable: it owns an operating-system handle. Use one instance per
 *       file and `close()` or let it destruct.
 * @warning Not thread-safe for concurrent `open`/`close`. `read_tensor` on a
 *          distinct reader instance per thread is safe; concurrent reads through
 *          one instance require external serialisation of the handle.
 */
class GgufReader {
public:
    /**
     * @brief Construct in the closed state.
     */
    GgufReader();

    /**
     * @brief Close the handle if open.
     */
    ~GgufReader();

    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    /**
     * @brief Open and parse a GGUF file from disk.
     * @param[in] path NUL-terminated path to the file, non-NULL.
     * @return 0 on success; EINVAL for a null path; EBUSY if already open;
     *         ENOENT if the file cannot be opened; ERANGE if the manifest is
     *         malformed or the declared file is too short.
     * @post On success `manifest()` and `file_size()` are populated and
     *       `is_open()` is true.
     */
    int open(const char* path);

    /**
     * @brief Close the file and release the handle and manifest.
     * @post Safe to call when already closed.
     */
    void close() noexcept;

    /**
     * @brief Report whether the reader holds an open file.
     * @return true when a file is open.
     */
    bool is_open() const noexcept;

    /**
     * @brief Access the parsed manifest as a mutable reference for validation.
     * @return Reference to the parsed manifest; empty when closed.
     * @pre `is_open()` is true.
     * @warning This exists so `gguf_validate_tensors`, which fills in
     *          `byte_size`, can run on an open reader's manifest. Treat the
     *          result as read-mostly; do not reorder tensors or edit metadata.
     */
    GgufFile& manifest_mutable() noexcept;

    /**
     * @brief Access the parsed manifest.
     * @return Reference to the parsed manifest; empty when closed.
     * @pre `is_open()` is true.
     */
    const GgufFile& manifest() const noexcept;

    /**
     * @brief Total size of the open file in bytes.
     * @return File size, or 0 when closed.
     */
    std::uint64_t file_size() const noexcept;

    /**
     * @brief Read one tensor's payload bytes into `out`.
     * @param[in]  info Tensor whose `offset` and `byte_size` were computed by
     *                   `gguf_validate_tensors`, non-NULL.
     * @param[out] out   Receives exactly `info.byte_size` bytes on success.
     * @return 0 on success; EINVAL for null arguments or an already-closed
     *         reader; ERANGE when the tensor extent is invalid or lies beyond
     *         the file.
     * @pre `info.byte_size` is non-zero and `info.offset + info.byte_size` does
     *      not overflow past `file_size()`.
     * @post On failure `out` is left cleared and unmodified.
     */
    int read_tensor(const GgufTensorInfo& info, std::vector<std::uint8_t>* out) const;

    /**
     * @brief Find a tensor by exact name.
     * @param[in] name Tensor name to match, e.g. "blk.0.attn_q.weight".
     * @return Pointer to the manifest entry, or nullptr when absent.
     */
    const GgufTensorInfo* find_tensor(std::string_view name) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rocminfer

#endif  // ROCMINFER_GGUF_READER_HPP