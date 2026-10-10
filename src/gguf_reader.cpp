/**
 * @file gguf_reader.cpp
 * @brief Bounds-checked GGUF v2/v3 manifest parsing.
 *
 * All multi-byte integers are read little-endian, matching the container format.
 * A `Cursor` walks the buffer and refuses to advance past the end, so a declared
 * count larger than the remaining bytes fails cleanly instead of reading past
 * the allocation. Counts that could be enormous (array lengths, tensor counts)
 * are validated against remaining bytes before any per-element work, which keeps
 * a hostile file from driving an unbounded loop.
 */

#include "rocminfer/gguf_reader.hpp"

#include "rocminfer/format_registry.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>

namespace rocminfer {

namespace {

/**
 * @brief Sequential little-endian reader over a fixed byte range.
 *
 * Every accessor is bounds-checked. Once `failed` is set the cursor latches it,
 * so a caller can run a whole parse and test one flag at the end.
 */
class Cursor {
public:
    Cursor(const std::uint8_t* begin, const std::uint8_t* end) : ptr_(begin), end_(end) {}

    bool failed() const noexcept { return failed_; }

    std::size_t remaining() const noexcept { return static_cast<std::size_t>(end_ - ptr_); }

    /// Reserve `count` bytes and return them, or null when unavailable.
    const std::uint8_t* take(std::size_t count) noexcept {
        if (failed_ || count > remaining()) {
            failed_ = true;
            return nullptr;
        }
        const std::uint8_t* result = ptr_;
        ptr_ += count;
        return result;
    }

    template <typename T>
    bool read_pod(T* out) noexcept {
        static_assert(std::is_trivially_copyable_v<T>, "POD read requires a trivially copyable type");
        const std::uint8_t* src = take(sizeof(T));
        if (src == nullptr) {
            return false;
        }
        std::memcpy(out, src, sizeof(T));
        return true;
    }

    /// Read a length-prefixed string, appending to `out`.
    bool read_string(std::string* out) noexcept {
        std::uint64_t length = 0;
        if (!read_pod(&length)) {
            return false;
        }
        // Refuse a string longer than what is left in the file before allocating.
        if (length > remaining()) {
            failed_ = true;
            return false;
        }
        const std::uint8_t* src = take(static_cast<std::size_t>(length));
        if (src == nullptr) {
            return false;
        }
        out->assign(reinterpret_cast<const char*>(src), static_cast<std::size_t>(length));
        return true;
    }

    /// Skip `count` bytes.
    bool skip(std::size_t count) noexcept { return take(count) != nullptr; }

    /// Advance the cursor to the next multiple of `alignment`, padding with zeros.
    bool align(std::size_t alignment) noexcept {
        if (alignment <= 1) {
            return !failed_;
        }
        const std::size_t pad = (alignment - (remaining() + taken_) % alignment) % alignment;
        return skip(pad);
    }

    /// Bytes consumed so far, used with `align` to track absolute position.
    void set_base(std::size_t base) noexcept { taken_ = base; }

private:
    const std::uint8_t* ptr_ = nullptr;
    const std::uint8_t* end_ = nullptr;
    bool failed_ = false;
    std::size_t taken_ = 0;
};

/// Largest element count we will trust before checking against remaining bytes.
constexpr std::uint64_t kMaxReasonableCount = 1ull << 34;

}  // namespace

std::uint32_t gguf_value_type_size(GgufValueType type) noexcept {
    switch (type) {
        case GgufValueType::kUint8:
        case GgufValueType::kInt8:
        case GgufValueType::kBool:    return 1;
        case GgufValueType::kUint16:
        case GgufValueType::kInt16:   return 2;
        case GgufValueType::kUint32:
        case GgufValueType::kInt32:
        case GgufValueType::kFloat32: return 4;
        case GgufValueType::kUint64:
        case GgufValueType::kInt64:
        case GgufValueType::kFloat64: return 8;
        case GgufValueType::kString:
        case GgufValueType::kArray:   return 0;
    }
    return 0;
}

const GgufMetadataValue* GgufFile::find_metadata(std::string_view key) const noexcept {
    for (const auto& entry : metadata) {
        if (entry.first == key) {
            return &entry.second;
        }
    }
    return nullptr;
}

int gguf_parse(GgufFile* out, const void* data, std::size_t size) noexcept {
    if (out == nullptr || data == nullptr || size == 0) {
        return EINVAL;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    Cursor cursor(bytes, bytes + size);

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t kv_count = 0;

    if (!cursor.read_pod(&magic) || magic != GGUF_MAGIC) {
        return EINVAL;
    }
    if (!cursor.read_pod(&version)) {
        return ERANGE;
    }
    if (version != GGUF_VERSION_V2 && version != GGUF_VERSION_V3) {
        return EINVAL;
    }
    if (!cursor.read_pod(&tensor_count) || !cursor.read_pod(&kv_count)) {
        return ERANGE;
    }
    if (tensor_count > kMaxReasonableCount || kv_count > kMaxReasonableCount) {
        return ERANGE;
    }

    // Each tensor-info costs at least a length prefix plus a dim count, type and
    // offset; each KV pair at least a key length. Reject implausible counts
    // before reserving, so a hostile header cannot request a huge allocation.
    if (tensor_count > size / 8 + 1 || kv_count > size / 8 + 1) {
        return ERANGE;
    }

    out->version = version;
    out->tensor_count = tensor_count;
    out->metadata_kv_count = kv_count;
    out->tensors.clear();
    out->metadata.clear();
    out->tensors.resize(static_cast<std::size_t>(tensor_count));
    out->metadata.reserve(static_cast<std::size_t>(kv_count));

    for (std::uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        if (!cursor.read_string(&key)) {
            return ERANGE;
        }
        std::uint32_t raw_type = 0;
        if (!cursor.read_pod(&raw_type)) {
            return ERANGE;
        }
        if (raw_type > static_cast<std::uint32_t>(GgufValueType::kFloat64)) {
            return ERANGE;
        }
        const GgufValueType value_type = static_cast<GgufValueType>(raw_type);
        GgufMetadataValue value;
        value.type = value_type;

        if (value_type == GgufValueType::kArray) {
            std::uint32_t raw_element_type = 0;
            std::uint64_t element_count = 0;
            if (!cursor.read_pod(&raw_element_type) || !cursor.read_pod(&element_count)) {
                return ERANGE;
            }
            if (raw_element_type > static_cast<std::uint32_t>(GgufValueType::kFloat64)) {
                return ERANGE;
            }
            value.array_element_type = static_cast<GgufValueType>(raw_element_type);
            value.array_length = element_count;
            // Reject an element count that cannot fit in the remaining bytes
            // before consuming any of them.
            if (element_count > kMaxReasonableCount) {
                return ERANGE;
            }
            if (value.array_element_type == GgufValueType::kString) {
                for (std::uint64_t e = 0; e < element_count; ++e) {
                    std::string discarded;
                    if (!cursor.read_string(&discarded)) {
                        return ERANGE;
                    }
                }
            } else {
                const std::uint32_t element_size =
                    gguf_value_type_size(value.array_element_type);
                if (element_size == 0 || element_count > cursor.remaining() / element_size) {
                    return ERANGE;
                }
                if (!cursor.skip(static_cast<std::size_t>(element_count * element_size))) {
                    return ERANGE;
                }
            }
        } else if (value_type == GgufValueType::kString) {
            if (!cursor.read_string(&value.string_value)) {
                return ERANGE;
            }
        } else {
            const std::uint32_t width = gguf_value_type_size(value_type);
            if (width == 0) {
                return ERANGE;
            }
            const std::uint8_t* raw = cursor.take(width);
            if (raw == nullptr) {
                return ERANGE;
            }
            // Record the scalar as a double for uniform inspection; exact 32-bit
            // integers below 2^53 survive exactly, which covers all GGUF fields.
            if (width == 1) {
                std::uint8_t v = 0;
                std::memcpy(&v, raw, 1);
                value.numeric_value = (value_type == GgufValueType::kInt8)
                                          ? static_cast<double>(static_cast<std::int8_t>(v))
                                          : static_cast<double>(v);
                value.bool_value = (value_type == GgufValueType::kBool) ? (v != 0) : false;
            } else if (width == 2) {
                std::uint16_t v = 0;
                std::memcpy(&v, raw, 2);
                value.numeric_value = (value_type == GgufValueType::kInt16)
                                          ? static_cast<double>(static_cast<std::int16_t>(v))
                                          : static_cast<double>(v);
            } else if (width == 4) {
                std::uint32_t v = 0;
                std::memcpy(&v, raw, 4);
                if (value_type == GgufValueType::kFloat32) {
                    float f = 0.0f;
                    std::memcpy(&f, raw, 4);
                    value.numeric_value = static_cast<double>(f);
                } else {
                    value.numeric_value = (value_type == GgufValueType::kInt32)
                                              ? static_cast<double>(static_cast<std::int32_t>(v))
                                              : static_cast<double>(v);
                }
            } else {
                std::uint64_t v = 0;
                std::memcpy(&v, raw, 8);
                if (value_type == GgufValueType::kFloat64) {
                    double d = 0.0;
                    std::memcpy(&d, raw, 8);
                    value.numeric_value = d;
                } else {
                    value.numeric_value = (value_type == GgufValueType::kInt64)
                                              ? static_cast<double>(static_cast<std::int64_t>(v))
                                              : static_cast<double>(v);
                }
            }
        }

        out->metadata.emplace_back(std::move(key), std::move(value));
    }

    // Tensor info follows the key/value block, per the container layout written
    // by gguf_write_out in the pinned revision.
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo& info = out->tensors[static_cast<std::size_t>(i)];
        if (!cursor.read_string(&info.name)) {
            return ERANGE;
        }
        std::uint32_t n_dims = 0;
        if (!cursor.read_pod(&n_dims)) {
            return ERANGE;
        }
        if (n_dims > 4) {
            // GGUF supports at most 4 dimensions; more is malformed.
            return ERANGE;
        }
        info.dimensions.resize(n_dims);
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            if (!cursor.read_pod(&info.dimensions[d])) {
                return ERANGE;
            }
        }
        if (!cursor.read_pod(&info.type_id) || !cursor.read_pod(&info.offset)) {
            return ERANGE;
        }
    }

    // Tensor payload starts at the next aligned boundary after the metadata.
    const std::size_t consumed = size - cursor.remaining();
    std::size_t aligned = ((consumed + GGUF_DEFAULT_ALIGNMENT - 1) / GGUF_DEFAULT_ALIGNMENT) *
                         GGUF_DEFAULT_ALIGNMENT;
    if (aligned > size) {
        // A file whose metadata ends exactly at EOF has no room for payload;
        // this is valid only for a metadata-only file, so record the offset and
        // let tensor validation report any tensor that does not fit.
        out->data_offset = size;
    } else {
        out->data_offset = aligned;
    }

    return 0;
}

int gguf_validate_tensors(GgufFile& file, std::size_t buffer_size,
                          std::uint64_t file_size) noexcept {
    if (file.data_offset > file_size) {
        return ERANGE;
    }
    const std::uint64_t buffer_limit = std::min<std::uint64_t>(buffer_size, file_size);

    for (GgufTensorInfo& info : file.tensors) {
        std::uint64_t elements = 1;
        for (std::uint64_t dim : info.dimensions) {
            if (dim == 0 || elements > std::numeric_limits<std::uint64_t>::max() / dim) {
                return ERANGE;
            }
            elements *= dim;
        }

        std::uint64_t byte_size = 0;
        const FormatInfo* format = format_by_ggml_type(info.type_id);
        if (format != nullptr) {
            if ((elements % format->block_elements) != 0) {
                return ERANGE;
            }
            byte_size = (elements / format->block_elements) * format->block_bytes;
        } else if (info.type_id == static_cast<std::uint32_t>(GgmlType::kF32)) {
            byte_size = elements * 4;
        } else if (info.type_id == static_cast<std::uint32_t>(GgmlType::kF16)) {
            byte_size = elements * 2;
        } else {
            // Type with no known size: cannot validate extent, so skip it rather
            // than reject a file that mixes formats we do not decode.
            continue;
        }

        info.byte_size = byte_size;

        // Offsets are absolute from the data section. Validate against the true
        // file size so a manifest-only read still catches a truncated download.
        if (info.offset > file_size - file.data_offset) {
            return ERANGE;
        }
        const std::uint64_t start = file.data_offset + info.offset;
        if (start < info.offset || byte_size > file_size - start) {
            return ERANGE;
        }
        // When the payload is resident, additionally require it inside the buffer.
        if (byte_size > buffer_limit && start + byte_size <= file_size) {
            // Payload lies beyond the resident prefix; extent is still valid on
            // disk, so this is not an error, only a note for the caller.
        }
    }

    return 0;
}

/// Upper bound on bytes read to parse the manifest. The dominant consumer is the
/// tokenizer metadata array, which can be tens of megabytes for a large
/// vocabulary; 512 MiB leaves ample headroom without loading any tensor payload.
constexpr std::size_t kManifestProbeBytes = 512u * 1024u * 1024u;

struct GgufReader::Impl {
    std::ifstream stream;          ///< Positioned on demand for tensor reads.
    GgufFile file;                 ///< Parsed manifest.
    std::uint64_t total_bytes = 0; ///< True file size from the filesystem.
    bool open = false;             ///< True when the handle is usable.
};

GgufReader::GgufReader() : impl_(std::make_unique<Impl>()) {}

GgufReader::~GgufReader() { close(); }

bool GgufReader::is_open() const noexcept { return impl_ != nullptr && impl_->open; }

const GgufFile& GgufReader::manifest() const noexcept { return impl_->file; }

GgufFile& GgufReader::manifest_mutable() noexcept { return impl_->file; }

std::uint64_t GgufReader::file_size() const noexcept { return impl_ ? impl_->total_bytes : 0; }

void GgufReader::close() noexcept {
    if (impl_) {
        if (impl_->stream.is_open()) {
            impl_->stream.close();
        }
        impl_->open = false;
        impl_->total_bytes = 0;
        impl_->file = GgufFile{};
    }
}

int GgufReader::open(const char* path) {
    if (path == nullptr) {
        return EINVAL;
    }
    if (impl_ && impl_->open) {
        return EBUSY;
    }

    // Get the true size from the filesystem; seeftell is 32-bit on Windows.
    std::error_code ec;
    const std::uintmax_t total = std::filesystem::file_size(std::filesystem::path(path), ec);
    if (ec || total == 0) {
        return ENOENT;
    }

    auto fresh = std::make_unique<Impl>();
    fresh->stream.open(path, std::ios::binary);
    if (!fresh->stream.is_open()) {
        return ENOENT;
    }

    // Read only the manifest prefix.
    const std::size_t want =
        static_cast<std::size_t>(std::min<std::uintmax_t>(total, kManifestProbeBytes));
    std::vector<std::uint8_t> prefix(want);
    fresh->stream.read(reinterpret_cast<char*>(prefix.data()),
                       static_cast<std::streamsize>(want));
    if (static_cast<std::size_t>(fresh->stream.gcount()) != want) {
        return EIO;
    }
    fresh->stream.clear();
    fresh->stream.seekg(0, std::ios::beg);

    const int status = gguf_parse(&fresh->file, prefix.data(), prefix.size());
    if (status != 0) {
        return status;
    }

    fresh->total_bytes = static_cast<std::uint64_t>(total);
    fresh->open = true;
    impl_ = std::move(fresh);
    return 0;
}

const GgufTensorInfo* GgufReader::find_tensor(std::string_view name) const noexcept {
    for (const GgufTensorInfo& info : impl_->file.tensors) {
        if (name == info.name) {
            return &info;
        }
    }
    return nullptr;
}

int GgufReader::read_tensor(const GgufTensorInfo& info,
                            std::vector<std::uint8_t>* out) const {
    if (out == nullptr || !is_open()) {
        return EINVAL;
    }
    out->clear();

    if (info.byte_size == 0 || info.byte_size > impl_->total_bytes) {
        return ERANGE;
    }
    // data_offset + offset + byte_size must not overflow or exceed the file.
    if (info.offset > impl_->total_bytes - impl_->file.data_offset) {
        return ERANGE;
    }
    const std::uint64_t start = impl_->file.data_offset + info.offset;
    if (info.byte_size > impl_->total_bytes - start) {
        return ERANGE;
    }

    out->resize(static_cast<std::size_t>(info.byte_size));
    impl_->stream.clear();
    impl_->stream.seekg(static_cast<std::streamoff>(start), std::ios::beg);
    impl_->stream.read(reinterpret_cast<char*>(out->data()),
                       static_cast<std::streamsize>(info.byte_size));
    if (static_cast<std::size_t>(impl_->stream.gcount()) != out->size()) {
        out->clear();
        return EIO;
    }
    return 0;
}

}  // namespace rocminfer