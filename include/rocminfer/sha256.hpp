/**
 * @file sha256.hpp
 * @brief Self-contained SHA-256 used to pin format goldens and artifact identity.
 *
 * Golden tests must be traceable to the exact upstream bytes they came from, so
 * the engine needs a digest it can rely on without pulling in a crypto library.
 * This is a plain FIPS 180-4 implementation used only for content addressing.
 *
 * @note This is not used for authentication, key derivation, or any adversarial
 *       setting. It identifies content so a later run can prove it read the same
 *       bytes.
 */

#ifndef ROCMINFER_SHA256_HPP
#define ROCMINFER_SHA256_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace rocminfer {

/**
 * @brief Streaming SHA-256 state.
 *
 * Feed bytes with `update` in any number of chunks, then call `hex_digest`
 * exactly once. The object is reusable by assigning a fresh state.
 */
class Sha256 {
public:
    /**
     * @brief Construct in the initial digest state.
     * @note Equivalent to an immediate `reset()`; provided so `Sha256::hex` and
     *       stack objects are usable without an explicit reset.
     */
    Sha256() noexcept;

    /**
     * @brief Reset to the initial digest state.
     */
    void reset() noexcept;

    /**
     * @brief Absorb a contiguous byte range into the digest.
     * @param[in] data Bytes to absorb; may be null only when `length` is zero.
     * @param[in] length Number of bytes at `data`.
     * @pre `data` points to at least `length` readable bytes.
     */
    void update(const void* data, std::size_t length) noexcept;

    /**
     * @brief Finalize and render the digest as lowercase hex.
     * @return 64-character NUL-terminated string owned by this object.
     * @post The object is left finalized; call `reset` before reusing it.
     */
    std::string hex_digest() noexcept;

    /**
     * @brief One-shot convenience wrapper.
     * @param[in] data   Bytes to digest; may be null only when `length` is zero.
     * @param[in] length Number of bytes at `data`.
     * @return 64-character lowercase hex digest.
     * @pre `data` points to at least `length` readable bytes.
     */
    static std::string hex(const void* data, std::size_t length) noexcept;

private:
    void compress(const std::uint8_t* block) noexcept;

    std::array<std::uint32_t, 8> state_{};
    std::uint8_t buffer_[64] = {};  ///< Partial block awaiting the next update.
    std::size_t buffered_ = 0;      ///< Bytes currently held in `buffer_`.
    std::uint64_t total_bits_ = 0;  ///< Total message length in bits.
    bool finalized_ = false;
};

}  // namespace rocminfer

#endif  // ROCMINFER_SHA256_HPP