/**
 * @file sha256.cpp
 * @brief Self-contained FIPS 180-4 SHA-256 for content-addressing goldens.
 *
 * Streaming implementation: absorb arbitrary-length chunks with `update`, then
 * finalize once with `hex_digest`. No heap allocation occurs in `update` or
 * `compress`; only the final hex rendering builds a string.
 *
 * @note Used only to prove a later run read the same bytes as a committed
 *       golden. Not for authentication or adversarial security.
 */

#include "rocminfer/sha256.hpp"

#include "rocminfer/detail/sha256_constants.hpp"

#include <cstring>

namespace rocminfer {

namespace {

/// FIPS 180-4 round constants. Derived from their definition rather than
/// transcribed; see detail/sha256_constants.hpp for why that distinction
/// matters and how the derivation is exact.
constexpr const std::uint32_t* kRoundConstants =
    rocminfer::detail::kSha256RoundConstants.data();

constexpr std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) noexcept {
    return (value >> bits) | (value << (32u - bits));
}

// Compile-time sanity: rotating right by 8 of 0x000000FF must be 0xFF000000.
static_assert(rotr(0x000000FFu, 8) == 0xFF000000u, "rotr must be exact");
static_assert(rotr(0x80000000u, 1) == 0x40000000u, "rotr must not shift by 32");
static_assert(rotr(0x00000001u, 2) == 0x40000000u, "rotr must be exact for high bits");

inline std::uint32_t load_be32(const std::uint8_t* p) noexcept {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

}  // namespace

Sha256::Sha256() noexcept { reset(); }

void Sha256::reset() noexcept {
    for (std::size_t i = 0; i < state_.size(); ++i) {
        state_[i] = rocminfer::detail::kSha256StateWords[i];
    }
    std::memset(buffer_, 0, sizeof(buffer_));
    buffered_ = 0;
    total_bits_ = 0;
    finalized_ = false;
}

void Sha256::compress(const std::uint8_t* block) noexcept {
    std::uint32_t w[64];
    for (std::size_t i = 0; i < 16; ++i) {
        w[i] = load_be32(block + i * 4);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
        const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(const void* data, std::size_t length) noexcept {
    if (finalized_ || (data == nullptr && length != 0)) {
        return;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    total_bits_ += static_cast<std::uint64_t>(length) * 8u;

    // Top up a partially filled buffer first.
    if (buffered_ > 0) {
        const std::size_t need = 64 - buffered_;
        const std::size_t take = (length < need) ? length : need;
        std::memcpy(buffer_ + buffered_, bytes, take);
        buffered_ += take;
        bytes += take;
        length -= take;
        if (buffered_ == 64) {
            compress(buffer_);
            buffered_ = 0;
        }
    }

    // Consume whole blocks directly from the caller's buffer.
    while (length >= 64) {
        compress(bytes);
        bytes += 64;
        length -= 64;
    }

    if (length > 0) {
        std::memcpy(buffer_, bytes, length);
        buffered_ = length;
    }
}

std::string Sha256::hex_digest() noexcept {
    if (!finalized_) {
        // Pad: 0x80, then zeros, then the 64-bit big-endian bit length.
        buffer_[buffered_] = 0x80u;
        ++buffered_;
        if (buffered_ > 56) {
            std::memset(buffer_ + buffered_, 0, 64 - buffered_);
            compress(buffer_);
            buffered_ = 0;
        }
        std::memset(buffer_ + buffered_, 0, 56 - buffered_);
        for (std::size_t i = 0; i < 8; ++i) {
            buffer_[56 + i] = static_cast<std::uint8_t>((total_bits_ >> (56 - i * 8)) & 0xFFu);
        }
        compress(buffer_);
        finalized_ = true;
    }

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::size_t i = 0; i < 8; ++i) {
        const std::uint32_t word = state_[i];
        for (int shift = 28; shift >= 0; shift -= 4) {
            out.push_back(kHex[(word >> shift) & 0x0Fu]);
        }
    }
    return out;
}

std::string Sha256::hex(const void* data, std::size_t length) noexcept {
    Sha256 context;
    context.update(data, length);
    return context.hex_digest();
}

}  // namespace rocminfer