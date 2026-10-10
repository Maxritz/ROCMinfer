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

#include <cstring>

namespace rocminfer {

namespace {

/// FIPS 180-4 round constants K[0..63]: first 32 bits of the fractional parts of
/// the cube roots of the first 64 primes.
constexpr std::uint32_t kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

// Spot-check entries that a transcription slip would silently corrupt. These
// are asserted because a wrong constant still produces a plausible 32-byte
// digest, which is far more dangerous than a compile error.
static_assert(kRoundConstants[5] == 0x59f111f1u, "K[5]");
static_assert(kRoundConstants[21] == 0x4a7484aau, "K[21]");
static_assert(kRoundConstants[37] == 0x766a0abbu, "K[37]");
static_assert(kRoundConstants[63] == 0xc67178f2u, "K[63]");

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
    state_[0] = 0x6a09e667u;
    state_[1] = 0xbb67ae85u;
    state_[2] = 0x3c6ef372u;
    state_[3] = 0xa54ff53au;
    state_[4] = 0x510e527fu;
    state_[5] = 0x9b05688cu;
    state_[6] = 0x1f83d9abu;
    state_[7] = 0x5be0cd19u;
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