/**
 * @file sha256_constants.hpp
 * @brief SHA-256 constants derived from their FIPS 180-4 definition.
 *
 * FIPS 180-4 does not publish SHA-256's round constants as an opaque table. It
 * *defines* them:
 *
 *   H(0)[i] = first 32 bits of the fractional part of sqrt(p_i),   i = 0..7
 *   K[i]    = first 32 bits of the fractional part of cbrt(p_i),  i = 0..63
 *
 * where p_i is the i-th prime. This header evaluates that definition with exact
 * integer arithmetic instead of carrying a transcribed table.
 *
 * ## Why this exists
 *
 * A hand-copied constant table is a silent-corruption hazard: three mistyped hex
 * digits still produce a well-formed 32-byte digest, so every self-consistency
 * check passes. Worse, a test whose expected digests were also written from
 * memory agrees with the defect instead of catching it. Deriving the constants
 * removes the transcription step entirely, so the table cannot be wrong.
 *
 * ## Method
 *
 * Scaling by a power of two moves the fractional part into the integer domain:
 * `floor(cbrt(p * 2^96)) == floor(cbrt(p) * 2^32)`, and because `floor(cbrt(p))`
 * is an integer, the low 32 bits are exactly the wanted fractional word. The
 * same argument gives `sqrt(p * 2^64)` for H. Both roots are computed by
 * integer binary search, so no floating-point rounding enters the result and the
 * computation is constexpr.
 *
 * @note `unsigned __int128` is required. Values stay below 2^109, well inside
 *       its range, and both supported toolchains provide it as an extension.
 * @warning Do not replace these tables with literals. That re-opens the exact
 *          defect this file exists to prevent.
 */

#ifndef ROCMINFER_DETAIL_SHA256_CONSTANTS_HPP
#define ROCMINFER_DETAIL_SHA256_CONSTANTS_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocminfer::detail {

/// Number of SHA-256 round constants.
inline constexpr std::size_t kSha256RoundCount = 64;

/// Number of initial hash words.
inline constexpr std::size_t kSha256StateCount = 8;

/**
 * @brief Return the `index`-th prime, counted from zero.
 * @param[in] index Zero-based prime index; 0 yields 2.
 * @return The prime itself.
 * @note Consteval because only constant evaluation is intended; runtime callers
 *       that need the same sequence must generate primes independently, which
 *       keeps the verification path from reusing this implementation.
 */
consteval std::uint32_t prime_at(std::size_t index) noexcept {
    std::uint32_t candidate = 2;
    std::size_t found = 0;
    for (;;) {
        bool is_prime = candidate >= 2;
        for (std::uint32_t d = 2; is_prime && d * d <= candidate; ++d) {
            if (candidate % d == 0) {
                is_prime = false;
            }
        }
        if (is_prime) {
            if (found == index) {
                return candidate;
            }
            ++found;
        }
        ++candidate;
    }
}

/// Integer cube root by binary search; exact, no floating point.
consteval unsigned __int128 integer_cuberoot(unsigned __int128 value) noexcept {
    unsigned __int128 high = 1;
    while (high * high * high <= value) {
        high <<= 1;
    }
    unsigned __int128 low = high >> 1;
    while (low < high) {
        const unsigned __int128 mid = (low + high + 1) >> 1;
        if (mid * mid * mid <= value) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

/// Integer square root by binary search; exact, no floating point.
consteval unsigned __int128 integer_sqrt(unsigned __int128 value) noexcept {
    unsigned __int128 high = 1;
    while (high * high <= value) {
        high <<= 1;
    }
    unsigned __int128 low = high >> 1;
    while (low < high) {
        const unsigned __int128 mid = (low + high + 1) >> 1;
        if (mid * mid <= value) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

/**
 * @brief Derive round constant `i` from the cube root of the i-th prime.
 * @param[in] i Round index in 0..63.
 * @return The 32-bit fractional word FIPS 180-4 specifies for K[i].
 */
consteval std::uint32_t derive_round_constant(std::size_t i) noexcept {
    const unsigned __int128 scaled =
        static_cast<unsigned __int128>(prime_at(i)) << 96;
    return static_cast<std::uint32_t>(integer_cuberoot(scaled) & 0xFFFFFFFFu);
}

/**
 * @brief Derive initial hash word `i` from the square root of the i-th prime.
 * @param[in] i Word index in 0..7.
 * @return The 32-bit fractional word FIPS 180-4 specifies for H(0)[i].
 */
consteval std::uint32_t derive_state_word(std::size_t i) noexcept {
    const unsigned __int128 scaled =
        static_cast<unsigned __int128>(prime_at(i)) << 64;
    return static_cast<std::uint32_t>(integer_sqrt(scaled) & 0xFFFFFFFFu);
}

/// Build the full round-constant table from the definition.
consteval std::array<std::uint32_t, kSha256RoundCount> make_round_constants() noexcept {
    std::array<std::uint32_t, kSha256RoundCount> table{};
    for (std::size_t i = 0; i < kSha256RoundCount; ++i) {
        table[i] = derive_round_constant(i);
    }
    return table;
}

/// Build the full initial-state table from the definition.
consteval std::array<std::uint32_t, kSha256StateCount> make_state_words() noexcept {
    std::array<std::uint32_t, kSha256StateCount> table{};
    for (std::size_t i = 0; i < kSha256StateCount; ++i) {
        table[i] = derive_state_word(i);
    }
    return table;
}

/// SHA-256 round constants, derived rather than transcribed.
inline constexpr std::array<std::uint32_t, kSha256RoundCount> kSha256RoundConstants =
    make_round_constants();

/// SHA-256 initial hash value, derived rather than transcribed.
inline constexpr std::array<std::uint32_t, kSha256StateCount> kSha256StateWords =
    make_state_words();

// Anchor the derivation to the values printed in FIPS 180-4 section 4.2.2 and
// 5.3.3. These four are cross-checked against an independent implementation,
// so they are documentation rather than the source of truth.
static_assert(kSha256RoundConstants[0] == 0x428a2f98u, "K[0] must match FIPS 180-4");
static_assert(kSha256RoundConstants[5] == 0x59f111f1u, "K[5] must match FIPS 180-4");
static_assert(kSha256RoundConstants[21] == 0x4a7484aau, "K[21] must match FIPS 180-4");
static_assert(kSha256RoundConstants[37] == 0x766a0abbu, "K[37] must match FIPS 180-4");
static_assert(kSha256RoundConstants[63] == 0xc67178f2u, "K[63] must match FIPS 180-4");
static_assert(kSha256StateWords[0] == 0x6a09e667u, "H[0] must match FIPS 180-4");
static_assert(kSha256StateWords[7] == 0x5be0cd19u, "H[7] must match FIPS 180-4");

}  // namespace rocminfer::detail

#endif  // ROCMINFER_DETAIL_SHA256_CONSTANTS_HPP