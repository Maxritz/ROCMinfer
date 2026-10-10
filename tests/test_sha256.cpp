/**
 * @file test_sha256.cpp
 * @brief Known-answer tests for the SHA-256 golden-digest implementation.
 *
 * Validates against the standard FIPS 180-4 / RFC 6234 vectors, including the
 * million-'a' vector that exercises the multi-block path and the boundary sizes
 * around the 64-byte block and 56-byte length field. These digests are committed
 * as golden-test anchors, so a subtle bug here would silently corrupt every
 * golden rather than fail loudly.
 */

#include "rocminfer/sha256.hpp"

#include "rocminfer/detail/sha256_constants.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "FAIL line " << line << ": " << expression << "\n";
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

void expect_digest(const std::string& input, const char* expected, int line) {
    const std::string actual = rocminfer::Sha256::hex(input.data(), input.size());
    if (actual != expected) {
        std::cerr << "FAIL line " << line << ": sha256(\"" << input.substr(0, 16)
                  << (input.size() > 16 ? "..." : "") << "\") expected " << expected
                  << " got " << actual << "\n";
        ++g_failures;
    }
}

#define EXPECT_DIGEST(input, expected) expect_digest((input), (expected), __LINE__)

/**
 * @brief Runtime prime generation, independent of the production derivation.
 * @param[in] index Zero-based prime index.
 * @return The `index`-th prime.
 * @note Deliberately not shared with the production `prime_at`, so a mistake in
 *       prime selection cannot hide by being wrong in both places at once.
 */
std::uint32_t prime_runtime(std::size_t index) {
    std::vector<std::uint32_t> primes;
    for (std::uint32_t candidate = 2; primes.size() <= index; ++candidate) {
        bool is_prime = true;
        for (std::uint32_t d = 2; d * d <= candidate; ++d) {
            if (candidate % d == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            primes.push_back(candidate);
        }
    }
    return primes[index];
}

/**
 * @brief Independent re-derivation of a SHA-256 round constant.
 * @param[in] index Round index in 0..63.
 * @return Fractional 32 bits of cbrt(p_index).
 * @note Uses floating point, deliberately. The production table is computed
 *       with exact integer arithmetic; verifying it through a different
 *       arithmetic path means a shared mistake would have to occur in two
 *       unrelated implementations to go unnoticed.
 */
std::uint32_t round_constant_via_float(std::size_t index) {
    const long double root = std::pow(static_cast<long double>(prime_runtime(index)),
                                      1.0L / 3.0L);
    const long double fraction = root - std::floor(root);
    return static_cast<std::uint32_t>(fraction * 4294967296.0L);
}

/**
 * @brief Independent re-derivation of an initial hash word.
 * @param[in] index Word index in 0..7.
 * @return Fractional 32 bits of sqrt(p_index).
 */
std::uint32_t state_word_via_float(std::size_t index) {
    const long double root = std::sqrt(static_cast<long double>(prime_runtime(index)));
    const long double fraction = root - std::floor(root);
    return static_cast<std::uint32_t>(fraction * 4294967296.0L);
}

void test_constants_agree_across_independent_derivations() {
    for (std::size_t i = 0; i < rocminfer::detail::kSha256RoundCount; ++i) {
        check(rocminfer::detail::kSha256RoundConstants[i] == round_constant_via_float(i),
              "round constant mismatch vs floating-point derivation", __LINE__);
    }
    for (std::size_t i = 0; i < rocminfer::detail::kSha256StateCount; ++i) {
        check(rocminfer::detail::kSha256StateWords[i] == state_word_via_float(i),
              "state word mismatch vs floating-point derivation", __LINE__);
    }
}

void test_known_vectors() {
    EXPECT_DIGEST("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_DIGEST("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
EXPECT_DIGEST("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // 145-byte message spanning three blocks: exercises multi-block compression,
    // a carried partial buffer, and length encoding above one block.
    EXPECT_DIGEST("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopqrstu"
                  "vwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk",
                  "244b7fbf2b76ac1dadb1be9eea21dab536a9b2628aa22d6b66557d617bbaf1c9");
    // The classic "a" x 1,000,000 vector (RFC 6234), exercises many blocks.
    EXPECT_DIGEST(std::string(1000000, 'a'),
                  "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

void test_streaming_equivalence() {
    // Feeding the same message in arbitrary chunk sizes must match one-shot.
    const std::string message =
        "The quick brown fox jumps over the lazy dog, repeatedly and at length, "
        "so that the streaming path crosses several 64-byte block boundaries.";
    const std::string one_shot = rocminfer::Sha256::hex(message.data(), message.size());

    for (std::size_t chunk : {1u, 3u, 7u, 32u, 55u, 56u, 63u, 64u, 65u, 127u, 1000u}) {
        rocminfer::Sha256 context;
        std::size_t offset = 0;
        while (offset < message.size()) {
            const std::size_t take = std::min(chunk, message.size() - offset);
            context.update(message.data() + offset, take);
            offset += take;
        }
        if (context.hex_digest() != one_shot) {
            std::cerr << "FAIL chunked streaming with chunk=" << chunk << "\n";
            ++g_failures;
        }
    }
}

void test_reset_allows_reuse() {
    rocminfer::Sha256 context;
    static_cast<void>(context.hex_digest());  // finalize
    context.reset();
    context.update("abc", 3);
    const std::string digest = context.hex_digest();
    CHECK(digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

}  // namespace

int main() {
    test_constants_agree_across_independent_derivations();
    test_known_vectors();
    test_streaming_equivalence();
    test_reset_allows_reuse();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all sha256 checks passed\n";
    return 0;
}