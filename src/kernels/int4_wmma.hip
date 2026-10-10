/**
 * @file int4_wmma.hip
 * @brief gfx1201 int4 x int4 WMMA tile kernel and host launcher.
 *
 * Computes one 16x16 tile `C = A * B` of signed int4 products accumulated in
 * int32. Three code paths, selected by the device target at compile time:
 *
 * - gfx1201: one `v_wmma_i32_16x16x16_iu4` instruction per 32-thread wave via
 *   the raw builtin `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12` with
 *   both sign flags true (signed x signed) and clamping off. rocWMMA has no
 *   int4 fragment, so the fragment is built explicitly: each lane holds one
 *   packed int32 (8 int4) of A, one of B, and 8 int32 accumulators.
 * - gfx1031: 32 lanes x 8 outputs, inner 16-dot via `__builtin_amdgcn_sdot8`
 *   (`v_dot8_i32_i4`), the same opcode the `int4_dot.hip` kernel uses.
 * - Anything else (including the host pass): identical distribution with a
 *   scalar nibble loop, so the file always parses and stays correct albeit
 *   slow.
 *
 * Lane mapping on the gfx1201 path (one wave, `lane = threadIdx.x`):
 * lane L reads A row `L % 16`, K-half `(L / 16) * 8` as one little-endian
 * int32 (nibble t holds `A[row][kbase + t]`, matching the engine int4 packing
 * where element k sits in the low nibble of byte k/2 for even k); B column
 * `L % 16`, same K-half, gathered nibble by nibble; accumulator slot s is
 * stored to `C[(L / 16) * 8 + s][L % 16]`. This is the wave32 16x16x16 WMMA
 * geometry (16 M-lanes x 2 K-groups for inputs, 2 M-groups x 16 N-lanes with 8
 * contiguous M outputs per lane), re-derived for the packed int4 operand
 * shape; the oracle test verifies it bit-exact on silicon.
 *
 * The builtin call shape (per-operand sign flags, no si4 fragment type)
 * follows `ck_tile/.../wmma/wmma_gfx12.hpp` (MIT, ROCm 7.16); the scatter,
 * gather, and store code here is re-derived, not copied.
 */

#include <rocminfer/int4_wmma.hpp>

#include <rocminfer/profiler.hpp>

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>

namespace rocminfer {

/// Packed 8-wide int32 vector exchanged with the iu4 WMMA builtin.
typedef int int32x8_t __attribute__((ext_vector_type(8)));

namespace {

/// True when `pointer` is a multiple of the power-of-two `alignment`.
bool is_aligned(const void* pointer, std::uintptr_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

}  // namespace

/**
 * @brief Load four consecutive little-endian bytes as one u32.
 * @param[in] p First byte; must be 4-byte aligned.
 * @return Bytes `p[0..3]` with `p[0]` in bits 0..7, so packed-int4 element k
 *         of the group sits in nibble k (bits `4*k..4*k+3`).
 * @pre `p` points to 4 readable device bytes.
 * @note External linkage (not anonymous): the gfx1201 device pass uses the
 *       WMMA branch while other passes use the per-lane branch, so an
 *       internal-linkage helper would read as unused in one pass and trip
 *       `-Werror`; external-linkage `inline` never does.
 */
__device__ inline std::uint32_t load_u32_le(const std::uint8_t* p) noexcept {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

/**
 * @brief Dot of one 16-wide A row against one 16-deep B column slice.
 * @param[in] a_row 8 packed bytes of A row m (elements 0..15, engine packing).
 * @param[in] b      Full 128-byte B tile, row-major packed.
 * @param[in] n      Column index, 0..15.
 * @return Exact sum over k of `s4(A[m][k]) * s4(B[k][n])`, at most 1024 in
 *         magnitude, so int32 is exact.
 * @pre `a_row` is 4-byte aligned; `n < 16`.
 * @note Inner product is `sdot8` on gfx1031, scalar nibbles elsewhere (the
 *       sdot8 builtin is rejected on gfx1201, hence the gate).
 */
__device__ inline std::int32_t dot_a_row_b_col(const std::uint8_t* a_row,
                                               const std::uint8_t* b,
                                               unsigned int n) noexcept {
    const std::uint32_t a1 = load_u32_le(a_row);
    const std::uint32_t a2 = load_u32_le(a_row + 4u);
    std::uint32_t b1 = 0u;
    std::uint32_t b2 = 0u;
    const unsigned int byte = n >> 1u;
    const unsigned int shift = ((n & 1u) != 0u) ? 4u : 0u;
#pragma unroll
    for (unsigned int k = 0u; k < 8u; ++k) {
        b1 |= ((static_cast<std::uint32_t>(b[k * 8u + byte]) >> shift) & 0xFu)
              << (4u * k);
        b2 |= ((static_cast<std::uint32_t>(b[(k + 8u) * 8u + byte]) >> shift) & 0xFu)
              << (4u * k);
    }
#if defined(__gfx1031__)
    std::int32_t acc = __builtin_amdgcn_sdot8(a1, b1, 0, false);
    acc += __builtin_amdgcn_sdot8(a2, b2, 0, false);
    return acc;
#else
    std::int64_t acc = 0;
#pragma unroll
    for (unsigned int t = 0u; t < 8u; ++t) {
        const std::uint32_t na1 = (a1 >> (4u * t)) & 0xFu;
        const std::uint32_t nb1 = (b1 >> (4u * t)) & 0xFu;
        const std::uint32_t na2 = (a2 >> (4u * t)) & 0xFu;
        const std::uint32_t nb2 = (b2 >> (4u * t)) & 0xFu;
        const int va1 = ((na1 & 8u) != 0u) ? (static_cast<int>(na1) - 16) : static_cast<int>(na1);
        const int vb1 = ((nb1 & 8u) != 0u) ? (static_cast<int>(nb1) - 16) : static_cast<int>(nb1);
        const int va2 = ((na2 & 8u) != 0u) ? (static_cast<int>(na2) - 16) : static_cast<int>(na2);
        const int vb2 = ((nb2 & 8u) != 0u) ? (static_cast<int>(nb2) - 16) : static_cast<int>(nb2);
        acc += static_cast<std::int64_t>(va1) * static_cast<std::int64_t>(vb1);
        acc += static_cast<std::int64_t>(va2) * static_cast<std::int64_t>(vb2);
    }
    return static_cast<std::int32_t>(acc);
#endif
}

/**
 * @brief Compute one 16x16 int32 output tile from two packed signed-int4 tiles.
 *
 * `C = A * B` with A, B each 16x16 row-major signed int4 (128 bytes each) and
 * C 16x16 row-major int32. Launched as a single wave of
 * `kInt4WmmaWaveThreads` threads; `threadIdx.x` is the lane id.
 *
 * @param[out] c Destination tile, 256 int32, device memory.
 * @param[in]  a LHS packed tile, 128 bytes, device memory.
 * @param[in]  b RHS packed tile, 128 bytes, device memory.
 * @return Nothing; writes all of `c`.
 * @pre Launched with exactly one wave (`<<<1, 32>>>`); `a`, `b`, `c` are
 *      4-byte aligned device pointers.
 * @note On gfx1201 every lane issues the iu4 builtin once and the wave
 *       computes the full tile cooperatively. On other targets each lane
 *       computes 8 outputs (`lane * 8 + s`) independently.
 * @warning The caller must synchronise the stream before reading `c`.
 */
__global__
void int4_wmma_16x16_kernel(int32_t* __restrict__ c,
                            const std::uint8_t* __restrict__ a,
                            const std::uint8_t* __restrict__ b) {
    const unsigned int lane = threadIdx.x;
#if defined(__gfx1201__)
    const unsigned int group = lane >> 4u;
    const unsigned int index = lane & 15u;

    const std::uint32_t a_word = load_u32_le(a + index * 8u + group * 4u);

    std::uint32_t b_word = 0u;
    const unsigned int b_byte = index >> 1u;
    const unsigned int b_shift = ((index & 1u) != 0u) ? 4u : 0u;
#pragma unroll
    for (unsigned int t = 0u; t < 8u; ++t) {
        const std::uint32_t byte =
            static_cast<std::uint32_t>(b[(group * 8u + t) * 8u + b_byte]);
        b_word |= ((byte >> b_shift) & 0xFu) << (4u * t);
    }

    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
        true, static_cast<std::int32_t>(a_word), true,
        static_cast<std::int32_t>(b_word), acc, false);

    const unsigned int m_base = group * 8u;
#pragma unroll
    for (unsigned int s = 0u; s < 8u; ++s) {
        c[(m_base + s) * 16u + index] = acc[s];
    }
#else
    for (unsigned int s = 0u; s < 8u; ++s) {
        const unsigned int o = lane * 8u + s;
        c[o] = dot_a_row_b_col(a + (o >> 4u) * 8u, b, o & 15u);
    }
#endif
}

/**
 * @brief Compute one 16x16 int32 output tile from two packed signed-int4 tiles.
 *
 * `C = A * B` with A, B each 16x16 row-major signed int4 (128 bytes each) and
 * C 16x16 row-major int32. A single wave emits one `v_wmma_i32_16x16x16_iu4`
 * instruction on gfx1201.
 *
 * @param[out] c      Output tile base, device memory, at least 1024 bytes,
 *                    4-byte aligned.
 * @param[in]  a      LHS tile, packed signed int4, device memory, 128 bytes,
 *                    4-byte aligned.
 * @param[in]  b      RHS tile, packed signed int4, device memory, 128 bytes,
 *                    4-byte aligned.
 * @param[in]  stream HIP stream to enqueue on, or nullptr for the default.
 * @return 0 on success; EINVAL if any pointer is null or not 4-byte aligned;
 *         EIO if the launch fails; ENOTSUP on a host-only build.
 * @pre A HIP context exists on the calling thread; `a`, `b`, `c` are device
 *      pointers.
 * @post On success the kernel is enqueued on `stream` and `c` holds `A*B`
 *       *once the stream has synchronised*; this call performs no
 *       synchronisation.
 * @warning `a`, `b`, `c` must not alias each other.
 */
int int4_wmma_16x16(int32_t* c, const void* a, const void* b, void* stream) {
    if (c == nullptr || a == nullptr || b == nullptr) {
        return EINVAL;
    }
    if (!is_aligned(c, 4u) || !is_aligned(a, 4u) || !is_aligned(b, 4u)) {
        return EINVAL;
    }
#ifdef ROCMINFER_HAVE_HIP
    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    int4_wmma_16x16_kernel<<<1, kInt4WmmaWaveThreads, 0, hip_stream>>>(
        c, static_cast<const std::uint8_t*>(a),
        static_cast<const std::uint8_t*>(b));
    const hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? 0 : EIO;
#else
    (void)stream;
    return ENOTSUP;
#endif
}

/**
 * @brief Traced wrapper around `int4_wmma_16x16` on the `int4-wmma` component.
 * @param[in]  recorder Tracer; must outlive the launch.
 * @param[out] c        Output tile, device memory.
 * @param[in]  a        LHS tile, device memory.
 * @param[in]  b        RHS tile, device memory.
 * @param[in]  stream   HIP stream, or nullptr.
 * @return Exactly what `int4_wmma_16x16` returns.
 * @pre The device kernel is compiled for a target that supports WMMA (gfx1201).
 * @post On success the launch is enqueued and the scope closed; device timing is
 *       pending until `TraceRecorder::resolve_pending` is called.
 * @warning Per the engine tracing contract, two `int4-wmma` scopes must not be
 *          nested at the same depth.
 */
int int4_wmma_16x16_traced(TraceRecorder& recorder, int32_t* c, const void* a,
                           const void* b, void* stream) {
    TraceScope scope(recorder, "int4-wmma", TraceDomain::kDevice, stream);
    return int4_wmma_16x16(c, a, b, stream);
}

}  // namespace rocminfer
