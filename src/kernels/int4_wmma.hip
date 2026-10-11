/**
 * @file int4_wmma.hip
 * @brief gfx1201 int4 x int4 WMMA tile kernel, host launcher, and broadcast
 *        matvec built on the same instruction.
 *
 * Two kernels live here, both exact signed-int4 x signed-int4 -> int32:
 *
 * - `int4_wmma_16x16_kernel` computes one 16x16 tile `C = A * B`.
 * - `int4_matvec_iu4_i32_kernel` computes the engine broadcast matvec
 *   `y[r] = sum_k s4(W[r,k]) * s4(x[k])` by driving that instruction over
 *   consecutive 16-channel slices of `x`, one wave per 16 output rows, with the
 *   partial sums accumulated in the WMMA's integer registers.
 *
 * Three code paths are selected by the device target at compile time:
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

#include <rocminfer/int4_matvec.hpp>
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

/**
 * @brief Broadcast integer matvec kernel: one wave owns 16 output rows.
 *
 * `y[row_base + m] = sum_{k<cols} s4(W[row_base+m][k]) * s4(x[k])`, computed
 * as `cols / 16` consecutive 16x16x16 IU4 WMMA tiles whose B operand is the
 * 16-channel slice of `x` replicated across all 16 columns.
 *
 * @param[out] y         Destination int32 sums; this block writes 16 entries.
 * @param[in]  W         Packed signed-int4 weight matrix, `row_bytes` per row.
 * @param[in]  x         Packed signed-int4 activation vector, `row_bytes` bytes.
 * @param[in]  row_bytes `cols / 2` packed bytes per weight row; multiple of 8.
 * @param[in]  k_steps   `cols / 16` 16-channel K steps per row.
 * @return Nothing; writes `y[row_base .. row_base + kInt4WmmaTile)`.
 * @pre Launched with `gridDim.x == rows / kInt4WmmaTile` and one wave per
 *      block, so `row_base + kInt4WmmaTile <= rows` needs no bounds test.
 * @note gfx1201: the A fragment of step `s` is loaded from
 *      `W[(row_base + lane % 16) * row_bytes + s * 8 + (lane / 16) * 4]`, the
 *      same expression `int4_wmma_16x16_kernel` uses with a runtime row
 *      stride. Because every column of the B tile holds `x[k]`, the B fragment
 *      is lane-independent: it is just the two 8-nibble words of `x` for this
 *      step, which is what `int4_wmma_16x16_kernel`'s nibble gather would have
 *      produced for any column index. Accumulator slot `s` of a lane in group
 *      `g` is therefore the partial dot of row `row_base + g*8 + s` against the
 *      step's slice, and it is stored to `y` without a column index.
 * @note gfx1031 and host: no matrix core, so one lane per output row
 *      (16 of the 32 lanes) accumulates the row's 8-weight groups with
 *      `__builtin_amdgcn_sdot8` on gfx1031 and a scalar nibble product
 *      elsewhere, the same gate `int4_matvec.hip` uses because the local ROCm
 *      7.16 compiler accepts that builtin on gfx1031 only. The builtin
 *      sign-extends every nibble, which is why only genuine two's-complement
 *      int4 may enter. This path is exact and deliberately not tuned: it is the
 *      portability path, not the hardware path.
 * @warning `y`, `W`, and `x` must not alias each other.
 */
__global__ __launch_bounds__(kInt4WmmaWaveThreads)
void int4_matvec_iu4_i32_kernel(std::int32_t* __restrict__ y,
                                const std::uint8_t* __restrict__ W,
                                const std::uint8_t* __restrict__ x,
                                std::int64_t row_bytes, std::int64_t k_steps) {
    const unsigned int lane = threadIdx.x;
    const std::int64_t row_base =
        static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(kInt4WmmaTile);
#if defined(__gfx1201__)
    const unsigned int group = lane >> 4u;
    const unsigned int index = lane & 15u;

    const std::uint8_t* __restrict__ w_row =
        W + (row_base + static_cast<std::int64_t>(index)) * row_bytes;

    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (std::int64_t step = 0; step < k_steps; ++step) {
        const std::int64_t byte_base = step * (kInt4WmmaTile / 2);
        const std::uint32_t a_word = load_u32_le(w_row + byte_base + group * 4u);
        const std::uint32_t b_word = load_u32_le(x + byte_base + group * 4u);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(
            true, static_cast<std::int32_t>(a_word), true,
            static_cast<std::int32_t>(b_word), acc, false);
    }

    const unsigned int m_base = group * 8u;
    #pragma unroll
    for (unsigned int s = 0u; s < 8u; ++s) {
        y[row_base + static_cast<std::int64_t>(m_base + s)] = acc[s];
    }
#else
    if (lane >= static_cast<unsigned int>(kInt4WmmaTile)) {
        return;
    }
    const std::uint8_t* __restrict__ w_row =
        W + (row_base + static_cast<std::int64_t>(lane)) * row_bytes;
    const std::int64_t groups = k_steps * (kInt4WmmaTile / kInt4MatvecGroup);
    std::int64_t acc = 0;
    for (std::int64_t g = 0; g < groups; ++g) {
        const std::int64_t group_bytes = kInt4MatvecGroup / 2;
        const std::uint32_t ww = load_u32_le(w_row + g * group_bytes);
        const std::uint32_t xx = load_u32_le(x + g * group_bytes);
        #if defined(__gfx1031__)
        acc += static_cast<std::int64_t>(__builtin_amdgcn_sdot8(ww, xx, 0, false));
        #else
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const std::uint32_t nw = (ww >> (4u * static_cast<std::uint32_t>(t))) & 0x0Fu;
            const std::uint32_t nx = (xx >> (4u * static_cast<std::uint32_t>(t))) & 0x0Fu;
            const int vw = (nw & 8u) ? (static_cast<int>(nw) - 16) : static_cast<int>(nw);
            const int vx = (nx & 8u) ? (static_cast<int>(nx) - 16) : static_cast<int>(nx);
            acc += static_cast<std::int64_t>(vw) * static_cast<std::int64_t>(vx);
        }
        #endif
    }
    y[row_base + static_cast<std::int64_t>(lane)] = static_cast<std::int32_t>(acc);
#endif
}

/**
 * @brief Compute `y = W * x` over packed signed int4 on the GPU (broadcast).
 *
 * The engine's INT4 matmul entry point on RDNA4: the same
 * `v_wmma_i32_16x16x16_iu4` instruction `int4_wmma_16x16` emits, applied to a
 * real `rows` x `cols` weight matrix and one shared activation vector instead
 * of a single 16x16 tile. `rows` and `cols` must both be multiples of
 * `kInt4WmmaTile`; see the header for the rationale and the reject cases.
 *
 * @param[out] y      Destination for `rows` int32 sums, device memory.
 * @param[in]  W      Packed signed-int4 weight matrix, device memory.
 * @param[in]  x      Packed signed-int4 activation vector, device memory.
 * @param[in]  rows   Number of output rows; positive multiple of 16.
 * @param[in]  cols   Weights per row; positive multiple of 16.
 * @param[in]  stream HIP stream to enqueue on, or nullptr for the default.
 * @return 0 on success; `EINVAL` on any rejected argument; `EIO` on launch
 *         failure; `ENOTSUP` on a host-only build.
 * @pre A HIP context exists on the calling thread and all three pointers are
 *      device pointers from the HIP allocator.
 * @post On success the kernel is enqueued on `stream`; `y[0..rows)` holds the
 *       sums after the stream synchronises. On failure nothing is enqueued and
 *       `y` is untouched.
 * @note `cols` being a multiple of `kInt4WmmaTile` is enforced in its byte form,
 *       `row_bytes % (kInt4WmmaTile / 2) == 0`, so the rule is read off the
 *       packed layout instead of restating the element count. `row_bytes == 0`
 *       rejects a `cols` that is not a multiple of 8 at all, `weight_bytes == 0`
 *       rejects a `rows * row_bytes` product that would overflow, and the rows
 *       check is direct because a wave owns exactly `kInt4WmmaTile` of them.
 * @warning `W`, `x`, and `y` must not alias each other. Operands are genuine
 *          two's-complement signed int4; `Model::matvec_int4x` rejects
 *          ROCmFP4 `Codebook10` tensors upstream of this entry point.
 */
int int4_matvec_iu4_i32(std::int32_t* y, const void* W, const void* x,
                        std::int64_t rows, std::int64_t cols, void* stream) {
#ifdef ROCMINFER_HAVE_HIP
    if (y == nullptr || W == nullptr || x == nullptr) {
        return EINVAL;
    }
    if (rows <= 0 || cols <= 0) {
        return EINVAL;
    }
    if ((rows % static_cast<std::int64_t>(kInt4WmmaTile)) != 0) {
        return EINVAL;
    }
    if (!is_aligned(y, 4u) || !is_aligned(W, 4u) || !is_aligned(x, 4u)) {
        return EINVAL;
    }
    if (rows > static_cast<std::int64_t>(0xFFFFFFFFu)) {
        return EINVAL;
    }

    const std::size_t row_bytes = int4_matvec_row_bytes(cols);
    const std::size_t weight_bytes = int4_matvec_weight_bytes(rows, cols);
    if (row_bytes == 0u || weight_bytes == 0u ||
        (row_bytes % (static_cast<std::size_t>(kInt4WmmaTile) / 2u)) != 0u) {
        return EINVAL;
    }

    (void)hipGetLastError();

    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned int>(rows / kInt4WmmaTile), 1u, 1u);
    const dim3 threads(kInt4WmmaWaveThreads, 1u, 1u);
    const std::int64_t k_steps = cols / static_cast<std::int64_t>(kInt4WmmaTile);

    hipLaunchKernelGGL(int4_matvec_iu4_i32_kernel, grid, threads, 0u, hip_stream, y,
                       static_cast<const std::uint8_t*>(W),
                       static_cast<const std::uint8_t*>(x),
                       static_cast<std::int64_t>(row_bytes), k_steps);

    const hipError_t launch_status = hipGetLastError();
    return (launch_status == hipSuccess) ? 0 : EIO;
#else
    (void)y;
    (void)W;
    (void)x;
    (void)rows;
    (void)cols;
    (void)stream;
    return ENOTSUP;
#endif
}

/**
 * @brief Traced wrapper around `int4_matvec_iu4_i32` on `int4-matvec-iu4`.
 * @param[in]  recorder Tracer; must outlive the launch.
 * @param[out] y        Destination for `rows` int32 sums, device memory.
 * @param[in]  W        Packed signed-int4 weight matrix, device memory.
 * @param[in]  x        Packed signed-int4 activation vector, device memory.
 * @param[in]  rows     Number of output rows; positive multiple of 16.
 * @param[in]  cols     Weights per row; positive multiple of 16.
 * @param[in]  stream   HIP stream, or nullptr.
 * @return Exactly what `int4_matvec_iu4_i32` returns.
 * @pre Every precondition of `int4_matvec_iu4_i32`; the device kernel is
 *      compiled for a matrix-core target (gfx1201) for the WMMA path, and the
 *      exact sdot8/scalar fallback covers every other target.
 * @post On success the launch is enqueued and the scope closed; device timing is
 *       pending until `TraceRecorder::resolve_pending` is called.
 * @warning Per the engine tracing contract, two `int4-matvec-iu4` scopes must
 *          not be nested at the same depth.
 */
int int4_matvec_iu4_i32_traced(TraceRecorder& recorder, std::int32_t* y,
                               const void* W, const void* x, std::int64_t rows,
                               std::int64_t cols, void* stream) {
    TraceScope scope(recorder, "int4-matvec-iu4", TraceDomain::kDevice, stream);
    return int4_matvec_iu4_i32(y, W, x, rows, cols, stream);
}

}  // namespace rocminfer
