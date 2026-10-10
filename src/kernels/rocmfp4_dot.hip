/**
 * @file rocmfp4_dot.hip
 * @brief Fused ROCmFP4 dequantize-and-dot HIP kernel and its host launcher.
 *
 * Compiles as one translation unit for both host and device. The device half is
 * a portable FP32 dequantize-and-FMA path with no matrix instruction, so the
 * same source builds unchanged for RDNA2 (`gfx1031`, Wave64) and RDNA4
 * (`gfx1201`, Wave32). MFMA is CDNA-only and is deliberately not used.
 *
 * See `rocmfp4_dot.hpp` for the public contract and the numerical bound.
 */

#include <rocminfer/rocmfp4_dot.hpp>

#include <rocminfer/profiler.hpp>
#include <rocminfer/rocmfp4_reference.hpp>

#include <hip/hip_runtime.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <mutex>

namespace rocminfer {

/// Device-side scale table, populated once per process by `upload_scale_table`.
/// Declared at namespace scope (external device linkage) so the WMMA kernel in
/// a separate translation unit can read the same verified table.
__device__ __constant__ float c_rocmfp4_scale[kScaleTableSize];

namespace {

/// Subnormal steps occupy the first seven table indices.
constexpr std::uint32_t kSubnormalEntries = 7u;

/// Highest exponent in the finite unsigned E4M3 encoding.
constexpr std::uint32_t kMaxExponent = 15u;

/**
 * @brief Compute `2^exponent` as an exact float.
 * @param[in] exponent Power of two, in -10..4 for this format.
 * @return Exactly representable power of two.
 * @note Uses the IEEE-754 bit pattern rather than `std::pow` so the result is
 *       `constexpr` and exact, with no rounding and no libm dependency on the
 *       device compilation pass.
 */
constexpr float pow2_exact(int exponent) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(exponent + 127) << 23);
}

/**
 * @brief Generate the UE4M3 half-scale table from the format definition.
 * @return 127 floats; indices 0..125 defined, index 126 (`0x7E`) left at zero.
 * @note Derived, never transcribed. The upstream quantizer's table is the
 *       reference, and `upload_scale_table` compares every entry against
 *       `rocminfer::rocmfp4_scale_to_float` before the table reaches the device,
 *       so a mistyped literal here fails loudly instead of decoding silently.
 */
consteval std::array<float, kScaleTableSize> build_scale_table() noexcept {
    std::array<float, kScaleTableSize> table{};

    for (std::uint32_t mantissa = 0; mantissa <= kSubnormalEntries; ++mantissa) {
        table[mantissa] = static_cast<float>(mantissa) * pow2_exact(-10);
    }

    for (std::uint32_t exponent = 1; exponent <= kMaxExponent; ++exponent) {
        for (std::uint32_t mantissa = 0; mantissa < 8u; ++mantissa) {
            const std::size_t index = 8u + (exponent - 1u) * 8u + mantissa;
            if (index >= kScaleTableSize - 1u) {
                continue;
            }
            table[index] = static_cast<float>(8u + mantissa) *
                        pow2_exact(static_cast<int>(exponent) - 11);
        }
    }

    return table;
}

/// Host mirror of the device scale table, used for the upload and the oracle check.
constexpr std::array<float, kScaleTableSize> kHostScaleTable = build_scale_table();

static_assert(kHostScaleTable[0] == 0.0f, "scale index 0 must decode to zero");
static_assert(kHostScaleTable[1] == 0.5f / 512.0f, "first subnormal step is 2^-10");
static_assert(kHostScaleTable[8] == 8.0f * pow2_exact(-10), "exponent 1 mantissa 0");
static_assert(kHostScaleTable[15] == 15.0f * pow2_exact(-10), "exponent 1 mantissa 7");
static_assert(kHostScaleTable[120] == 8.0f * pow2_exact(4), "exponent 15 mantissa 0");
static_assert(kHostScaleTable[125] == 13.0f * pow2_exact(4), "exponent 15 mantissa 5");
static_assert(kHostScaleTable[126] == 0.0f, "index 0x7E is valid but decodes to zero");

/**
 * @brief Look up one packed block's scale byte in the device table.
 * @param[in] byte Scale byte as stored in the packed block.
 * @return Finite FP32 multiplier, exactly equal to
 *         `rocminfer::rocmfp4_scale_to_float(byte)`.
 * @note Bytes above `ROCMFP4_SCALE_MAX` select index 0, which holds `0.0f`, so
 *       the lookup can never index out of bounds and reproduces the oracle's
 *       "invalid byte decodes to zero" behaviour without a branch on the table
 *       itself.
 */
__device__ inline float scale_from_byte(std::uint8_t byte) noexcept {
    const unsigned int index = (byte <= ROCMFP4_SCALE_MAX) ? byte : 0u;
    return c_rocmfp4_scale[index];
}

/**
 * @brief Read four packed nibble bytes from an arbitrarily 4-byte-aligned address.
 * @param[in] p Address of the first of four consecutive bytes, inside a block.
 * @return The four bytes as a little-endian `uint32_t`, byte `p+0` in bits 0..7.
 * @note Uses two naturally aligned 32-bit loads plus a funnel shift, so it
 *       needs only 4-byte alignment instead of the 16-byte alignment the packed
 *       block stride (17 or 18 bytes) does not guarantee. Reads never leave the
 *       containing block: the shift depends only on `block_base % 4`, so the two
 *       loaded words span `[block_base-3, block_base+5)`, well inside the
 *       18-byte block, and a block base below 4 bytes can only be the first
 *       block of the tensor, which is 4-aligned by construction.
 * @note Reading at `block_base+16` would over-read by three bytes, so scale
 *       bytes are fetched with plain byte loads instead.
 */
__device__ inline std::uint32_t load_nibble_word(const std::uint8_t* p) noexcept {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(p);
    const std::uint32_t shift = static_cast<std::uint32_t>(address & 3u) * 8u;
    const std::uint32_t* base =
        reinterpret_cast<const std::uint32_t*>(address & ~static_cast<std::uintptr_t>(3u));
    if (shift == 0u) {
        return base[0];
    }
    return (base[0] >> shift) | (base[1] << (32u - shift));
}

/**
 * @brief Decode one packed 4-bit code and apply its block scale.
 * @param[in] code  Packed code; only the low nibble participates.
 * @param[in] scale FP32 multiplier from `scale_from_byte`.
 * @return `decode_code(code) * scale`, bit-identical to the CPU oracle.
 * @note Bit 3 is the sign and bits 0..2 are a magnitude code remapped as
 *       `M <= 4 ? M : 2*M - 4`, giving magnitudes 0,1,2,3,4,6,8,10. Negating
 *       the product is exact in IEEE-754, so this matches the oracle's
 *       `(float)(-mag) * scale` bit for bit.
 */
__device__ inline float code_weight(std::uint32_t code, float scale) noexcept {
    const std::uint32_t mag3 = code & 7u;
    const std::uint32_t mag = (mag3 <= 4u) ? mag3 : (2u * mag3 - 4u);
    const float value = static_cast<float>(mag) * scale;
    return (code & 8u) != 0u ? -value : value;
}

/**
 * @brief Accumulate four weights of one block against four activations.
 * @param[in] nib    Four packed codes, byte `0` in bits 0..7, byte `3` in bits
 *                   24..31, matching the layout of `load_nibble_word`. For the
 *                   upper half of a block the caller passes the word shifted
 *                   right by four, so `(nib >> 8) & 0xF` still selects the code
 *                   belonging to the second activation.
 * @param[in] scale  FP32 block scale.
 * @param[in] x0..x3 Four consecutive activations, `x0` matching the first code.
 * @return `sum_{t=0..3} code_weight(nib_t, scale) * x_t`.
 * @note The high nibble of each byte is *not* masked out before the shift-based
 *       extraction because `code_weight` already isolates the sign bit and the
 *       three magnitude bits, so masking would be redundant work.
 */
__device__ inline float dot4(std::uint32_t nib, float scale, float x0, float x1, float x2,
                             float x3) noexcept {
    return code_weight(nib & 0xFu, scale) * x0 + code_weight((nib >> 8) & 0xFu, scale) * x1 +
           code_weight((nib >> 16) & 0xFu, scale) * x2 +
           code_weight((nib >> 24) & 0xFu, scale) * x3;
}

/**
 * @brief Fused dequantize-and-dot kernel: one workgroup per output row.
 * @param[out] out          Destination for `rows` FP32 dot products.
 * @param[in]  weights      Packed blocks, row-major, unpadded.
 * @param[in]  x            Activations, `cols` FP32 elements.
 * @param[in]  blocks_per_row `cols / 32`.
 * @param[in]  block_bytes  18 for the dual-scale layout, 17 for FAST.
 * @param[in]  dual_scale   Wave-uniform layout selector.
 * @return Nothing; writes `out[blockIdx.x]`.
 * @note Launched with exactly `rows` workgroups of `kRocmFP4DotThreads`
 *       threads, so `rows` is not a kernel argument and `blockIdx.x` needs no
 *       bounds test.
 * @note This is a matrix-vector product, not a batched matrix-matrix one: every
 *       output row is contracted against the *same* `cols`-element activation
 *       vector, so `x` is never offset by `row`. Striding `x` here would read
 *       `rows * cols` floats from a buffer the contract sizes at `cols`.
 * @note Byte `j` of a block carries weight `j` in its low nibble and weight
 *       `j + 16` in its high nibble, so the two halves are consumed as
 *       `dot4(q, d0, ...)` and `dot4(q >> 4, d1, ...)` over the same four words.
 * @note The shared-memory reduction rather than a shuffle reduction is what
 *       makes one source valid on both Wave64 and Wave32 targets.
 */
__global__ __launch_bounds__(kRocmFP4DotThreads) void rocmfp4_dot_f32_kernel(
    float* __restrict__ out, const std::uint8_t* __restrict__ weights,
    const float* __restrict__ x, std::int64_t blocks_per_row, std::int64_t block_bytes,
    bool dual_scale) {
    static_assert(kRocmFP4DotThreads == 256u, "kernel launch width changed without review");
    static_assert((kRocmFP4DotThreads & (kRocmFP4DotThreads - 1u)) == 0u,
                  "the tree reduction requires a power-of-two workgroup size");

    const unsigned int tid = threadIdx.x;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);

    const std::uint8_t* __restrict__ row_bytes = weights + row * blocks_per_row * block_bytes;

    float acc = 0.0f;
    for (std::int64_t b = static_cast<std::int64_t>(tid); b < blocks_per_row;
         b += static_cast<std::int64_t>(kRocmFP4DotThreads)) {
        const std::uint8_t* __restrict__ block = row_bytes + b * block_bytes;

        const std::uint32_t q0 = load_nibble_word(block + 0);
        const std::uint32_t q1 = load_nibble_word(block + 4);
        const std::uint32_t q2 = load_nibble_word(block + 8);
        const std::uint32_t q3 = load_nibble_word(block + 12);

        const float d0 = scale_from_byte(block[QS_ROCMFP4]);
        const float d1 = dual_scale ? scale_from_byte(block[QS_ROCMFP4 + 1u]) : d0;

        const float* __restrict__ xp = x + b * static_cast<std::int64_t>(QK_ROCMFP4);
        const float4 lo0 = *reinterpret_cast<const float4*>(xp + 0);
        const float4 lo1 = *reinterpret_cast<const float4*>(xp + 4);
        const float4 lo2 = *reinterpret_cast<const float4*>(xp + 8);
        const float4 lo3 = *reinterpret_cast<const float4*>(xp + 12);
        const float4 hi0 = *reinterpret_cast<const float4*>(xp + 16);
        const float4 hi1 = *reinterpret_cast<const float4*>(xp + 20);
        const float4 hi2 = *reinterpret_cast<const float4*>(xp + 24);
        const float4 hi3 = *reinterpret_cast<const float4*>(xp + 28);

        acc += dot4(q0, d0, lo0.x, lo0.y, lo0.z, lo0.w);
        acc += dot4(q1, d0, lo1.x, lo1.y, lo1.z, lo1.w);
        acc += dot4(q2, d0, lo2.x, lo2.y, lo2.z, lo2.w);
        acc += dot4(q3, d0, lo3.x, lo3.y, lo3.z, lo3.w);
        acc += dot4(q0 >> 4, d1, hi0.x, hi0.y, hi0.z, hi0.w);
        acc += dot4(q1 >> 4, d1, hi1.x, hi1.y, hi1.z, hi1.w);
        acc += dot4(q2 >> 4, d1, hi2.x, hi2.y, hi2.z, hi2.w);
        acc += dot4(q3 >> 4, d1, hi3.x, hi3.y, hi3.z, hi3.w);
    }

    __shared__ float partial[kRocmFP4DotThreads];
    partial[tid] = acc;
    __syncthreads();
    for (unsigned int stride = kRocmFP4DotThreads / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        out[row] = partial[0];
    }
}

}  // namespace

/**
 * @brief Upload and verify the device scale table.
 * @return 0 on success, `EILSEQ` if any table entry disagrees with the CPU
 *         oracle, `EIO` if the copy fails.
 * @note Called exactly once per process from `rocmfp4_ensure_scale_table`.
 *       Verification runs before the upload so a drifted table never reaches the
 *       device. Declared `inline` to match its header and satisfy the
 *       single-definition rule across the host/device passes.
 */
int upload_scale_table() noexcept {
    for (std::size_t i = 0; i < kScaleTableSize; ++i) {
        const float oracle =
            rocmfp4_scale_to_float(static_cast<std::uint8_t>(i));
        if (kHostScaleTable[i] != oracle) {
            return EILSEQ;
        }
    }

    const hipError_t status =
        hipMemcpyToSymbol(c_rocmfp4_scale, kHostScaleTable.data(), sizeof(kHostScaleTable), 0,
                          hipMemcpyHostToDevice);
    return (status == hipSuccess) ? 0 : EIO;
}

/**
 * @brief Ensure the device scale table is present, exactly once per process.
 * @return 0 when the table is loaded and verified, otherwise the sticky failure
 *         code recorded by the first attempt.
 * @note `std::call_once` publishes the stored status to every later caller, so a
 *       failure stays visible instead of being retried-and-forgotten on the next
 *       call. Exposed publicly so the WMMA kernel launcher can require it
 *       independently of the FMA kernel.
 */
int rocmfp4_ensure_scale_table() noexcept {
    struct State {
        std::once_flag once;
        int status = EIO;
    };
    static State state{};

    std::call_once(state.once, []() noexcept { state.status = upload_scale_table(); });
    return state.status;
}

/// True when `value` is a multiple of `alignment`, for a power-of-two alignment.
bool is_aligned(const void* pointer, std::uintptr_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

/// True when `a * b` is representable, for non-negative `b`.
bool product_fits(std::int64_t a, std::int64_t b) noexcept {
    if (a <= 0 || b <= 0) {
        return false;
    }
    return a <= (INT64_MAX / b);
}

std::size_t rocmfp4_dot_row_bytes(std::int64_t cols, bool dual_scale) noexcept {
    if (cols <= 0 || (cols % static_cast<std::int64_t>(QK_ROCMFP4)) != 0) {
        return 0u;
    }
    const std::int64_t block_bytes = dual_scale ? static_cast<std::int64_t>(QK_ROCMFP4_BYTES)
                                                : static_cast<std::int64_t>(QK_ROCMFP4_FAST_BYTES);
    const std::int64_t blocks_per_row = cols / static_cast<std::int64_t>(QK_ROCMFP4);
    if (!product_fits(blocks_per_row, block_bytes)) {
        return 0u;
    }
    return static_cast<std::size_t>(blocks_per_row * block_bytes);
}

std::size_t rocmfp4_dot_weight_bytes(std::int64_t rows, std::int64_t cols,
                                    bool dual_scale) noexcept {
    const std::size_t row_bytes = rocmfp4_dot_row_bytes(cols, dual_scale);
    if (row_bytes == 0u || rows <= 0) {
        return 0u;
    }
    if (rows > static_cast<std::int64_t>(INT64_MAX / static_cast<std::int64_t>(row_bytes))) {
        return 0u;
    }
    return static_cast<std::size_t>(rows) * row_bytes;
}

int rocmfp4_dot_f32(float* out, const void* weights, const float* x, std::int64_t rows,
                    std::int64_t cols, bool dual_scale, void* stream) {
    if (out == nullptr || weights == nullptr || x == nullptr) {
        return EINVAL;
    }
    if (rows <= 0 || cols <= 0 || (cols % static_cast<std::int64_t>(QK_ROCMFP4)) != 0) {
        return EINVAL;
    }
    if (!is_aligned(out, 4u) || !is_aligned(weights, 4u) || !is_aligned(x, 16u)) {
        return EINVAL;
    }

    const std::int64_t block_bytes = dual_scale ? static_cast<std::int64_t>(QK_ROCMFP4_BYTES)
                                                : static_cast<std::int64_t>(QK_ROCMFP4_FAST_BYTES);
    const std::int64_t blocks_per_row = cols / static_cast<std::int64_t>(QK_ROCMFP4);
    if (!product_fits(rows, blocks_per_row) ||
        !product_fits(rows * blocks_per_row, block_bytes)) {
        return EINVAL;
    }
    if (rows > static_cast<std::int64_t>(0xFFFFFFFFu)) {
        return EINVAL;
    }

    const int table_status = rocmfp4_ensure_scale_table();
    if (table_status != 0) {
        return table_status;
    }

    (void)hipGetLastError();

    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned int>(rows), 1u, 1u);
    const dim3 threads(kRocmFP4DotThreads, 1u, 1u);

    hipLaunchKernelGGL(rocmfp4_dot_f32_kernel, grid, threads, 0u, hip_stream,
                       reinterpret_cast<float*>(out),
                       static_cast<const std::uint8_t*>(weights), x,
                       blocks_per_row, block_bytes, dual_scale);

    const hipError_t launch_status = hipGetLastError();
    return (launch_status == hipSuccess) ? 0 : EIO;
}

int rocmfp4_dot_f32_traced(TraceRecorder& recorder, float* out, const void* weights,
                           const float* x, std::int64_t rows, std::int64_t cols,
                           bool dual_scale, void* stream) {
    TraceScope scope(recorder, "quant-dot", TraceDomain::kDevice, stream);
    return rocmfp4_dot_f32(out, weights, x, rows, cols, dual_scale, stream);
}

}  // namespace rocminfer