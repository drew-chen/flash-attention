#include "../cuda_utils.h"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

/*
My next steps involve experimenting with tensor cores. As tensor cores are
optimized for fp16 (relative to the higher accuracy tf32) and fp16 storage
reduces memory bandwidth by half, it is naturally to update the input
Q, K and V to fp16 and keep accumulation to fp32 to maintain precision relative
to v4. As a fair baseline
for future tensor core performance, my existing v4 will be adapted to
use fp16 inputs and fp32 accumulation.

flash_forward_v4_fp16_cuda_launch expects raw pointers for tensors shaped as:

- q: [B, H, M, D]
- k: [B, H, N, D]
- v: [B, H, N, D]

- out: [B, H, M, D]

  For each batch element b and head h, out[b, h] =
  softmax((q[b, h] * k[b, h]^T) / sqrt(D)) * v[b, h].

Dimensions:

- (batch_size) B: batch size. How many independent sequences are processed together.
- (num_heads) H: number of attention heads per sequence.
- (query_seq_len) M: number of query/output rows.
- (kv_seq_len) N: number of key/value rows. Self-attention uses M = N, while
  cross-attention may use different lengths.
- (head_dim) D: head dimension. Size of the per-token vector inside one head.

V4 FP16 implementation:

V4 FP16 uses the same D=64 FA2-style algorithm and warp ownership as V3 and V4.
V3's Algorithm section is the canonical ownership and distribution reference.
The algorithm I use to choose between fp16 and fp32 is pretty simple. Inputs to
matrix multiplications should aim to be fp16, which uses less global and shared
memory and gets them ready for tensor cores. If a value may exceed max(fp16) =
65,504, it should be fp32. Values built from many accumulations or reductions
should also be fp32, since fp16 rounding error can add up even when the final
value fits in fp16.

This makes Q, K, V, and the copy of P used by the matrix multiplication fp16.
The QK scores, softmax state m and l, and output accumulators stay fp32. P is an
interesting case because its unnormalized value is

    p_ij = exp(s_ij - m_i_new).

Safe softmax subtracts the updated row maximum, so m_i_new >= s_ij and the
exponent is always zero or negative. exp(0) is 1, while exp of a negative value
is a positive fraction below 1. This means 0 < p_ij <= 1 for valid entries
(masked entries use zero). Even though P has not been normalized by l yet, it
cannot overflow fp16. Very small probabilities can still round or underflow,
so I compute exp and the l reduction in fp32, then store a rounded fp16 copy of
P for the PV matrix multiplication.

PV accumulates into fp32 output registers. At the end I normalize by l in fp32
and round once to fp16 when writing the final output.
*/

namespace flash_attention {
namespace {

constexpr int B_r = 64;
constexpr int B_c = 32;
constexpr int WARP_SIZE = 32;

constexpr int HEAD_DIM = 64;
// C++20's std::sqrt is not constexpr, so spell out 1 / sqrt(64).
constexpr float SOFTMAX_SCALE = 1.0F / 8.0F;
// Offset consecutive K rows by one shared-memory bank width.
constexpr int K_SHARED_STRIDE = HEAD_DIM + 2;
constexpr int THREAD_BLOCK_SZ = 256;
constexpr int NUM_WARPS = THREAD_BLOCK_SZ / WARP_SIZE;
constexpr int WARP_TILE_ROWS = B_r / NUM_WARPS;
static_assert(HEAD_DIM == 64, "V4 FP16 is specialized for D=64");
static_assert(THREAD_BLOCK_SZ % WARP_SIZE == 0,
              "V4 FP16 thread block size must contain a whole number of warps");
static_assert(B_r % NUM_WARPS == 0, "V4 FP16 Q rows must divide evenly among warps");

enum class ReductionOp : std::uint8_t { SUM, MAX };

// Activates all lanes in a warp shuffle (all bits 1).
constexpr unsigned FULL_WARP_MASK = ~0U;

// Keep bounded tile, row, and lane arithmetic 32-bit. Widen explicitly to
// std::size_t only when forming global-memory offsets.
struct TileParams {
    static constexpr std::size_t sQ_offset = 0;
    static constexpr std::size_t sKV_shared_offset =
        sQ_offset + (static_cast<std::size_t>(B_r) * HEAD_DIM);
    static constexpr std::size_t sP_offset =
        sKV_shared_offset + (static_cast<std::size_t>(B_c) * K_SHARED_STRIDE);
    static constexpr std::size_t total_elements = sP_offset + (static_cast<std::size_t>(B_r) * B_c);
    static constexpr std::size_t total_bytes = total_elements * sizeof(half);
};

struct FlashForwardKernelParams {
    const half *const gQ; // [B, H, M, 64] Global memory query pointer.
    const half *const gK; // [B, H, N, 64] Global memory key pointer.
    const half *const gV; // [B, H, N, 64] Global memory value pointer.
    half *const gO;       // [B, H, M, 64] Global memory output pointer.
    const int H;           // Number of heads.
    const int M;           // Query sequence length.
    const int N;           // K/V sequence length.
};

// "Global memory instructions support reading or writing words of
// size equal to 1, 2, 4, 8, or 16 bytes." Such an access compiles to one global
// memory instruction only when naturally aligned. Eight halves form a supported
// 16-byte word, but a plain half[8] aggregate is only 2-byte aligned, so
// alignas(16) supplies the type alignment required for a single 128-bit load.
// The binding separately checks that the actual addresses meet that requirement.
// https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#device-memory-accesses
struct alignas(16) Half8 {
    half values[8];
};
static_assert(sizeof(Half8) == 16, "Half8 must hold one 16-byte global-memory word");
static_assert(alignof(Half8) == 16, "Half8 must be naturally aligned for a 16-byte load");

/**
 * Returns the flat offset for a row tile within the batch and head selected by
 * blockIdx.x and blockIdx.y. Pass blockIdx.z for a Q/O tile and the K/V-loop
 * index for a K/V tile.
 */
__device__ __forceinline__ std::size_t get_tile_offset(const int num_heads,
                                                       const int total_rows,
                                                       const int tile_rows,
                                                       const int tile_idx) {
    const int batch_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int row_offset =
        (((batch_idx * num_heads) + head_idx) * total_rows) + (tile_idx * tile_rows);
    return static_cast<std::size_t>(row_offset) * HEAD_DIM;
}

/**
 * Loads one D=64 tile with 16-byte global reads. Each read contains eight half
 * values. s_row_stride may include K's shared-memory padding. The C++ binding
 * checks input alignment.
 */
__device__ void load_shared_tile_vectorized(half *const smem_ptr,
                                            const half *const gmem_ptr,
                                            const int H,
                                            const int total_rows,
                                            const int tile_rows,
                                            const int tile_i,
                                            const int s_row_stride) {
    static_assert(HEAD_DIM == 64, "V4 FP16 vectorized tile loads assume D=64");
    constexpr int VECTOR_WIDTH = 8;
    constexpr int VECTORS_PER_ROW = HEAD_DIM / VECTOR_WIDTH;
    const int vector_tile_size = tile_rows * VECTORS_PER_ROW;
    const int g_tile_start = tile_i * tile_rows;
    const std::size_t g_tile_offset = get_tile_offset(H, total_rows, tile_rows, tile_i);
    const auto *const g_tile_vectors = reinterpret_cast<const Half8 *>(gmem_ptr + g_tile_offset);

    for (int vector_i = static_cast<int>(threadIdx.x); vector_i < vector_tile_size;
         vector_i += blockDim.x) {
        const int s_row = vector_i / VECTORS_PER_ROW;
        const int vector_col = vector_i % VECTORS_PER_ROW;
        Half8 values{};
        if (g_tile_start + s_row < total_rows) {
            values = g_tile_vectors[vector_i];
        }
        half *const s_values = smem_ptr + (s_row * s_row_stride) + (vector_col * VECTOR_WIDTH);
#pragma unroll
        for (int value_i = 0; value_i < VECTOR_WIDTH; ++value_i) {
            s_values[value_i] = values.values[value_i];
        }
    }
}

/** Loads Q_i [B_r, D]. */
__device__ half *load_Q_i(const FlashForwardKernelParams &p,
                          half *const smem_ptr,
                          const int tile_i) {
    half *const sQ_i = smem_ptr + TileParams::sQ_offset;
    load_shared_tile_vectorized(sQ_i, p.gQ, p.H, p.M, B_r, tile_i, HEAD_DIM);
    return sQ_i;
}

/**
 * Loads K_j from K [B, H, N, D] into shared memory with padded K rows.
 * The two half-precision padding elements per row avoid bank conflicts when
 * warp lanes read one K column during QK. V later reuses this allocation with
 * its normal [B_c, D] row-major layout.
 */
__device__ half *load_K_j(const FlashForwardKernelParams &p,
                          half *const smem_ptr,
                          const int tile_j) {
    half *const sK_j = smem_ptr + TileParams::sKV_shared_offset;
    load_shared_tile_vectorized(sK_j, p.gK, p.H, p.N, B_c, tile_j, K_SHARED_STRIDE);
    return sK_j;
}

/** Loads V_j [B_c, D] after K's aliased shared storage is no longer needed. */
__device__ half *load_V_j(const FlashForwardKernelParams &p,
                          half *const smem_ptr,
                          const int tile_j) {
    half *const sV_j = smem_ptr + TileParams::sKV_shared_offset;
    load_shared_tile_vectorized(sV_j, p.gV, p.H, p.N, B_c, tile_j, HEAD_DIM);
    return sV_j;
}

/**
 * Reduces one value from each warp lane using Op.
 * The complete result is returned by the lane with threadIdx.x % 32 == 0.
 */
template <ReductionOp Op> __device__ __forceinline__ float warp_reduce(float initial_value) {
    float value = initial_value;
    // Warp reduction pattern: compare 32 elements across 32 threads
    for (int offset = 16; offset >= 1; offset /= 2) {
        const float other =
            __shfl_down_sync(FULL_WARP_MASK, value, static_cast<unsigned int>(offset));
        if constexpr (Op == ReductionOp::SUM) {
            value += other;
        } else if constexpr (Op == ReductionOp::MAX) {
            value = fmaxf(value, other);
        }
    }
    return value;
}

/**
 * Reduces across a warp and broadcasts the result back to every lane.
 */
template <ReductionOp Op> __device__ __forceinline__ float warp_allreduce(float initial_value) {
    return __shfl_sync(FULL_WARP_MASK, warp_reduce<Op>(initial_value), 0);
}

/**
 * Calculates the current Q/K tile's scores for this warp.
 *
 * The warp logically computes
 *
 *     S_ij[row_start : row_start + WARP_TILE_ROWS, :]
 *         = sQ_i[row_start : row_start + WARP_TILE_ROWS, :] @ sK_j^T
 *
 * with shape [WARP_TILE_ROWS, D] @ [D, B_c] = [WARP_TILE_ROWS, B_c].
 * Returns wS_ij which contains S_ij[:, lane].
 */
__device__ __forceinline__ std::array<float, WARP_TILE_ROWS>
score_ij(const half *const sQ_i,
         const half *const sK_j,
         const int row_start,
         const int lane,
         const bool key_valid) {
    std::array<float, WARP_TILE_ROWS> wS_ij{};
    if (!key_valid) {
        return wS_ij;
    }

    for (int d = 0; d < HEAD_DIM; ++d) {
        const float key = __half2float(sK_j[(lane * K_SHARED_STRIDE) + d]);
        for (int warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
            const int row = row_start + warp_row;
            const float query = __half2float(sQ_i[(row * HEAD_DIM) + d]);
            wS_ij[static_cast<std::size_t>(warp_row)] += query * key;
        }
    }
    return wS_ij;
}

/**
 * Updates this warp's online-softmax state and writes its P rows to shared memory.
 */
__device__ __forceinline__ void online_softmax_ij(half *const sP_ij_unnormalized,
                                                  const int M,
                                                  const int query_start,
                                                  const int row_start,
                                                  const int lane,
                                                  const bool key_valid,
                                                  std::array<float, WARP_TILE_ROWS> &wS_ij,
                                                  float (&wM_i_replicated)[WARP_TILE_ROWS],
                                                  float (&wL_i_replicated)[WARP_TILE_ROWS],
                                                  float (&wO_i_left_unnormalized)[WARP_TILE_ROWS],
                                                  float (&wO_i_right_unnormalized)[WARP_TILE_ROWS]) {
    for (int warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
        const int row = row_start + warp_row;
        const bool query_valid = query_start + row < M;
        float &score = wS_ij[static_cast<std::size_t>(warp_row)];
        score = query_valid && key_valid ? score * SOFTMAX_SCALE
                                         : -std::numeric_limits<float>::infinity();

        // Each lane contributes its score against sK_j[lane, :] for this owned
        // Q row. The reductions broadcast the row result back to every lane.
        const float key_tile_row_max = warp_allreduce<ReductionOp::MAX>(score);
        const float m_new =
            query_valid ? fmaxf(wM_i_replicated[warp_row], key_tile_row_max) : 0.0F;
        const float p_value = query_valid && key_valid ? expf(score - m_new) : 0.0F;
        const float key_tile_row_sum = warp_allreduce<ReductionOp::SUM>(p_value);
        const float old_scale = query_valid ? expf(wM_i_replicated[warp_row] - m_new) : 0.0F;

        wL_i_replicated[warp_row] =
            (old_scale * wL_i_replicated[warp_row]) + key_tile_row_sum;
        wO_i_left_unnormalized[warp_row] *= old_scale;
        wO_i_right_unnormalized[warp_row] *= old_scale;
        wM_i_replicated[warp_row] = m_new;
        sP_ij_unnormalized[(row * B_c) + lane] = __float2half_rn(p_value);
    }
}

/**
 * Accumulates the current P/V tile's contribution to this warp's output.
 *
 * The warp logically computes
 *
 *     wO += sP_ij_unnormalized[row_start : row_start + WARP_TILE_ROWS, :] @ sV_j
 *
 * with shape [WARP_TILE_ROWS, B_c] @ [B_c, D] = [WARP_TILE_ROWS, D].
 */
__device__ __forceinline__ void accumulate_PV_into_O_i(
    const half *const sP_ij_unnormalized,
    const half *const sV_j,
    const int row_start,
    const int lane,
    float (&wO_i_left_unnormalized)[WARP_TILE_ROWS],
    float (&wO_i_right_unnormalized)[WARP_TILE_ROWS]) {
    for (int key_row = 0; key_row < B_c; ++key_row) {
        const float value_left = __half2float(sV_j[(key_row * HEAD_DIM) + lane]);
        const float value_right =
            __half2float(sV_j[(key_row * HEAD_DIM) + lane + WARP_SIZE]);
        for (int warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
            const int row = row_start + warp_row;
            const float p_value = __half2float(sP_ij_unnormalized[(row * B_c) + key_row]);
            wO_i_left_unnormalized[warp_row] += p_value * value_left;
            wO_i_right_unnormalized[warp_row] += p_value * value_right;
        }
    }
}

/**
 * Normalizes and stores this lane's two output dimensions.
 */
__device__ __forceinline__ void normalize_and_save_O_i(const FlashForwardKernelParams &p,
                                                       const int query_start,
                                                       const int row_start,
                                                       const int lane,
                                                       const float (&wL_i_replicated)
                                                           [WARP_TILE_ROWS],
                                                       const float (&wO_i_left_unnormalized)
                                                           [WARP_TILE_ROWS],
                                                       const float (&wO_i_right_unnormalized)
                                                           [WARP_TILE_ROWS]) {
    const int batch_head = (static_cast<int>(blockIdx.x) * p.H) + static_cast<int>(blockIdx.y);
    const std::size_t output_head_offset =
        static_cast<std::size_t>(batch_head) * static_cast<std::size_t>(p.M) * HEAD_DIM;
    const std::size_t lane_offset = static_cast<std::size_t>(lane);
    for (int warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
        const int global_row = query_start + row_start + warp_row;
        if (global_row < p.M) {
            const std::size_t output_row_offset =
                output_head_offset + (static_cast<std::size_t>(global_row) * HEAD_DIM);
            p.gO[output_row_offset + lane_offset] = __float2half_rn(
                wO_i_left_unnormalized[warp_row] / wL_i_replicated[warp_row]);
            p.gO[output_row_offset + lane_offset + WARP_SIZE] = __float2half_rn(
                wO_i_right_unnormalized[warp_row] / wL_i_replicated[warp_row]);
        }
    }
}

/**
 * D=64 FA2-style kernel. Each warp owns eight query/output rows.
 *
 * For every Q row owned by a warp, each lane computes one score. Lane 0 uses
 * K row 0 within the current block, lane 1 uses K row 1, and so on. Each lane
 * also accumulates two output dimensions. Running softmax state and output
 * accumulators stay in registers. Shared memory contains Q and P. K and V use
 * separate names for the same shared storage because their lifetimes do not
 * overlap.
 */
__global__ void forward_d64(FlashForwardKernelParams p) {
    extern __shared__ half smem[];

    half *const sP_ij_unnormalized = smem + TileParams::sP_offset;

    const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
    const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
    // Each warp owns a contiguous set of Q/output rows, and every lane loops
    // over all rows owned by its warp.
    const int row_start = warp * WARP_TILE_ROWS;
    const int query_tile = static_cast<int>(blockIdx.z);
    const int query_start = query_tile * B_r;
    const int key_tiles = detail::ceil_div<int>(p.N, B_c);

    // Every lane keeps the same max and denominator for each warp-owned row.
    // Warp reductions keep these copies in sync.
    float wM_i_replicated[WARP_TILE_ROWS];
    float wL_i_replicated[WARP_TILE_ROWS];
    // These arrays correspond to the two iterations of:
    // for (int output_dim = lane; output_dim < HEAD_DIM; output_dim += WARP_SIZE)
    // D=64 is twice the warp size, so two accumulators per lane cover every
    // output dimension without an inter-lane reduction.
    // This lane accumulates one value per warp-owned row at output index lane.
    float wO_i_left_unnormalized[WARP_TILE_ROWS] = {};
    // It also accumulates one value per row at output index lane + WARP_SIZE.
    float wO_i_right_unnormalized[WARP_TILE_ROWS] = {};

    for (int row = 0; row < WARP_TILE_ROWS; ++row) {
        wM_i_replicated[row] = -std::numeric_limits<float>::infinity();
        wL_i_replicated[row] = 0.0F;
    }

    half *const sQ_i = load_Q_i(p, smem, query_tile);

    for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
        half *const sK_j = load_K_j(p, smem, key_tile);
        __syncthreads();

        const bool key_valid = (key_tile * B_c) + lane < p.N;
        // For every Q row owned by this warp, wS_ij stores this lane's score
        // against sK_j[lane, :].
        auto wS_ij =
            score_ij(sQ_i, sK_j, row_start, lane, key_valid);
        online_softmax_ij(sP_ij_unnormalized, p.M, query_start, row_start, lane, key_valid, wS_ij,
                          wM_i_replicated, wL_i_replicated, wO_i_left_unnormalized,
                          wO_i_right_unnormalized);
        // Finish reading sK_j before its shared storage is reused for sV_j.
        __syncthreads();

        half *const sV_j = load_V_j(p, smem, key_tile);
        __syncthreads();

        accumulate_PV_into_O_i(sP_ij_unnormalized, sV_j, row_start, lane,
                               wO_i_left_unnormalized, wO_i_right_unnormalized);
        // Finish reading sV_j before the next sK_j reuses its shared storage.
        __syncthreads();
    }

    normalize_and_save_O_i(p, query_start, row_start, lane, wL_i_replicated, wO_i_left_unnormalized,
                           wO_i_right_unnormalized);
}

} // namespace

/**
 * Configures and launches the tiled attention kernel. The kernel keeps its
 * running softmax state in registers.
 */
void flash_forward_v4_fp16_cuda_launch(const c10::Half *const gQ,
                                    const c10::Half *const gK,
                                    const c10::Half *const gV,
                                    c10::Half *const gO,
                                    int B,
                                    int H,
                                    int M,
                                    int N,
                                    int D) {
    TORCH_CHECK(D == static_cast<int>(HEAD_DIM), "V4 FP16 CUDA kernel supports only D=64");

    // c10::Half is PyTorch's portable fp16 type while half is CUDA-specific.
    // Their pointer types are unrelated, so the shared fp16 representation is
    // reinterpreted once at the launcher boundary instead of converted element by element.
    FlashForwardKernelParams p{
        reinterpret_cast<const half *>(gQ),
        reinterpret_cast<const half *>(gK),
        reinterpret_cast<const half *>(gV),
        reinterpret_cast<half *>(gO),
        H,
        M,
        N,
    };

    const int query_tiles = detail::ceil_div<int>(M, B_r);
    // One block per (batch, head, Q tile).
    dim3 grid{static_cast<unsigned int>(B), static_cast<unsigned int>(H),
              static_cast<unsigned int>(query_tiles)};

    // Prefer the largest shared-memory carveout to maximize residency for the
    // 16.125 KiB half-precision tile allocation.
    C10_CUDA_CHECK(cudaFuncSetAttribute(forward_d64, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));

    dim3 block{THREAD_BLOCK_SZ};
    forward_d64<<<grid, block, TileParams::total_bytes, c10::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace flash_attention
