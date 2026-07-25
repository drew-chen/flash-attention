#include "../cuda_utils.h"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

/*
These notes describe my simplified implementation of Tri Dao's flash attention 2.

flash_forward_v4_cuda_launch expects raw pointers for tensors shaped as:

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

V4 implementation:

V4 uses the same D=64 FA2-style algorithm and warp ownership as V3. It adds a
few lower-level optimizations:

1. Copy aligned Q/K/V data with float4 global loads.

2. Pad each shared K row from 64 to 65 floats so lanes reading the same K column
   use different shared-memory banks. V still reuses this allocation with its
   normal row-major layout.

3. Force-inline several small device stages. Loops are not forced to unroll because
   that increased register use and made the kernel slower.

4. Use 32-bit int values for bounded tile, row, warp, lane, and loop indices to
   avoid unnecessary 64-bit arithmetic in hot loops. Global tensor offsets stay
   std::size_t and are widened explicitly after the bounded index calculations.

Like V3, the kernel aims to fit at least two blocks per SM and does not use
tensor cores or an asynchronous pipeline.

The code below implements the D=64 specialization described above. Other head
dimensions are redirected to V2 by the C++ binding.

Variables are named similar to the FA1 paper: the s prefix means shared mem
ptr, the g prefix means global mem ptr, and _i means a subscript of i.
*/

namespace flash_attention {
namespace {

constexpr int B_r = 64;
constexpr int B_c = 32;
constexpr int WARP_SIZE = 32;

constexpr int HEAD_DIM = 64;
// 1 / sqrt(HEAD_DIM) for the D=64 specialization.
constexpr float SOFTMAX_SCALE = 1.0F / 8.0F;
// A stride of 65 makes lanes reading one K column use different shared-memory banks.
constexpr int K_SHARED_STRIDE = HEAD_DIM + 1;
constexpr int THREAD_BLOCK_SZ = 256;
constexpr int NUM_WARPS = THREAD_BLOCK_SZ / WARP_SIZE;
constexpr int ROWS_PER_WARP = B_r / NUM_WARPS;
static_assert(HEAD_DIM == 64, "V4 is specialized for D=64");
static_assert(THREAD_BLOCK_SZ % WARP_SIZE == 0,
              "V4 thread block size must contain a whole number of warps");
static_assert(B_r % NUM_WARPS == 0, "V4 Q rows must divide evenly among warps");

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
    static constexpr std::size_t total_elements =
        sP_offset + (static_cast<std::size_t>(B_r) * B_c);
    static constexpr std::size_t total_bytes = total_elements * sizeof(float);
};

struct FlashForwardKernelParams {
    const float *const gQ; // [B, H, M, 64] Global memory query pointer.
    const float *const gK; // [B, H, N, 64] Global memory key pointer.
    const float *const gV; // [B, H, N, 64] Global memory value pointer.
    float *const gO;       // [B, H, M, 64] Global memory output pointer.
    const int H;           // Number of heads.
    const int M;           // Query sequence length.
    const int N;           // K/V sequence length.
};

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
 * Loads one D=64 tile with float4 global reads. s_row_stride may include K's
 * shared-memory padding. The C++ binding checks input alignment.
 */
__device__ void load_shared_tile_vectorized(float *const smem_ptr,
                                            const float *const gmem_ptr,
                                            const int H,
                                            const int total_rows,
                                            const int tile_rows,
                                            const int tile_i,
                                            const int s_row_stride) {
    static_assert(HEAD_DIM == 64, "V4 vectorized tile loads assume D=64");
    constexpr int VECTOR_WIDTH = 4;
    constexpr int VECTORS_PER_ROW = HEAD_DIM / VECTOR_WIDTH;
    const int vector_tile_size = tile_rows * VECTORS_PER_ROW;
    const int g_tile_start = tile_i * tile_rows;
    const std::size_t g_tile_offset = get_tile_offset(H, total_rows, tile_rows, tile_i);
    const auto *const g_tile_vectors = reinterpret_cast<const float4 *>(gmem_ptr + g_tile_offset);

    for (int vector_i = static_cast<int>(threadIdx.x); vector_i < vector_tile_size;
         vector_i += blockDim.x) {
        const int s_row = vector_i / VECTORS_PER_ROW;
        const int vector_col = vector_i % VECTORS_PER_ROW;
        const float4 values = g_tile_start + s_row < total_rows
                                  ? g_tile_vectors[vector_i]
                                  : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        float *const s_values = smem_ptr + (s_row * s_row_stride) + (vector_col * VECTOR_WIDTH);
        s_values[0] = values.x;
        s_values[1] = values.y;
        s_values[2] = values.z;
        s_values[3] = values.w;
    }
}

/** Loads Q_i [B_r, D]. */
__device__ float *load_Q_i(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           const int tile_i) {
    float *const sQ_i = smem_ptr + TileParams::sQ_offset;
    load_shared_tile_vectorized(sQ_i, p.gQ, p.H, p.M, B_r, tile_i, HEAD_DIM);
    return sQ_i;
}

/**
 * Loads K_j from K [B, H, N, D] into shared memory with padded K rows.
 * The extra element per row avoids bank conflicts when warp lanes read one K
 * column during QK. V later reuses this allocation with its normal [B_c, D]
 * row-major layout.
 */
__device__ float *load_K_j(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           const int tile_j) {
    float *const sK_j = smem_ptr + TileParams::sKV_shared_offset;
    load_shared_tile_vectorized(sK_j, p.gK, p.H, p.N, B_c, tile_j, K_SHARED_STRIDE);
    return sK_j;
}

/** Loads V_j [B_c, D] after K's aliased shared storage is no longer needed. */
__device__ float *load_V_j(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           const int tile_j) {
    float *const sV_j = smem_ptr + TileParams::sKV_shared_offset;
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
 * Calculates this lane's score column for the rows owned by its warp.
 */
__device__ __forceinline__ void score_ij(const float *const sQ_i,
                                         const float *const sK_j,
                                         const int row_start,
                                         const int lane,
                                         const bool key_valid,
                                         float (&wS_ij)[ROWS_PER_WARP]) {
    if (!key_valid) {
        return;
    }

    for (int k = 0; k < HEAD_DIM; ++k) {
        const float key = sK_j[(lane * K_SHARED_STRIDE) + k];
        for (int owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
            const int row = row_start + owned_row;
            wS_ij[owned_row] += sQ_i[(row * HEAD_DIM) + k] * key;
        }
    }
}

/**
 * Updates this warp's online-softmax state and writes its P rows to shared memory.
 */
__device__ __forceinline__ void online_softmax_ij(float *const sP_ij_unnormalized,
                                                  const int M,
                                                  const int query_start,
                                                  const int row_start,
                                                  const int lane,
                                                  const bool key_valid,
                                                  float (&wS_ij)[ROWS_PER_WARP],
                                                  float (&wM_i_replicated)[ROWS_PER_WARP],
                                                  float (&wL_i_replicated)[ROWS_PER_WARP],
                                                  float (&wO_i_left_unnormalized)[ROWS_PER_WARP],
                                                  float (&wO_i_right_unnormalized)[ROWS_PER_WARP]) {
    for (int owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
        const int row = row_start + owned_row;
        const bool query_valid = query_start + row < M;
        wS_ij[owned_row] = query_valid && key_valid ? wS_ij[owned_row] * SOFTMAX_SCALE
                                                    : -std::numeric_limits<float>::infinity();

        const float tile_max = warp_allreduce<ReductionOp::MAX>(wS_ij[owned_row]);
        const float m_new = query_valid ? fmaxf(wM_i_replicated[owned_row], tile_max) : 0.0F;
        const float p_value = query_valid && key_valid ? expf(wS_ij[owned_row] - m_new) : 0.0F;
        const float tile_sum = warp_allreduce<ReductionOp::SUM>(p_value);
        const float old_scale = query_valid ? expf(wM_i_replicated[owned_row] - m_new) : 0.0F;

        wL_i_replicated[owned_row] = (old_scale * wL_i_replicated[owned_row]) + tile_sum;
        wO_i_left_unnormalized[owned_row] *= old_scale;
        wO_i_right_unnormalized[owned_row] *= old_scale;
        wM_i_replicated[owned_row] = m_new;
        sP_ij_unnormalized[(row * B_c) + lane] = p_value;
    }
}

/**
 * Accumulates this lane's left and right output columns from the current P/V tile.
 */
__device__ __forceinline__ void accumulate_O_i(const float *const sP_ij_unnormalized,
                                               const float *const sV_j,
                                               const int row_start,
                                               const int lane,
                                               float (&wO_i_left_unnormalized)[ROWS_PER_WARP],
                                               float (&wO_i_right_unnormalized)[ROWS_PER_WARP]) {
    for (int key_row = 0; key_row < B_c; ++key_row) {
        const float value_left = sV_j[(key_row * HEAD_DIM) + lane];
        const float value_right = sV_j[(key_row * HEAD_DIM) + lane + WARP_SIZE];
        for (int owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
            const int row = row_start + owned_row;
            const float p_value = sP_ij_unnormalized[(row * B_c) + key_row];
            wO_i_left_unnormalized[owned_row] += p_value * value_left;
            wO_i_right_unnormalized[owned_row] += p_value * value_right;
        }
    }
}

/**
 * Normalizes and stores this lane's output columns.
 */
__device__ __forceinline__ void normalize_and_save_O_i(const FlashForwardKernelParams &p,
                                                       const int query_start,
                                                       const int row_start,
                                                       const int lane,
                                                       const float (&wL_i_replicated)
                                                           [ROWS_PER_WARP],
                                                       const float (&wO_i_left_unnormalized)
                                                           [ROWS_PER_WARP],
                                                       const float (&wO_i_right_unnormalized)
                                                           [ROWS_PER_WARP]) {
    const int batch_head = (static_cast<int>(blockIdx.x) * p.H) + static_cast<int>(blockIdx.y);
    const std::size_t output_head_offset =
        static_cast<std::size_t>(batch_head) * static_cast<std::size_t>(p.M) * HEAD_DIM;
    const std::size_t lane_offset = static_cast<std::size_t>(lane);
    for (int owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
        const int global_row = query_start + row_start + owned_row;
        if (global_row < p.M) {
            const std::size_t output_row_offset =
                output_head_offset + (static_cast<std::size_t>(global_row) * HEAD_DIM);
            p.gO[output_row_offset + lane_offset] =
                wO_i_left_unnormalized[owned_row] / wL_i_replicated[owned_row];
            p.gO[output_row_offset + lane_offset + WARP_SIZE] =
                wO_i_right_unnormalized[owned_row] / wL_i_replicated[owned_row];
        }
    }
}

/**
 * D=64 FA2-style kernel. Each warp owns eight complete query/output rows.
 *
 * A lane holds one score column and two output columns for each owned row.
 * Running softmax state and output accumulators stay in registers. Shared
 * memory contains Q and P. K and V use separate names for the same shared
 * storage because their lifetimes do not overlap.
 */
__global__ void forward_d64(FlashForwardKernelParams p) {
    extern __shared__ float smem[];

    float *const sP_ij_unnormalized = smem + TileParams::sP_offset;

    const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;
    const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
    // Warps split Q/output rows. Their lanes split the K/V work, so every lane
    // loops over the rows owned by its warp instead of owning one complete row.
    const int row_start = warp * ROWS_PER_WARP;
    const int query_tile = static_cast<int>(blockIdx.z);
    const int query_start = query_tile * B_r;
    const int key_tiles = detail::ceil_div<int>(p.N, B_c);

    // Replicated across lanes after warp reductions: each lane needs the same
    // softmax state for ROWS_PER_WARP rows.
    float wM_i_replicated[ROWS_PER_WARP];
    float wL_i_replicated[ROWS_PER_WARP];
    // For this lane, wO_i_left_unnormalized contains ROWS_PER_WARP outputs for one column
    // in 0-31. Collectively across the warp, it contains all columns 0-31.
    float wO_i_left_unnormalized[ROWS_PER_WARP] = {};
    // For this lane, wO_i_right_unnormalized contains ROWS_PER_WARP outputs for one column
    // in 32-63. Collectively across the warp, it contains all columns 32-63.
    float wO_i_right_unnormalized[ROWS_PER_WARP] = {};

    for (int row = 0; row < ROWS_PER_WARP; ++row) {
        wM_i_replicated[row] = -std::numeric_limits<float>::infinity();
        wL_i_replicated[row] = 0.0F;
    }

    float *const sQ_i = load_Q_i(p, smem, query_tile);

    for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
        float *const sK_j = load_K_j(p, smem, key_tile);
        __syncthreads();

        const bool key_valid = (key_tile * B_c) + lane < p.N;
        // For this lane, wS_ij contains ROWS_PER_WARP scores for one column.
        // Collectively across the warp, it contains all B_c score columns.
        float wS_ij[ROWS_PER_WARP] = {};
        score_ij(sQ_i, sK_j, row_start, lane, key_valid, wS_ij);
        online_softmax_ij(sP_ij_unnormalized, p.M, query_start, row_start, lane, key_valid, wS_ij,
                          wM_i_replicated, wL_i_replicated, wO_i_left_unnormalized,
                          wO_i_right_unnormalized);
        // Finish reading sK_j before its shared storage is reused for sV_j.
        __syncthreads();

        float *const sV_j = load_V_j(p, smem, key_tile);
        __syncthreads();

        accumulate_O_i(sP_ij_unnormalized, sV_j, row_start, lane, wO_i_left_unnormalized,
                       wO_i_right_unnormalized);
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
void flash_forward_v4_cuda_launch(const float *gQ,
                                  const float *gK,
                                  const float *gV,
                                  float *gO,
                                  int B,
                                  int H,
                                  int M,
                                  int N,
                                  int D) {
    TORCH_CHECK(D == static_cast<int>(HEAD_DIM), "V4 CUDA kernel supports only D=64");

    FlashForwardKernelParams p{
        gQ,
        gK,
        gV,
        gO,
        H,
        M,
        N,
    };

    const int query_tiles = detail::ceil_div<int>(M, B_r);
    // One block per (batch, head, Q tile).
    dim3 grid{static_cast<unsigned int>(B), static_cast<unsigned int>(H),
              static_cast<unsigned int>(query_tiles)};

    // Prefer the largest shared-memory carveout so three 32.125 KiB blocks can reside on one SM;
    // this trades some L1 capacity for the occupancy targeted by this kernel.
    C10_CUDA_CHECK(cudaFuncSetAttribute(forward_d64, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));

    dim3 block{THREAD_BLOCK_SZ};
    forward_d64<<<grid, block, TileParams::total_bytes, c10::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace flash_attention
