#include "../cuda_utils.h"
#include <array>
#include <c10/cuda/CUDAException.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

/*
V3 warp-local FlashAttention-2 forward kernel.

API:
- q: [B, H, M, D]
- k: [B, H, N, D]
- v: [B, H, N, D]
- out: [B, H, M, D]

For each batch element b and head h:

    out[b, h] = softmax((q[b, h] @ k[b, h]^T) / sqrt(D)) @ v[b, h]

Dimensions:
- (batch_size) B: batch size.
- (num_heads) H: number of attention heads.
- (query_seq_len) M: number of query/output rows.
- (kv_seq_len) N: number of key/value rows. M and N may differ.
- (head_dim) D: head dimension.

Inputs and output are contiguous CUDA float32 tensors. The optimized kernel is
specialized for D=64; `flash_forward_v3()` in flash.cpp redirects other head
dimensions to V2.

Algorithm and optimization notes: docs/v3-extended-notes.md
*/

namespace flash_attention {
namespace {

constexpr std::size_t B_r = 64;
constexpr std::size_t B_c = 32;
constexpr std::size_t WARP_SIZE = 32;

constexpr std::size_t HEAD_DIM = 64;
// C++20's std::sqrt is not constexpr, so spell out 1 / sqrt(64).
constexpr float SOFTMAX_SCALE = 1.0F / 8.0F;
constexpr int THREAD_BLOCK_SZ = 256;
constexpr std::size_t NUM_WARPS = THREAD_BLOCK_SZ / WARP_SIZE;
constexpr std::size_t WARP_TILE_ROWS = B_r / NUM_WARPS;
static_assert(HEAD_DIM == 64, "V3 is specialized for D=64");
static_assert(THREAD_BLOCK_SZ % WARP_SIZE == 0,
              "V3 thread block size must contain a whole number of warps");
static_assert(B_r % NUM_WARPS == 0, "V3 Q rows must divide evenly among warps");
static_assert(B_c == WARP_SIZE, "V3 maps one K row and score column to each warp lane");

enum class ReductionOp : std::uint8_t { SUM, MAX };

// Marks all 32 lanes as participants in a warp shuffle.
constexpr unsigned FULL_WARP_MASK = ~0U;

struct TileParams {
    static constexpr std::size_t sQ_offset = 0;
    static constexpr std::size_t sKV_shared_offset = sQ_offset + (B_r * HEAD_DIM);
    static constexpr std::size_t sP_offset = sKV_shared_offset + (B_c * HEAD_DIM);
    static constexpr std::size_t total_elements = sP_offset + (B_r * B_c);
    static constexpr std::size_t total_bytes = total_elements * sizeof(float);
};

struct FlashForwardKernelParams {
    const float *const gQ; // [B, H, M, 64] Global memory query pointer.
    const float *const gK; // [B, H, N, 64] Global memory key pointer.
    const float *const gV; // [B, H, N, 64] Global memory value pointer.
    float *const gO;       // [B, H, M, 64] Global memory output pointer.
    const std::size_t H;   // Number of heads.
    const std::size_t M;   // Query sequence length.
    const std::size_t N;   // K/V sequence length.
};

/**
 * Returns the flat offset for a row tile within the batch and head selected by
 * blockIdx.x and blockIdx.y. Pass blockIdx.z for a Q/O tile and the K/V-loop
 * index for a K/V tile.
 */
__device__ std::size_t get_tile_offset(const std::size_t H,
                                       const std::size_t total_rows,
                                       const std::size_t total_cols,
                                       const std::size_t tile_rows,
                                       const std::size_t tile_idx) {
    const std::size_t batch_idx = static_cast<std::size_t>(blockIdx.x);
    const std::size_t head_idx = static_cast<std::size_t>(blockIdx.y);
    const std::size_t batch_head_offset = ((batch_idx * H) + head_idx) * total_rows * total_cols;
    return batch_head_offset + (tile_idx * tile_rows * total_cols);
}

/**
 * Uses the thread block to load one contiguous data tile from global to shared memory.
 *
 * gmem_ptr has shape [B, H, total_rows, total_cols]. blockIdx selects [B, H],
 * and tile_i selects the [tile_rows, total_cols] segment of [total_rows, total_cols].
 * Out-of-range rows use pad.
 * The caller must synchronize before consuming or reusing shared memory.
 * Setting total_cols == 1 loads a 1D tile (row-major).
 */
__device__ void load_shared_tile(float *const smem_ptr,
                                 const float *const gmem_ptr,
                                 const std::size_t H,
                                 const std::size_t total_rows,
                                 const std::size_t total_cols,
                                 const std::size_t tile_rows,
                                 const std::size_t tile_i,
                                 const float pad = 0.0F) {
    const std::size_t s_tile_size = tile_rows * total_cols;
    const std::size_t g_tile_start = tile_i * tile_rows;
    const std::size_t g_tile_offset = get_tile_offset(H, total_rows, total_cols, tile_rows, tile_i);

    for (std::size_t flattened_i = static_cast<std::size_t>(threadIdx.x); flattened_i < s_tile_size;
         flattened_i += blockDim.x) {
        const std::size_t s_row = flattened_i / total_cols;
        // The tile and global column indices are the same.
        const std::size_t col = flattened_i % total_cols;

        // Convert the tile row into a global row.
        const std::size_t g_row = g_tile_start + s_row;
        if (g_row < total_rows) {
            const std::size_t g_idx = g_tile_offset + (s_row * total_cols) + col;
            smem_ptr[flattened_i] = gmem_ptr[g_idx];
        } else {
            smem_ptr[flattened_i] = pad;
        }
    }
}

/**
 * Loads Q_i [B_r, D] from Q [B, H, M, D].
 */
__device__ float *load_Q_i(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           const std::size_t tile_i) {
    float *const sQ_i = smem_ptr + TileParams::sQ_offset;
    load_shared_tile(sQ_i, p.gQ, p.H, p.M, HEAD_DIM, B_r, tile_i);
    return sQ_i;
}

/**
 * Loads K_j [B_c, D] from K [B, H, N, D].
 */
__device__ float *load_K_j(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           const std::size_t tile_j) {
    float *const sK_j = smem_ptr + TileParams::sKV_shared_offset;
    load_shared_tile(sK_j, p.gK, p.H, p.N, HEAD_DIM, B_c, tile_j);
    return sK_j;
}

/**
 * Loads V_j [B_c, D] from V [B, H, N, D].
 *
 * V reuses K's shared-memory storage after the score and P are calculated.
 */
__device__ float *load_V_j(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           const std::size_t tile_j) {
    float *const sV_j = smem_ptr + TileParams::sKV_shared_offset;
    load_shared_tile(sV_j, p.gV, p.H, p.N, HEAD_DIM, B_c, tile_j);
    return sV_j;
}

/**
 * Reduces one value from each warp lane using Op.
 * The complete result is returned by the lane with threadIdx.x % 32 == 0.
 */
template <ReductionOp Op> __device__ float warp_reduce(float initial_value) {
    float value = initial_value;
    // Combine one value from each of the 32 lanes.
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
template <ReductionOp Op> __device__ float warp_allreduce(float initial_value) {
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
 * B_c equals WARP_SIZE, so lane l computes column l of S_ij.
 * Returns wS_ij which contains S_ij[:, lane].
 */
__device__ std::array<float, WARP_TILE_ROWS> score_ij(const float *const sQ_i,
                                                      const float *const sK_j,
                                                      const std::size_t row_start,
                                                      const std::size_t lane,
                                                      const bool key_valid) {
    std::array<float, WARP_TILE_ROWS> wS_ij{};
    if (!key_valid) {
        return wS_ij;
    }
    // Dot this lane's K row with every Q row owned by the warp to produce
    // one column of S_ij.
    for (std::size_t d = 0; d < HEAD_DIM; ++d) {
        const float key = sK_j[(lane * HEAD_DIM) + d];
        for (std::size_t warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
            const std::size_t row = row_start + warp_row;
            wS_ij[warp_row] += sQ_i[(row * HEAD_DIM) + d] * key;
        }
    }
    return wS_ij;
}

/**
 * Updates this warp's online-softmax state and writes its P rows to shared memory.
 */
__device__ void online_softmax_ij(float *const sP_ij_unnormalized,
                                  const std::size_t M,
                                  const std::size_t query_start,
                                  const std::size_t row_start,
                                  const std::size_t lane,
                                  const bool key_valid,
                                  std::array<float, WARP_TILE_ROWS> &wS_ij,
                                  float (&wM_i_replicated)[WARP_TILE_ROWS],
                                  float (&wL_i_replicated)[WARP_TILE_ROWS],
                                  float (&wO_i_left_unnormalized)[WARP_TILE_ROWS],
                                  float (&wO_i_right_unnormalized)[WARP_TILE_ROWS]) {
    for (std::size_t warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
        const std::size_t row = row_start + warp_row;
        const bool query_valid = query_start + row < M;
        wS_ij[warp_row] = query_valid && key_valid ? wS_ij[warp_row] * SOFTMAX_SCALE
                                                   : -std::numeric_limits<float>::infinity();

        // Each lane contributes its score for one K row in the current tile.
        // The reductions produce this Q row's max and sum for the current K
        // tile, then broadcast each result back to every lane.
        const float key_tile_row_max = warp_allreduce<ReductionOp::MAX>(wS_ij[warp_row]);
        const float m_new = query_valid ? fmaxf(wM_i_replicated[warp_row], key_tile_row_max) : 0.0F;
        const float p_value = query_valid && key_valid ? expf(wS_ij[warp_row] - m_new) : 0.0F;
        const float key_tile_row_sum = warp_allreduce<ReductionOp::SUM>(p_value);
        const float old_scale = query_valid ? expf(wM_i_replicated[warp_row] - m_new) : 0.0F;

        wL_i_replicated[warp_row] = (old_scale * wL_i_replicated[warp_row]) + key_tile_row_sum;
        wO_i_left_unnormalized[warp_row] *= old_scale;
        wO_i_right_unnormalized[warp_row] *= old_scale;
        wM_i_replicated[warp_row] = m_new;
        sP_ij_unnormalized[(row * B_c) + lane] = p_value;
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
__device__ void accumulate_PV_into_wO_i(const float *const sP_ij_unnormalized,
                                        const float *const sV_j,
                                        const std::size_t row_start,
                                        const std::size_t lane,
                                        float (&wO_i_left_unnormalized)[WARP_TILE_ROWS],
                                        float (&wO_i_right_unnormalized)[WARP_TILE_ROWS]) {
    for (std::size_t key_row = 0; key_row < B_c; ++key_row) {
        // Each lane computes two output columns so all 32 lanes span D=64.
        const float value_left = sV_j[(key_row * HEAD_DIM) + lane];
        const float value_right = sV_j[(key_row * HEAD_DIM) + lane + WARP_SIZE];
        for (std::size_t warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
            const std::size_t row = row_start + warp_row;
            const float p_value = sP_ij_unnormalized[(row * B_c) + key_row];
            wO_i_left_unnormalized[warp_row] += p_value * value_left;
            wO_i_right_unnormalized[warp_row] += p_value * value_right;
        }
    }
}

/**
 * Normalizes and stores this lane's two output dimensions.
 */
__device__ void normalize_and_save_O_i(const FlashForwardKernelParams &p,
                                       const std::size_t query_start,
                                       const std::size_t row_start,
                                       const std::size_t lane,
                                       const float (&wL_i_replicated)[WARP_TILE_ROWS],
                                       const float (&wO_i_left_unnormalized)[WARP_TILE_ROWS],
                                       const float (&wO_i_right_unnormalized)[WARP_TILE_ROWS]) {
    const std::size_t batch_head =
        (static_cast<std::size_t>(blockIdx.x) * p.H) + static_cast<std::size_t>(blockIdx.y);
    const std::size_t output_head_offset = batch_head * p.M * HEAD_DIM;
    for (std::size_t warp_row = 0; warp_row < WARP_TILE_ROWS; ++warp_row) {
        const std::size_t global_row = query_start + row_start + warp_row;
        if (global_row < p.M) {
            const std::size_t output_row_offset = output_head_offset + (global_row * HEAD_DIM);
            p.gO[output_row_offset + lane] =
                wO_i_left_unnormalized[warp_row] / wL_i_replicated[warp_row];
            p.gO[output_row_offset + lane + WARP_SIZE] =
                wO_i_right_unnormalized[warp_row] / wL_i_replicated[warp_row];
        }
    }
}

/**
 * D=64 FA2-style kernel. Each warp owns eight query/output rows.
 *
 * For every Q row owned by a warp, each lane computes one score. Lane 0 uses
 * K row 0 within the current block, lane 1 uses K row 1, and so on. Each lane
 * also accumulates two output dimensions.
 * Running softmax state and output accumulators stay in registers. Shared memory
 * contains Q and P. K and V use separate names for the same shared storage
 * because their lifetimes do not overlap.
 */
__global__ void forward_d64(FlashForwardKernelParams p) {
    extern __shared__ float smem[];

    float *const sP_ij_unnormalized = smem + TileParams::sP_offset;

    const std::size_t warp = static_cast<std::size_t>(threadIdx.x) / WARP_SIZE;
    const std::size_t lane = static_cast<std::size_t>(threadIdx.x) % WARP_SIZE;
    // Each warp owns a contiguous set of Q/output rows, and every lane loops
    // over all rows owned by its warp.
    const std::size_t row_start = warp * WARP_TILE_ROWS;
    const std::size_t query_tile = static_cast<std::size_t>(blockIdx.z);
    const std::size_t query_start = query_tile * B_r;
    const std::size_t key_tiles = detail::ceil_div<std::size_t>(p.N, B_c);

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

    for (std::size_t row = 0; row < WARP_TILE_ROWS; ++row) {
        wM_i_replicated[row] = -std::numeric_limits<float>::infinity();
        wL_i_replicated[row] = 0.0F;
    }

    float *const sQ_i = load_Q_i(p, smem, query_tile);

    for (std::size_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
        float *const sK_j = load_K_j(p, smem, key_tile);
        __syncthreads();

        const bool key_valid = (key_tile * B_c) + lane < p.N;
        // For every Q row owned by this warp, wS_ij stores this lane's score
        // against the K tile row with the same index as the lane.
        auto wS_ij = score_ij(sQ_i, sK_j, row_start, lane, key_valid);
        online_softmax_ij(sP_ij_unnormalized, p.M, query_start, row_start, lane, key_valid, wS_ij,
                          wM_i_replicated, wL_i_replicated, wO_i_left_unnormalized,
                          wO_i_right_unnormalized);
        // Finish reading sK_j before its shared storage is reused for sV_j.
        __syncthreads();

        float *const sV_j = load_V_j(p, smem, key_tile);
        __syncthreads();

        accumulate_PV_into_wO_i(sP_ij_unnormalized, sV_j, row_start, lane, wO_i_left_unnormalized,
                                wO_i_right_unnormalized);
        // Finish reading sV_j before the next sK_j reuses its shared storage.
        __syncthreads();
    }

    normalize_and_save_O_i(p, query_start, row_start, lane, wL_i_replicated, wO_i_left_unnormalized,
                           wO_i_right_unnormalized);
}

} // namespace

/**
 * Allocates the running softmax state and launches the tiled attention kernel.
 */
void flash_forward_v3_cuda_launch(const float *gQ,
                                  const float *gK,
                                  const float *gV,
                                  float *gO,
                                  int B,
                                  int H,
                                  int M,
                                  int N,
                                  int D) {
    TORCH_CHECK(D == static_cast<int>(HEAD_DIM), "V3 CUDA kernel supports only D=64");

    FlashForwardKernelParams p{
        gQ,
        gK,
        gV,
        gO,
        static_cast<std::size_t>(H),
        static_cast<std::size_t>(M),
        static_cast<std::size_t>(N),
    };

    const std::size_t query_tiles = detail::ceil_div<std::size_t>(static_cast<std::size_t>(M), B_r);
    // One block per (batch, head, Q tile).
    dim3 grid{static_cast<unsigned int>(B), static_cast<unsigned int>(H),
              static_cast<unsigned int>(query_tiles)};

    // Prefer the largest shared-memory carveout so three 32 KiB blocks can reside on one SM;
    // this trades some L1 capacity for the occupancy targeted by this kernel.
    C10_CUDA_CHECK(cudaFuncSetAttribute(forward_d64, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));

    dim3 block{THREAD_BLOCK_SZ};
    forward_d64<<<grid, block, TileParams::total_bytes>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace flash_attention
