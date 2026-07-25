#include "../cuda_utils.h"
#include <c10/cuda/CUDAException.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

/*
These notes describe my simplified implementation of Tri Dao's flash attention 2.

flash_forward_v3_cuda_launch expects raw pointers for tensors shaped as:

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

V3 implementation:

This implements the warp partitioning described by FA2.

V3 starts as a working copy of V2 and follows FA2's warp ownership more closely,
without tensor cores or an asynchronous pipeline. The main goal is to fit at
least two blocks per SM on the RTX 4080.

1. We chose D = 64 for convenience so the warp-local state can use simple
   compile-time arrays. Use 256 threads (eight warps) per block; with B_r = 64,
   each warp owns eight complete query/output rows.

2. Keep the running maximum, denominator, and unnormalized output in the owning
   warp's registers. Lanes cooperatively calculate QK and PV for those rows, so
   no inter-warp reduction is needed.

3. Keep Q in shared memory for the full K/V loop, reuse one shared buffer for
   K then V, and keep the unnormalized P tile in shared memory for the PV step.

4. Preserve V2's query-tile grid parallelism, delayed output normalization,
   cross-attention support, and partial-tile bounds handling.

Algorithm:

Initialize:
    B_c = 32    # Number of score cols and K/V rows
                # processed per data tile (can be tuned).
    B_r = 64    # Number of score rows and Q rows
                # processed per data tile (can be tuned).

    T_c = ceil(N / B_c)  # Number of K/V blocks
    T_r = ceil(M / B_r)  # Number of Q blocks

    # This represents conceptual state for one Q tile. Each warp owns the
    # entries corresponding to one or more complete query rows.

    wM_i_replicated = fill(B_r, -infinity) # (B_r): Running maximum state.
    wL_i_replicated = zeros(B_r)           # (B_r): Running softmax denominator.
    wO_i_unnormalized = zeros(B_r, D)      # (B_r x D): Running output numerator.



# Each thread block selects rows of queries and calculates rows of O output.

# -- Load shared memory 2D tile dim (B_r x D) --

s_Q_i = Q[b, h, i*B_r: min((i + 1)*B_r, M), :]


for each K/V block j = 0 to T_c - 1:

    s_K_j = K[b, h, j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Save up to B_c K rows into SRAM.


    wS_ij = s_Q_i @ s_K_j^T / sqrt(D)

        # Each warp owns complete score rows. Each lane stores one score
        # column for its rows, so the complete score tile is register-local.


    wM_i_new = max(wM_i_replicated, rowmax(wS_ij))

        # (B_r). Each warp calculates the entries for its owned rows, then
        # replicates the result across its lanes with warp shuffles.


    sP_ij_unnormalized = exp(wS_ij - wM_i_new)

        # (B_r x B_c): Calculate the attention-weight numerator using the
        # updated max, but do not apply the softmax denominator. V3 stores
        # this intermediate in shared memory.


    wM_i_rescale_factor = exp(wM_i_replicated - wM_i_new)

        # (B_r): By multiplying by this constant, the exp scale of the previous
        iteration is updated to this iteration's max.

    wL_i_replicated = wM_i_rescale_factor * wL_i_replicated
                        + rowsum(sP_ij_unnormalized)

        # (B_r): Update running denominator. By the end of the algorithm,
        # wL_i_replicated is the denominator for each score row.

    # -- Compute running output --

    # sV_j reuses the shared storage previously occupied by sK_j.
    s_V_j = V[b, h, j*B_c: min((j + 1)*B_c, N), :]

    wO_i_unnormalized = wO_i_unnormalized * wM_i_rescale_factor
                          + sP_ij_unnormalized @ s_V_j

        # (B_r x D): Rescale softmax numerator for running output then
        # add this tile to the running output.

    wM_i_replicated = wM_i_new

# Apply the softmax denominator once after processing every K/V tile.
O[b, h, i*B_r: min((i + 1)*B_r, M), :] = wO_i_unnormalized / wL_i_replicated

---

The code below implements the D=64 specialization described above. Other head
dimensions are redirected to V2 by the C++ binding.

Variables are named similar to the FA1 paper: the s prefix means shared mem
ptr, the g prefix means global mem ptr, and _i means a subscript of i.
*/

namespace flash_attention {
namespace {

constexpr std::size_t B_r = 64;
constexpr std::size_t B_c = 32;
constexpr std::size_t WARP_SIZE = 32;

constexpr std::size_t HEAD_DIM = 64;
constexpr int THREAD_BLOCK_SZ = 256;
constexpr std::size_t NUM_WARPS = THREAD_BLOCK_SZ / WARP_SIZE;
constexpr std::size_t ROWS_PER_WARP = B_r / NUM_WARPS;
static_assert(HEAD_DIM == 64, "V3 is specialized for D=64");
static_assert(THREAD_BLOCK_SZ % WARP_SIZE == 0,
              "V3 thread block size must contain a whole number of warps");
static_assert(B_r % NUM_WARPS == 0, "V3 Q rows must divide evenly among warps");

enum class ReductionOp : std::uint8_t { SUM, MAX };

// Activates all warps in a warp shuffle (all bits 1).
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
__device__ std::size_t get_tile_offset(const std::size_t num_heads,
                                       const std::size_t total_rows,
                                       const std::size_t total_cols,
                                       const std::size_t tile_rows,
                                       const std::size_t tile_idx) {
    const std::size_t batch_idx = static_cast<std::size_t>(blockIdx.x);
    const std::size_t head_idx = static_cast<std::size_t>(blockIdx.y);
    const std::size_t batch_head_offset =
        ((batch_idx * num_heads) + head_idx) * total_rows * total_cols;
    return batch_head_offset + (tile_idx * tile_rows * total_cols);
}

/**
 * Uses threadblock to load one contiguous data tile from global to shared memory.
 *
 * gmem_ptr has shape [B, H, total_rows, total_cols]. blockIdx selects [B, H],
 * and tile_i selects the [tile_rows, total_cols] segment of [total_rows, total_cols]
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
        // Note: tile col idx == global co idx
        const std::size_t col = flattened_i % total_cols;

        // Convert tile row into global row
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
template <ReductionOp Op> __device__ float warp_allreduce(float initial_value) {
    return __shfl_sync(FULL_WARP_MASK, warp_reduce<Op>(initial_value), 0);
}

/**
 * Calculates this lane's score column for the rows owned by its warp.
 */
__device__ void score_ij(const float *const sQ_i,
                         const float *const sK_j,
                         const std::size_t row_start,
                         const std::size_t lane,
                         const bool key_valid,
                         float (&wS_ij)[ROWS_PER_WARP]) {
    if (!key_valid) {
        return;
    }

    for (std::size_t k = 0; k < HEAD_DIM; ++k) {
        const float key = sK_j[(lane * HEAD_DIM) + k];
        for (std::size_t owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
            const std::size_t row = row_start + owned_row;
            wS_ij[owned_row] += sQ_i[(row * HEAD_DIM) + k] * key;
        }
    }
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
                                  float (&wS_ij)[ROWS_PER_WARP],
                                  float (&wM_i_replicated)[ROWS_PER_WARP],
                                  float (&wL_i_replicated)[ROWS_PER_WARP],
                                  float (&wO_i_left_unnormalized)[ROWS_PER_WARP],
                                  float (&wO_i_right_unnormalized)[ROWS_PER_WARP]) {
    for (std::size_t owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
        const std::size_t row = row_start + owned_row;
        const bool query_valid = query_start + row < M;
        wS_ij[owned_row] = query_valid && key_valid ? wS_ij[owned_row] * 0.125F
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
__device__ void accumulate_O_i(const float *const sP_ij_unnormalized,
                               const float *const sV_j,
                               const std::size_t row_start,
                               const std::size_t lane,
                               float (&wO_i_left_unnormalized)[ROWS_PER_WARP],
                               float (&wO_i_right_unnormalized)[ROWS_PER_WARP]) {
    for (std::size_t key_row = 0; key_row < B_c; ++key_row) {
        const float value_left = sV_j[(key_row * HEAD_DIM) + lane];
        const float value_right = sV_j[(key_row * HEAD_DIM) + lane + WARP_SIZE];
        for (std::size_t owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
            const std::size_t row = row_start + owned_row;
            const float p_value = sP_ij_unnormalized[(row * B_c) + key_row];
            wO_i_left_unnormalized[owned_row] += p_value * value_left;
            wO_i_right_unnormalized[owned_row] += p_value * value_right;
        }
    }
}

/**
 * Normalizes and stores this lane's output columns.
 */
__device__ void normalize_and_save_O_i(const FlashForwardKernelParams &p,
                                       const std::size_t query_start,
                                       const std::size_t row_start,
                                       const std::size_t lane,
                                       const float (&wL_i_replicated)[ROWS_PER_WARP],
                                       const float (&wO_i_left_unnormalized)[ROWS_PER_WARP],
                                       const float (&wO_i_right_unnormalized)[ROWS_PER_WARP]) {
    const std::size_t batch_head =
        (static_cast<std::size_t>(blockIdx.x) * p.H) + static_cast<std::size_t>(blockIdx.y);
    const std::size_t output_head_offset = batch_head * p.M * HEAD_DIM;
    for (std::size_t owned_row = 0; owned_row < ROWS_PER_WARP; ++owned_row) {
        const std::size_t global_row = query_start + row_start + owned_row;
        if (global_row < p.M) {
            const std::size_t output_row_offset = output_head_offset + (global_row * HEAD_DIM);
            p.gO[output_row_offset + lane] =
                wO_i_left_unnormalized[owned_row] / wL_i_replicated[owned_row];
            p.gO[output_row_offset + lane + WARP_SIZE] =
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

    const std::size_t warp = static_cast<std::size_t>(threadIdx.x) / WARP_SIZE;
    const std::size_t lane = static_cast<std::size_t>(threadIdx.x) % WARP_SIZE;
    // Warps split Q/output rows. Their lanes split the K/V work, so every lane
    // loops over the rows owned by its warp instead of owning one complete row.
    const std::size_t row_start = warp * ROWS_PER_WARP;
    const std::size_t query_tile = static_cast<std::size_t>(blockIdx.z);
    const std::size_t query_start = query_tile * B_r;
    const std::size_t key_tiles = detail::ceil_div<std::size_t>(p.N, B_c);

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

    for (std::size_t row = 0; row < ROWS_PER_WARP; ++row) {
        wM_i_replicated[row] = -std::numeric_limits<float>::infinity();
        wL_i_replicated[row] = 0.0F;
    }

    float *const sQ_i = load_Q_i(p, smem, query_tile);

    for (std::size_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
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
