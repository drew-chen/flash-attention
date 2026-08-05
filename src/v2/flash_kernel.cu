#include "../cuda_utils.h"
#include <c10/cuda/CUDAException.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

/*
V2 simplified FlashAttention-2 forward kernel.

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

Inputs and output are contiguous CUDA float32 tensors.

Algorithm and optimization notes: docs/v2-extended-notes.md
*/

namespace flash_attention {
namespace {

constexpr int THREAD_BLOCK_SZ = 128;
// TODO: Reuse shared-memory buffers more tightly, then calculate B_R and B_C dynamically.
// Data tile size != thread block size
constexpr std::size_t B_r = 64;
constexpr std::size_t B_c = 32;
constexpr std::size_t WARP_SIZE = 32;
static_assert(THREAD_BLOCK_SZ % WARP_SIZE == 0,
              "Thread block size must contain a whole number of warps");
constexpr std::size_t NUM_WARPS = THREAD_BLOCK_SZ / WARP_SIZE;
static_assert(B_r % NUM_WARPS == 0, "Q rows must divide evenly among warps");

enum class ReductionOp : std::uint8_t { SUM, MAX };
enum class RhsAccess : std::uint8_t { ROW, COLUMN };

// Activates all warps in a warp shuffle (all bits 1).
constexpr unsigned FULL_WARP_MASK = ~0U;

// Stores values and offsets into shared-memory tiles
struct TileParams {
    TileParams(const std::size_t M, const std::size_t N, const std::size_t D) {
        TORCH_CHECK(M > 0, "M must be positive");
        TORCH_CHECK(N > 0, "N must be positive");
        TORCH_CHECK(D > 0, "D must be positive");

        int device;
        C10_CUDA_CHECK(cudaGetDevice(&device));

        int max_smem_bytes;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                              device));

        std::size_t offset = 0;

        this->sQ_offset = offset;
        offset += B_r * D;

        this->sK_offset = offset;
        offset += B_c * D;

        this->sV_offset = offset;
        offset += B_c * D;

        this->sScore_offset = offset;
        offset += B_r * B_c;

        this->sM_offset = offset;
        offset += B_r;

        this->sL_offset = offset;
        offset += B_r;

        this->sO_offset = offset;
        offset += B_r * D;

        this->sMnew_offset = offset;
        offset += B_r;

        this->sLrescaled_offset = offset;
        offset += B_r;

        this->total_elements = offset;

        this->total_bytes = offset * sizeof(float);
        TORCH_CHECK(total_bytes <= static_cast<std::size_t>(max_smem_bytes),
                    "Shared-memory layout exceeds device capacity");
    }

    // Offsets to access shared memory buffers
    std::size_t sQ_offset;         // [B_r, D]
    std::size_t sK_offset;         // [B_c, D]
    std::size_t sV_offset;         // [B_c, D]
    std::size_t sScore_offset;     // [B_r, B_c]. Can be re-used for P_ij.
    std::size_t sM_offset;         // [B_r]
    std::size_t sL_offset;         // [B_r]
    std::size_t sO_offset;         // [B_r, D]
    std::size_t sMnew_offset;      // [B_r]
    std::size_t sLrescaled_offset; // [B_r]
    std::size_t total_elements;
    std::size_t total_bytes;
};

struct FlashForwardKernelParams {
    const float *const gQ;          // [B, H, M, D] Global memory query pointer.
    const float *const gK;          // [B, H, N, D] Global memory key pointer.
    const float *const gV;          // [B, H, N, D] Global memory value pointer.
    float *const gO;                // [B, H, M, D] Global memory output pointer.
    const std::size_t B;            // Batch size.
    const std::size_t H;            // Number of heads.
    const std::size_t M;            // Query sequence length.
    const std::size_t N;            // K/V sequence length.
    const std::size_t D;            // Head dimension.
    const TileParams tile{M, N, D}; // Shared-memory tile parameters.
};

/**
 * Returns the flat offset for a row tile within the batch and head selected by
 * blockIdx.x and blockIdx.y. Pass blockIdx.z for a Q/O tile and the K/V-loop
 * index for a K/V tile.
 */
__device__ __forceinline__ std::size_t get_tile_offset(const std::size_t num_heads,
                                                       const std::size_t total_rows,
                                                       const std::size_t total_cols,
                                                       const std::size_t tile_rows,
                                                       const std::size_t tile_idx) {
    const std::size_t batch_idx = static_cast<std::size_t>(blockIdx.x);
    const std::size_t head_idx = static_cast<std::size_t>(blockIdx.y);
    const std::size_t batch_head_offset =
        (batch_idx * num_heads + head_idx) * total_rows * total_cols;
    return batch_head_offset + tile_idx * tile_rows * total_cols;
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
    const std::size_t g_tile_offset =
        get_tile_offset(H, total_rows, total_cols, tile_rows, tile_i);

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
 * Inverse of load_shared_tile.
 *
 * gmem_ptr has shape [B, H, total_rows, total_cols]. blockIdx selects [B, H],
 * and tile_i selects the [tile_rows, total_cols] segment of [total_rows, total_cols].
 * The source shared-memory tile must be synchronized before this call.
 * The caller must synchronize afterward before reusing or overwriting it.
 * Setting total_cols == 1 saves a 1D tile (row-major).
 */
__device__ void save_shared_tile(const float *const smem_ptr,
                                 float *const gmem_ptr,
                                 const std::size_t H,
                                 const std::size_t total_rows,
                                 const std::size_t total_cols,
                                 const std::size_t tile_rows,
                                 const std::size_t tile_i) {
    const std::size_t s_tile_size = tile_rows * total_cols;
    const std::size_t g_tile_start = tile_i * tile_rows;
    const std::size_t g_tile_offset =
        get_tile_offset(H, total_rows, total_cols, tile_rows, tile_i);

    for (std::size_t flattened_i = static_cast<std::size_t>(threadIdx.x); flattened_i < s_tile_size;
         flattened_i += blockDim.x) {
        const std::size_t s_row = flattened_i / total_cols;
        // Note: tile col idx == global co idx
        const std::size_t col = flattened_i % total_cols;

        // Convert tile row into global row
        const std::size_t g_row = g_tile_start + s_row;
        if (g_row < total_rows) {
            const std::size_t g_idx = g_tile_offset + (s_row * total_cols) + col;
            gmem_ptr[g_idx] = smem_ptr[flattened_i];
        }
    }
}

/**
 * Loads Q_i [B_r, D] from Q [B, H, M, D].
 */
__device__ float *load_Q_i(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           std::size_t tile_i) {
    float *const sQ_i = smem_ptr + p.tile.sQ_offset;
    load_shared_tile(sQ_i, p.gQ, p.H, p.M, p.D, B_r, tile_i);
    return sQ_i;
}

/**
 * Loads K_j [B_c, D] from K [B, H, N, D].
 */
__device__ float *load_K_j(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           std::size_t tile_j) {
    float *const sK_j = smem_ptr + p.tile.sK_offset;
    load_shared_tile(sK_j, p.gK, p.H, p.N, p.D, B_c, tile_j);
    return sK_j;
}

/**
 * Loads V_j [B_c, D] from V [B, H, N, D].
 */
__device__ float *load_V_j(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           std::size_t tile_j) {
    float *const sV_j = smem_ptr + p.tile.sV_offset;
    load_shared_tile(sV_j, p.gV, p.H, p.N, p.D, B_c, tile_j);
    return sV_j;
}

/**
 * Calculates one scaled dot product between a row of X and a row or column of Y serially.
 * X is [X_rows, C_dim]. Y is [Y_cols, C_dim] for ROW access and
 * [C_dim, Y_cols] for COLUMN access.
 */
template <RhsAccess Access>
__device__ float scaled_dotprod(const float *const X,
                                const float *const Y,
                                const std::size_t row,
                                const std::size_t col,
                                const std::size_t Y_cols,
                                const std::size_t C_dim,
                                const float scale = 1.0F) {
    float dot_prod = 0.0F;
    for (std::size_t c = 0; c < C_dim; c++) {
        std::size_t Y_idx;
        if constexpr (Access == RhsAccess::ROW) {
            Y_idx = (col * C_dim) + c;
        } else {
            Y_idx = (c * Y_cols) + col;
        }
        dot_prod += X[(row * C_dim) + c] * Y[Y_idx];
    }
    return dot_prod * scale;
}

/**
 * Calculates Q_i @ K_j^T / sqrt(D) and returns the [B_r, B_c] shared-memory result.
 */
__device__ float *score_ij(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           const float *const sQ_i,
                           const float *const sK_j,
                           const std::size_t tile_j) {
    float *const sScore_ij = smem_ptr + p.tile.sScore_offset;
    // Note: rsqrt (reciprocol) not sqrt
    const float scale = rsqrt(static_cast<float>(p.D));
    // Zero padding on QKV is valid for the dot product, but padded scores must be -infinity.
    // -inf is the identity for the later rowmax operation and we also need
    // out of bounds scores to not affect the softmax's denominator's exp
    // sum, so we require exp(out of bounds val) == 0, so out of bounds val = -inf.
    constexpr float padding = -std::numeric_limits<float>::infinity();
    for (std::size_t flattened_score_idx = threadIdx.x; flattened_score_idx < B_r * B_c;
         flattened_score_idx += blockDim.x) {
        const std::size_t row = flattened_score_idx / B_c;
        const std::size_t col = flattened_score_idx % B_c;
        const std::size_t global_col = (tile_j * B_c) + col;

        sScore_ij[flattened_score_idx] =
            global_col < p.N ? scaled_dotprod<RhsAccess::ROW>(sQ_i, sK_j, row, col, B_c, p.D, scale)
                             : padding;
    }
    return sScore_ij;
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
 * Calculates the row-wise max of m_i [B_r] and S_ij [B_r, B_c].
 * Returns the sMnew_i [B_r] shared-memory result.
 */
__device__ float *warp_rowmax(const FlashForwardKernelParams &p,
                              float *const smem_ptr,
                              const float *const sM_i,
                              const float *const sScore_ij) {
    static_assert(B_c == 32, "Warp reduction assumes one warp fits one row");
    float *const sMnew_i = smem_ptr + p.tile.sMnew_offset;
    const std::size_t initial_row = threadIdx.x / 32;
    const std::size_t col = threadIdx.x % 32;
    // We can process this many rows in parallel
    const std::size_t block_num_warps = blockDim.x / 32;
    // Each warp maps to saving the result for one row
    for (std::size_t row = initial_row; row < B_r; row += block_num_warps) {
        float score_rowmax = warp_reduce<ReductionOp::MAX>(sScore_ij[(row * B_c) + col]);
        // Only the first of every warp has access to the warp reduce result
        if (threadIdx.x % 32 == 0) {
            sMnew_i[row] = fmaxf(sM_i[row], score_rowmax);
        }
    }
    return sMnew_i;
}

/**
 * Calculates l_i * e^(m_i - mnew_i) to update L_i's exponent scale.
 * Returns the sLrescaled_i [B_r] shared-memory result.
 */
__device__ float *rescale_L_i(const FlashForwardKernelParams &p,
                              float *const smem_ptr,
                              const float *const sL_i,
                              const float *const sM_i,
                              const float *const sMnew_i) {
    static_assert(B_r <= THREAD_BLOCK_SZ, "Implementation assumes a row fits in the threadblock.");
    float *const sLrescaled_i = smem_ptr + p.tile.sLrescaled_offset;
    const std::size_t tid = threadIdx.x;
    if (tid < B_r) {
        sLrescaled_i[tid] = sL_i[tid] * expf(sM_i[tid] - sMnew_i[tid]);
    }
    return sLrescaled_i;
}

/**
 * Mutates L_i to store rowsum(e^(S_ij - mnew_i)) + rescaled L_i [B_r].
 */
__device__ void add_Lnew_i_contribution(float *const sL_i,
                                        const float *const sScore_ij,
                                        const float *const sLrescaled_i,
                                        const float *const sMnew_i) {
    static_assert(B_c == 32, "Warp reduction assumes one warp fits one row");
    const std::size_t initial_row = threadIdx.x / 32;
    const std::size_t col = threadIdx.x % 32;
    // We can process this many rows in parallel
    const std::size_t block_num_warps = blockDim.x / 32;
    // Each warp maps to saving the result for one row
    for (std::size_t row = initial_row; row < B_r; row += block_num_warps) {
        const float exp_val = expf(sScore_ij[(row * B_c) + col] - sMnew_i[row]);
        float Lnew_i_contribution = warp_reduce<ReductionOp::SUM>(exp_val);

        // Only the first of every warp has access to the warp reduce result
        if (threadIdx.x % 32 == 0) {
            sL_i[row] = Lnew_i_contribution + sLrescaled_i[row];
        }
    }
}

/**
 * Mutates the score [B_r, B_c] to calculate unnormalized probabilities
 * exp(S_ij - mnew_i).
 */
__device__ void compute_unscaled_P_ij(const FlashForwardKernelParams &p,
                                      float *const sScore_ij,
                                      const float *const sMnew_i,
                                      const std::size_t tile_i,
                                      const std::size_t tile_j) {
    for (std::size_t flattened_score_idx = threadIdx.x; flattened_score_idx < B_r * B_c;
         flattened_score_idx += blockDim.x) {

        const std::size_t row = flattened_score_idx / B_c;
        const std::size_t col = flattened_score_idx % B_c;
        const std::size_t global_row = (tile_i * B_r) + row;
        const std::size_t global_col = (tile_j * B_c) + col;
        // Out of bounds rows are 0 since P_ij is later used for matmul.
        // Technically the col bounds check is redundant due to S_ij's -inf padding.
        if (global_row < p.M && global_col < p.N) {
            sScore_ij[(row * B_c) + col] =
                expf(sScore_ij[(row * B_c) + col] - sMnew_i[row]);
        } else {
            sScore_ij[(row * B_c) + col] = 0;
        }
    }
}

/**
 * Rescales the unnormalized running output when the running maximum changes.
 */
__device__ void warp_rescale_O_i(const FlashForwardKernelParams &p,
                                 float *const sO_i,
                                 const float *const sM_i,
                                 const float *const sMnew_i,
                                 const std::size_t tile_i) {
    const std::size_t warp = threadIdx.x / WARP_SIZE;
    const std::size_t lane = threadIdx.x % WARP_SIZE;
    for (std::size_t row = warp; row < B_r; row += NUM_WARPS) {
        const std::size_t global_row = (tile_i * B_r) + row;
        const float scale = expf(sM_i[row] - sMnew_i[row]);
        for (std::size_t col = lane; col < p.D; col += WARP_SIZE) {
            sO_i[row * p.D + col] =
                global_row < p.M ? sO_i[row * p.D + col] * scale : 0.0F;
        }
    }
}

/**
 * Mutates sO_rescaled to have this tile's output contribution P_ij @ V_j + rescaled O_i [B_r, D].
 */
__device__ void warp_accumulate_O_i(const FlashForwardKernelParams &p,
                                    float *const sO_rescaled_i,
                                    const float *const sP_ij,
                                    const float *const sV_j) {
    const std::size_t warp = threadIdx.x / WARP_SIZE;
    const std::size_t lane = threadIdx.x % WARP_SIZE;
    for (std::size_t row = warp; row < B_r; row += NUM_WARPS) {
        for (std::size_t col = lane; col < p.D; col += WARP_SIZE) {
            sO_rescaled_i[(row * p.D) + col] +=
                scaled_dotprod<RhsAccess::COLUMN>(sP_ij, sV_j, row, col, p.D, B_c);
        }
    }
}

/**
 * Calculates one query tile of attention output per block.
 */
__global__ void forward(FlashForwardKernelParams p) {
    extern __shared__ float smem[];

    const std::size_t T_c = detail::ceil_div<std::size_t>(p.N, B_c);
    const std::size_t i = blockIdx.z;

    float *sQ_i = load_Q_i(p, smem, i);
    float *sO_i = smem + p.tile.sO_offset;
    float *sM_i = smem + p.tile.sM_offset;
    float *sL_i = smem + p.tile.sL_offset;
    for (std::size_t idx = threadIdx.x; idx < B_r * p.D; idx += blockDim.x) {
        sO_i[idx] = 0.0F;
    }
    if (threadIdx.x < B_r) {
        sM_i[threadIdx.x] = -std::numeric_limits<float>::infinity();
        sL_i[threadIdx.x] = 0.0F;
    }
    __syncthreads();

    for (std::size_t j = 0; j < T_c; j++) {
        float *sK_j = load_K_j(p, smem, j);
        float *sV_j = load_V_j(p, smem, j);
        __syncthreads();
        float *sScore_ij = score_ij(p, smem, sQ_i, sK_j, j);
        __syncthreads();
        float *sMnew_i = warp_rowmax(p, smem, sM_i, sScore_ij);
        __syncthreads();
        float *sLrescaled_i = rescale_L_i(p, smem, sL_i, sM_i, sMnew_i);
        __syncthreads();
        add_Lnew_i_contribution(sL_i, sScore_ij, sLrescaled_i, sMnew_i);
        compute_unscaled_P_ij(p, sScore_ij, sMnew_i, i, j);
        __syncthreads();
        warp_rescale_O_i(p, sO_i, sM_i, sMnew_i, i);
        __syncthreads();
        warp_accumulate_O_i(p, sO_i, sScore_ij, sV_j);
        if (threadIdx.x < B_r) {
            sM_i[threadIdx.x] = sMnew_i[threadIdx.x];
        }
        __syncthreads();
    }

    const std::size_t warp = threadIdx.x / WARP_SIZE;
    const std::size_t lane = threadIdx.x % WARP_SIZE;
    for (std::size_t row = warp; row < B_r; row += NUM_WARPS) {
        for (std::size_t col = lane; col < p.D; col += WARP_SIZE) {
            sO_i[(row * p.D) + col] /= sL_i[row];
        }
    }
    __syncthreads();
    save_shared_tile(sO_i, p.gO, p.H, p.M, p.D, B_r, i);
}

} // namespace

/**
 * Allocates the running softmax state and launches the tiled attention kernel.
 */
void flash_forward_v2_cuda_launch(const float *gQ,
                                  const float *gK,
                                  const float *gV,
                                  float *gO,
                                  int B,
                                  int H,
                                  int M,
                                  int N,
                                  int D) {

    FlashForwardKernelParams p{
        gQ,
        gK,
        gV,
        gO,
        static_cast<std::size_t>(B),
        static_cast<std::size_t>(H),
        static_cast<std::size_t>(M),
        static_cast<std::size_t>(N),
        static_cast<std::size_t>(D),
    };

    C10_CUDA_CHECK(cudaFuncSetAttribute(forward, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(p.tile.total_bytes)));

    dim3 block{THREAD_BLOCK_SZ};
    const std::size_t query_tiles =
        detail::ceil_div<std::size_t>(static_cast<std::size_t>(M), B_r);
    // One block per (batch, head, Q tile).
    dim3 grid{static_cast<unsigned int>(B),
              static_cast<unsigned int>(H),
              static_cast<unsigned int>(query_tiles)};

    forward<<<grid, block, p.tile.total_bytes>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace flash_attention
