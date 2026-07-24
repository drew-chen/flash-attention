#include "../cuda_utils.h"
#include <c10/cuda/CUDAException.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

/*
In these notes, a tiled flash attention is constructed step-by-step from
normal attention. These notes skip how the reccurence relation used
is derived.

Algorithms:
1. Standard attention
2. Single-row, two-pass online-softmax attention
3. Single-row, single-pass online-softmax attention
4. FlashAttention (tiled)

Algorithm 4 is the one that is implemented.

flash_forward_cuda_launch expects raw pointers for tensors shaped as:

- q: [B, H, N, D]
- k: [B, H, N, D]
- v: [B, H, N, D]

- out: [B, H, N, D]

  For each batch element b and head h, out[b, h] =
  softmax((q[b, h] * k[b, h]^T) / sqrt(D)) * v[b, h].

Dimensions:

- (batch_size) B: batch size. How many independent sequences are processed together.
- (num_heads) H: number of attention heads per sequence.
- (seq_len) N: self-attention sequence length. In general attention, Q has M rows
  and K/V have N rows; this implementation is specialized to M = N = seq_len.
  Can also be thought of as context length.
- (head_dim) D: head dimension. Size of the per-token vector inside one head.


Assumes self-attention: q, k, v are CUDA float32 contiguous tensors with the same
shape [B, H, N, D], so M = N = seq_len.

Indexing convention: tensor indices are zero-based. A loop over N elements uses
i = 0 to N - 1. Recurrence state 0 is the empty-prefix state, so processing
tensor element i advances recurrence state i to state i + 1.

1. Standard attention algorithm:

S = QK^T / sqrt(D) (pre-softmax logits, ie, score)
P = row_softmax(S) (attention probabilities/weight matrix)
O = PV  (self-attention output)


2. Single-row, two-pass online-softmax attention:

Algorithm for one row of output O[b, h, k, :], with b, h, and query row k fixed.
This algorithm avoids materializing the full attention matrix, but saves one score row x.

Notes taken from Zihao Ye's "From Online Softmax to FlashAttention".

Derivation of online softmax's recurrence relation is not shown.


Pass 1:

Initialize:
    m_0 = -infinity  # Running maximum of the processed logits.
    l_0 = 0          # Running numerically stable softmax denominator.

for i = 0 to N - 1:
    x_i = dot(Q[k, :], K[i, :]) / sqrt(D)  # Scalar logit for query k and the current key i.
    m_{i+1} = max(m_i, x_i)                # Running maximum for the score row.
    l_{i+1} = l_i*e^(m_i - m_{i+1})        # Update the running max-shifted softmax normalizer.
            + e^(x_i - m_{i+1})
save x_i values for this score row

Pass 2:

Initialize:
    o_0 = zeros(D)  # Running partial attention-output row vector.

for i = 0 to N - 1:
    a_i = e^(x_i - m_N)/l_N         # Calculate the numerically stable attention weight.

    o_{i+1} = o_i + a_i*V[i, :]     # Accumulate the weighted value row.
                                    # Over N iterations, this is equivalent to row vector
                                    # a * matrix V, since the full attention-weight row
                                    # a is dotted with each column of V;
                                    # equivalently, each row V[i, :] is scaled by a_i
                                    # before the rows are summed.

    O[b, h, k, :] = o_N             # Save row vector output


3. Single-row, single-pass online-softmax attention

Using a flash attention recurrence relationship yields:

Initialize:
    m_0 = -infinity  # Running maximum of the processed logits.
    l_0 = 0          # Running numerically stable softmax denominator.
    o_0 = zeros(D)   # Running normalized attention-output row vector.

for i = 0 to N - 1:
    x_i = dot(Q[k, :], K[i, :]) / sqrt(D)           # Scalar logit for query k for each key vector.
    m_{i+1} = max(m_i, x_i)                         # Update the running maximum.
    rescaled_l_i = l_i*e^(m_i - m_{i+1})            # If prev max was the same, do nothing,
                                                    # otherwise, correct it's exponent scale.

    l_{i+1} = rescaled_l_i + e^(x_i - m_{i+1})      # Update attention row's running
                                                    # sum's softmax denominator.

    old_output_contribution = o_i * l_i * e^(m_i - m_{i+1}) / l_{i+1}

        # Remove prev output's denominator l_i
        # then correct it's exponent scale and set the newly updated denominator l_{i+1}.

    o_{i+1} = old_output_contribution + (e^(x_i - m_{i+1})/l_{i+1})*V[i, :]

        # Add this row vector to the running sum output row vector

O[b, h, k, :] = o_N                 # Save row vector output

4. FlashAttention (tiled)

Unlike the previous examples, this is for the entire output rather than a row.
Furthermore, the notation is adjusted from the paper to more closely align with
CUDA. Furthermore, the subscript annotation is for indexing into block-level
state rather than for state transitions like above.

Divide Q, K, and V along the sequence dimension and load into shared memory 2D tiles
of (B_r x D), (B_c x D), and (B_c x D) respectively, where (B_r, B_c) are the
dimensions of the score tile calculated by processing the queries, keys, and values
in the tile.

Ex: Q (M x D) is composed of T_r tiles, labelled Q_i (B_r x D) by stacking along the sequence dim.
Q = [
    ---Q_0---
    ---Q_1---
    ...
    --- Q_{T_r - 1}
]

Through algebra similar to how online-softmax is performed, attention can be performed
one tile at a time with a running output. Only the current Q and K/V tiles and running
state slices need to be simultaneously loaded into SRAM.


This avoids us from needing to handle ops with the entire N / M elements loaded per row
thus we don't need to materialize the (M x N) attention score matrix (self-attention has N=M).


Intuition on running state:

After iterating over the entire sequence / all blocks, the running output is same
as the naive attention output. The running state m_i, l_i, and O_i lives in global
memory between tile updates. The current tiles are loaded into SRAM, where m_i and l_i
have one scalar per query and O_i has one D-element row per query.

The m_i and l_i state must be stored as vectors, not single variables, because we do not
actually calculate the result for one query's attention before moving onto the next; we
incrementally build running state for a block of queries.

The outer loop iterates over K/V tiles, thus for a fixed K/V tile, we iterate over all
query tiles to calculate the running output. This is an inversion of the
per-query output POV but is mathematically equivalent and done to keep the heavier
data movement of the K/V tiles on the outer loop rather than inner loop.


Initialize:
    B_c = floor(SRAM_capacity_elements/(4*D))   # Number of score cols and K/V rows
                                                # processed per data tile.
    B_r = min(B_c, D)                           # Number of score rows and Q rows
                                                # processed per data tile.

    T_c = ceil(N / B_c)  # Number of K/V blocks
    T_r = ceil(M / B_r)  # Number of Q blocks

    # Allocate a shared-memory workspace for the current Q, K/V, score, output,
    # and recurrence-state tiles. Allocate global-memory backing for m and l.
    # O serves as both the output and the backing for the running output state.

    m_i = fill(B_r, -infinity) # (B_r): Shared running maximum state for a Q tile.
    l_i = zeros(B_r)           # (B_r): Shared running softmax denominator state.
    O_i = zeros(B_r, D)        # (B_r x D): Shared running attention output state.


# Keep K/V as the outer loop so each loaded K/V tile is reused across all query tiles.

# Outer loop iterates with j, to stay consistent with flash attention paper

for each K/V block j = 0 to T_c - 1:

    K_j = K[j*B_c: min((j + 1)*B_c, N), :]
    V_j = V[j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Save up to B_c rows of K and V into SRAM.

    for each Q block i = 0 to T_r - 1:
        # -- Load shared memory 2D tile dim (B_r x D) --
        Q_i = Q[i*B_r: min((i + 1)*B_r, M), :]

            # (B_r x D): Choose up to B_r rows of Q.

        # Load this Q tile's current m_i, l_i, and O_i into shared memory, using
        # the initialized defaults for its first update and globally saved state later.


        # -- Calculate the block-local scores using SRAM --

        S_ij = Q_i @ K_j^T / sqrt(D)

            # (B_r x B_c): S_ij is a score matrix, calculating the score for each Q
            # row in the block by dotting with each K row in the block.
            # A row of S_ij is only the score of a query against B_c rows of
            # K rather than all N key rows.
            # Note 1: The tranpose is conceptual and done by changing indexing,
            #   rather than a separate device function call.
            # Note 2: The rest of the work here is building a running output
            #   from the segment S_ij which eventually yields the same result
            #   using S.

        # -- Setup block-local variables for block-local softmax --

        # (this is in parallel for each query in the block).

        mlocal_ij = rowmax(S_ij)

            # (B_r x 1): block local rowmax to eventually calculate the running max for a query

        mnew_i = max(m_i, mlocal_ij)

            # (B_r): Update the vector so each element corresponds to
            # the running max score for each query in the block
            # By the end of the entire algorithm, m_i will contain
            # rowmax(S), which is equivalent to what
            # would've happened had we fully materialized S and performed
            # rowsoftmax.

        rescaled_l_i = l_i*e^(m_i - mnew_i)

            # (B_r): This represents the old block local rowsoftmax's denominator
            # prior to the contribution of this the scores using this query block.
            # Conceptually, we are updating the a query's rowsoftmax denominator
            # by using the scores of new K/V rows. This is confusing because
            # we have the inner loop iterate over queries for memory performance,
            # while I phrase these concepts from a fixed query perspective.
            # If prev max was the same, do nothing, otherwise, correct its exponent scale
            # to use the updated max.

        new_l_i_contribution = rowsum(e^(S_ij - mnew_i))

            # (B_r): Broadcast apply exp operations and subtract each row of S_ij with it's
            # corresponding rowmax then calculate the row sum.
            # This is this tile's contribution to the softmax denominator for
            # each query.

        lnew_i = rescaled_l_i + new_l_i_contribution

            # (B_r): Update running denominator. By the end of the algorithm,
            # l_i is the denominator for each row of rowsoftmax(S).

        # -- Calculate this query block's contribution to the running output --

        P_ij = exp(S_ij - mnew_i) / lnew_i

            # (B_r x B_c) Calculate softmax row-wise to scale tile scores into probabilities
            # using mnew_i and lnew_i, broadcast across their corresponding rows.
            # For each row of S_ij, subtract with corrresponding row max then divide
            # by corresponding denominator. Each row of S_ij again represents
            # the score from one query when scorred against the K/V rows in this tile.
            # Output is a 2D matrix the same size as S_ij.

        new_output_contribution = P_ij  @  V_j

            # (B_r x D): Conceptually, this does this:
            # For every row of P_ij, ie for every query in the sequence tile
            #   dot product with every value in the tile and sum them,
            #   to effectively perform a weighted sum of probabilities.
            # Finally, by parallelizing across B_c rows of V_j, we finish the calc
            # for this tile.
            # This basically means we attend each query with every key and value
            # within our tile.


        # -- Adjust running output for this query with the updated running max --

        rescaled_O_i = O_i * rescaled_l_i / lnew_i

            # (B_r x D): For the running O_i matrix, update the scalars' safe
            # softmax factors via scaling so that they are re-calculated with mnew_i.
            # Do this by multiplying by the rescaled old denominator contribution,
            # then dividing by the new denominator lnew_i.


        # -- Save running output for this query --

        Onew_i = rescaled_O_i + new_output_contribution

            # (B_r x D): Sum matrices to update tile's running output

        # -- Save running state to global memory --

        m[b, h, i*B_r: min((i + 1)*B_r, M)] = mnew_i
        l[b, h, i*B_r: min((i + 1)*B_r, M)] = lnew_i
        O[b, h, i*B_r: min((i + 1)*B_r, M), :] = Onew_i

            # Save the updated state. O is finalized after the last K/V tile.

        # A future optimization is to store the O tile without the li denominator
        # and perform the division on a second pass on the final O tile, reducing # FLOPs
        # of needing to constantly rescale.

---

In the following impl, variables are named similar to the FA1 paper
and s prefix means shared mem ptr, g prefix means global mem ptr,
and finally _i means a subscript of i.
*/

namespace flash_attention {
namespace {

constexpr int THREAD_BLOCK_SZ = 128;
// TODO: Reuse SRAM buffers more tightly, then calculate B_R and B_C dynamically.
// Data tile size != thread block size
constexpr std::size_t B_r = 32;
constexpr std::size_t B_c = 32;

enum class ReductionOp : std::uint8_t { SUM, MAX };
enum class RhsAccess : std::uint8_t { ROW, COLUMN };

// Activates all warps in a warp shuffle (all bits 1).
constexpr unsigned FULL_WARP_MASK = ~0U;

// Stores values and offsets into SRAM tiles
struct TileParams {
    TileParams(const std::size_t M, const std::size_t N, const std::size_t D) {
        TORCH_CHECK(M > 0, "M must be positive");
        TORCH_CHECK(N > 0, "N must be positive");
        TORCH_CHECK(D > 0, "D must be positive");

        int device;
        C10_CUDA_CHECK(cudaGetDevice(&device));

        int max_smem_bytes;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_bytes, cudaDevAttrMaxSharedMemoryPerBlock,
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
    float *const gM;                // [B, H, M] Global memory running-max pointer.
    float *const gL;                // [B, H, M] Global memory normalizer pointer.
    const std::size_t B;            // Batch size.
    const std::size_t H;            // Number of heads.
    const std::size_t M;            // Query sequence length.
    const std::size_t N;            // K/V sequence length.
    const std::size_t D;            // Head dimension.
    const TileParams tile{M, N, D}; // Shared-memory tile parameters.
};

/**
 * Returns the flat offset for the batch and head selected by blockIdx.
 */
__device__ __forceinline__ std::size_t get_batch_head_offset(const std::size_t num_heads,
                                                             const std::size_t total_rows,
                                                             const std::size_t total_cols) {
    const std::size_t batch_idx = static_cast<std::size_t>(blockIdx.x);
    const std::size_t head_idx = static_cast<std::size_t>(blockIdx.y);
    return (batch_idx * num_heads + head_idx) * total_rows * total_cols;
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
    const std::size_t batch_head_offset = get_batch_head_offset(H, total_rows, total_cols);

    for (std::size_t flattened_i = static_cast<std::size_t>(threadIdx.x); flattened_i < s_tile_size;
         flattened_i += blockDim.x) {
        const std::size_t s_row = flattened_i / total_cols;
        // Note: tile col idx == global co idx
        const std::size_t col = flattened_i % total_cols;

        // Convert tile row into global row
        const std::size_t g_row = g_tile_start + s_row;
        if (g_row < total_rows) {
            const std::size_t g_idx = batch_head_offset + (g_row * total_cols) + col;
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
    const std::size_t batch_head_offset = get_batch_head_offset(H, total_rows, total_cols);

    for (std::size_t flattened_i = static_cast<std::size_t>(threadIdx.x); flattened_i < s_tile_size;
         flattened_i += blockDim.x) {
        const std::size_t s_row = flattened_i / total_cols;
        // Note: tile col idx == global co idx
        const std::size_t col = flattened_i % total_cols;

        // Convert tile row into global row
        const std::size_t g_row = g_tile_start + s_row;
        if (g_row < total_rows) {
            const std::size_t g_idx = batch_head_offset + (g_row * total_cols) + col;
            gmem_ptr[g_idx] = smem_ptr[flattened_i];
        }
    }
}

/**
 * Cooperatively fills a contiguous shared-memory tile with a scalar value.
 */
__device__ __forceinline__ void set_smem(float *const smem_ptr,
                                         const std::size_t num_elements,
                                         const float value) {
    for (std::size_t i = static_cast<std::size_t>(threadIdx.x); i < num_elements;
         i += static_cast<std::size_t>(blockDim.x)) {
        smem_ptr[i] = value;
    }
}

/**
 * Initializes row vector m_i [B_r] in shared memory to -infinity.
 */
__device__ float *init_M_i_neginf(const FlashForwardKernelParams &p, float *const smem_ptr) {
    float *const sM_i = smem_ptr + p.tile.sM_offset;
    set_smem(sM_i, B_r, -std::numeric_limits<float>::infinity());
    return sM_i;
}

/**
 * Initializes row vector l_i [B_r] in shared memory to zero.
 */
__device__ float *init_L_i_zero(const FlashForwardKernelParams &p, float *const smem_ptr) {
    float *const sL_i = smem_ptr + p.tile.sL_offset;
    set_smem(sL_i, B_r, 0.0F);
    return sL_i;
}

/**
 * Initializes O_i [B_r, D] in shared memory to zero.
 */
__device__ float *init_O_i_zero(const FlashForwardKernelParams &p, float *const smem_ptr) {
    float *const sO_i = smem_ptr + p.tile.sO_offset;
    set_smem(sO_i, B_r * p.D, 0.0F);
    return sO_i;
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
 * Loads row vector m_i [B_r] from m [B, H, M], padding with -infinity.
 */
__device__ float *load_M_i(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           std::size_t tile_i) {
    float *const sM_i = smem_ptr + p.tile.sM_offset;
    // total_cols == 1 since m_i is 1D
    load_shared_tile(sM_i, p.gM, p.H, p.M, std::size_t{1}, B_r, tile_i,
                     -std::numeric_limits<float>::infinity());
    return sM_i;
}

/**
 * Loads row vector l_i [B_r] from l [B, H, M], padding with zero.
 */
__device__ float *load_L_i(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           std::size_t tile_i) {
    float *const sL_i = smem_ptr + p.tile.sL_offset;
    // total_cols == 1 since l_i is 1D
    load_shared_tile(sL_i, p.gL, p.H, p.M, std::size_t{1}, B_r, tile_i);
    return sL_i;
}

/**
 * Loads O_i [B_r, D] from O [B, H, M, D].
 */
__device__ float *load_O_i(const FlashForwardKernelParams &p,
                           float *const smem_ptr,
                           std::size_t tile_i) {
    float *const sO_i = smem_ptr + p.tile.sO_offset;
    load_shared_tile(sO_i, p.gO, p.H, p.M, p.D, B_r, tile_i);
    return sO_i;
}

/**
 * Saves updated m_i [B_r] from shared memory to m [B, H, M].
 */
__device__ void save_M_i(const FlashForwardKernelParams &p,
                         const float *const smem_ptr,
                         std::size_t tile_i) {
    const float *const sM_i = smem_ptr + p.tile.sMnew_offset;
    save_shared_tile(sM_i, p.gM, p.H, p.M, std::size_t{1}, B_r, tile_i);
}

/**
 * Saves updated l_i [B_r] from shared memory to l [B, H, M].
 */
__device__ void save_L_i(const FlashForwardKernelParams &p,
                         const float *const smem_ptr,
                         std::size_t tile_i) {
    const float *const sL_i = smem_ptr + p.tile.sL_offset;
    save_shared_tile(sL_i, p.gL, p.H, p.M, std::size_t{1}, B_r, tile_i);
}

/**
 * Saves updated O_i [B_r, D] from shared memory to O [B, H, M, D].
 */
__device__ void save_O_i(const FlashForwardKernelParams &p,
                         const float *const smem_ptr,
                         std::size_t tile_i) {
    const float *const sO_i = smem_ptr + p.tile.sO_offset;
    save_shared_tile(sO_i, p.gO, p.H, p.M, p.D, B_r, tile_i);
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
__device__ float *rowmax(const FlashForwardKernelParams &p,
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
 * Mutates the score [B_r, B_c] to calculate the running probabilites
 * softmax exp(S_ij - mnew_i) / lnew_i.
 */
__device__ void softmax(const FlashForwardKernelParams &p,
                        float *const sScore_ij,
                        const float *const sLnew_i,
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
                expf(sScore_ij[(row * B_c) + col] - sMnew_i[row]) / sLnew_i[row];
        } else {
            sScore_ij[(row * B_c) + col] = 0;
        }
    }
}

/**
 * Mutates running output sO_i [B_r, D] to have the updated softmax scaling
 * O_i * rescaled_l_i / lnew_i.
 */
__device__ void rescale_O_i(const FlashForwardKernelParams &p,
                            float *const sO_i,
                            const float *const sLrescaled_i,
                            const float *const sLnew_i,
                            const std::size_t tile_i) {
    for (std::size_t flattened_output_idx = threadIdx.x; flattened_output_idx < B_r * p.D;
         flattened_output_idx += blockDim.x) {

        const std::size_t row = flattened_output_idx / p.D;
        const std::size_t global_row = (tile_i * B_r) + row;
        if (global_row < p.M) {
            sO_i[flattened_output_idx] =
                sO_i[flattened_output_idx] * sLrescaled_i[row] / sLnew_i[row];
        } else {
            sO_i[flattened_output_idx] = 0.0F;
        }
    }
}

/**
 * Mutates sO_rescaled to have this tile's output contribution P_ij @ V_j + rescaled O_i [B_r, D].
 */
__device__ void add_Onew_i_contribution(const FlashForwardKernelParams &p,
                                        float *const sO_rescaled_i,
                                        const float *const sP_ij,
                                        const float *const sV_j) {
    for (std::size_t flattened_output_idx = threadIdx.x; flattened_output_idx < B_r * p.D;
         flattened_output_idx += blockDim.x) {
        const std::size_t row = flattened_output_idx / p.D;
        const std::size_t col = flattened_output_idx % p.D;
        sO_rescaled_i[flattened_output_idx] +=
            scaled_dotprod<RhsAccess::COLUMN>(sP_ij, sV_j, row, col, p.D, B_c);
    }
}

/**
 * Calculates one (batch, head) of attention output per block using algorithm 4.
 */
__global__ void forward(FlashForwardKernelParams p) {
    // This must be partitioned into Q_i, K_j, etc using p.tile
    extern __shared__ float smem[];

    const std::size_t T_r = detail::ceil_div<std::size_t>(p.M, B_r);
    const std::size_t T_c = detail::ceil_div<std::size_t>(p.N, B_c);

    for (std::size_t j = 0; j < T_c; j++) {
        float *sK_j = load_K_j(p, smem, j);
        float *sV_j = load_V_j(p, smem, j);

        for (std::size_t i = 0; i < T_r; i++) {
            float *sQ_i = load_Q_i(p, smem, i);
            float *sO_i;
            float *sM_i;
            float *sL_i;

            if (j == 0) {
                sO_i = init_O_i_zero(p, smem);
                sM_i = init_M_i_neginf(p, smem);
                sL_i = init_L_i_zero(p, smem);
            } else {
                sO_i = load_O_i(p, smem, i);
                sM_i = load_M_i(p, smem, i);
                sL_i = load_L_i(p, smem, i);
            }
            __syncthreads();
            float *sScore_ij = score_ij(p, smem, sQ_i, sK_j, j);
            __syncthreads();
            float *sMnew_i = rowmax(p, smem, sM_i, sScore_ij);
            __syncthreads();
            float *sLrescaled_i = rescale_L_i(p, smem, sL_i, sM_i, sMnew_i);
            __syncthreads();
            // sL_i is mutated to contain sLnew_i
            add_Lnew_i_contribution(sL_i, sScore_ij, sLrescaled_i, sMnew_i);
            __syncthreads();
            // Use sL_i as sLnew_i and mutate S_ij to contain P_ij
            softmax(p, sScore_ij, sL_i, sMnew_i, i, j);
            __syncthreads();
            float *sP_ij = sScore_ij;
            // sO_i is mutated to contain rescaled_O_i
            rescale_O_i(p, sO_i, sLrescaled_i, sL_i, i);
            __syncthreads();
            add_Onew_i_contribution(p, sO_i, sP_ij, sV_j);
            __syncthreads();
            save_M_i(p, smem, i);
            save_L_i(p, smem, i);
            save_O_i(p, smem, i);
            __syncthreads();
        }
    }
}

} // namespace

/**
 * Allocates the running softmax state and launches the tiled attention kernel.
 */
void flash_forward_v1_cuda_launch(const float *gQ,
                                  const float *gK,
                                  const float *gV,
                                  float *gO,
                                  int B,
                                  int H,
                                  int M,
                                  int N,
                                  int D) {

    // Q, K, and V inputs and the O output are already allocated in global memory.
    // O also serves as the persistent running output state.
    // S_ij is temporary on-chip storage and is reused in place as P_ij.
    // Therefore, only the full m and l arrays [B, H, M] need new global allocations.
    // Their current tiles m_i and l_i are loaded on chip and written back between
    // K/V-tile iterations because all query-row state cannot fit on chip at once.

    float *gM;
    float *gL;
    const std::size_t state_elements =
        static_cast<std::size_t>(B) * static_cast<std::size_t>(H) * static_cast<std::size_t>(M);
    cudaMalloc(&gM, state_elements * sizeof(float));
    cudaMalloc(&gL, state_elements * sizeof(float));

    FlashForwardKernelParams p{
        gQ,
        gK,
        gV,
        gO,
        gM,
        gL,
        static_cast<std::size_t>(B),
        static_cast<std::size_t>(H),
        static_cast<std::size_t>(M),
        static_cast<std::size_t>(N),
        static_cast<std::size_t>(D),
    };

    dim3 block{THREAD_BLOCK_SZ};
    // Grid for [B, H, N, D] outputs: (D tiles, N tiles, B * H).
    dim3 grid{static_cast<unsigned int>(B), static_cast<unsigned int>(H)};

    forward<<<grid, block, p.tile.total_bytes>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaFree(gM);
    cudaFree(gL);
}

} // namespace flash_attention
