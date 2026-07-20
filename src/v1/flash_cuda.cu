#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>

#include "../cuda_utils.h"
#include "flash"

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
                                                # processed per tile.
    B_r = min(B_c, D)                           # Number of score rows and Q rows
                                                # processed per tile.

    # Full running state, initialized in global memory for every batch and head.
    m[b, h, :] = fill(M, -infinity) # (M): Running maximum for each query row.
    l[b, h, :] = zeros(M)           # (M): Running stable softmax denominator for each query row.
    O[b, h, :, :] = zeros(M, D)     # (M, D): Running normalized attention output.

    T_c = ceil(N / B_c)  # Number of K/V blocks
    T_r = ceil(M / B_r)  # Number of Q blocks


# Keep K/V as the outer loop so each loaded K/V tile is reused across all query tiles.

# Outer loop iterates with j, to stay consistent with flash attention paper

for each K/V block j = 0 to T_c - 1:

    K_j = K[j*B_c: min((j + 1)*B_c, N), :]
    V_j = V[j*B_c: min((j + 1)*B_c, N), :]

        # (B_c x D): Save up to B_c rows of K and V into SRAM.

    for each Q block i = 0 to T_r - 1:
        # -- Load shared memory 2D tile dim (B_r x D) --
        Q_i = Q[i*B_r: min((i + 1)*B_r, M), :]
        O_i = O[b, h, i*B_r: min((i + 1)*B_r, M), :]

            # (B_r x D): Choose up to B_r rows of Q.
            # Load this query tile's running state from global memory.


        m_i = m[b, h, i*B_r: min((i + 1)*B_r, M)]
        l_i = l[b, h, i*B_r: min((i + 1)*B_r, M)]

            # (B_r) Load in per-query running state.


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

            # (B_r): block local rowmax to eventually calculate the running max for a query

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

        old_output_contribution = O_i * rescaled_l_i / lnew_i

            # (B_r x D): For the running O_i matrix, update the scalars' safe
            # softmax factors via scaling so that they are re-calculated with mnew_i.
            # Do this by multiplying by the rescaled old denominator contribution,
            # then dividing by the new denominator lnew_i.


        # -- Save running output for this query --

        Onew_i = old_output_contribution + new_output_contribution

            # (B_r x D): Sum matrices to update tile's running output

        # -- Save running state to global memory --

        m[b, h, i*B_r: min((i + 1)*B_r, M)] = mnew_i
        l[b, h, i*B_r: min((i + 1)*B_r, M)] = lnew_i
        O[b, h, i*B_r: min((i + 1)*B_r, M), :] = Onew_i

            # Save the updated state. O is finalized after the last K/V tile.

        # A future optimization is to store the O tile without the li denominator
        # and perform the division on a second pass on the final O tile, reducing # FLOPs
        # of needing to constantly rescale.



*/

namespace flash_attention {

constexpr int BLOCK_SZ = 128;

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

        // smem is a float array, so the following sizing and ffset params are
        // calculated in terms of elements not bytes
        const std::size_t capacity_elements =
            static_cast<std::size_t>(max_smem_bytes) / sizeof(float);

        // sizes chosen according to FA1 paper
        this->B_c = std::min(detail::ceil_div<std::size_t>(capacity_elements, 4 * D), N);
        this->B_r = std::min(std::min(this->B_c, D), M);

        std::size_t offset = 0;

        this->sQ_offset = offset;
        offset += this->B_r * D;

        this->sK_offset = offset;
        offset += this->B_c * D;

        this->sV_offset = offset;
        offset += this->B_c * D;

        this->sS_offset = offset;
        offset += this->B_r * this->B_c;

        this->sM_offset = offset;
        offset += this->B_r;

        this->sL_offset = offset;
        offset += this->B_r;

        this->sO_offset = offset;
        offset += this->B_r * D;

        this->scratch_0_offset = offset;
        offset += this->B_r;

        this->scratch_1_offset = offset;
        offset += this->B_r;

        this->total_elements = offset;

        this->total_bytes = offset * sizeof(float);
        TORCH_CHECK(total_bytes <= static_cast<std::size_t>(max_smem_bytes),
                    "Shared-memory layout exceeds device capacity");
    }

    std::size_t B_c;
    std::size_t B_r;
    // Offsets to access shared memory buffers
    std::size_t sQ_offset;        // [B_r, D]
    std::size_t sK_offset;        // [B_c, D]
    std::size_t sV_offset;        // [B_c, D]
    std::size_t sS_offset;        // [B_r, B_c]. Can be re-used for P_ij.
    std::size_t sM_offset;        // [B_r]
    std::size_t sL_offset;        // [B_r]
    std::size_t sO_offset;        // [B_r, D]
    std::size_t scratch_0_offset; // [B_r]
    std::size_t scratch_1_offset; // [B_r]
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
 * Cooperatively loads one contiguous row tile from global to shared memory.
 *
 * g_ptr has shape [B, H, total_rows, total_cols]. blockIdx selects [B, H],
 * and tile_i selects [tile_rows, total_cols]. Out-of-range rows use pad.
 * The caller must synchronize before consuming or reusing shared memory.
 */
__device__ void load_shared_tile(float *const s_ptr,
                                 const float *const g_ptr,
                                 const std::size_t H,
                                 const std::size_t total_rows,
                                 const std::size_t total_cols,
                                 const std::size_t tile_rows,
                                 const std::size_t tile_i,
                                 const float pad = 0.0F) {
    const std::size_t s_tile_size = tile_rows * total_cols;
    const std::size_t batch_idx = static_cast<std::size_t>(blockIdx.x);
    const std::size_t head_idx = static_cast<std::size_t>(blockIdx.y);
    const std::size_t batch_head_offset = (batch_idx * H + head_idx) * total_rows * total_cols;
    const std::size_t g_tile_start = tile_i * tile_rows;

    for (std::size_t i = static_cast<std::size_t>(threadIdx.x); i < s_tile_size;
         i += static_cast<std::size_t>(blockDim.x)) {
        const std::size_t s_row = i / total_cols;
        // Note: tile col idx == global co idx
        const std::size_t col = i % total_cols;

        // Convert tile row into global row
        const std::size_t g_row = g_tile_start + s_row;
        if (g_row < total_rows) {
            const std::size_t g_idx = batch_head_offset + (g_row * total_cols) + col;
            s_ptr[i] = g_ptr[g_idx];
        } else {
            s_ptr[i] = pad;
        }
    }
}

// Loads Q_i [B_r, D] from Q [B, H, M, D].
__device__ float *load_Q_i(const FlashForwardKernelParams &p, float *smem_ptr, std::size_t tile_i) {
    float *const smem_tile_ptr = smem_ptr + p.tile.sQ_offset;
    load_shared_tile(smem_tile_ptr, p.gQ, p.H, p.M, p.D, p.tile.B_r, tile_i);
    return smem_tile_ptr;
}

// Loads K_j [B_c, D] from K [B, H, N, D].
__device__ float *load_K_j(const FlashForwardKernelParams &p, float *smem_ptr, std::size_t tile_j) {
    float *const smem_tile_ptr = smem_ptr + p.tile.sK_offset;
    load_shared_tile(smem_tile_ptr, p.gK, p.H, p.N, p.D, p.tile.B_c, tile_j);
    return smem_tile_ptr;
}

// Loads V_j [B_c, D] from V [B, H, N, D].
__device__ float *load_V_j(const FlashForwardKernelParams &p, float *smem_ptr, std::size_t tile_j) {
    float *const smem_tile_ptr = smem_ptr + p.tile.sV_offset;
    load_shared_tile(smem_tile_ptr, p.gV, p.H, p.N, p.D, p.tile.B_c, tile_j);
    return smem_tile_ptr;
}

// Loads m_i [B_r] from m [B, H, M], padding with -infinity.
__device__ float *load_M_i(const FlashForwardKernelParams &p, float *smem_ptr, std::size_t tile_i) {
    float *const smem_tile_ptr = smem_ptr + p.tile.sM_offset;
    load_shared_tile(smem_tile_ptr, p.gM, p.H, p.M, std::size_t{1}, p.tile.B_r, tile_i,
                     -std::numeric_limits<float>::infinity());
    return smem_tile_ptr;
}


// Loads l_i [B_r] from l [B, H, M], padding with zero.
__device__ float *load_L_i(const FlashForwardKernelParams &p, float *smem_ptr, std::size_t tile_i) {
    float *const smem_tile_ptr = smem_ptr + p.tile.sL_offset;
    load_shared_tile(smem_tile_ptr, p.gL, p.H, p.M, std::size_t{1}, p.tile.B_r, tile_i);
    return smem_tile_ptr;
}

// Loads O_i [B_r, D] from O [B, H, M, D].
__device__ float *load_O_i(const FlashForwardKernelParams &p, float *smem_ptr, std::size_t tile_i) {
    float *const smem_tile_ptr = smem_ptr + p.tile.sO_offset;
    load_shared_tile(smem_tile_ptr, p.gO, p.H, p.M, p.D, p.tile.B_r, tile_i);
    return smem_tile_ptr;
}

/**
A block of this kernel calculates one (batch, head) of attention output
by following the pseudocode in algorithm 4 in the notes at the top of the file.
*/
__global__ void forward(FlashForwardKernelParams p) {
    // This must be partitioned into Q_i, K_j, etc using p.tile
    extern __shared__ float smem[];

    const std::size_t T_r = detail::ceil_div<std::size_t>(p.M, p.tile.B_r);
    const std::size_t T_c = detail::ceil_div<std::size_t>(p.N, p.tile.B_c);

    for (std::size_t kv_tile_idx = 0; kv_tile_idx < T_c; kv_tile_idx++) {
        float *sK = load_K_j(p, smem, kv_tile_idx);
        float *sV = load_V_j(p, smem, kv_tile_idx);

        for (std::size_t q_tile_idx = 0; q_tile_idx < T_r; q_tile_idx++) {
            float *sQ = load_Q_i(p, smem, q_tile_idx);
            // TODO: For the first K/V tile, initialize sO = 0, sM = -infinity, and sL = 0
            // instead of loading uninitialized running state from global memory.
            float *sO = load_O_i(p, smem, q_tile_idx);
            float *sM = load_M_i(p, smem, q_tile_idx);
            float *sL = load_L_i(p, smem, q_tile_idx);
            __syncthreads();
            row_s_matmul(p, s_Q, s_K, smem);
        }
    }
}

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

    dim3 block{BLOCK_SZ};
    // Grid for [B, H, N, D] outputs: (D tiles, N tiles, B * H).
    dim3 grid{static_cast<unsigned int>(B), static_cast<unsigned int>(H)};

    forward<<<grid, block, p.tile.total_bytes>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaFree(gM);
    cudaFree(gL);
}

} // namespace flash_attention
