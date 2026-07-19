#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

struct FlashForwardKernelParams {
    const float *q;
    const float *k;
    const float *v;
    float *const out;
    // global memory ptr allocated for a draft v0 implementation
    float *const intermediate_score;
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
};


/*
In these notes, a tiled flash attention is constructed step-by-step from
normal attention. These notes skip how the reccurence relation used
is derived.

A very naive attention calculation (not flash attention).
Is performed by many small kernel launches.

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
                                    # Over N iterations, this is equivalent to row vector a * matrix V,
                                    # since the full attention-weight row a is dotted with each column of V;
                                    # equivalently, each row V[i, :] is scaled by a_i before the rows are summed.
O[b, h, k, :] = o_N                 # Save row vector output


3. Single-row scalar FlashAttention recurrence:

Using a flash attention recurrence relationship yields:

Initialize:
    m_0 = -infinity  # Running maximum of the processed logits.
    l_0 = 0          # Running numerically stable softmax denominator.
    o_0 = zeros(D)   # Running normalized attention-output row vector.

for i = 0 to N - 1:
    x_i = dot(Q[k, :], K[i, :]) / sqrt(D)           # Scalar logit for query k for each key vector.
    m_{i+1} = max(m_i, x_i)                         # Update the running maximum.
    rescaled_l_i = l_i*e^(m_i - m_{i+1})            # If prev max was the same, do nothing, otherwise, correct it's exponent scale.
    l_{i+1} = rescaled_l_i + e^(x_i - m_{i+1})      # Update attention row's running sum's softmax denominator.

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
of (B_M x D), (B_N x D), and (B_N x D) respectively.

Ex: Q (M x D) is composed of T_r tiles, labelled Q_i (B_M x D) by stacking along the sequence dim.
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
    B_N = floor(SRAM_capacity_elements/(4*D)) # Number of score cols and K/V rows processed per tile.
    B_M = min(B_N, D)                         # Number of score rows and Q rows processed per tile.

    # Full running state, initialized in global memory for every batch and head.
    m[b, h, :] = fill(M, -infinity) # (M): Running maximum for each query row.
    l[b, h, :] = zeros(M)           # (M): Running stable softmax denominator for each query row.
    O[b, h, :, :] = zeros(M, D)     # (M, D): Running normalized attention output.

    T_c = ceil(N / B_N)  # Number of K/V blocks
    T_r = ceil(M / B_M)  # Number of Q blocks


# Keep K/V as the outer loop so each loaded K/V tile is reused across all query tiles.

# Outer loop iterates with j, to stay consistent with flash attention paper

for each K/V block j = 0 to T_c - 1:

    K_j = K[j*B_N: min((j + 1)*B_N, N), :]
    V_j = V[j*B_N: min((j + 1)*B_N, N), :]

        # (B_N x D): Save up to B_N rows of K and V into SRAM.

    for each Q block i = 0 to T_r - 1:
        # -- Load shared memory 2D tile dim (B_M x D) --
        Q_i = Q[i*B_M: min((i + 1)*B_M, M), :]
        m_i = m[b, h, i*B_M: min((i + 1)*B_M, M)]
        l_i = l[b, h, i*B_M: min((i + 1)*B_M, M)]
        O_i = O[b, h, i*B_M: min((i + 1)*B_M, M), :]

            # (B_M x D): Choose up to B_M rows of Q.
            # Load this query tile's running state from global memory.


        # -- Calculate the block-local scores using SRAM --

        S_ij = Q_i @ K_j^T / sqrt(D)

            # (B_M x B_N): S_ij is a score matrix, calculating the score for each Q
            # row in the block by dotting with each K row in the block.
            # A row of S_ij is only the score of a query against B_N rows of
            # K rather than all N key rows.
            # Note 1: The tranpose is conceptual and done by changing indexing,
            #   rather than a separate device function call.
            # Note 2: The rest of the work here is building a running output
            #   from the segment S_ij which eventually yields the same result
            #   using S.

        # -- Setup block-local variables for block-local softmax --

        # (this is in parallel for each query in the block).

        mlocal_ij = rowmax(S_ij)

            # (B_M): block local rowmax to eventually calculate the running max for a query

        mnew_i = max(m_i, mlocal_ij)

            # (B_M): Update the vector so each element corresponds to
            # the running max score for each query in the block
            # By the end of the entire algorithm, m_i will contain
            # rowmax(S), which is equivalent to what
            # would've happened had we fully materialized S and performed
            # rowsoftmax.

        rescaled_l_i = l_i*e^(m_i - mnew_i)

            # (B_M): This represents the old block local rowsoftmax's denominator
            # prior to the contribution of this the scores using this query block.
            # Conceptually, we are updating the a query's rowsoftmax denominator
            # by using the scores of new K/V rows. This is confusing because
            # we have the inner loop iterate over queries for memory performance,
            # while I phrase these concepts from a fixed query perspective.
            # If prev max was the same, do nothing, otherwise, correct its exponent scale
            # to use the updated max.

        new_l_i_contribution = rowsum(e^(S_ij - mnew_i))

            # (B_M): Broadcast apply exp operations and subtract each row of S_ij with it's corresponding
            # rowmax then calculate the row sum.
            # This is this tile's contribution to the softmax denominator for each query.

        lnew_i = rescaled_l_i + new_l_i_contribution

            # (B_M): Update running denominator. By the end of the algorithm,
            # l_i is the denominator for each row of rowsoftmax(S).

        # -- Calculate this query block's contribution to the running output --

        P_ij = exp(S_ij - mnew_i) / lnew_i

            # (B_M x B_N) Calculate softmax row-wise to scale tile scores into probabilities
            # using mnew_i and lnew_i, broadcast across their corresponding rows.
            # For each row of S_ij, subtract with corrresponding row max then divide
            # by corresponding denominator. Each row of S_ij again represents
            # the score from one query when scorred against the K/V rows in this tile.
            # Output is a 2D matrix the same size as S_ij.

        new_output_contribution = P_ij  @  V_j

            # (B_M x D): Conceptually, this does this:
            # For every row of P_ij, ie for every query in the sequence tile
            #   dot product with every value in the tile and sum them,
            #   to effectively perform a weighted sum of probabilities.
            # Finally, by parallelizing across B_N rows of V_j, we finish the calc
            # for this tile.
            # This basically means we attend each query with every key and value
            # within our tile.


        # -- Adjust running output for this query with the updated running max --

        old_output_contribution = O_i * rescaled_l_i / lnew_i

            # (B_M x D): For the running O_i matrix, update the scalars' safe
            # softmax factors via scaling so that they are re-calculated with mnew_i.
            # Do this by multiplying by the rescaled old denominator contribution,
            # then dividing by the new denominator lnew_i.


        # -- Save running output for this query --

        Onew_i = old_output_contribution + new_output_contribution

            # (B_M x D): Sum matrices to update tile's running output

        # -- Save running state to global memory --

        m[b, h, i*B_M: min((i + 1)*B_M, M)] = mnew_i
        l[b, h, i*B_M: min((i + 1)*B_M, M)] = lnew_i
        O[b, h, i*B_M: min((i + 1)*B_M, M), :] = Onew_i

            # Save the updated state. O is finalized after the last K/V tile.

        # A future optimization is to store the O tile without the li denominator
        # and perform the division on a second pass on the final O tile, reducing # FLOPs
        # of needing to constantly rescale.



*/
void naive_forward_v0_cuda_launch(const float *q,
                                  const float *k,
                                  const float *v,
                                  float *out,
                                  int batch_size,
                                  int num_heads,
                                  int seq_len,
                                  int head_dim) {
    float *intermediate_score;
    // [B, H]
    const std::size_t batch_head_count =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(num_heads);
    // [B, H, N, D] for q, k, v, out, and k_transpose (whose per-head view is [D, N]).
    const std::size_t batch_head_tensor_size =
        batch_head_count * static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(head_dim);
    // [B, H, M, N] for QK^T scores and row-wise softmax probabilities;
    // self-attention specializes this to [B, H, N, N].
    const std::size_t batch_head_score_size =
        batch_head_count * static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(seq_len);
    cudaMalloc(&intermediate_score, batch_head_score_size * sizeof(float));
    float *scaled_score;
    cudaMalloc(&scaled_score, batch_head_score_size * sizeof(float));
    float *k_transpose;
    cudaMalloc(&k_transpose, batch_head_tensor_size * sizeof(float));

    dim3 block{TILE_SZ, TILE_SZ, 1};
    // Grid for [B, H, N, D] outputs: (D tiles, N tiles, B * H).
    dim3 qkv_grid{static_cast<unsigned int>(ceil_div(head_dim, TILE_SZ)),
                  static_cast<unsigned int>(ceil_div(seq_len, TILE_SZ)),
                  static_cast<unsigned int>(batch_head_count)};
    // Grid for [B, H, N, N] score/probability outputs: (N tiles, N tiles, B * H).
    dim3 score_grid{static_cast<unsigned int>(ceil_div(seq_len, TILE_SZ)),
                    static_cast<unsigned int>(ceil_div(seq_len, TILE_SZ)),
                    static_cast<unsigned int>(batch_head_count)};

    transpose<<<qkv_grid, block>>>(k, k_transpose, num_heads, seq_len, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    matmul<<<score_grid, block>>>(q, k_transpose, intermediate_score, num_heads, seq_len, head_dim,
                                  seq_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaMemcpy(scaled_score, intermediate_score,
                              batch_head_score_size * sizeof(float), cudaMemcpyDeviceToDevice));
    scale<<<score_grid, block>>>(scaled_score, num_heads, seq_len, seq_len,
                                 1.0F / static_cast<float>(std::sqrt(head_dim)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    softmax_rows_launch(scaled_score, intermediate_score, batch_size, num_heads, seq_len, seq_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    matmul<<<qkv_grid, block>>>(intermediate_score, v, out, num_heads, seq_len, seq_len, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cudaFree(k_transpose);
    cudaFree(scaled_score);
    cudaFree(intermediate_score);
}
