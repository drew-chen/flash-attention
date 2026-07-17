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
- (head_dim) D: head dimension. Size of the per-token vector inside one head.


Assumes self-attention: q, k, v are CUDA float32 contiguous tensors with the same
shape [B, H, N, D], so M = N = seq_len.

1. Standard attention algorithm:

X = QK^T / sqrt(D) (pre-softmax logits)
A = row_softmax(X) (attention-weight matrix)
O = AV  (self-attention output)


2. Single-row, two-pass online-softmax attention:

Algorithm for one row of output O[b, h, k, :], with b, h, and query row k fixed.
This algorithm avoids materializing the full attention matrix, but saves one score row x.

Notes taken from Zihao Ye's "From Online Softmax to FlashAttention".

Derivation of online softmax's recurrence relation is not shown.


Pass 1:

Initialize:
    m_0 = -infinity  # Running maximum of the processed logits.
    l_0 = 0          # Running numerically stable softmax denominator.

for i = 1 to N:
    x_i = dot(Q[k, :], K[i, :]) / sqrt(D)  # Scalar logit for query k and the current key i.
    m_i = max(m_{i-1}, x_i)                # Running maximum for the score row.
    l_i = l_{i-1}*e^(m_{i-1} - m_i)        # Update the running max-shifted softmax normalizer.
        + e^(x_i - m_i)
save x_i values for this score row

Pass 2:

Initialize:
    o_0 = zeros(D)  # Running partial attention-output row vector.

for i = 1 to N:
    a_i = e^(x_i - m_N)/l_N         # Calculate the numerically stable attention weight.
    o_i = o_{i-1} + a_i*V[i, :]     # Accumulate the weighted value row.
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

for i = 1 to N:
    x_i = dot(Q[k, :], K[i, :]) / sqrt(D)           # Scalar logit for query k for each key vector.
    m_i = max(m_{i-1}, x_i)                         # Update the running maximum.
    rescaled_l_{i-1} = l_{i-1}*e^(m_{i-1} - m_i)  # If prev max was the same, do nothing, otherwise, correct it's exponent scale.
    l_i = rescaled_l_{i-1} + e^(x_i - m_i)        # Update attention row's running sum's softmax denominator.



    old_output_contribution = o_{i-1} * l_{i-1} * e^(m_{i-1} - m_i) / l_i

        # Remove prev output's denominator l_{i-1}
        # then correct it's exponent scale and set the newly updated denominator l_i.

    o_i = old_output_contribution + (e^(x_i - m_i)/l_i)*V[i, :]

        # Add this row vector to the running sum output row vector

O[b, h, k, :] = o_N                 # Save row vector output

4. FlashAttention (tiled)

TODO: tile in 2d rather than 1d

Perform flash attention a block b at a time

Initialize:
    m_0 = -infinity  # Running maximum of the processed logits.
    l_0 = 0          # Running numerically stable softmax denominator.
    o_0 = zeros(D)   # Running normalized attention-output row vector.


for i = 1 to ceil(N/b):
                                                       # Calculate one row of block
    xb_i = dot(Q[k, :], K[(i - 1)b:ib, :]) / sqrt(D)   # multiply query vector with b key vectors (key matrix with b rows), yielding col vector size b
    m_i = max(m_{i-1}, *xb_i)                          # Update the running maximum with the largest scalar in xb_i (ie, across entire block).
    rescaled_l_{i-1} = l_{i-1}*e^(m_{i-1} - m_i)       # (Unchanged) If prev max was the same, do nothing, otherwise, correct it's exponent scale.
    l_i = rescaled_l_{i-1} + ∑e^(xb_j - m_i)           # Update attention row's running sum's softmax denominator with terms in this block.



    old_output_contribution = o_{i-1} * l_{i-1} * e^(m_{i-1} - m_i) / l_i

        # (Unchanged) Remove prev output's denominator l_{i-1}.
        # then correct it's exponent scale and set the newly updated denominator l_i.

    o_i = old_output_contribution + ∑(e^(x_j - m_i)/l_i)*V[(i - 1)b + j, :] 

        # Add each row vector in the block to the running sum output row vector

O[b, h, k, :] = o_N                 # Save row vector output

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
