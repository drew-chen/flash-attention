#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "flash_cuda_helpers.cuh"
#include "softmax.cuh"
#include <cmath>
#include <cstddef>

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
