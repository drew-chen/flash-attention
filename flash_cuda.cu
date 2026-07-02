#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

/*

flash_forward_cuda_launch expects raw pointers for tensors shaped as:

- q: [B, H, N, D]
- k: [B, H, N, D]
- v: [B, H, N, D]

- out: [B, H, N, D]

  For each batch element b and head h, out[b, h] =
  softmax((q[b, h] * k[b, h]^T) / sqrt(D)) * v[b, h].

Dimensions:

- B: batch size. How many independent sequences are processed together.
- H: number of attention heads per sequence.
- N: sequence length. How many token positions each head attends over.
- D: head dimension. Size of the per-token vector inside one head.
*/
__global__ void flash_forward_kernel(const float *q, const float *k, const float *v, float *out,
                                     int batch_size, int num_heads, int seq_len, int head_dim) {
    (void)q;
    (void)k;
    (void)v;
    (void)out;
    (void)batch_size;
    (void)num_heads;
    (void)seq_len;
    (void)head_dim;
    // Kernel skeleton only. Real FlashAttention work will go here.
}

constexpr unsigned int ceil_div(unsigned int a, unsigned int b) { return a / b + (a % b != 0); }

/*
Assumes q, k, v are CUDA float32 contiguous tensors with shape [B, H, N, D].
*/
void flash_forward_cuda_launch(const float *q, const float *k, const float *v, float *out,
                               int batch_size, int num_heads, int seq_len, int head_dim) {
    // x,y fast dims of head dim and seq_len, followed by flattened batch and heads
    dim3 block{16, 16, 1};
    dim3 grid{ceil_div(static_cast<unsigned int>(head_dim), block.x),
              ceil_div(static_cast<unsigned int>(seq_len), block.y),
              static_cast<unsigned int>(batch_size * num_heads)};
    flash_forward_kernel<<<grid, block>>>(q, k, v, out, batch_size, num_heads, seq_len, head_dim);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
