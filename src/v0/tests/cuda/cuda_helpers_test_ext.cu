#include <c10/cuda/CUDAException.h>

#include "../../flash_cuda_helpers.cuh"
#include "../../softmax.cuh"

int ceil_div_host(int a, int b) { return ceil_div(a, b); }

void transpose_cuda_launch(const float *input,
                           float *output,
                           int batch_size,
                           int num_heads,
                           int input_width,
                           int input_height) {
    dim3 block{TILE_SZ, TILE_SZ, 1};
    dim3 grid{static_cast<unsigned int>(ceil_div(input_width, TILE_SZ)),
              static_cast<unsigned int>(ceil_div(input_height, TILE_SZ)),
              static_cast<unsigned int>(batch_size * num_heads)};
    transpose<<<grid, block>>>(input, output, num_heads, input_height, input_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scale_cuda_launch(float *data,
                       int batch_size,
                       int num_heads,
                       int input_width,
                       int input_height,
                       float factor) {
    dim3 block{TILE_SZ, TILE_SZ, 1};
    dim3 grid{static_cast<unsigned int>(ceil_div(input_width, TILE_SZ)),
              static_cast<unsigned int>(ceil_div(input_height, TILE_SZ)),
              static_cast<unsigned int>(batch_size * num_heads)};
    scale<<<grid, block>>>(data, num_heads, input_width, input_height, factor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_test_cuda_launch(const float *input,
                              float *output,
                              int batch_size,
                              int num_heads,
                              int rows,
                              int head_dim) {
    softmax_rows_launch(input, output, batch_size, num_heads, rows, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void matmul_cuda_launch(const float *left,
                        const float *right,
                        float *output,
                        int batch_size,
                        int num_heads,
                        int left_height,
                        int shared_dim,
                        int right_width) {
    dim3 block{TILE_SZ, TILE_SZ, 1};
    dim3 grid{static_cast<unsigned int>(ceil_div(right_width, TILE_SZ)),
              static_cast<unsigned int>(ceil_div(left_height, TILE_SZ)),
              static_cast<unsigned int>(batch_size * num_heads)};
    matmul<<<grid, block>>>(left, right, output, num_heads, left_height, shared_dim, right_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
