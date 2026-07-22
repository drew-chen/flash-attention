#include <c10/cuda/CUDAException.h>

#include "../../flash_cuda_helpers.cuh"
#include "../../softmax.cuh"

void transpose_cuda_launch(const float *input,
                           float *output,
                           int batch_size,
                           int num_heads,
                           int input_width,
                           int input_height) {
    dim3 block{flash_attention::detail::TILE_SZ, flash_attention::detail::TILE_SZ, 1};
    dim3 grid{static_cast<unsigned int>(flash_attention::detail::ceil_div<int>(
                  input_width, flash_attention::detail::TILE_SZ)),
              static_cast<unsigned int>(flash_attention::detail::ceil_div<int>(
                  input_height, flash_attention::detail::TILE_SZ)),
              static_cast<unsigned int>(batch_size * num_heads)};
    flash_attention::detail::transpose<<<grid, block>>>(input, output, num_heads, input_height,
                                                         input_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scale_cuda_launch(float *data,
                       int batch_size,
                       int num_heads,
                       int input_width,
                       int input_height,
                       float factor) {
    dim3 block{flash_attention::detail::TILE_SZ, flash_attention::detail::TILE_SZ, 1};
    dim3 grid{static_cast<unsigned int>(flash_attention::detail::ceil_div<int>(
                  input_width, flash_attention::detail::TILE_SZ)),
              static_cast<unsigned int>(flash_attention::detail::ceil_div<int>(
                  input_height, flash_attention::detail::TILE_SZ)),
              static_cast<unsigned int>(batch_size * num_heads)};
    flash_attention::detail::scale<<<grid, block>>>(data, num_heads, input_width, input_height,
                                                     factor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_test_cuda_launch(const float *input,
                              float *output,
                              int batch_size,
                              int num_heads,
                              int rows,
                              int head_dim) {
    flash_attention::detail::softmax_rows_launch(input, output, batch_size, num_heads, rows,
                                                  head_dim);
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
    dim3 block{flash_attention::detail::TILE_SZ, flash_attention::detail::TILE_SZ, 1};
    dim3 grid{static_cast<unsigned int>(flash_attention::detail::ceil_div<int>(
                  right_width, flash_attention::detail::TILE_SZ)),
              static_cast<unsigned int>(flash_attention::detail::ceil_div<int>(
                  left_height, flash_attention::detail::TILE_SZ)),
              static_cast<unsigned int>(batch_size * num_heads)};
    flash_attention::detail::matmul<<<grid, block>>>(left, right, output, num_heads, left_height,
                                                      shared_dim, right_width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
