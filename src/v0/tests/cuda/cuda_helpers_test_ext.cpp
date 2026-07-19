#include <torch/extension.h>

#include "../../../cuda_utils.h"

void transpose_cuda_launch(const float *input,
                           float *output,
                           int batch_size,
                           int num_heads,
                           int input_width,
                           int input_height);
void scale_cuda_launch(float *data,
                       int batch_size,
                       int num_heads,
                       int input_width,
                       int input_height,
                       float factor);
void softmax_test_cuda_launch(const float *input,
                              float *output,
                              int batch_size,
                              int num_heads,
                              int rows,
                              int head_dim);
void matmul_cuda_launch(const float *left,
                        const float *right,
                        float *output,
                        int batch_size,
                        int num_heads,
                        int left_height,
                        int shared_dim,
                        int right_width);
torch::Tensor transpose_cuda(const torch::Tensor &input) {
    flash_attention::detail::check_cuda_float32_contiguous_dim(input, "input", 4);

    const auto batch_size = static_cast<int>(input.size(0));
    const auto num_heads = static_cast<int>(input.size(1));
    const auto input_height = static_cast<int>(input.size(2));
    const auto input_width = static_cast<int>(input.size(3));
    auto output =
        torch::empty({input.size(0), input.size(1), input.size(3), input.size(2)}, input.options());
    transpose_cuda_launch(input.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
                          batch_size, num_heads, input_width, input_height);
    return output;
}

torch::Tensor scale_cuda(const torch::Tensor &input, double factor) {
    flash_attention::detail::check_cuda_float32_contiguous_dim(input, "input", 4);

    const auto batch_size = static_cast<int>(input.size(0));
    const auto num_heads = static_cast<int>(input.size(1));
    const auto input_height = static_cast<int>(input.size(2));
    const auto input_width = static_cast<int>(input.size(3));
    auto output = input.clone();
    scale_cuda_launch(output.mutable_data_ptr<float>(), batch_size, num_heads, input_width,
                      input_height, static_cast<float>(factor));
    return output;
}

torch::Tensor softmax_cuda(const torch::Tensor &input) {
    flash_attention::detail::check_cuda_float32_contiguous_dim(input, "input", 4);

    const auto batch_size = static_cast<int>(input.size(0));
    const auto num_heads = static_cast<int>(input.size(1));
    const auto rows = static_cast<int>(input.size(2));
    const auto head_dim = static_cast<int>(input.size(3));
    auto output = torch::zeros_like(input);
    softmax_test_cuda_launch(input.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
                             batch_size, num_heads, rows, head_dim);
    return output;
}

torch::Tensor matmul_cuda(const torch::Tensor &left, const torch::Tensor &right) {
    flash_attention::detail::check_cuda_float32_contiguous_dim(left, "left", 4);
    flash_attention::detail::check_cuda_float32_contiguous_dim(right, "right", 4);
    TORCH_CHECK(left.size(0) == right.size(0), "left and right batch sizes must match");
    TORCH_CHECK(left.size(1) == right.size(1), "left and right head counts must match");
    TORCH_CHECK(left.size(3) == right.size(2), "left width must equal right height");

    const auto batch_size = static_cast<int>(left.size(0));
    const auto num_heads = static_cast<int>(left.size(1));
    const auto left_height = static_cast<int>(left.size(2));
    const auto shared_dim = static_cast<int>(left.size(3));
    const auto right_width = static_cast<int>(right.size(3));
    auto output =
        torch::zeros({left.size(0), left.size(1), left.size(2), right.size(3)}, left.options());
    matmul_cuda_launch(left.const_data_ptr<float>(), right.const_data_ptr<float>(),
                       output.mutable_data_ptr<float>(), batch_size, num_heads, left_height,
                       shared_dim, right_width);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_cuda", &transpose_cuda, "Test wrapper for the tiled CUDA transpose helper");
    m.def("scale_cuda", &scale_cuda, "Test wrapper for the CUDA scale helper");
    m.def("softmax_cuda", &softmax_cuda, "Test wrapper for the CUDA softmax helper");
    m.def("matmul_cuda", &matmul_cuda, "Test wrapper for the CUDA matmul helper");
}
