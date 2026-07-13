#pragma once

#include <torch/extension.h>

inline void check_cuda_float32_contiguous(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_dim(const torch::Tensor &tensor, const char *name, int dim) {
    TORCH_CHECK(tensor.dim() == dim, name, " must be ", dim, "D");
}

inline void check_cuda_float32_contiguous_dim(const torch::Tensor &tensor, const char *name,
                                              int dim) {
    check_cuda_float32_contiguous(tensor, name);
    check_dim(tensor, name, dim);
}
