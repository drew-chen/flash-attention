#pragma once

#include <concepts>

#include <torch/extension.h>

namespace flash_attention::detail {

// Performs integer division of a by b, rounding up.
template<std::integral T, std::integral U>
constexpr auto ceil_div(T a, U b) {
    return (a / b) + static_cast<decltype(a / b)>(a % b != 0);
}

inline void check_cuda_float32_contiguous(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_dim(const torch::Tensor &tensor, const char *name, int dim) {
    TORCH_CHECK(tensor.dim() == dim, name, " must be ", dim, "D");
}

inline void check_cuda_float32_contiguous_dim(const torch::Tensor &tensor,
                                              const char *name,
                                              int dim) {
    check_cuda_float32_contiguous(tensor, name);
    check_dim(tensor, name, dim);
}

}  // namespace flash_attention::detail
