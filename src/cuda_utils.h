#pragma once

#include <concepts>

#include <torch/extension.h>

namespace flash_attention::detail {

// Performs integer division of a by b, rounding up then casts to T.
template<std::integral T>
constexpr auto ceil_div(std::integral auto a, std::integral auto b) {
    return static_cast<T>((a / b) + static_cast<decltype(a / b)>(a % b != 0));
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

inline void validate_attention_inputs(const torch::Tensor &q,
                                      const torch::Tensor &k,
                                      const torch::Tensor &v) {
    check_cuda_float32_contiguous_dim(q, "q", 4);
    check_cuda_float32_contiguous_dim(k, "k", 4);
    check_cuda_float32_contiguous_dim(v, "v", 4);
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must be on the same CUDA device");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.size(0) == k.size(0), "q and k must have the same batch size");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k must have the same number of heads");
    TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head dimension");
    TORCH_CHECK(k.size(2) > 0, "K/V sequence length N must be greater than zero");
}

// Returns true when the output has no elements and no CUDA kernel launch is needed.
inline bool output_is_empty(const torch::Tensor &q) {
    const bool B_is_zero = q.size(0) == 0;
    const bool H_is_zero = q.size(1) == 0;
    const bool M_is_zero = q.size(2) == 0;
    const bool D_is_zero = q.size(3) == 0;
    return B_is_zero || H_is_zero || M_is_zero || D_is_zero;
}

}  // namespace flash_attention::detail
