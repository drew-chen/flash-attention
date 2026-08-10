#pragma once

#include <concepts>
#include <cstdint>

#include <torch/extension.h>

namespace flash_attention::detail {

// Performs integer division of a by b, rounding up then casts to T.
template <std::integral T> constexpr auto ceil_div(std::integral auto a, std::integral auto b) {
    return static_cast<T>((a / b) + static_cast<decltype(a / b)>(a % b != 0));
}

inline void check_cuda_contiguous(const torch::Tensor &tensor,
                                  const char *name,
                                  c10::ScalarType scalar_type,
                                  const char *scalar_type_name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " must be ", scalar_type_name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_dim(const torch::Tensor &tensor, const char *name, int dim) {
    TORCH_CHECK(tensor.dim() == dim, name, " must be ", dim, "D");
}

inline void check_cuda_contiguous_dim(const torch::Tensor &tensor,
                                      const char *name,
                                      int dim,
                                      c10::ScalarType scalar_type,
                                      const char *scalar_type_name) {
    check_cuda_contiguous(tensor, name, scalar_type, scalar_type_name);
    check_dim(tensor, name, dim);
}

inline void check_cuda_float32_contiguous_dim(const torch::Tensor &tensor,
                                              const char *name,
                                              int dim) {
    check_cuda_contiguous_dim(tensor, name, dim, torch::kFloat32, "float32");
}

inline void validate_attention_inputs_with_type(const torch::Tensor &q,
                                                const torch::Tensor &k,
                                                const torch::Tensor &v,
                                                c10::ScalarType scalar_type,
                                                const char *scalar_type_name) {
    check_cuda_contiguous_dim(q, "q", 4, scalar_type, scalar_type_name);
    check_cuda_contiguous_dim(k, "k", 4, scalar_type, scalar_type_name);
    check_cuda_contiguous_dim(v, "v", 4, scalar_type, scalar_type_name);
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must be on the same CUDA device");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.size(0) == k.size(0), "q and k must have the same batch size");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k must have the same number of heads");
    TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head dimension");
    TORCH_CHECK(k.size(2) > 0, "K/V sequence length N must be greater than zero");
}

inline void validate_attention_inputs(const torch::Tensor &q,
                                      const torch::Tensor &k,
                                      const torch::Tensor &v) {
    validate_attention_inputs_with_type(q, k, v, torch::kFloat32, "float32");
}

inline void validate_attention_inputs_float16(const torch::Tensor &q,
                                              const torch::Tensor &k,
                                              const torch::Tensor &v) {
    validate_attention_inputs_with_type(q, k, v, torch::kFloat16, "float16");
}

template <typename scalar_t>
inline bool are_aligned(const torch::Tensor &q,
                        const torch::Tensor &k,
                        const torch::Tensor &v,
                        std::uintptr_t alignment) {
    const auto is_aligned = [alignment](const torch::Tensor &tensor) {
        return reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr<scalar_t>()) % alignment == 0;
    };
    return is_aligned(q) && is_aligned(k) && is_aligned(v);
}

template <typename scalar_t>
inline void check_head_dim_and_alignment(const torch::Tensor &q,
                                         const torch::Tensor &k,
                                         const torch::Tensor &v,
                                         const char *kernel_name,
                                         int head_dim,
                                         std::uintptr_t alignment) {
    TORCH_CHECK(q.size(3) == head_dim, kernel_name, " supports only head dimension D=", head_dim);
    TORCH_CHECK(are_aligned<scalar_t>(q, k, v, alignment), kernel_name,
                " requires q, k, and v base addresses to be ", alignment, "-byte aligned");
}

// Returns true when the output has no elements and no CUDA kernel launch is needed.
inline bool output_is_empty(const torch::Tensor &q) {
    const bool B_is_zero = q.size(0) == 0;
    const bool H_is_zero = q.size(1) == 0;
    const bool M_is_zero = q.size(2) == 0;
    const bool D_is_zero = q.size(3) == 0;
    return B_is_zero || H_is_zero || M_is_zero || D_is_zero;
}

template <typename scalar_t, typename Launcher>
inline torch::Tensor launch_attention(const torch::Tensor &q,
                                      const torch::Tensor &k,
                                      const torch::Tensor &v,
                                      Launcher launcher) {
    auto out = torch::empty_like(q);

    if (output_is_empty(q)) {
        return out;
    }

    launcher(q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
             v.const_data_ptr<scalar_t>(), out.mutable_data_ptr<scalar_t>(),
             static_cast<int>(q.size(0)), static_cast<int>(q.size(1)), static_cast<int>(q.size(2)),
             static_cast<int>(k.size(2)), static_cast<int>(q.size(3)));

    return out;
}

} // namespace flash_attention::detail
