#include <torch/extension.h>

#include "../cuda_utils.h"
#include <cstdint>

namespace flash_attention {

void flash_forward_v5_cuda_launch(const c10::Half *q,
                                  const c10::Half *k,
                                  const c10::Half *v,
                                  c10::Half *out,
                                  int batch_size,
                                  int num_heads,
                                  int query_seq_len,
                                  int kv_seq_len,
                                  int head_dim);
torch::Tensor flash_forward_v5_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v);

void check_cuda_float16_contiguous_dim(const torch::Tensor &tensor, const char *name, int dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    detail::check_dim(tensor, name, dim);
}

void validate_attention_inputs_float16(const torch::Tensor &q,
                                       const torch::Tensor &k,
                                       const torch::Tensor &v) {
    check_cuda_float16_contiguous_dim(q, "q", 4);
    check_cuda_float16_contiguous_dim(k, "k", 4);
    check_cuda_float16_contiguous_dim(v, "v", 4);
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must be on the same CUDA device");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.size(0) == k.size(0), "q and k must have the same batch size");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k must have the same number of heads");
    TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head dimension");
    TORCH_CHECK(k.size(2) > 0, "K/V sequence length N must be greater than zero");
}

bool is_vector_load_aligned(const torch::Tensor &tensor) {
    // The kernel loads eight FP16 elements in each 16-byte transaction.
    constexpr std::uintptr_t VECTOR_LOAD_ALIGNMENT = 16;
    return reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr<c10::Half>()) %
               VECTOR_LOAD_ALIGNMENT ==
           0;
}

bool can_use_v5_vector_loads(const torch::Tensor &q,
                             const torch::Tensor &k,
                             const torch::Tensor &v) {
    return is_vector_load_aligned(q) && is_vector_load_aligned(k) && is_vector_load_aligned(v);
}

void check_v5_kernel_requirements(const torch::Tensor &q,
                                  const torch::Tensor &k,
                                  const torch::Tensor &v) {
    TORCH_CHECK(q.size(3) == 64, "V5 supports only head dimension D=64");
    TORCH_CHECK(can_use_v5_vector_loads(q, k, v),
                "V5 requires q, k, and v base addresses to be 16-byte aligned");
}

torch::Tensor flash_forward_v5(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    validate_attention_inputs_float16(q, k, v);
    check_v5_kernel_requirements(q, k, v);
    return flash_forward_v5_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v5_unchecked(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    check_v5_kernel_requirements(q, k, v);
    return flash_forward_v5_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v5_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v) {
    // Assumes q is [B, H, M, D] and k/v are [B, H, N, D].
    auto out = torch::empty_like(q);

    if (detail::output_is_empty(q)) {
        return out;
    }

    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int query_seq_len = static_cast<int>(q.size(2));
    const int kv_seq_len = static_cast<int>(k.size(2));
    const int head_dim = static_cast<int>(q.size(3));

    flash_forward_v5_cuda_launch(q.const_data_ptr<c10::Half>(), k.const_data_ptr<c10::Half>(),
                                 v.const_data_ptr<c10::Half>(), out.mutable_data_ptr<c10::Half>(),
                                 batch_size, num_heads, query_seq_len, kv_seq_len, head_dim);

    return out;
}

} // namespace flash_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention::flash_forward_v5,
          "V5 FlashAttention forward. Expects q [B, H, M, D] and k/v [B, H, N, D] CUDA "
          "float16 contiguous tensors with D=64 and 16-byte-aligned base addresses.");
    m.def("forward_unchecked", &flash_attention::flash_forward_v5_unchecked,
          "V5 FlashAttention forward without input validation. "
          "Requires q [B, H, M, 64] and k/v [B, H, N, 64] CUDA float16 contiguous tensors "
          "with 16-byte-aligned base addresses.");
}
