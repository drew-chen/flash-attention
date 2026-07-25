#include <torch/extension.h>

#include "../cuda_utils.h"

namespace flash_attention {

void naive_forward_v0_cuda_launch(const float *q,
                                  const float *k,
                                  const float *v,
                                  float *out,
                                  int batch_size,
                                  int num_heads,
                                  int seq_len,
                                  int head_dim);
torch::Tensor flash_forward_v0_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v);

// Validates the public contract for flash_attention_v0.forward(...):
// CUDA float32 contiguous self-attention tensors with shape [B, H, N, D]
// (the general attention dimensions are specialized to M = N).
void validate_flash_inputs(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v) {
    flash_attention::detail::check_cuda_float32_contiguous_dim(q, "q", 4);
    flash_attention::detail::check_cuda_float32_contiguous_dim(k, "k", 4);
    flash_attention::detail::check_cuda_float32_contiguous_dim(v, "v", 4);
    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.size(2) > 0, "sequence length N must be greater than zero");
}

torch::Tensor flash_forward_v0(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    validate_flash_inputs(q, k, v);
    return flash_forward_v0_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v0_unchecked(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    return flash_forward_v0_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v0_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v) {
    // Assumes the validated contiguous self-attention [B, H, N, D] contract (M = N).
    auto out = torch::empty_like(q);

    if (detail::output_is_empty(q)) {
        return out;
    }

    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int seq_len = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));

    naive_forward_v0_cuda_launch(q.const_data_ptr<float>(), k.const_data_ptr<float>(),
                                 v.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
                                 batch_size, num_heads, seq_len, head_dim);

    return out;
}

}  // namespace flash_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention::flash_forward_v0,
          "V0 FlashAttention forward. Expects CUDA float32 contiguous tensors of shape [B, H, N, "
          "D].");
    m.def("forward_unchecked", &flash_attention::flash_forward_v0_unchecked,
          "V0 FlashAttention forward without input validation. "
          "Requires CUDA float32 contiguous [B, H, N, D] inputs.");
}
