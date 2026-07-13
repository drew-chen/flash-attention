#include <torch/extension.h>

#include "../cuda_utils.h"

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

// Validates the public contract for flash_attention.forward_v0(...):
// CUDA float32 contiguous tensors with shape [B, H, N, D].
void validate_flash_inputs(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v) {
    check_cuda_float32_contiguous_dim(q, "q", 4);
    check_cuda_float32_contiguous_dim(k, "k", 4);
    check_cuda_float32_contiguous_dim(v, "v", 4);
    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have identical shape [B, H, N, D]");
}

torch::Tensor flash_forward_v0(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    validate_flash_inputs(q, k, v);
    return flash_forward_v0_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v0_assume_valid(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    return flash_forward_v0_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v0_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v) {
    // Assumes q, k, v already satisfy the validated contiguous [B, H, N, D] contract.
    auto out = torch::empty_like(q);

    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int seq_len = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));

    naive_forward_v0_cuda_launch(q.const_data_ptr<float>(), k.const_data_ptr<float>(),
                                 v.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
                                 batch_size, num_heads, seq_len, head_dim);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_v0", &flash_forward_v0,
          "V0 FlashAttention forward. Expects CUDA float32 contiguous tensors of shape [B, H, N, "
          "D].");
    m.def("forward_v0_assume_valid", &flash_forward_v0_assume_valid,
          "V0 FlashAttention forward assuming CUDA float32 contiguous [B, H, N, D] inputs.");
}
