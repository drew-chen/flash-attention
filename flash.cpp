#include <torch/extension.h>

void flash_forward_cuda_launch(const float *q, const float *k, const float *v, float *out,
                               int batch_size, int num_heads, int seq_len, int head_dim);
torch::Tensor flash_forward_pytorch_cuda(const torch::Tensor &q, const torch::Tensor &k,
                                         const torch::Tensor &v);

// Validates the public contract for flash_attention.forward(...):
// CUDA float32 contiguous tensors with shape [B, H, N, D].
void validate_flash_inputs(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must have shape [B, H, N, D]");
    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 && k.scalar_type() == torch::kFloat32 &&
                    v.scalar_type() == torch::kFloat32,
                "q, k, v must be float32 tensors");
}

torch::Tensor flash_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    validate_flash_inputs(q, k, v);
    return flash_forward_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_assume_valid(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    return flash_forward_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_pytorch_cuda(const torch::Tensor &q, const torch::Tensor &k,
                                         const torch::Tensor &v) {
    // Assumes q, k, v already satisfy the validated contiguous [B, H, N, D] contract.
    auto out = torch::empty_like(q);

    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int seq_len = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));

    flash_forward_cuda_launch(q.const_data_ptr<float>(), k.const_data_ptr<float>(),
                              v.const_data_ptr<float>(), out.mutable_data_ptr<float>(), batch_size,
                              num_heads, seq_len, head_dim);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_forward,
          "FlashAttention forward. Expects CUDA float32 contiguous tensors of shape [B, H, N, D].");
    m.def("forward_assume_valid", &flash_forward_assume_valid,
          "FlashAttention forward assuming CUDA float32 contiguous [B, H, N, D] inputs.");
}
