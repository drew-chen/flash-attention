#include <torch/extension.h>

#include "../cuda_utils.h"
#include <cstdint>

namespace flash_attention {

void flash_forward_v4_cuda_launch(const float *q,
                                  const float *k,
                                  const float *v,
                                  float *out,
                                  int batch_size,
                                  int num_heads,
                                  int query_seq_len,
                                  int kv_seq_len,
                                  int head_dim);
torch::Tensor flash_forward_v4_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v);

// Validates the public contract for flash_attention_v4.forward(...):
// CUDA float32 contiguous attention tensors: q [B, H, M, D] and k/v [B, H, N, D].
void validate_flash_inputs(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v) {
    flash_attention::detail::check_cuda_float32_contiguous_dim(q, "q", 4);
    flash_attention::detail::check_cuda_float32_contiguous_dim(k, "k", 4);
    flash_attention::detail::check_cuda_float32_contiguous_dim(v, "v", 4);
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have identical shape [B, H, N, D]");
    TORCH_CHECK(q.size(0) == k.size(0), "q and k must have the same batch size");
    TORCH_CHECK(q.size(1) == k.size(1), "q and k must have the same number of heads");
    TORCH_CHECK(q.size(3) == k.size(3), "q and k must have the same head dimension");
}

torch::Tensor call_v2(const char *entrypoint,
                      const torch::Tensor &q,
                      const torch::Tensor &k,
                      const torch::Tensor &v) {
    return pybind11::module_::import("flash_attention_v2")
        .attr(entrypoint)(q, k, v)
        .cast<torch::Tensor>();
}

bool is_float4_aligned(const torch::Tensor &tensor) {
    constexpr std::uintptr_t FLOAT4_ALIGNMENT = 4 * sizeof(float);
    return reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr<float>()) % FLOAT4_ALIGNMENT == 0;
}

bool can_use_v4_vector_loads(const torch::Tensor &q,
                             const torch::Tensor &k,
                             const torch::Tensor &v) {
    return is_float4_aligned(q) && is_float4_aligned(k) && is_float4_aligned(v);
}

torch::Tensor flash_forward_v4(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    validate_flash_inputs(q, k, v);
    if (q.size(3) != 64 || !can_use_v4_vector_loads(q, k, v)) {
        return call_v2("forward", q, k, v);
    }
    return flash_forward_v4_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v4_unchecked(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    if (q.size(3) != 64 || !can_use_v4_vector_loads(q, k, v)) {
        return call_v2("forward_unchecked", q, k, v);
    }
    return flash_forward_v4_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v4_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v) {
    // Assumes q is [B, H, M, D] and k/v are [B, H, N, D].
    auto out = torch::empty_like(q);

    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int query_seq_len = static_cast<int>(q.size(2));
    const int kv_seq_len = static_cast<int>(k.size(2));
    const int head_dim = static_cast<int>(q.size(3));

    flash_forward_v4_cuda_launch(q.const_data_ptr<float>(), k.const_data_ptr<float>(),
                                 v.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
                                 batch_size, num_heads, query_seq_len, kv_seq_len, head_dim);

    return out;
}

}  // namespace flash_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention::flash_forward_v4,
          "V4 FlashAttention forward. Expects q [B, H, M, D] and k/v [B, H, N, D] CUDA "
          "float32 contiguous tensors.");
    m.def("forward_unchecked", &flash_attention::flash_forward_v4_unchecked,
          "V4 FlashAttention forward without input validation. "
          "Requires q [B, H, M, D] and k/v [B, H, N, D] CUDA float32 contiguous tensors.");
}
