#include <torch/extension.h>

#include "../cuda_utils.h"

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
torch::Tensor call_v2(const char *entrypoint,
                      const torch::Tensor &q,
                      const torch::Tensor &k,
                      const torch::Tensor &v) {
    return pybind11::module_::import("flash_attention_v2")
        .attr(entrypoint)(q, k, v)
        .cast<torch::Tensor>();
}

bool can_use_v4_vector_loads(const torch::Tensor &q,
                             const torch::Tensor &k,
                             const torch::Tensor &v) {
    constexpr std::uintptr_t FLOAT4_ALIGNMENT = 4 * sizeof(float);
    return detail::are_aligned<float>(q, k, v, FLOAT4_ALIGNMENT);
}

torch::Tensor flash_forward_v4(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    detail::validate_attention_inputs(q, k, v);
    if (q.size(3) != 64 || !can_use_v4_vector_loads(q, k, v)) {
        return call_v2("forward", q, k, v);
    }
    return detail::launch_attention<float>(q, k, v, flash_forward_v4_cuda_launch);
}

torch::Tensor flash_forward_v4_unchecked(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    if (q.size(3) != 64 || !can_use_v4_vector_loads(q, k, v)) {
        return call_v2("forward_unchecked", q, k, v);
    }
    return detail::launch_attention<float>(q, k, v, flash_forward_v4_cuda_launch);
}

} // namespace flash_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention::flash_forward_v4,
          "V4 FlashAttention forward. Expects q [B, H, M, D] and k/v [B, H, N, D] CUDA "
          "float32 contiguous tensors.");
    m.def("forward_unchecked", &flash_attention::flash_forward_v4_unchecked,
          "V4 FlashAttention forward without input validation. "
          "Requires q [B, H, M, D] and k/v [B, H, N, D] CUDA float32 contiguous tensors.");
}
