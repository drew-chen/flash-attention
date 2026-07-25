#include <torch/extension.h>

#include "../cuda_utils.h"

namespace flash_attention {

void flash_forward_v3_cuda_launch(const float *q,
                                  const float *k,
                                  const float *v,
                                  float *out,
                                  int batch_size,
                                  int num_heads,
                                  int query_seq_len,
                                  int kv_seq_len,
                                  int head_dim);
torch::Tensor flash_forward_v3_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v);

torch::Tensor call_v2(const char *entrypoint,
                      const torch::Tensor &q,
                      const torch::Tensor &k,
                      const torch::Tensor &v) {
    return pybind11::module_::import("flash_attention_v2")
        .attr(entrypoint)(q, k, v)
        .cast<torch::Tensor>();
}

torch::Tensor flash_forward_v3(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    detail::validate_attention_inputs(q, k, v);
    if (q.size(3) != 64) {
        return call_v2("forward", q, k, v);
    }
    return flash_forward_v3_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v3_unchecked(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    if (q.size(3) != 64) {
        return call_v2("forward_unchecked", q, k, v);
    }
    return flash_forward_v3_pytorch_cuda(q, k, v);
}

torch::Tensor flash_forward_v3_pytorch_cuda(const torch::Tensor &q,
                                            const torch::Tensor &k,
                                            const torch::Tensor &v) {
    // Assumes q is [B, H, M, D] and k/v are [B, H, N, D].
    auto out = torch::empty_like(q);

    const int batch_size = static_cast<int>(q.size(0));
    const int num_heads = static_cast<int>(q.size(1));
    const int query_seq_len = static_cast<int>(q.size(2));
    const int kv_seq_len = static_cast<int>(k.size(2));
    const int head_dim = static_cast<int>(q.size(3));

    flash_forward_v3_cuda_launch(q.const_data_ptr<float>(), k.const_data_ptr<float>(),
                                 v.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
                                 batch_size, num_heads, query_seq_len, kv_seq_len, head_dim);

    return out;
}

}  // namespace flash_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention::flash_forward_v3,
          "V3 FlashAttention forward. Expects q [B, H, M, D] and k/v [B, H, N, D] CUDA "
          "float32 contiguous tensors.");
    m.def("forward_unchecked", &flash_attention::flash_forward_v3_unchecked,
          "V3 FlashAttention forward without input validation. "
          "Requires q [B, H, M, D] and k/v [B, H, N, D] CUDA float32 contiguous tensors.");
}
