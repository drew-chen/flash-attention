#include <torch/extension.h>

#include "../cuda_utils.h"

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
torch::Tensor flash_forward_v5(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    detail::validate_attention_inputs_float16(q, k, v);
    detail::check_head_dim_and_alignment<c10::Half>(q, k, v, "V5", 64, 16);
    return detail::launch_attention<c10::Half>(q, k, v, flash_forward_v5_cuda_launch);
}

torch::Tensor flash_forward_v5_unchecked(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    detail::check_head_dim_and_alignment<c10::Half>(q, k, v, "V5", 64, 16);
    return detail::launch_attention<c10::Half>(q, k, v, flash_forward_v5_cuda_launch);
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
