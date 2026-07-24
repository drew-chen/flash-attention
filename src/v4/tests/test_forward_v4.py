import pytest
import torch

import flash_attention_v4
from src.baseline import forward as baseline_forward


@pytest.mark.parametrize(
    "batch_size,num_heads,query_seq_len,kv_seq_len,head_dim",
    [
        (2, 3, 32, 32, 16),
        (2, 3, 33, 17, 17),
        (1, 2, 65, 33, 64),
        (1, 12, 512, 512, 64),
    ],
    ids=[
        "self-attention-v2-redirect",
        "cross-attention-v2-redirect",
        "d64-cross-attention-partial-tiles",
        "large-self-attention",
    ],
)
def test_forward_v4_matches_pytorch_reference(
    batch_size, num_heads, query_seq_len, kv_seq_len, head_dim
):
    """V4 contract: q may have M rows while k/v have N rows."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    q = torch.randn(batch_size, num_heads, query_seq_len, head_dim, device="cuda")
    k = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device="cuda")
    v = torch.randn_like(k)

    expected = baseline_forward(q, k, v)
    actual = flash_attention_v4.forward(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_forward_v4_redirects_misaligned_contiguous_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    shape = (1, 2, 65, 64)
    numel = 1 * 2 * 65 * 64

    def misaligned_tensor():
        tensor = torch.empty(numel + 1, device="cuda")[1:].view(shape)
        return tensor.normal_()

    q = misaligned_tensor()
    k = misaligned_tensor()
    v = misaligned_tensor()
    assert q.is_contiguous()
    assert q.data_ptr() % 16 != 0

    expected = baseline_forward(q, k, v)
    actual = flash_attention_v4.forward(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
