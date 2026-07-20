import pytest
import torch

from src.baseline import forward as baseline_forward
import flash_attention


@pytest.mark.xfail(reason="the v1 FlashAttention kernel is still in progress", strict=True)
@pytest.mark.parametrize(
    "batch_size,num_heads,query_seq_len,kv_seq_len,head_dim",
    [
        (2, 3, 32, 32, 16),
        (2, 3, 33, 17, 17),
        (1, 12, 512, 512, 64),
    ],
    ids=["self-attention", "cross-attention-partial-tiles", "large-self-attention"],
)
def test_forward_v1_matches_pytorch_reference(
    batch_size, num_heads, query_seq_len, kv_seq_len, head_dim
):
    """V1 contract: q may have M rows while k/v have N rows."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    q = torch.randn(batch_size, num_heads, query_seq_len, head_dim, device="cuda")
    k = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, device="cuda")
    v = torch.randn_like(k)

    expected = baseline_forward(q, k, v)
    # The v1 binding currently retains this entry-point name while it is developed.
    actual = flash_attention.forward_v0(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
