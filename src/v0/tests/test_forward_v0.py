import pytest
import torch

from src.baseline import forward as baseline_forward
import flash_attention_v0

@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,head_dim",
    [
        (2, 3, 32, 16),
        (2, 3, 33, 17),
        (1, 12, 512, 64),
    ],
    ids=["tile-aligned", "partial-tiles", "large-self-attention"],
)
def test_forward_v0_matches_pytorch_reference(batch_size, num_heads, seq_len, head_dim):
    """Public V0 correctness contract."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = baseline_forward(q, k, v)
    actual = flash_attention_v0.forward(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
