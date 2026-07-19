import pytest
    import torch

from src.baseline import forward as baseline_forward
import flash_attention

@pytest.mark.parametrize(
    "seq_len,head_dim",
    [(32, 16), (33, 17)],
    ids=["tile-aligned", "partial-tiles"],
)
def test_forward_v0_matches_pytorch_reference(seq_len, head_dim):
    """Public V0 correctness contract."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")



    q = torch.randn(2, 3, seq_len, head_dim, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = baseline_forward(q, k, v)
    actual = flash_attention.forward_v0(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
