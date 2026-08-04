import pytest
import torch

import flash_attention_v5
from src.baseline import forward as baseline_forward


@pytest.mark.parametrize(
    "batch_size,num_heads,query_seq_len,kv_seq_len,head_dim",
    [
        (1, 2, 65, 33, 64),
        (1, 12, 512, 512, 64),
    ],
    ids=[
        "d64-cross-attention-partial-tiles",
        "large-self-attention",
    ],
)
def test_forward_v5_matches_pytorch_reference(
    batch_size, num_heads, query_seq_len, kv_seq_len, head_dim
):
    """V5 contract: q may have M rows while k/v have N rows."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    q = torch.randn(
        batch_size, num_heads, query_seq_len, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        batch_size, num_heads, kv_seq_len, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn_like(k)

    expected = baseline_forward(q, k, v)
    actual = flash_attention_v5.forward(q, k, v)

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_forward_v5_rejects_float32_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    q = torch.randn(1, 1, 32, 64, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(RuntimeError, match="q must be float16"):
        flash_attention_v5.forward(q, k, v)


def test_forward_v5_rejects_unsupported_head_dimension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    q = torch.randn(1, 1, 32, 32, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(RuntimeError, match="D=64"):
        flash_attention_v5.forward(q, k, v)


def test_forward_v5_rejects_misaligned_contiguous_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    shape = (1, 2, 65, 64)
    numel = 1 * 2 * 65 * 64

    def misaligned_tensor():
        tensor = torch.empty(numel + 1, device="cuda", dtype=torch.float16)[1:].view(shape)
        return tensor.normal_()

    q = misaligned_tensor()
    k = misaligned_tensor()
    v = misaligned_tensor()
    assert q.is_contiguous()
    assert q.data_ptr() % 16 != 0

    with pytest.raises(RuntimeError, match="16-byte aligned"):
        flash_attention_v5.forward(q, k, v)
