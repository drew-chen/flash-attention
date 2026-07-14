import torch

from src.baseline import forward


def test_forward_matches_explicit_attention_expression():
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = torch.softmax((q @ k.transpose(-1, -2)) / (q.size(-1) ** 0.5), dim=-1) @ v

    torch.testing.assert_close(forward(q, k, v), expected)
