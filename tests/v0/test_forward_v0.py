def test_forward_v0_matches_pytorch_reference():
    """Public V0 correctness contract; enable once the three V0 kernels exist."""
    import torch

    import flash_attention

    q = torch.randn(2, 3, 32, 16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = torch.softmax((q @ k.transpose(-1, -2)) / (q.size(-1) ** 0.5), dim=-1) @ v
    actual = flash_attention.forward_v0(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
