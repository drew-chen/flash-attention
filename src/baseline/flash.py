"""Explicit, unfused PyTorch attention reference."""

import torch

from src.tensor_types import AttentionTensor


def forward(q: AttentionTensor, k: AttentionTensor, v: AttentionTensor) -> AttentionTensor:
    """Compute non-causal scaled dot-product attention for `[B, H, N, D]` tensors.

    This deliberately uses PyTorch's explicit matmul, scaling, softmax, and
    matmul operations. It is the correctness reference for the CUDA versions,
    not an attempt to implement FlashAttention.
    """
    return torch.softmax((q @ k.transpose(-1, -2)) / (q.size(-1) ** 0.5), dim=-1) @ v
