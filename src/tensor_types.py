"""Shared Python tensor type aliases for the attention implementations."""

from typing import TypeAlias

import torch
from jaxtyping import Float

# The general formulation uses Q [B, H, M, D] and K/V [B, H, N, D]. The current
# project implements self-attention only, so M = N and Q, K, V, and O share this
# FP32 [B, H, N, D] shape. Jaxtyping labels connect matching dimensions.
AttentionTensor: TypeAlias = Float[torch.Tensor, "batch heads sequence head_dim"]
