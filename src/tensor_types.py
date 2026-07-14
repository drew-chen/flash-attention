"""Shared Python tensor type aliases for the attention implementations."""

from typing import TypeAlias

import torch
from jaxtyping import Float

# The current project scope uses the same FP32 [B, H, N, D] shape for Q, K, V,
# and the output. Jaxtyping labels connect matching dimensions across arguments.
AttentionTensor: TypeAlias = Float[torch.Tensor, "batch heads sequence head_dim"]
