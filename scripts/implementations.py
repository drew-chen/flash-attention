"""Shared names and compute properties for attention implementations."""

IMPLEMENTATION_LABELS = {
    "baseline": "Explicit PyTorch",
    "sdpa": "SDPA FP32",
    "sdpa-fp16": "SDPA FP16",
    "v0": "V0",
    "v1": "V1",
    "v2": "V2",
    "v3": "V3",
    "v4": "V4",
    "v4-fp16": "V4 FP16",
    "v5": "V5 WMMA",
    "v6": "V6 WMMA",
}

# Only the exceptional storage dtype is listed; the FP32 set is derived.
FP16_IMPLEMENTATIONS = frozenset({"v4-fp16", "sdpa-fp16", "v5", "v6"})
FP32_IMPLEMENTATIONS = frozenset(IMPLEMENTATION_LABELS) - FP16_IMPLEMENTATIONS

# V4 FP16 has FP16 storage but does not use Tensor Cores.
TENSOR_CORE_IMPLEMENTATIONS = FP16_IMPLEMENTATIONS - {"v4-fp16"}
