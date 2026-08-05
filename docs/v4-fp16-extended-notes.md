# V4 FP16 extended notes

## Motivation

V4 FP16 is a storage baseline for the tensor-core work in V5. Tensor cores are
optimized for FP16 operands, and FP16 storage halves the bytes required by Q, K,
V, and P relative to FP32. Adapting V4 to FP16 storage isolates those effects
before its scalar matrix products are replaced with WMMA.

The implementation retains V4's D=64 FA2-style algorithm and warp ownership.
[V3's extended notes](v3-extended-notes.md) are the canonical ownership and
distribution reference.

## Precision choices

Inputs to matrix multiplications use FP16 storage. Values that may exceed the
maximum finite FP16 value, 65,504, remain FP32. Values formed by long
accumulations or reductions also remain FP32 because FP16 rounding error can
accumulate even when the final result fits in FP16.

Q, K, V, and the copy of P consumed by the matrix multiplication therefore use
FP16. QK scores, the online-softmax state m and l, and output accumulators remain
FP32.

P is safe to store in FP16 because its unnormalized value is

```text
p_ij = exp(s_ij - m_i_new)
```

Safe softmax subtracts the updated row maximum, so `m_i_new >= s_ij` and the
exponent is always zero or negative. Therefore `0 < p_ij <= 1` for valid
entries, while masked entries use zero. P cannot overflow FP16, although very
small probabilities can still round or underflow.

The kernel computes the exponential and denominator reduction in FP32, then
stores a rounded FP16 copy of P for the PV matrix multiplication. PV accumulates
into FP32 output registers. Finally, the kernel normalizes by l in FP32 and
rounds once to FP16 when writing the output.
