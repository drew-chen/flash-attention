# FlashAttention

Pedagogical project for a handwritten CUDA FlashAttention implementation. PyTorch
provides the Python binding and correctness reference.

## Optimization worklog

**Primary shape:** `B=2, H=8, M=N=2048, D=64, dtype=float32, causal=false`

| Version | Latency (µs) | Δ vs. baseline | FLOPs per second (TFLOP/s)* | DRAM bandwidth (GB/s)* | Arithmetic Intensity (FLOP/byte)* | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 2869.55 | — | 9.202 | 552.211 | 16.664 | Reference measurement |
| V0 | 13752.51 | +379.3% | 2.140 | 354.121 | 6.044 | Draft non-flash attention implementation is worse than baseline |
| V1 |  ---: | ---: | ---: | ---: | ---:  | Fused kernel, online softmax, skipping tranpose |

\* See the [profiling appendix](#appendix-profiling-unfused-implementations).

## Setup

Requires Python 3.12, a CUDA-enabled PyTorch build, a compatible CUDA toolkit
with `nvcc`, and a CUDA-compatible C++20 compiler. Create an environment and
install the Python dependencies:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Build the extension (repeat after changing C++ or CUDA files):

```bash
python setup.py build_ext --inplace
```

To target a different GPU architecture:

```bash
TORCH_CUDA_ARCH_LIST=<value> python setup.py build_ext --inplace
```

## Python usage

```python
import torch
import flash_attention

q = torch.randn(2, 3, 32, 16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attention.forward_v0(q, k, v)
```

`forward_v0` implements self-attention only: it accepts contiguous CUDA `float32`
Q, K, and V tensors, each shaped `[B, H, N, D]`, where the general attention
dimensions satisfy `M = N`. Its output has the same shape.
Use `src.baseline.forward` as the PyTorch correctness reference.

## Tests

```bash
python -m pytest -q
```

For CUDA memory checks:

```bash
compute-sanitizer --target-processes all python -m pytest -q
```

## Benchmarks

```bash
python benchmark.py
```

The fixed suite covers self-attention shapes `M=N=512, 1024, 2048` at
`B=2, H=8, D=64`, with 25
warmups and 100 timed calls.

## Appendix: profiling unfused implementations

Set `implementation` to `v0` or `baseline`:

```bash
implementation=v0
```

Collect duration and DRAM traffic for one complete NVTX-marked forward range:

```bash
sudo /usr/local/cuda/bin/ncu \
  --replay-mode range \
  --nvtx \
  --nvtx-include "flash_attention.${implementation}/" \
  --metrics dram__bytes.sum,gpu__time_duration.sum \
  --export "/tmp/flash_${implementation}_s2048_range" \
  --force-overwrite \
  python profile_cuda.py "$implementation"
```

Collect the FP32 instruction counts for each kernel in that range:

```bash
sudo /usr/local/cuda/bin/ncu \
  --replay-mode kernel \
  --nvtx \
  --nvtx-include "flash_attention.${implementation}/" \
  --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
  --export "/tmp/flash_${implementation}_s2048_kernels" \
  --force-overwrite \
  python profile_cuda.py "$implementation"
```

Combine the reports as follows:

```text
FP32 operations = Σ(FADD + FMUL + 2 × FFMA)
FLOPs per second = FP32 operations / range duration
DRAM bandwidth = range DRAM bytes / range duration
Arithmetic intensity = FP32 operations / range DRAM bytes
```

FP32 operations (`FADD + FMUL + 2 × FFMA`) are summed across each
implementation's kernel profiles; duration and DRAM bytes cover its complete
NVTX-marked forward range. Nsight cannot collect FP32 instruction counts for a
multi-kernel range, which is why the kernel and range reports are collected
separately. Special-function operations such as `exp` are not included in the
FP32 operation count.
