# FlashAttention

Pedagogical project for a handwritten CUDA FlashAttention implementation. PyTorch
provides the Python binding and correctness reference.

## Optimization worklog

**Primary shape:** `B=4, H=12, M=N=2048, D=64, dtype=float32, causal=false`

`causal=false` means every query may attend to every key; no triangular future-token mask is
applied.

| Version | Latency (µs) | Δ vs. baseline | FLOPs per second (TFLOP/s)* | DRAM bandwidth (GB/s)* | Arithmetic Intensity (FLOP/byte)* | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 8418.95 | — | --- | --- | --- | Reference measurement |
| V0 | 33890.02 | +302.5% | --- | --- | --- | Draft non-flash attention implementation is worse than baseline |
| V1 | 172278.32 | +1946.3% | 0.336 | 0.416 | 809.701 | Fused online-softmax kernel follows FA 1 |

V1's high DRAM arithmetic intensity but low compute throughput indicates that it is limited by
compute execution and insufficient parallelism rather than DRAM bandwidth.

\* V1 hardware metrics were measured on an RTX 4080. See the
[profiling appendix](#appendix-profiling-fused-implementations) to refresh them.

## Setup

Requires Python 3.12, a CUDA-enabled PyTorch build, a compatible CUDA toolkit
with `nvcc`, and a CUDA-compatible C++20 compiler. Create an environment and
install the Python dependencies:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Build both versioned extensions (repeat after changing C++ or CUDA files):

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
import flash_attention_v0
import flash_attention_v1

q = torch.randn(2, 3, 32, 16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

out_v0 = flash_attention_v0.forward(q, k, v)
out_v1 = flash_attention_v1.forward(q, k, v)
```

Both accept contiguous CUDA `float32` tensors. V0 requires self-attention with
`M = N`; V1 accepts Q `[B, H, M, D]` and K/V `[B, H, N, D]`.
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
`B=4, H=12, D=64`, with 25 warmups and 100 timed calls.

V1 launches one block per `(batch, head)`, so this shape launches only 48 blocks. Nsight Compute
flags this grid as too small for the profiled RTX 4080's 76 SMs: it provides only 0.32 full waves
and leaves some SMs with no work. The report measures 10.64% SM throughput but only 0.06% DRAM
throughput, confirming that DRAM bandwidth is not the primary limiter. Its workload-distribution
rule also estimates a 23.4% speedup from eliminating the resulting SM imbalance. Increasing
parallelism within or across query tiles is therefore the first optimization target.

## Appendix: profiling fused implementations

Profile v1 with Nsight Compute's built-in roofline section:

```bash
implementation=v1
sudo /usr/local/cuda/bin/ncu \
  --set roofline \
  --replay-mode kernel \
  --nvtx \
  --nvtx-include "flash_attention.${implementation}/" \
  --export "/tmp/flash_${implementation}_s2048_roofline" \
  --force-overwrite \
  python profile_cuda.py "$implementation"
```

Open the report in Nsight Compute:

```bash
ncu-ui /tmp/flash_v1_s2048_roofline.ncu-rep
```

Or print the single-precision roofline overview in the terminal:

```bash
/usr/local/cuda/bin/ncu \
  --import /tmp/flash_v1_s2048_roofline.ncu-rep \
  --page details \
  --section SpeedOfLight_RooflineChart \
  --print-details all
```

Nsight derives the kernel's achieved FLOP/s, memory traffic, and arithmetic
intensity and displays them together on the roofline chart. This maps cleanly to
v1 and later fused implementations because one attention kernel represents the
complete forward operation. V0 remains in the latency benchmark, but is omitted
from roofline profiling because its separate kernels do not form one roofline
point.
