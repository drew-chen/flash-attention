# FlashAttention

This is a project meant for learning. The cuda code is handwritten but the surrounding utils such as the binding to pytorch is not.


## Optimization worklog

Use the same primary shape for every iteration; record the small shape suite only
when a change is worth keeping. Latency is the decision metric; the other values
help explain whether the kernel is compute- or memory-bound.

**Primary shape:** `B=__, H=__, S=__, D=__, dtype=__, causal=__`

| Iteration | Change | Latency (µs) | Δ vs. baseline | TFLOP/s | GB/s | AI (FLOP/B) | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | Baseline |  | — |  |  |  |  |

| Iteration | S=512 latency | S=2K latency | S=8K latency | S=32K latency | Regression? |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 (baseline) |  |  |  |  | No |

Keep two plots alongside this log:

- **Latency by sequence length:** one line per retained iteration.
- **Roofline:** baseline and current-best kernel, with points labeled by sequence length.

## Setup

Create and activate a Python 3.12 virtual environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch numpy jaxtyping pytest
```

Build the CUDA extension:

```bash
python setup.py build_ext --inplace
```

Run this once initially and again after changing C++ or CUDA sources/headers in
`src/` (including shared CUDA utilities). Python-only edits—such as the
baseline, benchmark, tests, or README—do not require rebuilding.

The extension requires a CUDA-capable PyTorch installation and a compatible CUDA toolkit. To target a different GPU architecture, set `TORCH_CUDA_ARCH_LIST` when building:

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

`forward_v0` accepts contiguous CUDA `float32` tensors with shape `[B, H, N, D]`.

The explicit PyTorch correctness reference is available as `src.baseline.forward`:

```python
from src.baseline import forward as baseline_forward

out = baseline_forward(q, k, v)
```

## Tests

Run the full test suite:

```bash
python -m pytest -q
```

Run individual test groups:

```bash
python -m pytest -q src/baseline/tests
python -m pytest -q src/v0/tests/test_cuda_helpers.py
python -m pytest -q src/v0/tests/test_forward_v0.py
```

Run CUDA memory checks:

```bash
compute-sanitizer --target-processes all python -m pytest -q
```

## Benchmarks

Build the extension, then run every registered implementation with GPU-side CUDA
events:

```bash
python benchmark/benchmark.py
```

The default suite is `S=512, 1024, 2048` at `B=2, H=8, D=64`, with 25 warm-up
calls and 100 timed repetitions per implementation and shape. It currently runs
`baseline` and `v0`; future registered implementations are included
automatically.

The output is a Markdown table like this (values depend on the GPU):

```text
GPU timing: B=2, H=8, D=64, dtype=float32, warmup=25, repetitions=100
| Implementation | Sequence length | Latency (us) | Effective TFLOP/s |
| :------------- | --------------: | -----------: | ----------------: |
| baseline       |             512 |        <...> |             <...> |
| v0             |             512 |        <...> |             <...> |
| baseline       |            1024 |        <...> |             <...> |
| v0             |            1024 |        <...> |             <...> |
```

### Choose a shape or implementation

To override the default shape suite, timed repetitions, or implementation set:

```bash
python benchmark/benchmark.py --seq-lens 2048 --repetitions 100
python benchmark/benchmark.py --implementations baseline
python benchmark/benchmark.py --implementations v0
```

The reported TFLOP/s counts the two matrix multiplications only; latency is the
primary comparison metric.
