# FlashAttention

Pedagogical project for building handwritten CUDA attention kernels. PyTorch provides the Python
binding and correctness reference.

## Usage

```python
import torch
import flash_attention_v1

q = torch.randn(2, 3, 32, 16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attention_v1.forward(q, k, v)
```

## Implementations

| Version | Design | Purpose |
| --- | --- | --- |
| Baseline | Explicit PyTorch matmul, scaling, softmax, and matmul | Correctness and latency reference |
| V0 | Unfused CUDA kernels that materialize the attention matrix | Naive CUDA starting point |
| V1 | One fused CUDA kernel with tiled online softmax | First FlashAttention-style implementation |

The CUDA implementations accept contiguous CUDA `float32` tensors. V0 requires self-attention
with `M = N`; V1 accepts Q `[B, H, M, D]` and K/V `[B, H, N, D]`.

## Results

Primary shape: `B=4, H=12, M=N=2048, D=64, dtype=float32, causal=false`.

### Latency

| Version | Latency (µs) | Δ vs. baseline |
| --- | ---: | ---: |
| Baseline | 8418.95 | — |
| V0 | 33890.02 | +302.5% |
| V1 | 172278.32 | +1946.3% |

### V1 profiling

The fused v1 kernel was profiled on an original RTX 4080:

| Metric | Measured | RTX 4080 reference |
| --- | ---: | ---: |
| FP32 throughput | 0.336 TFLOP/s | 48.7 TFLOP/s peak |
| DRAM bandwidth | 0.416 GB/s | 716.8 GB/s peak |
| Arithmetic intensity | 809.701 FLOP/byte | 67.9 FLOP/byte ridge point |
| Occupancy | 8.33% achieved | 16.67% theoretical |
| Grid | 48 blocks | 76 SMs |
| Shared memory | 38.40 KB/block | Limits residency to 2 blocks/SM |

The hardware references come from NVIDIA's
[Ada GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf).

### Notes

#### Baseline

The explicit PyTorch implementation is the correctness and latency reference.

#### V0

V0 is slower than the PyTorch baseline despite using tiled CUDA kernels.

#### V1

V1 avoids the full attention matrix, but one block per `(batch, head)` produces only 48 blocks, so
some of the RTX 4080's 76 SMs receive no work. Shared memory caps residency at eight theoretical
warps per SM; the kernel achieves four.

The next architectural step is to parallelize independent query tiles, following the
FlashAttention-2 work partition rather than tuning the current 48-block launch.

## Setup

Requires Python 3.12, CUDA-enabled PyTorch, a compatible CUDA toolkit with `nvcc`, and a
CUDA-compatible C++20 compiler.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

To target a different GPU architecture:

```bash
TORCH_CUDA_ARCH_LIST=<value> python setup.py build_ext --inplace
```

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

The fixed suite measures `M=N=512, 1024, 2048` at `B=4, H=12, D=64`, using 25 warmups and 100
timed calls.

## Profiling

Profiling is restricted to fused implementations. Baseline and v0 launch multiple kernels, so a
single roofline or occupancy value would be ambiguous.

`profile.sh` captures the roofline, occupancy, and launch statistics in one report. Profile one
fused implementation:

```bash
./profile.sh v1
```

Or profile every registered fused implementation:

```bash
./profile.sh all
```

`all` prints a skip message for baseline and v0 because they are unfused. Passing either one
directly is an error. If non-admin GPU performance counters are disabled, run the script with
`sudo`; it uses the repository's virtual environment by absolute path.

Reports are saved as `/tmp/flash_<implementation>_s2048_profile.ncu-rep`.

Open the report:

```bash
ncu-ui /tmp/flash_v1_s2048_profile.ncu-rep
```

Print occupancy and launch statistics:

```bash
/usr/local/cuda/bin/ncu \
  --import /tmp/flash_v1_s2048_profile.ncu-rep \
  --page details \
  --section Occupancy \
  --section LaunchStats
```

Print the roofline overview:

```bash
/usr/local/cuda/bin/ncu \
  --import /tmp/flash_v1_s2048_profile.ncu-rep \
  --page details \
  --section SpeedOfLight_RooflineChart \
  --print-details all
```
