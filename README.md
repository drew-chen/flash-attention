# FlashAttention

Pedagogical project implementing multi-head attention forward passes with handwritten CUDA kernels.
Surrounding infrastructure (profiling script, PyTorch binding etc) is not hand-written.

## Usage

```python
import torch
import flash_attention_v1

q = torch.randn(2, 3, 32, 16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attention_v1.forward(q, k, v)
```

## Results

Primary shape: `B=4, H=12, M=N=2048, D=64, dtype=float32` (all implementations are non-causal).

### Implementations and latency

| Version | Latency (µs) | Δ vs. baseline | Design |
| --- | ---: | ---: | --- |
| Baseline | 9295.93 | — | Explicit PyTorch attention; correctness and latency reference |
| V0 | 36074.19 | +288.1% | Unfused kernels that materialize attention; naive CUDA starting point |
| V1 | 182577.85 | +1864.1% | Fused tiled online softmax; first FlashAttention-style implementation |
| V2 | 103111.38 | +1009.2% | FA2-style query/warp partitioning with shared state; warp-local optimization starting point |

The CUDA implementations accept contiguous CUDA `float32` tensors. V0 requires self-attention
with `M = N`; V1 and V2 accept Q `[B, H, M, D]` and K/V `[B, H, N, D]`.

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

### V2 profiling

The fused v2 kernel was profiled on the same GPU:

| Metric | Measured | RTX 4080 reference |
| --- | ---: | ---: |
| FP32 throughput | 0.562 TFLOP/s | 48.7 TFLOP/s peak |
| DRAM bandwidth | 9.10 GB/s | 716.8 GB/s peak |
| Arithmetic intensity | 61.70 FLOP/byte | 67.9 FLOP/byte ridge point |
| Occupancy | 8.33% achieved | 8.33% theoretical |
| Grid | 1,536 blocks | 76 SMs |
| Shared memory | 58.37 KB/block | Limits residency to 1 block/SM |

The hardware references come from NVIDIA's
[Ada GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf).

### Notes

#### Baseline

The explicit PyTorch implementation is the correctness and latency reference.

#### V0

V0 is slower than the PyTorch baseline despite using tiled CUDA kernels.

#### V1

V1 follows the FA1 and avoids the full attention matrix, avoids unnecessary transposes, and performs warp reductions but one of the main issues is it's grid setup. It uses block per `(batch, head)` which produces only 48 blocks for the benchmarked shape, so some of the RTX 4080's 76 SMs receive no work. In fact ideally, we have more than 76 blocks running at a time. As for occupancy, shared memory caps residency at eight theoretical
warps per SM while the kernel achieves four.

The next architectural step is to parallelize independent query tiles, following the
FlashAttention-2 work partition rather than tuning the current 48-block launch.

#### V2

V2 adds FA2-style query-tile parallelism and assigns complete score rows to
warps, increasing the benchmark grid from 48 to 1,536 blocks. However, this
simplified implementation does not completely follow FA2's warp-local storage
strategy. It keeps `m`, `l`, `O`, and temporary softmax state in shared memory
alongside the Q/K/V and S/P tiles.

The resulting 58.37 KB shared-memory allocation permits only one 128-thread
block, or four active warps, per SM. This limits both theoretical and achieved
occupancy to 8.33%. A more complete FA2 implementation would keep the running
softmax state and output accumulators in warp/thread registers, reuse shared
buffers more aggressively, and reserve shared memory primarily for staging
matrix tiles.

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

The shape and timing counts can be overridden for larger, shorter benchmark runs:

```bash
python benchmark.py \
  --batch-size 1 \
  --num-heads 32 \
  --head-dim 64 \
  --seq-lens 4096 \
  --warmup 5 \
  --repetitions 10
```

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
