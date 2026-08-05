# Operations

## Setup

Requires Python 3.12, CUDA-enabled PyTorch, a compatible CUDA toolkit with
`nvcc`, and a CUDA-compatible C++20 compiler.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

After adding, renaming, or moving a CUDA extension, force an in-place rebuild so
stale incremental build artifacts are not reused:

```bash
python setup.py build_ext --inplace --force
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

## Accuracy

`accuracy.py` compares SDPA, SDPA FP16, V4, V4 FP16, and V5 with the FP32
PyTorch reference and reports max absolute error, mean absolute error, RMSE,
and relative L2 error. Run all implementations:

```bash
python accuracy.py
```

Or measure only V5:

```bash
python accuracy.py v5
```

The shape and random seed can be overridden:

```bash
python accuracy.py v5 \
  --batch-size 1 \
  --num-heads 12 \
  --query-seq-len 1024 \
  --kv-seq-len 1024 \
  --head-dim 64 \
  --seed 0
```

## Benchmarks

The fixed suite measures `M=N=512, 1024, 2048` at `B=4, H=12, D=64`, using
25 warmups and 50 timed calls.

Benchmark every registered implementation:

```bash
python benchmark.py all
```

Or benchmark one implementation:

```bash
python benchmark.py v5
```

Benchmark the dtype-matched FP16 PyTorch reference with:

```bash
python benchmark.py sdpa-fp16
```

The required implementation argument accepts `all`, `baseline`, `sdpa`,
`sdpa-fp16`, and `v0` through `v5`, plus `v4-fp16`.

The shape and timing counts can be overridden for larger, shorter benchmark
runs:

```bash
python benchmark.py v5 \
  --batch-size 1 \
  --num-heads 32 \
  --head-dim 64 \
  --seq-lens 4096 \
  --warmup 5 \
  --repetitions 10
```

## Profiling

Use `profile.sh` for detailed inspection of one fused CUDA kernel. Use
`roofline.py` for the combined application-level comparison, including the
multi-kernel PyTorch and SDPA references. V0 is excluded from both comparisons
because it launches multiple project kernels.

### Kernel details

Collect one detailed report, or all fused-kernel reports:

```bash
./profile.sh v5
./profile.sh all
```

Reports are saved as `/tmp/flash_<implementation>_s2048_profile.ncu-rep`. Set
the report to inspect, then print its duration, occupancy, registers, and shared
memory:

```bash
ncu_bin=/usr/local/cuda/bin/ncu
report=/tmp/flash_v4-fp16_s2048_profile.ncu-rep

"$ncu_bin" \
  --import "$report" \
  --page details \
  --section SpeedOfLight \
  --section Occupancy \
  --section LaunchStats
```

Print FP32 operations and DRAM bandwidth:

```bash
"$ncu_bin" \
  --import "$report" \
  --page details \
  --section SpeedOfLight_RooflineChart \
  --print-details all
```

For V5's dense FP16-to-FP32 Tensor Core values, use its report and Tensor Core
section instead:

```bash
report=/tmp/flash_v5_s2048_profile.ncu-rep

"$ncu_bin" \
  --import "$report" \
  --page details \
  --section SpeedOfLight_HierarchicalTensorRooflineChart \
  --print-details all
```

Print the shared-memory counters:

```bash
"$ncu_bin" \
  --import "$report" \
  --page details \
  --section MemoryWorkloadAnalysis_Tables \
  --print-details all
```

The README uses these calculations:

- `TFLOP/s = operations/cycle × SM GHz ÷ 1000`
- `FLOP/byte = TFLOP/s × 1000 ÷ DRAM GB/s`
- `bank-conflict % = 100 × bank conflicts ÷ wavefronts`

For scalar FP32, operations/cycle includes FFMA operations plus FADD and FMUL
instructions. For V5, use the `fp16` source, `fp32` destination, sparsity-off
row. Run `ncu-ui "$report"` only when interactive investigation is useful.

Profiling uses input seed 0, flushes caches between replay passes, and locks
clocks to the supported boost frequency. Pass `--set full` to `profile.sh` only
when the larger report is needed.

### Application roofline

Collect all required reports and generate `roofline-data.json` and
`roofline.png`:

```bash
python roofline.py --profile
```

Regenerate from existing reports, or refresh only selected implementations:

```bash
python roofline.py
python roofline.py --profile --implementations v5
```

The collector measures only elapsed time and DRAM bytes. Every point uses the
same algorithmic QK+PV work, while V5's built-in Nsight sections provide the
FP32, dense FP16 Tensor Core, and DRAM ceilings. Minimal reports are kept
separately as `/tmp/flash_<implementation>_s2048_roofline.ncu-rep`.
