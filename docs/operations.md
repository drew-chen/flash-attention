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

Profiling commands cover the project's fused implementations. Baseline and V0
launch multiple kernels, so a single roofline or occupancy value would be
ambiguous. The SDPA baselines are handled inside PyTorch and are not profiled
here.

`profile.sh` uses Nsight Compute's `detailed` set by default and adds scheduler,
warp-state, and detailed memory-workload sections. The resulting report includes
the roofline, occupancy, launch statistics, stalls, and shared-memory conflict
counters. Profile one fused implementation:

```bash
./profile.sh v5
```

Or profile every registered fused implementation:

```bash
./profile.sh all
```

`all` skips baseline, both SDPA baselines, and V0. Passing one of them directly
is an error. If non-admin GPU performance counters are disabled, run the script
with `sudo`. It uses the repository's virtual environment by absolute path.

Use a different Nsight Compute section set when needed:

```bash
./profile.sh v5 --set full
```

Reports are saved as `/tmp/flash_<implementation>_s2048_profile.ncu-rep`.

Open the report:

```bash
ncu-ui /tmp/flash_v5_s2048_profile.ncu-rep
```

Open all six fused-kernel reports:

```bash
ncu-ui \
  /tmp/flash_v1_s2048_profile.ncu-rep \
  /tmp/flash_v2_s2048_profile.ncu-rep \
  /tmp/flash_v3_s2048_profile.ncu-rep \
  /tmp/flash_v4_s2048_profile.ncu-rep \
  /tmp/flash_v4-fp16_s2048_profile.ncu-rep \
  /tmp/flash_v5_s2048_profile.ncu-rep
```

Print occupancy and launch statistics:

```bash
/usr/local/cuda/bin/ncu \
  --import /tmp/flash_v5_s2048_profile.ncu-rep \
  --page details \
  --section Occupancy \
  --section LaunchStats
```

Print the roofline overview:

```bash
/usr/local/cuda/bin/ncu \
  --import /tmp/flash_v5_s2048_profile.ncu-rep \
  --page details \
  --section SpeedOfLight_RooflineChart \
  --print-details all
```
