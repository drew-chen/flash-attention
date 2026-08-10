"""Measure attention latency with randomized, interleaved sample rounds.

For each shape, every implementation is warmed up first. Each timing round then
measures one batch from every implementation in randomized order.
"""

import argparse
import random
import statistics
import time

import torch

import flash_attention_v0
import flash_attention_v1
import flash_attention_v2
import flash_attention_v3
import flash_attention_v4
import flash_attention_v4_fp16
import flash_attention_v5
import flash_attention_v6
from scripts.implementations import FP16_IMPLEMENTATIONS
from scripts.workload import (
    BATCH_SIZE,
    HEAD_DIM,
    NUM_HEADS,
    PRIMARY_SEQ_LEN,
)
from src.baseline import forward as baseline_forward

SEQ_LENS = (512, 1024, PRIMARY_SEQ_LEN)
WARMUP = 25
WARMUP_MS = 500.0
REPETITIONS = 50
SAMPLES = 7


IMPLEMENTATIONS = {
    "baseline": baseline_forward,
    "sdpa": torch.nn.functional.scaled_dot_product_attention,
    "sdpa-fp16": torch.nn.functional.scaled_dot_product_attention,
    "v0": flash_attention_v0.forward_unchecked,
    "v1": flash_attention_v1.forward_unchecked,
    "v2": flash_attention_v2.forward_unchecked,
    "v3": flash_attention_v3.forward_unchecked,
    "v4": flash_attention_v4.forward_unchecked,
    "v4-fp16": flash_attention_v4_fp16.forward_unchecked,
    "v5": flash_attention_v5.forward_unchecked,
    "v6": flash_attention_v6.forward_unchecked,
}


def make_inputs(seq_len, dtype):
    q = torch.randn(
        BATCH_SIZE,
        NUM_HEADS,
        seq_len,
        HEAD_DIM,
        device="cuda",
        dtype=dtype,
    )
    return q, torch.randn_like(q), torch.randn_like(q)


def warmup_cuda_call(call):
    """Warm until both the call-count and elapsed-time requirements are met."""
    calls = 0
    start_time = time.perf_counter()
    while calls < WARMUP or (time.perf_counter() - start_time) * 1_000 < WARMUP_MS:
        # Synchronize in small batches so elapsed wall time reflects completed
        # GPU work rather than how quickly Python can enqueue kernels.
        remaining_calls = WARMUP - calls
        batch_calls = min(10, remaining_calls) if remaining_calls > 0 else 10
        for _ in range(batch_calls):
            call()
        calls += batch_calls
        torch.cuda.synchronize()


def time_cuda_batch(call):
    """Return one CUDA-event measurement averaged across the configured calls."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPETITIONS):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000 / REPETITIONS


def benchmark_cuda_calls(calls, seed):
    """Time CUDA calls in randomized, interleaved sample rounds."""
    latencies = {version: [] for version, _, _ in calls}
    rng = random.Random(seed)
    for _ in range(SAMPLES):
        for version, _, call in rng.sample(calls, k=len(calls)):
            latencies[version].append(time_cuda_batch(call))
    return latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "implementations",
        choices=("all", *IMPLEMENTATIONS),
        nargs="+",
        help="one or more implementations to benchmark, or 'all'",
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    if "all" in args.implementations and len(args.implementations) != 1:
        parser.error("'all' cannot be combined with named implementations")
    implementations = tuple(
        IMPLEMENTATIONS.items()
        if args.implementations == ["all"]
        else ((name, IMPLEMENTATIONS[name]) for name in args.implementations)
    )
    sequence_lengths = ", ".join(str(seq_len) for seq_len in args.seq_lens)
    implementation_names = ", ".join(version for version, _ in implementations)

    print("| Parameter | Value |")
    print("| --- | --- |")
    print(f"| Implementations | {implementation_names} |")
    print(f"| Batch size (B) | {BATCH_SIZE} |")
    print(f"| Heads (H) | {NUM_HEADS} |")
    print(f"| Query lengths (M) | {sequence_lengths} |")
    print(f"| K/V lengths (N) | {sequence_lengths} |")
    print(f"| Head dimension (D) | {HEAD_DIM} |")
    print(f"| Minimum warmup calls | {WARMUP} |")
    print(f"| Minimum warmup duration | {WARMUP_MS:.0f} ms |")
    print(f"| Timed calls per sample | {REPETITIONS} |")
    print(f"| Samples | {SAMPLES} |")
    print()
    print("| Version | Dtype | M | N | Median (µs) | Min (µs) | Max (µs) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    for seq_len in args.seq_lens:
        inputs_by_dtype = {}
        calls = []
        for version, forward in implementations:
            dtype = torch.float16 if version in FP16_IMPLEMENTATIONS else torch.float32
            if dtype not in inputs_by_dtype:
                inputs_by_dtype[dtype] = make_inputs(seq_len, dtype)
            inputs = inputs_by_dtype[dtype]
            call = lambda forward=forward, inputs=inputs: forward(*inputs)
            warmup_cuda_call(call)
            calls.append((version, dtype, call))

        latencies = benchmark_cuda_calls(calls, seed=seq_len)

        for version, dtype, _ in calls:
            samples = latencies[version]
            dtype_name = str(dtype).removeprefix("torch.")
            print(
                f"| {version} | {dtype_name} | {seq_len} | {seq_len} | "
                f"{statistics.median(samples):.2f} | {min(samples):.2f} | {max(samples):.2f} |"
            )


if __name__ == "__main__":
    main()
