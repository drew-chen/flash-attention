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
from src.baseline import forward as baseline_forward

BATCH_SIZE = 4
NUM_HEADS = 12
HEAD_DIM = 64
SEQ_LENS = (512, 1024, 2048)
WARMUP = 25
WARMUP_MS = 500.0
REPETITIONS = 50
SAMPLES = 7


def sdpa_forward(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


IMPLEMENTATIONS = {
    "baseline": baseline_forward,
    "sdpa": sdpa_forward,
    "sdpa-fp16": sdpa_forward,
    "v0": flash_attention_v0.forward_unchecked,
    "v1": flash_attention_v1.forward_unchecked,
    "v2": flash_attention_v2.forward_unchecked,
    "v3": flash_attention_v3.forward_unchecked,
    "v4": flash_attention_v4.forward_unchecked,
    "v4-fp16": flash_attention_v4_fp16.forward_unchecked,
    "v5": flash_attention_v5.forward_unchecked,
    "v6": flash_attention_v6.forward_unchecked,
}
FP16_IMPLEMENTATIONS = (
    "sdpa-fp16",
    "v4-fp16",
    "v5",
    "v6",
)


def make_inputs(
    seq_len,
    batch_size=BATCH_SIZE,
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    dtype=torch.float32,
):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    return q, torch.randn_like(q), torch.randn_like(q)


def warmup_cuda_call(call, minimum_calls=WARMUP, minimum_duration_ms=WARMUP_MS):
    """Warm until both the call-count and elapsed-time requirements are met."""
    calls = 0
    start_time = time.perf_counter()
    while calls < minimum_calls or (time.perf_counter() - start_time) * 1_000 < minimum_duration_ms:
        # Synchronize in small batches so elapsed wall time reflects completed
        # GPU work rather than how quickly Python can enqueue kernels.
        remaining_calls = minimum_calls - calls
        batch_calls = min(10, remaining_calls) if remaining_calls > 0 else 10
        for _ in range(batch_calls):
            call()
        calls += batch_calls
        torch.cuda.synchronize()


def time_cuda_batch(call, repetitions=REPETITIONS):
    """Return one CUDA-event measurement averaged across `repetitions` calls."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000 / repetitions


def benchmark_cuda_calls(calls, repetitions=REPETITIONS, samples=SAMPLES, seed=0):
    """Time CUDA calls in randomized, interleaved sample rounds."""
    latencies = {version: [] for version, _, _ in calls}
    rng = random.Random(seed)
    for _ in range(samples):
        for version, _, call in rng.sample(calls, k=len(calls)):
            latencies[version].append(time_cuda_batch(call, repetitions))
    return latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "implementations",
        choices=("all", *IMPLEMENTATIONS),
        nargs="+",
        help="one or more implementations to benchmark, or 'all'",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--warmup-ms", type=float, default=WARMUP_MS)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    if "all" in args.implementations and len(args.implementations) != 1:
        parser.error("'all' cannot be combined with named implementations")
    if args.warmup < 0 or args.warmup_ms < 0:
        parser.error("warmup counts and durations must be non-negative")
    if args.repetitions <= 0 or args.samples <= 0:
        parser.error("repetitions and samples must be positive")

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
    print(f"| Batch size (B) | {args.batch_size} |")
    print(f"| Heads (H) | {args.num_heads} |")
    print(f"| Query lengths (M) | {sequence_lengths} |")
    print(f"| K/V lengths (N) | {sequence_lengths} |")
    print(f"| Head dimension (D) | {args.head_dim} |")
    print(f"| Minimum warmup calls | {args.warmup} |")
    print(f"| Minimum warmup duration | {args.warmup_ms:.0f} ms |")
    print(f"| Timed calls per sample | {args.repetitions} |")
    print(f"| Samples | {args.samples} |")
    print()
    print("| Version | Dtype | M | N | Median (µs) | Min (µs) | Max (µs) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    for seq_len in args.seq_lens:
        inputs_by_dtype = {}
        calls = []
        for version, forward in implementations:
            dtype = torch.float16 if version in FP16_IMPLEMENTATIONS else torch.float32
            if dtype not in inputs_by_dtype:
                inputs_by_dtype[dtype] = make_inputs(
                    seq_len,
                    batch_size=args.batch_size,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                )
            inputs = inputs_by_dtype[dtype]
            call = lambda forward=forward, inputs=inputs: forward(*inputs)
            warmup_cuda_call(
                call,
                minimum_calls=args.warmup,
                minimum_duration_ms=args.warmup_ms,
            )
            calls.append((version, dtype, call))

        latencies = benchmark_cuda_calls(
            calls,
            repetitions=args.repetitions,
            samples=args.samples,
            seed=seq_len,
        )

        for version, dtype, _ in calls:
            samples = latencies[version]
            dtype_name = str(dtype).removeprefix("torch.")
            print(
                f"| {version} | {dtype_name} | {seq_len} | {seq_len} | "
                f"{statistics.median(samples):.2f} | {min(samples):.2f} | {max(samples):.2f} |"
            )


if __name__ == "__main__":
    main()
