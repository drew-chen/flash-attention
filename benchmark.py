"""Measure attention latency on the fixed benchmark suite."""

import argparse

import torch

import flash_attention_v0
import flash_attention_v1
import flash_attention_v2
import flash_attention_v3
import flash_attention_v4
import flash_attention_v4_fp16
import flash_attention_v5
from src.baseline import forward as baseline_forward

BATCH_SIZE = 4
NUM_HEADS = 12
HEAD_DIM = 64
SEQ_LENS = (512, 1024, 2048)
WARMUP = 25
REPETITIONS = 100


def sdpa_forward(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


IMPLEMENTATIONS = {
    "baseline": baseline_forward,
    "sdpa": sdpa_forward,
    "v0": flash_attention_v0.forward_unchecked,
    "v1": flash_attention_v1.forward_unchecked,
    "v2": flash_attention_v2.forward_unchecked,
    "v3": flash_attention_v3.forward_unchecked,
    "v4": flash_attention_v4.forward_unchecked,
    "v4-fp16": flash_attention_v4_fp16.forward_unchecked,
    "v5": flash_attention_v5.forward_unchecked,
}


def make_inputs(
    seq_len,
    batch_size=BATCH_SIZE,
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    dtype=torch.float32,
):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    return q, torch.randn_like(q), torch.randn_like(q)


def time_cuda_call(call, warmup=WARMUP, repetitions=REPETITIONS):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000 / repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "implementation",
        choices=("all", *IMPLEMENTATIONS),
        help="implementation to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    implementations = tuple(
        IMPLEMENTATIONS.items()
        if args.implementation == "all"
        else ((args.implementation, IMPLEMENTATIONS[args.implementation]),)
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
    print(f"| Warmups | {args.warmup} |")
    print(f"| Timed calls | {args.repetitions} |")
    print()
    print("| Version | Dtype | M | N | Latency (µs) |")
    print("| --- | --- | ---: | ---: | ---: |")

    for seq_len in args.seq_lens:
        for version, forward in implementations:
            dtype = torch.float16 if version in ("v4-fp16", "v5") else torch.float32
            q, k, v = make_inputs(
                seq_len,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                dtype=dtype,
            )
            latency = time_cuda_call(
                lambda: forward(q, k, v),
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            dtype_name = str(dtype).removeprefix("torch.")
            print(f"| {version} | {dtype_name} | {seq_len} | {seq_len} | {latency:.2f} |")


if __name__ == "__main__":
    main()
