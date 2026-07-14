"""CUDA-event benchmark for the baseline and CUDA implementations.

For each requested shape, the benchmark creates one Q/K/V input set and gives
that identical set to every selected implementation. Each implementation first
runs warm-up rounds, which are excluded from the result and let CUDA setup and
caches settle. It then uses CUDA events to time a batch of repeated calls,
waits for the end event, and divides total GPU time by the repetition count.
The output is therefore mean GPU latency per call, plus an approximate effective
throughput for the two attention matrix multiplications.

Run from the repository root after building the extension:
    python benchmark/benchmark.py
"""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import Callable, Final

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.baseline import forward as baseline_forward
from src.tensor_types import AttentionTensor
import flash_attention

AttentionForward = Callable[[AttentionTensor, AttentionTensor, AttentionTensor], AttentionTensor]


class Implementation(str, Enum):
    BASELINE = "baseline"
    V0 = "v0"


IMPLEMENTATION_FUNCTIONS: Final[dict[Implementation, AttentionForward]] = {
    Implementation.BASELINE: baseline_forward,
    Implementation.V0: flash_attention.forward_v0_unchecked,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--implementations",
        type=Implementation,
        choices=tuple(implementation.value for implementation in Implementation),
        nargs="+",
        default=list(Implementation),
        help="Implementations to benchmark (default: baseline v0).",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Sequence lengths to benchmark (default: 512 1024 2048).",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = (args.batch_size, args.num_heads, args.head_dim, *args.seq_lens)
    if any(value <= 0 for value in values):
        raise ValueError("All dimensions must be positive.")
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("--warmup must be non-negative and --repetitions must be positive.")


def get_implementation(implementation: Implementation | str) -> AttentionForward:
    """Return an implementation function from the registry."""
    try:
        return IMPLEMENTATION_FUNCTIONS[Implementation(implementation)]
    except ValueError as exc:
        raise ValueError(f"Unknown implementation: {implementation}") from exc


def time_cuda_call(call: Callable[[], AttentionTensor], warmup: int, repetitions: int) -> float:
    """Return mean GPU elapsed time in microseconds for one call."""
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


def attention_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> int:
    # QK^T and PV each cost approximately 2 * B * H * S^2 * D FLOPs.
    return 4 * batch_size * num_heads * seq_len * seq_len * head_dim


def format_markdown_table(rows: list[tuple[str, str, str, str]]) -> str:
    """Return a padded Markdown table that is aligned in a monospaced terminal."""
    headers = ("Implementation", "Sequence length", "Latency (us)", "Effective TFLOP/s")
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    alignments = (
        ":" + "-" * (widths[0] - 1),
        "-" * (widths[1] - 1) + ":",
        "-" * (widths[2] - 1) + ":",
        "-" * (widths[3] - 1) + ":",
    )

    def row_line(values: tuple[str, str, str, str]) -> str:
        return (
            f"| {values[0]:<{widths[0]}} | {values[1]:>{widths[1]}} | "
            f"{values[2]:>{widths[2]}} | {values[3]:>{widths[3]}} |"
        )

    return "\n".join((row_line(headers), row_line(alignments), *(row_line(row) for row in rows)))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")
    implementations = [
        (implementation, get_implementation(implementation)) for implementation in args.implementations
    ]

    print(
        f"GPU timing: B={args.batch_size}, H={args.num_heads}, "
        f"D={args.head_dim}, dtype=float32, warmup={args.warmup}, "
        f"repetitions={args.repetitions}"
    )
    results = []

    for seq_len in args.seq_lens:
        q = torch.randn(args.batch_size, args.num_heads, seq_len, args.head_dim, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for implementation, forward in implementations:
            latency_us = time_cuda_call(
                lambda: forward(q, k, v), args.warmup, args.repetitions
            )
            tflops = attention_flops(args.batch_size, args.num_heads, seq_len, args.head_dim)
            tflops /= latency_us * 1_000_000
            results.append((implementation.value, str(seq_len), f"{latency_us:.2f}", f"{tflops:.3f}"))

    print(format_markdown_table(results))


if __name__ == "__main__":
    main()
