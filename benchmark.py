"""Measure attention latency on the fixed benchmark suite."""

import torch

import flash_attention_v0
import flash_attention_v1
import flash_attention_v2
from src.baseline import forward as baseline_forward

BATCH_SIZE = 4
NUM_HEADS = 12
HEAD_DIM = 64
SEQ_LENS = (512, 1024, 2048)
WARMUP = 25
REPETITIONS = 100

IMPLEMENTATIONS = {
    "baseline": baseline_forward,
    "v0": flash_attention_v0.forward_unchecked,
    "v1": flash_attention_v1.forward_unchecked,
    "v2": flash_attention_v2.forward_unchecked,
}


def make_inputs(seq_len, batch_size=BATCH_SIZE, num_heads=NUM_HEADS):
    q = torch.randn(batch_size, num_heads, seq_len, HEAD_DIM, device="cuda")
    return q, torch.randn_like(q), torch.randn_like(q)


def time_cuda_call(call):
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPETITIONS):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000 / REPETITIONS


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    print("| Version | Sequence length | Latency (µs) |")
    print("| --- | ---: | ---: |")
    for seq_len in SEQ_LENS:
        q, k, v = make_inputs(seq_len)
        for version, forward in IMPLEMENTATIONS.items():
            latency = time_cuda_call(lambda: forward(q, k, v))
            print(f"| {version} | {seq_len} | {latency:.2f} |")


if __name__ == "__main__":
    main()
