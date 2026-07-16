"""Measure attention latency on the fixed benchmark suite."""

import torch

import flash_attention
from src.baseline import forward as baseline_forward

BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64
SEQ_LENS = (512, 1024, 2048)
WARMUP = 25
REPETITIONS = 100

IMPLEMENTATIONS = {
    "baseline": baseline_forward,
    "v0": flash_attention.forward_v0_unchecked,
}


def make_inputs(seq_len):
    q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM, device="cuda")
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
