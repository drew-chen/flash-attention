"""Compare CUDA attention implementations with an FP32 PyTorch reference."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch

import flash_attention_v4
import flash_attention_v4_fp16
import flash_attention_v5
from src.baseline import forward as reference_forward


BATCH_SIZE = 4
NUM_HEADS = 12
QUERY_SEQ_LEN = 2048
KV_SEQ_LEN = 2048
HEAD_DIM = 64
SEED = 0


@dataclass(frozen=True)
class Implementation:
    forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    input_dtype: torch.dtype


IMPLEMENTATIONS = {
    "sdpa": Implementation(
        torch.nn.functional.scaled_dot_product_attention, torch.float32
    ),
    "sdpa-fp16": Implementation(
        torch.nn.functional.scaled_dot_product_attention, torch.float16
    ),
    "v4": Implementation(flash_attention_v4.forward_unchecked, torch.float32),
    "v4-fp16": Implementation(flash_attention_v4_fp16.forward_unchecked, torch.float16),
    "v5": Implementation(flash_attention_v5.forward_unchecked, torch.float16),
}


def calculate_metrics(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, ...]:
    """Return max absolute error, mean absolute error, RMSE, and relative L2 error."""
    error = output.float() - reference
    absolute_error = error.abs()
    reference_l2 = torch.linalg.vector_norm(reference)
    # Relative L2 is scale-aware without dividing by individual reference
    # values, where values near zero would make pointwise relative error explode.
    relative_l2 = torch.linalg.vector_norm(error) / reference_l2
    return (
        absolute_error.max().item(),
        absolute_error.mean().item(),
        error.square().mean().sqrt().item(),
        relative_l2.item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("implementations", choices=IMPLEMENTATIONS, nargs="*")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--query-seq-len", type=int, default=QUERY_SEQ_LEN)
    parser.add_argument("--kv-seq-len", type=int, default=KV_SEQ_LEN)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    q_shape = (
        args.batch_size,
        args.num_heads,
        args.query_seq_len,
        args.head_dim,
    )
    kv_shape = (
        args.batch_size,
        args.num_heads,
        args.kv_seq_len,
        args.head_dim,
    )

    # Generate FP16 values first, then promote those exact values for FP32 runs.
    # This measures kernel/output precision without including initial input quantization.
    q_fp16 = torch.randn(q_shape, device="cuda", dtype=torch.float16)
    k_fp16 = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    v_fp16 = torch.randn_like(k_fp16)
    q_fp32, k_fp32, v_fp32 = q_fp16.float(), k_fp16.float(), v_fp16.float()

    with torch.inference_mode():
        reference = reference_forward(q_fp32, k_fp32, v_fp32)

        print("| Implementation | Input | Output | Max abs | Mean abs | RMSE | Relative L2 |")
        print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        implementation_names = args.implementations or IMPLEMENTATIONS
        for name in implementation_names:
            implementation = IMPLEMENTATIONS[name]
            inputs = (
                (q_fp16, k_fp16, v_fp16)
                if implementation.input_dtype == torch.float16
                else (q_fp32, k_fp32, v_fp32)
            )
            output = implementation.forward(*inputs)
            max_abs, mean_abs, rmse, relative_l2 = calculate_metrics(output, reference)
            input_dtype = str(implementation.input_dtype).removeprefix("torch.")
            output_dtype = str(output.dtype).removeprefix("torch.")
            print(
                f"| {name} | {input_dtype} | {output_dtype} | {max_abs:.3e} | "
                f"{mean_abs:.3e} | {rmse:.3e} | {relative_l2:.3e} |"
            )


if __name__ == "__main__":
    main()
