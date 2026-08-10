"""Compare CUDA attention implementations with an FP32 PyTorch reference."""

import argparse

import torch

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
from src.baseline import forward as reference_forward


IMPLEMENTATIONS = {
    "sdpa": torch.nn.functional.scaled_dot_product_attention,
    "sdpa-fp16": torch.nn.functional.scaled_dot_product_attention,
    "v4": flash_attention_v4.forward_unchecked,
    "v4-fp16": flash_attention_v4_fp16.forward_unchecked,
    "v5": flash_attention_v5.forward_unchecked,
    "v6": flash_attention_v6.forward_unchecked,
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    shape = (BATCH_SIZE, NUM_HEADS, PRIMARY_SEQ_LEN, HEAD_DIM)

    # Generate FP16 values first, then promote those exact values for FP32 runs.
    # This measures kernel/output precision without including initial input quantization.
    q_fp16 = torch.randn(shape, device="cuda", dtype=torch.float16)
    k_fp16 = torch.randn_like(q_fp16)
    v_fp16 = torch.randn_like(q_fp16)
    q_fp32, k_fp32, v_fp32 = q_fp16.float(), k_fp16.float(), v_fp16.float()

    with torch.inference_mode():
        reference = reference_forward(q_fp32, k_fp32, v_fp32)

        print("| Implementation | Input | Output | Max abs | Mean abs | RMSE | Relative L2 |")
        print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        implementation_names = args.implementations or IMPLEMENTATIONS
        for name in implementation_names:
            input_dtype = (
                torch.float16 if name in FP16_IMPLEMENTATIONS else torch.float32
            )
            inputs = (
                (q_fp16, k_fp16, v_fp16)
                if input_dtype == torch.float16
                else (q_fp32, k_fp32, v_fp32)
            )
            output = IMPLEMENTATIONS[name](*inputs)
            max_abs, mean_abs, rmse, relative_l2 = calculate_metrics(output, reference)
            input_dtype_name = str(input_dtype).removeprefix("torch.")
            output_dtype = str(output.dtype).removeprefix("torch.")
            print(
                f"| {name} | {input_dtype_name} | {output_dtype} | {max_abs:.3e} | "
                f"{mean_abs:.3e} | {rmse:.3e} | {relative_l2:.3e} |"
            )


if __name__ == "__main__":
    main()
