"""Expose one attention forward pass to NVIDIA Nsight Compute.

`profile.sh` runs this script under `ncu`. The NVTX (NVIDIA Tools
Extension) range names the forward pass so the profiler can select its kernels
and exclude setup and warm-up work.
"""

import argparse

import torch

from benchmark import FP16_IMPLEMENTATIONS, IMPLEMENTATIONS, make_inputs

SEQ_LEN = 2048
PROFILE_IMPLEMENTATIONS = (
    "baseline",
    "sdpa",
    "sdpa-fp16",
    "v1",
    "v2",
    "v3",
    "v4",
    "v4-fp16",
    "v5",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("implementation", choices=PROFILE_IMPLEMENTATIONS)
    implementation = parser.parse_args().implementation

    forward = IMPLEMENTATIONS[implementation]
    dtype = torch.float16 if implementation in FP16_IMPLEMENTATIONS else torch.float32
    torch.manual_seed(0)
    q, k, v = make_inputs(SEQ_LEN, dtype=dtype)

    # Warm up before entering the range selected by the profiler.
    forward(q, k, v)
    torch.cuda.synchronize()

    # profile.sh tells ncu to collect kernels launched in this named NVTX range.
    with torch.cuda.nvtx.range(f"flash_attention.{implementation}"):
        forward(q, k, v)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
