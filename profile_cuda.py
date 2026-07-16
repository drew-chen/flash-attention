"""Expose one attention forward pass as an NVTX range."""

import argparse

import torch

from benchmark import IMPLEMENTATIONS, make_inputs

SEQ_LEN = 2048


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("implementation", choices=IMPLEMENTATIONS)
    implementation = parser.parse_args().implementation

    forward = IMPLEMENTATIONS[implementation]
    q, k, v = make_inputs(SEQ_LEN)
    forward(q, k, v)
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range(f"flash_attention.{implementation}"):
        forward(q, k, v)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
