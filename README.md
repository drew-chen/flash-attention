# FlashAttention Learning Project
TODO: extend to flash attention 2

This project is a small learning scaffold for understanding attention, tiled attention, and how a future CUDA extension can connect into PyTorch.

## Python usage

```python
import flash_attention

out = flash_attention.forward_v0(q, k, v)
```

`flash_attention.forward_v0(...)` is the validated public V0 entrypoint. It checks
that `q`, `k`, and `v` are CUDA `float32`, contiguous, and shaped `[B, H, N, D]`
before dispatching to the CUDA implementation, which assumes that contract.

## Setup

Use a local virtual environment. This repo was validated in a CUDA-capable environment on July 2, 2026 with:

- Python 3.12.13 in `.venv`
- PyTorch `2.12.1+cu130`
- local `nvcc` `13.2`
- NVIDIA GeForce RTX 4080

### Create a virtual environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```

### Install dependencies

```bash
python -m pip install torch numpy jaxtyping pytest
```

The system `python3` on this machine is 3.14, so the repo uses a Python 3.12 virtual environment instead of the global interpreter.

## Verify the environment

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected on this machine:

- PyTorch reports `2.12.1+cu130`
- `torch.version.cuda` reports `13.0`
- `torch.cuda.is_available()` reports `True`

## VS Code

This repo includes committed workspace settings in `.vscode/` for:

- the repo-local Python interpreter at `.venv/bin/python`
- `clangd`-based resolution of PyTorch C++ extension headers such as `torch/extension.h`
- CUDA headers from `/usr/local/cuda/include`

Install these VS Code extensions for this workspace:

- `ms-python.python`
- `ms-python.vscode-pylance`
- `llvm-vs-code-extensions.vscode-clangd`
- `xaver.clang-format`

If VS Code still shows stale diagnostics after opening the repo, reload the window and run `clangd: Restart language server`.

### Reproduce VS Code C++/CUDA diagnostics from the CLI

VS Code's C++/CUDA diagnostics come from `clangd`. Run the same parser/checker
outside the editor with:

```bash
clangd --check=v0/flash_cuda.cu --log=verbose
```

The important line in the output is `All checks completed, 0 errors`. If the CLI
output and VS Code disagree, reload VS Code and run `clangd: Restart language
server`.

The V0 PyTorch/Tensor adapter lives in `v0/flash.cpp`; `v0/flash_cuda.cu` should stay as
raw CUDA kernel/launcher code. That keeps `clangd` diagnostics for `.cu` files
simple and avoids parsing PyTorch's tensor API through CUDA tooling.

## How Python reaches CUDA

`flash_attention.forward_v0(...)` is exposed through a PyTorch C++/CUDA extension:

1. Python calls `flash_attention.forward_v0(q, k, v)`.
2. `v0/flash.cpp` exposes that function with pybind11 and validates the tensors.
3. `v0/flash.cpp` passes raw tensor pointers to the CUDA launcher in `v0/flash_cuda.cu`.
4. The CUDA launcher configures `grid`/`block` dimensions.
5. `flash_forward_v0_kernel<<<grid, block>>>(...)` runs on the GPU.

V0 kernels use `grid.z` to select a `(batch, head)` pair from contiguous
`[B, H, ...]` tensors before operating on their local tile. Shared indexing
utilities live alongside the helper kernels in `v0/flash_cuda_helpers.cuh`.

The helper unit tests build a small test-only extension around reusable global
kernels in `v0/flash_cuda_helpers.cuh`. They cover batched/headed
transpose, scale, softmax, and matmul behavior independently of `forward_v0`.

## Build and test V0

```bash
python setup.py build_ext --inplace
python -m pytest -q tests/v0/test_forward_v0.py
```

`test_forward_v0.py` exercises only the public `flash_attention.forward_v0(...)`
contract. It is intentionally skipped until V0's QK, softmax, and PV kernels are
implemented.

Run the helper unit tests:

```bash
python -m pytest -q tests/v0/test_cuda_helpers.py
```

For CUDA memory checking, run the full test suite through NVIDIA Compute Sanitizer:

```bash
compute-sanitizer --target-processes all python -m pytest -q
```

`setup.py` defaults `TORCH_CUDA_ARCH_LIST` to `8.9`, which means CUDA compute
capability 8.9 / `sm_89`. That targets Ada Lovelace GPUs such as the RTX 4080.
Override it for a different GPU with:

```bash
TORCH_CUDA_ARCH_LIST=<value> python setup.py build_ext --inplace
```

The repository includes a minimal extension scaffold:

- `setup.py`
- `v0/flash.cpp`
- `v0/flash_cuda.cu`
- `v0/flash_cuda_helpers.cuh`
- `tests/v0/test_forward_v0.py`
- `tests/v0/test_cuda_helpers.py`

That CUDA extension path is still intentionally incomplete, so build or runtime failures there should be treated as expected during development rather than environment setup failures.
