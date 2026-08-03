import os
from pathlib import Path
import sys

from setuptools import setup


ROOT = Path(__file__).resolve().parent
DEFAULT_CUDA_ARCH_LIST = "8.9"
VERSION_SUFFIXES = ("0", "1", "2", "3", "4", "4_fp16")

def fail(message):
    raise SystemExit(f"\n[flash-attention setup] {message}\n")


HOST_WARNING_FLAGS = [
    "-std=c++20",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror=return-type",
]

NVCC_WARNING_FLAGS = [
    "-std=c++20",
    "--compiler-options",
    "-Wall,-Wextra,-Wpedantic,-Werror=return-type",
]


def load_torch_build_bits():
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            fail(
                "PyTorch is not installed in the active Python environment.\n"
                f"Python: {sys.executable}\n\n"
                "Install CPU-only PyTorch for the learning path:\n"
                "  python -m pip install torch\n\n"
                "If you want to build the CUDA extension later, install a CUDA-enabled "
                "PyTorch wheel that matches your toolchain before rerunning:\n"
                "  python setup.py build_ext --inplace"
            )
        raise

    return torch, BuildExtension, CUDAExtension, CUDA_HOME


def build_extension_modules():
    torch, _, CUDAExtension, CUDA_HOME = load_torch_build_bits()

    if CUDA_HOME is None:
        fail(
            "CUDA toolkit was not detected by torch.utils.cpp_extension.\n"
            f"Python: {sys.executable}\n"
            f"PyTorch: {torch.__version__}\n\n"
            "This repo has two modes:\n"
                "  1. CPU-only learning path: use torch_flash.naive_attention and skip setup.py\n"
            "  2. CUDA extension path: install a CUDA-enabled PyTorch build and a matching CUDA toolkit"
        )

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_CUDA_ARCH_LIST)

    def make_flash_extension(version):
        version_dir = ROOT / f"src/v{version}"
        return CUDAExtension(
            name=f"flash_attention_v{version}",
            sources=[
                str(version_dir / "flash.cpp"),
                str(version_dir / "flash_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": HOST_WARNING_FLAGS,
                "nvcc": NVCC_WARNING_FLAGS,
            },
        )

    return [make_flash_extension(version) for version in VERSION_SUFFIXES]


_, BuildExtension, _, _ = load_torch_build_bits()

setup(
    name="flash_attention",
    ext_modules=build_extension_modules(),
    cmdclass={"build_ext": BuildExtension},
)
