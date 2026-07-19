from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
TILE_SIZE = 16

def assert_close_with_values(actual, expected, **kwargs):
    try:
        torch.testing.assert_close(actual, expected, **kwargs)
    except AssertionError as exc:
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()
        raise AssertionError(
            f"{exc}\n\nactual:\n{actual_cpu}\n\nexpected:\n{expected_cpu}"
        ) from exc


@pytest.fixture(scope="session")
def cuda_helpers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    return load(
        name="flash_attention_cuda_helper_tests",
        sources=[
            str(ROOT / "cuda/cuda_helpers_test_ext.cpp"),
            str(ROOT / "cuda/cuda_helpers_test_ext.cu"),
        ],
        extra_cflags=["-std=c++20"],
        extra_cuda_cflags=["-std=c++20"],
        verbose=False,
    )


@pytest.fixture
def cuda_rng():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    return torch.Generator(device="cuda").manual_seed(1234)


@pytest.mark.parametrize(
    "batch_size,num_heads,height,width",
    [
        pytest.param(1, 1, TILE_SIZE, TILE_SIZE, id="single-batch-single-head"),
        pytest.param(1, 2, 16, 32, id="single-batch-multiple-heads"),
        pytest.param(2, 1, 32, 16, id="multiple-batches-single-head"),
        pytest.param(2, 3, 32, 32, id="multiple-batches-multiple-heads"),
        pytest.param(3, 2, 64, 48, id="rectangular-multiple-batches-and-heads"),
        pytest.param(2, 3, 15, 16, id="partial-height"),
        pytest.param(2, 3, 16, 15, id="partial-width"),
        pytest.param(2, 3, 17, 31, id="partial-height-and-width"),
    ],
)
def test_transpose(cuda_helpers, batch_size, num_heads, height, width):
    x = torch.arange(batch_size * num_heads * height * width, device="cuda", dtype=torch.float32).reshape(
        batch_size, num_heads, height, width
    )

    actual = cuda_helpers.transpose_cuda(x)
    expected = x.transpose(-1, -2).contiguous()

    assert_close_with_values(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch_size,num_heads,height,width",
    [
        pytest.param(1, 1, 1, 1, id="single-element"),
        pytest.param(1, 2, 16, 32, id="multiple-heads-wide"),
        pytest.param(2, 1, 32, 16, id="multiple-batches-tall"),
        pytest.param(2, 3, 31, 19, id="multiple-batches-and-heads"),
    ],
)
def test_scale(cuda_helpers, cuda_rng, batch_size, num_heads, height, width):
    x = torch.randn((batch_size, num_heads, height, width), device="cuda", dtype=torch.float32,
                    generator=cuda_rng)

    actual = cuda_helpers.scale_cuda(x, 0.5)
    expected = x * 0.5

    assert_close_with_values(actual, expected, rtol=0, atol=0)

@pytest.mark.parametrize("batch_size,num_heads,rows,width", [
    pytest.param(1, 1, 5, 16, id="single-batch-single-head"),
    pytest.param(2, 3, 5, 16, id="multiple-batches-and-heads"),
    pytest.param(2, 3, 5, 17, id="partial-width"),
    pytest.param(1, 1, 5, 4096, id="row-wider-than-a-block"),
])
def test_softmax(cuda_helpers, cuda_rng, batch_size, num_heads, rows, width):
    x = torch.cat(
        [
            torch.randn((batch_size, num_heads, rows - 1, width), device="cuda", dtype=torch.float32,
                        generator=cuda_rng),
            torch.arange(width, device="cuda", dtype=torch.float32)
            .reshape(1, 1, 1, width)
            .expand(batch_size, num_heads, 1, width),
        ],
        dim=2,
    )

    actual = cuda_helpers.softmax_cuda(x)
    expected = torch.softmax(x, dim=-1)

    assert_close_with_values(actual, expected, rtol=1e-6, atol=1e-6)
    assert_close_with_values(actual.sum(dim=-1), torch.ones_like(actual[..., 0]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,num_heads,left_shape,right_shape",
    [
        pytest.param(1, 1, (1, 1), (1, 1), id="single-element"),
        pytest.param(1, 2, (1, 17), (17, 1), id="multiple-heads-thin-oob"),
        pytest.param(2, 1, (15, 17), (17, 19), id="multiple-batches-partial-output"),
        pytest.param(2, 3, (17, 31), (31, 19), id="multiple-batches-and-heads-uneven"),
    ],
)
def test_matmul(cuda_helpers, cuda_rng, batch_size, num_heads, left_shape, right_shape):
    left = torch.randn(
        (batch_size, num_heads, *left_shape),
        device="cuda",
        dtype=torch.float32,
        generator=cuda_rng,
    )
    right = torch.randn(
        (batch_size, num_heads, *right_shape),
        device="cuda",
        dtype=torch.float32,
        generator=cuda_rng,
    )
    actual = cuda_helpers.matmul_cuda(left, right)
    expected = torch.matmul(left.float(), right.float())

    assert actual.dtype == torch.float32
    assert expected.dtype == torch.float32

    assert_close_with_values(actual, expected, rtol=1e-6, atol=1e-5)
