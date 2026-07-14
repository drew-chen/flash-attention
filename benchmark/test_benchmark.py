from argparse import Namespace

import pytest
import torch

from benchmark import benchmark


def benchmark_args(**overrides):
    values = {
        "batch_size": 1,
        "num_heads": 1,
        "head_dim": 64,
        "seq_lens": [128],
        "warmup": 0,
        "repetitions": 1,
        "implementations": [benchmark.Implementation.BASELINE],
    }
    values.update(overrides)
    return Namespace(**values)


def test_baseline_is_selectable():
    forward = benchmark.get_implementation(benchmark.Implementation.BASELINE)
    q = torch.randn(1, 1, 4, 8)
    output = forward(q, q, q)

    assert output.shape == q.shape


def test_benchmark_accepts_non_tile_dimensions_for_v0():
    benchmark.validate_args(
        benchmark_args(
            head_dim=15,
            seq_lens=[17],
            implementations=[benchmark.Implementation.BASELINE, benchmark.Implementation.V0],
        )
    )


def test_unknown_implementation_is_rejected():
    with pytest.raises(ValueError, match="Unknown implementation"):
        benchmark.get_implementation("v999")


def test_implementation_enum_is_a_string_and_has_a_registered_function():
    assert benchmark.Implementation.BASELINE == "baseline"
    assert benchmark.IMPLEMENTATION_FUNCTIONS[benchmark.Implementation.BASELINE] is benchmark.get_implementation(
        "baseline"
    )


def test_markdown_table_has_even_terminal_spacing():
    table = benchmark.format_markdown_table(
        [("baseline", "256", "45.08", "5.955"), ("v0", "1024", "4291.41", "1.001")]
    )
    lines = table.splitlines()

    assert all(len(line) == len(lines[0]) for line in lines)
    assert "| baseline       |             256 |" in table
