"""Collect minimal Nsight Compute data and generate the README roofline chart.

Users do not need to know Nsight Compute metric names. Run with ``--profile``
to refresh every report, or reuse reports in ``/tmp`` without it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
REPORT_DIR = Path("/tmp")
FP32_IMPLEMENTATIONS = (
    "baseline",
    "sdpa",
    "v1",
    "v2",
    "v3",
    "v4",
    "v4-fp16",
)
REFERENCE_IMPLEMENTATIONS = ("baseline", "sdpa")
SDPA_FP16 = "sdpa-fp16"
FUSED_IMPLEMENTATIONS = ("v1", "v2", "v3", "v4", "v4-fp16", "v5")
ALL_PROFILE_IMPLEMENTATIONS = (
    *REFERENCE_IMPLEMENTATIONS,
    SDPA_FP16,
    *FUSED_IMPLEMENTATIONS,
)

BATCH_SIZE = 4
NUM_HEADS = 12
SEQ_LEN = 2048
HEAD_DIM = 64
# These are the only raw counters the custom application-level chart needs.
# Hardware ceilings come from NVIDIA's built-in roofline sections below.
DURATION_METRIC = "gpu__time_duration.sum"
DRAM_BYTES_METRIC = "dram__bytes.sum"
APPLICATION_METRICS = (DURATION_METRIC, DRAM_BYTES_METRIC)
ROOFLINE_SECTIONS = (
    "SpeedOfLight_RooflineChart",
    "SpeedOfLight_HierarchicalTensorRooflineChart",
)

# QK and PV each perform 2 * B * H * M * N * D FLOPs.
ATTENTION_FLOPS = 4 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM


def report_path(implementation: str) -> Path:
    return REPORT_DIR / f"flash_{implementation}_s2048_roofline.ncu-rep"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def common_ncu_options() -> list[str]:
    return ["--cache-control", "all", "--clock-control", "boost"]


def profile_reports(ncu: str, implementations: tuple[str, ...]) -> None:
    for implementation in implementations:
        output = str(report_path(implementation)).removesuffix(".ncu-rep")
        command = [
            ncu,
            "--metrics",
            ",".join(APPLICATION_METRICS),
            "--replay-mode",
            "range" if implementation in REFERENCE_IMPLEMENTATIONS else "kernel",
            *common_ncu_options(),
        ]
        if implementation == "v5":
            for section in ROOFLINE_SECTIONS:
                command.extend(("--section", section))
        command.extend(
            (
                "--nvtx",
                "--nvtx-include",
                f"flash_attention.{implementation}/",
                "--export",
                output,
                "--force-overwrite",
                str(PYTHON_BIN),
                "profile_cuda.py",
                implementation,
            )
        )
        run(command)


def raw_metrics(ncu: str, report: Path) -> dict[str, float | str]:
    if not report.exists():
        raise FileNotFoundError(
            f"missing {report}; run `{PYTHON_BIN} roofline.py --profile` first"
        )
    completed = subprocess.run(
        [
            ncu,
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    if len(rows) != 3:
        raise RuntimeError(f"expected one result in {report}, found {max(0, len(rows) - 2)}")
    metrics: dict[str, float | str] = {}
    for name, value in zip(rows[0], rows[2], strict=True):
        if not value:
            continue
        try:
            metrics[name] = float(value)
        except ValueError:
            metrics[name] = value
    return metrics


def number(metrics: dict[str, float | str], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, float):
        raise KeyError(f"required NCU metric is missing: {name}")
    return value


def application_point(metrics: dict[str, float | str]) -> dict[str, float]:
    duration_s = number(metrics, DURATION_METRIC) * 1e-9
    dram_bytes = number(metrics, DRAM_BYTES_METRIC)
    dram_bytes_per_s = dram_bytes / duration_s
    return {
        "arithmetic_intensity_flop_per_byte": ATTENTION_FLOPS / dram_bytes,
        "performance_tflop_per_s": ATTENTION_FLOPS / duration_s / 1e12,
        "duration_ms": duration_s * 1e3,
        "dram_bandwidth_gb_per_s": dram_bytes_per_s / 1e9,
        "dram_traffic_gb": dram_bytes / 1e9,
    }


def detail_rows(ncu: str, report: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            ncu,
            "--import",
            str(report),
            "--page",
            "details",
            "--section",
            "SpeedOfLight_RooflineChart",
            "--section",
            "SpeedOfLight_HierarchicalTensorRooflineChart",
            "--print-details",
            "all",
            "--print-units",
            "base",
            "--print-fp",
            "--csv",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(completed.stdout)))


def detail_value(rows: list[dict[str, str]], body: str, metric: str) -> float:
    matches = [
        row
        for row in rows
        if row["Body Item Label"] == body and row["Metric Name"] == metric
    ]
    if len(matches) != 1:
        raise KeyError(f"expected one built-in roofline value for {body!r} / {metric!r}")
    return float(matches[0]["Metric Value"])


def hardware_limits(ncu: str, report: Path) -> dict[str, float]:
    # This is the one adapter coupled to NVIDIA's human-readable section schema.
    # Callers and users work only with the semantic values returned below.
    rows = detail_rows(ncu, report)
    fp32_body = "Single Precision Roofline"
    tensor_body = "DRAM Roofline (Src:fp16 Dst:fp32 Sparsity:off)"
    fp32_ops_per_cycle = detail_value(
        rows, fp32_body, "Theoretical Predicated-On FFMA Operations"
    )
    fp32_frequency = detail_value(rows, fp32_body, "SM Frequency")
    tensor_ops_per_cycle = detail_value(rows, tensor_body, "Theoretical Tensor Operations")
    tensor_frequency = detail_value(rows, tensor_body, "SM Frequency")
    dram_bytes_per_cycle = detail_value(
        rows, fp32_body, "Theoretical DRAM Bytes Accessible"
    )
    dram_frequency = detail_value(rows, fp32_body, "DRAM Frequency")
    return {
        "fp32_peak_tflop_per_s": fp32_ops_per_cycle * fp32_frequency / 1e12,
        "fp16_tensor_peak_tflop_per_s": tensor_ops_per_cycle
        * tensor_frequency
        / 1e12,
        "dram_peak_gb_per_s": dram_bytes_per_cycle * dram_frequency / 1e9,
    }


def logspace(start: float, stop: float, count: int) -> list[float]:
    first, last = math.log10(start), math.log10(stop)
    return [10 ** (first + (last - first) * index / (count - 1)) for index in range(count)]


def configure_axes(axis, title: str, ylabel: str, xmax: float, ymax: float) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator

    def format_log(value, _position):
        if value >= 1000:
            return f"{value:,.0f}"
        if value >= 1:
            return f"{value:g}"
        return f"{value:.2g}"

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.1, xmax)
    axis.set_ylim(0.01, ymax)
    axis.set_title(title, pad=16)
    axis.set_xlabel("Arithmetic Intensity [FLOP/byte]", labelpad=10)
    axis.set_ylabel(ylabel, labelpad=10)
    axis.grid(which="major", color="#CBD5E1", linewidth=0.85)
    axis.grid(which="minor", color="#E2E8F0", linewidth=0.45, alpha=0.55)
    axis.xaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.xaxis.set_major_formatter(FuncFormatter(format_log))
    axis.yaxis.set_major_formatter(FuncFormatter(format_log))
    axis.tick_params(axis="both", which="both", colors="#475569")
    for spine in axis.spines.values():
        spine.set_color("#94A3B8")


def plot_roofline(
    output: Path,
    title: str,
    fp32_peak_tflops: float,
    tensor_peak_tflops: float,
    peak_bandwidth: float,
    points: list[tuple[str, float, float, str, str]],
    xmax: float,
    ymax: float,
    ylabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install requirements.txt to generate roofline charts") from error

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 17,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "legend.fontsize": 10,
        }
    )
    figure, axis = plt.subplots(figsize=(12, 6.75), dpi=160)
    x_values = logspace(0.1, xmax, 600)
    fp32_roof = [min(fp32_peak_tflops, x * peak_bandwidth / 1000) for x in x_values]
    tensor_roof = [min(tensor_peak_tflops, x * peak_bandwidth / 1000) for x in x_values]
    axis.plot(
        x_values,
        fp32_roof,
        color="#27364A",
        linewidth=2.8,
        label="FP32 roof",
    )
    axis.plot(
        x_values,
        tensor_roof,
        color="#7C3AED",
        linewidth=2.4,
        linestyle="--",
        label="FP16 Tensor Core roof",
    )
    annotation_offsets = {
        "SDPA FP32": (14, -18),
        "SDPA FP16": (12, 8),
        "V4": (8, 10),
    }
    for label, intensity, performance, color, marker in points:
        axis.scatter(
            intensity,
            performance,
            s=105,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
        axis.annotate(
            label,
            (intensity, performance),
            xytext=annotation_offsets.get(label, (8, 7)),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
        )
    configure_axes(axis, title, ylabel, xmax, ymax)
    from matplotlib.lines import Line2D

    handles, legend_labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], marker="o", linestyle="none", color="#475569", markersize=8),
            Line2D([], [], marker="D", linestyle="none", color="#7C3AED", markersize=8),
        ]
    )
    legend_labels.extend(["Uses FP32 ceiling", "Uses FP16 Tensor Core ceiling"])
    axis.legend(
        handles,
        legend_labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        ncol=2,
    )
    figure.patch.set_facecolor("white")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect fresh Nsight Compute reports before extracting and plotting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT,
        help="directory for JSON and PNG outputs (default: repository root)",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=ALL_PROFILE_IMPLEMENTATIONS,
        default=ALL_PROFILE_IMPLEMENTATIONS,
        metavar="NAME",
        help="with --profile, refresh only these implementations (default: all)",
    )
    args = parser.parse_args()

    ncu = os.environ.get("FLASH_ATTN_NCU_BIN", "/usr/local/cuda/bin/ncu")
    if args.profile:
        profile_reports(ncu, tuple(args.implementations))

    required_reports = (*FP32_IMPLEMENTATIONS, "sdpa-fp16", "v5")
    metrics = {
        name: raw_metrics(ncu, report_path(name))
        for name in dict.fromkeys(required_reports)
    }
    limits = hardware_limits(ncu, report_path("v5"))
    all_implementations = (*FP32_IMPLEMENTATIONS, "sdpa-fp16", "v5")
    application_points = {
        name: application_point(metrics[name]) for name in all_implementations
    }
    data = {
        "shape": {
            "batch": BATCH_SIZE,
            "heads": NUM_HEADS,
            "m": SEQ_LEN,
            "n": SEQ_LEN,
            "d": HEAD_DIM,
        },
        "algorithmic_attention_flops": ATTENTION_FLOPS,
        "application_roofline": {
            "definition": (
                "algorithmic QK+PV FLOPs divided by complete-call duration "
                "and measured DRAM traffic"
            ),
            "hardware_limits": {
                "fp32_peak_tflop_per_s": limits["fp32_peak_tflop_per_s"],
                "fp16_tensor_peak_tflop_per_s": limits[
                    "fp16_tensor_peak_tflop_per_s"
                ],
                "dram_peak_gb_per_s": limits["dram_peak_gb_per_s"],
            },
            "points": application_points,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_output = args.output_dir / "roofline-data.json"
    json_output.write_text(json.dumps(data, indent=2) + "\n")

    colors = {
        "baseline": "#DC2626",
        "sdpa": "#F97316",
        "v1": "#D946EF",
        "v2": "#D4A000",
        "v3": "#16A34A",
        "v4": "#0284C7",
        "v4-fp16": "#0891B2",
        "sdpa-fp16": "#EA580C",
        "v5": "#7C3AED",
    }
    labels = {
        "baseline": "PyTorch",
        "sdpa": "SDPA FP32",
        "v1": "V1",
        "v2": "V2",
        "v3": "V3",
        "v4": "V4",
        "v4-fp16": "V4 FP16",
        "sdpa-fp16": "SDPA FP16",
        "v5": "V5 WMMA",
    }
    plot_points = [
        (
            labels[name],
            application_points[name]["arithmetic_intensity_flop_per_byte"],
            application_points[name]["performance_tflop_per_s"],
            colors[name],
            "D" if name in {"sdpa-fp16", "v5"} else "o",
        )
        for name in all_implementations
    ]
    plot_roofline(
        args.output_dir / "roofline.png",
        "FlashAttention Roofline — RTX 4080",
        limits["fp32_peak_tflop_per_s"],
        limits["fp16_tensor_peak_tflop_per_s"],
        limits["dram_peak_gb_per_s"],
        plot_points,
        5000,
        200,
        "Effective Performance [TFLOP/s]",
    )

    print(f"Wrote {json_output}")
    print(f"Wrote {args.output_dir / 'roofline.png'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
