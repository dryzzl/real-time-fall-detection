"""Dependency-free streaming stress harness.

This measures the synthetic frame/pipeline/JSON-record path, NOT fall-model
inference, detection accuracy, camera latency, or end-to-end alert latency.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys
import tracemalloc

from .config import RuntimeConfig
from .pipeline import run_pipeline
from .sources import synthetic_frames


def _positive_limit(value: float | None, name: str) -> None:
    if value is not None and (
        type(value) not in (int, float) or not isfinite(value) or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def run_stress(
    config: RuntimeConfig,
    *,
    max_peak_memory_mib: float | None = None,
    min_throughput_fps: float | None = None,
) -> dict[str, object]:
    """Exercise a bounded synthetic stream without retaining frame records.

    Memory is the peak of Python allocations tracked by tracemalloc. It does
    not include native decoder/ONNX/GPU allocations or process RSS.
    """
    if not isinstance(config, RuntimeConfig):
        raise ValueError("config must be a RuntimeConfig")
    _positive_limit(max_peak_memory_mib, "max_peak_memory_mib")
    _positive_limit(min_throughput_fps, "min_throughput_fps")

    frame_count = 0
    unexpected_predictions = 0
    malformed_records = 0

    def consume(record: dict) -> None:
        nonlocal frame_count, unexpected_predictions, malformed_records
        if record.get("event") != "frame" or record.get("frame_index") != frame_count:
            malformed_records += 1
        if (
            record.get("state") != "unknown"
            or record.get("confidence") is not None
            or record.get("reason") != "detection_not_implemented"
        ):
            unexpected_predictions += 1
        frame_count += 1

    # Do not interrupt another tool's existing allocation tracing.
    tracing_was_active = tracemalloc.is_tracing()
    if not tracing_was_active:
        tracemalloc.start()
    try:
        summary = run_pipeline(synthetic_frames(config), consume)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not tracing_was_active:
            tracemalloc.stop()

    peak_mib = peak_bytes / (1024 * 1024)
    checks = {
        "all_frames_processed": summary.frames_processed == config.frames == frame_count,
        "ordered_records": malformed_records == 0,
        "no_unverified_fall_predictions": unexpected_predictions == 0,
    }
    if max_peak_memory_mib is not None:
        checks["peak_python_memory_within_limit"] = peak_mib <= max_peak_memory_mib
    if min_throughput_fps is not None:
        checks["throughput_meets_minimum"] = summary.throughput_fps >= min_throughput_fps

    return {
        "event": "synthetic_stream_stress",
        "synthetic": True,
        "benchmark_scope": "frame_generation_and_pipeline_without_pose_or_lstm",
        "fall_detection_available": False,
        "frames_requested": config.frames,
        "frames_processed": summary.frames_processed,
        "frame_size_bytes": config.width * config.height * 3,
        "bytes_streamed": summary.frames_processed * config.width * config.height * 3,
        "elapsed_seconds": round(summary.elapsed_seconds, 6),
        "throughput_fps": round(summary.throughput_fps, 3),
        "peak_traced_python_memory_mib": round(peak_mib, 4),
        "memory_scope": "tracemalloc_python_allocations_only",
        "tracemalloc_already_active": tracing_was_active,
        "unexpected_predictions": unexpected_predictions,
        "malformed_records": malformed_records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stress-test synthetic streaming only (not the fall detector)."
    )
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-peak-memory-mib", type=float)
    parser.add_argument("--min-throughput-fps", type=float)
    parser.add_argument("--report", type=Path, help="Write JSON report to a new file only")
    args = parser.parse_args(argv)
    try:
        config = RuntimeConfig(
            frames=args.frames, width=args.width, height=args.height, fps=args.fps
        )
        report = run_stress(
            config,
            max_peak_memory_mib=args.max_peak_memory_mib,
            min_throughput_fps=args.min_throughput_fps,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
        print(json.dumps(report, allow_nan=False))
        return 0 if report["passed"] else 1
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
