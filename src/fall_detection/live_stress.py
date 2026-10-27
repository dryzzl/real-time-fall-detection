"""Synthetic multi-person stress benchmark for the live pose-to-state pipeline.

No camera, ONNX weights, trained LSTM, or real fall footage is exercised.
"""
from __future__ import annotations

import argparse
import json
from math import ceil, isfinite, sqrt
from pathlib import Path
import sys
import tracemalloc

from .contracts import FramePacket
from .features import FeatureConfig
from .live import run_live_pipeline
from .pose import Keypoint, PoseDetection


class SyntheticPoseEstimator:
    """Stable upright poses with two-frame occlusions of the first person."""

    def __init__(self, people: int, width: int, height: int) -> None:
        self.people = people
        columns = ceil(sqrt(people))
        rows = ceil(people / columns)
        self.positions = tuple(
            ((index % columns + 0.5) * width / columns,
             (index // columns + 0.5) * height / rows)
            for index in range(people)
        )
        self.box_width = width / columns * 0.24
        self.box_height = height / rows * 0.68

    def estimate(self, frame: FramePacket) -> tuple[PoseDetection, ...]:
        result = []
        for index, (cx, cy) in enumerate(self.positions):
            if index == 0 and frame.index % 19 in (8, 9):
                continue
            half_w, half_h = self.box_width / 2, self.box_height / 2
            points = tuple(
                Keypoint(cx, cy - half_h * 0.5 + key * half_h / 17, 0.95)
                for key in range(17)
            )
            result.append(PoseDetection(
                (cx - half_w, cy - half_h, cx + half_w, cy + half_h),
                0.95, points,
            ))
        return tuple(result)


def run_live_stress(
    *, frames: int = 300, people: int = 3, width: int = 160,
    height: int = 120, fps: float = 30.0,
    max_peak_memory_mib: float | None = None,
) -> dict[str, object]:
    """Exercise live tracking, bounded features and state decisions, without ML."""
    if type(frames) is not int or not 1 <= frames <= 1_000_000:
        raise ValueError("frames must be an integer between 1 and 1000000")
    if type(people) is not int or not 1 <= people <= 16:
        raise ValueError("people must be an integer between 1 and 16")
    if any(type(value) is not int or value < 24 * ceil(sqrt(people))
           for value in (width, height)):
        raise ValueError("frame dimensions must each be >= 24 * ceil(sqrt(people))")
    if type(fps) not in (int, float) or not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if max_peak_memory_mib is not None and (
        type(max_peak_memory_mib) not in (int, float)
        or not isfinite(max_peak_memory_mib) or max_peak_memory_mib <= 0
    ):
        raise ValueError("max_peak_memory_mib must be finite and positive")

    estimator = SyntheticPoseEstimator(people, width, height)
    feature_config = FeatureConfig(window_size=30, min_observed_samples=8)
    records = 0
    missing_observations = 0
    max_window_samples = 0
    malformed_records = 0
    unsupported_predictions = 0
    unexpected_track_ids = 0
    raw_data_leaks = 0

    def consume(record: dict) -> None:
        nonlocal records, missing_observations, max_window_samples
        nonlocal malformed_records, unsupported_predictions, unexpected_track_ids, raw_data_leaks
        if record.get("event") != "frame" or record.get("frame_index") != records:
            malformed_records += 1
        if (record.get("state") != "unknown"
                or record.get("fall_detection_available") is not False
                or record.get("confidence") is not None):
            unsupported_predictions += 1
        observed = record.get("people", [])
        ids = [person["track_id"] for person in observed]
        if (record.get("tracked_person_count") != people
                or len(ids) != len(set(ids))):
            malformed_records += 1
        if set(ids) != set(range(1, people + 1)):
            unexpected_track_ids += 1
        for person in observed:
            max_window_samples = max(max_window_samples, person["window_samples"])
            if not person["observed"]:
                missing_observations += 1
                if person["state"] != "unknown":
                    unsupported_predictions += 1
        if "bgr" in record or any("keypoints" in person for person in observed):
            raw_data_leaks += 1
        records += 1

    def packets():
        for index in range(frames):
            yield FramePacket(
                index=index, timestamp_seconds=index / fps,
                width=width, height=height,
                bgr=bytes(width * height * 3),
                source="synthetic_live_stress",
            )

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        summary = run_live_pipeline(
            packets(), consume, estimator=estimator, feature_config=feature_config
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    peak_mib = peak_bytes / (1024 * 1024)
    checks = {
        "all_frames_processed": records == frames == summary.frames_processed,
        "ordered_complete_records": malformed_records == 0,
        "stable_person_ids": unexpected_track_ids == 0,
        "bounded_temporal_windows": max_window_samples <= feature_config.window_size,
        "no_unverified_fall_predictions": unsupported_predictions == 0,
        "no_raw_pose_or_pixel_data": raw_data_leaks == 0,
    }
    if frames >= 10:
        checks["occlusions_exercised"] = missing_observations > 0
    if max_peak_memory_mib is not None:
        checks["peak_python_memory_within_limit"] = peak_mib <= max_peak_memory_mib

    return {
        "event": "synthetic_live_pose_stress",
        "synthetic": True,
        "benchmark_scope": "fake_pose_tracking_features_heuristic_states_only",
        "fall_detection_available": False,
        "frames_requested": frames,
        "frames_processed": summary.frames_processed,
        "people": people,
        "missing_observations": missing_observations,
        "max_window_samples": max_window_samples,
        "bytes_streamed": frames * width * height * 3,
        "elapsed_seconds": round(summary.elapsed_seconds, 6),
        "throughput_fps": round(summary.throughput_fps, 3),
        "peak_traced_python_memory_mib": round(peak_mib, 4),
        "memory_scope": "tracemalloc_python_allocations_only",
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic live-pipeline stress test; NOT model accuracy or camera latency."
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--people", type=int, default=3)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-peak-memory-mib", type=float)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_live_stress(
            frames=args.frames, people=args.people,
            width=args.width, height=args.height, fps=args.fps,
            max_peak_memory_mib=args.max_peak_memory_mib,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(report, allow_nan=False))
        return 0 if report["passed"] else 1
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
