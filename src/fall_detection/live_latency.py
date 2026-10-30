"""Bounded per-frame latency profiling for the streaming software pipeline.

Measurements exclude frame acquisition, UI preview, and alert delivery. They
are diagnostics, not camera-to-alert latency or real-world accuracy evidence.
"""
from __future__ import annotations

from collections import deque
from math import ceil, isfinite
from typing import Mapping

STAGES = ("pose_tracking_features_ms", "classification_ms", "record_emit_ms", "total_ms")


class FrameLatencyCollector:
    """Keep a bounded window of recent stage timings in milliseconds."""

    def __init__(self, max_samples: int = 2048) -> None:
        if type(max_samples) is not int or not 1 <= max_samples <= 100_000:
            raise ValueError("max_samples must be an integer between 1 and 100000")
        self.max_samples = max_samples
        self.frames_seen = 0
        self._samples: deque[dict[str, float]] = deque(maxlen=max_samples)

    def record(self, timings: Mapping[str, float]) -> None:
        if set(timings) != set(STAGES):
            raise ValueError("latency sample has missing or unexpected stages")
        sample = {}
        for stage in STAGES:
            value = timings[stage]
            if type(value) not in (float, int) or not isfinite(value) or value < 0:
                raise ValueError(f"{stage} must be a finite nonnegative number")
            sample[stage] = float(value)
        component_sum = sum(sample[key] for key in STAGES[:-1])
        if abs(component_sum - sample["total_ms"]) > max(1e-6, sample["total_ms"] * 1e-6):
            raise ValueError("latency stages must sum to total_ms")
        self._samples.append(sample)
        self.frames_seen += 1

    def report(self) -> dict[str, object]:
        """Nearest-rank percentiles of the latest max_samples frames only."""
        stats: dict[str, object] = {}
        for stage in STAGES:
            values = sorted(sample[stage] for sample in self._samples)
            if not values:
                stats[stage] = None
                continue
            n = len(values)
            stats[stage] = {
                "min": round(values[0], 6),
                "mean": round(sum(values) / n, 6),
                "p50": round(values[ceil(0.50 * n) - 1], 6),
                "p95": round(values[ceil(0.95 * n) - 1], 6),
                "p99": round(values[ceil(0.99 * n) - 1], 6),
                "max": round(values[-1], 6),
            }
        return {
            "event": "live_processing_latency_profile",
            "scope": "pose_tracking_features_classification_record_emit_only",
            "excludes": ["camera_capture", "video_decode", "preview", "alert_delivery"],
            "camera_to_alert_latency_measured": False,
            "fall_detection_available": False,
            "sample_policy": "most_recent_frames_bounded",
            "frames_seen": self.frames_seen,
            "frames_sampled": len(self._samples),
            "max_samples": self.max_samples,
            "latency_ms": stats,
        }
