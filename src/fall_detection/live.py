"""Streaming pose, multi-person tracking and temporal baseline integration.

Heuristic per-person states are unvalidated; no trained-model claims or alerts.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
import time
from typing import Protocol

from .contracts import FallState, FramePacket
from .features import FeatureConfig, PoseTrackingFeatureStage, TemporalFeatureBank
from .pipeline import RunSummary
from .state import StateConfig, StateDecision, TemporalStateClassifier
from .tracking import PersonTracker, PoseTrackingStage, TrackerConfig


class PoseEstimator(Protocol):
    def estimate(self, frame: FramePacket) -> tuple: ...


def run_live_pipeline(
    frames: Iterable[FramePacket],
    emit: Callable[[dict], None],
    *,
    estimator: PoseEstimator,
    tracker_config: TrackerConfig | None = None,
    feature_config: FeatureConfig | None = None,
    state_config: StateConfig | None = None,
    on_frame: Callable[[FramePacket, tuple[StateDecision, ...]], bool] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunSummary:
    """Emit bounded per-person pose/temporal decisions, never raw pixels."""
    stage = PoseTrackingFeatureStage(
        PoseTrackingStage(estimator, PersonTracker(tracker_config)),
        TemporalFeatureBank(feature_config),
    )
    classifier = TemporalStateClassifier(state_config)
    started = clock()
    count = 0
    first_timestamp = 0.0
    last_timestamp = -1.0
    last_index = -1
    for frame in frames:
        if frame.index <= last_index or frame.timestamp_seconds <= last_timestamp:
            raise ValueError("frame indices and timestamps must increase strictly")
        if count == 0:
            first_timestamp = frame.timestamp_seconds
        windows = stage.process(frame)
        decisions = classifier.classify_many(windows)
        people = [
            {
                **decision.as_record(),
                "observed": window.latest.observed,
                "window_ready": window.ready,
                "window_samples": len(window.samples),
            }
            for window, decision in zip(windows, decisions)
        ]
        emit({
            "event": "frame",
            "source": frame.source,
            "frame_index": frame.index,
            "timestamp_seconds": frame.timestamp_seconds,
            "timestamp_basis": frame.timestamp_basis,
            "width": frame.width,
            "height": frame.height,
            "state": FallState.UNKNOWN.value,
            "confidence": None,
            "reason": "per_person_heuristic_baseline_only",
            "state_source": "heuristic_baseline_unvalidated",
            "fall_detection_available": False,
            "tracked_person_count": len(people),
            "people": people,
        })
        count += 1
        last_index = frame.index
        last_timestamp = frame.timestamp_seconds
        if on_frame is not None and not on_frame(frame, decisions):
            break
    elapsed = max(0.0, clock() - started)
    return RunSummary(
        frames_processed=count,
        source_duration_seconds=last_timestamp - first_timestamp if count else 0.0,
        elapsed_seconds=elapsed,
        throughput_fps=count / elapsed if elapsed > 0 else 0.0,
    )
