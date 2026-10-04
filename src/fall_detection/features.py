"""Normalized pose/motion features and bounded per-person temporal windows."""

from collections import deque
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
import tomllib
from typing import Protocol

from .contracts import FramePacket
from .pose import Keypoint, PoseDetection
from .tracking import PersonTracker, TrackSnapshot


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the deterministic temporal feature contract."""

    window_size: int = 30
    keypoint_count: int = 17
    min_keypoint_confidence: float = 0.30
    min_observed_samples: int = 8
    max_motion_gap_seconds: float = 1.0

    def __post_init__(self) -> None:
        if type(self.window_size) is not int or not 1 <= self.window_size <= 10_000:
            raise ValueError("window_size must be an integer between 1 and 10000")
        if type(self.keypoint_count) is not int or not 1 <= self.keypoint_count <= 100:
            raise ValueError("keypoint_count must be an integer between 1 and 100")
        if (type(self.min_keypoint_confidence) not in (int, float)
                or not isfinite(self.min_keypoint_confidence)
                or not 0 <= self.min_keypoint_confidence <= 1):
            raise ValueError("min_keypoint_confidence must be a finite number between 0 and 1")
        if (type(self.min_observed_samples) is not int
                or not 1 <= self.min_observed_samples <= self.window_size):
            raise ValueError("min_observed_samples must be between 1 and window_size")
        if (type(self.max_motion_gap_seconds) not in (int, float)
                or not isfinite(self.max_motion_gap_seconds)
                or self.max_motion_gap_seconds <= 0):
            raise ValueError("max_motion_gap_seconds must be a finite number greater than 0")


def load_feature_config(path: Path | None = None, **overrides: object) -> FeatureConfig:
    settings: dict[str, object] = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"features"} or not isinstance(document["features"], dict):
            raise ValueError("feature configuration must contain only a [features] table")
        settings.update(document["features"])
    allowed = {field.name for field in fields(FeatureConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise ValueError("unknown feature settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return FeatureConfig(**settings)


@dataclass(frozen=True)
class PoseFeatureSample:
    """One timestamped observation; missing values remain explicit rather than imputed."""

    track_id: int
    frame_index: int
    timestamp_seconds: float
    observed: bool
    detection_confidence: float | None
    box_center_xy: tuple[float, float] | None
    box_size_wh: tuple[float, float] | None
    box_aspect_ratio: float | None
    center_velocity_xy_per_second: tuple[float, float] | None
    keypoints_xy: tuple[tuple[float | None, float | None], ...]
    keypoint_mask: tuple[bool, ...]

    def as_record(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "observed": self.observed,
            "detection_confidence": self.detection_confidence,
            "box_center_xy": list(self.box_center_xy) if self.box_center_xy else None,
            "box_size_wh": list(self.box_size_wh) if self.box_size_wh else None,
            "box_aspect_ratio": self.box_aspect_ratio,
            "center_velocity_xy_per_second": (
                list(self.center_velocity_xy_per_second)
                if self.center_velocity_xy_per_second else None
            ),
            "keypoints_xy": [list(point) for point in self.keypoints_xy],
            "keypoint_mask": list(self.keypoint_mask),
        }


@dataclass(frozen=True)
class TemporalFeatureWindow:
    track_id: int
    samples: tuple[PoseFeatureSample, ...]
    observed_count: int
    ready: bool

    @property
    def latest(self) -> PoseFeatureSample:
        return self.samples[-1]

    def as_record(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "sample_count": len(self.samples),
            "observed_count": self.observed_count,
            "ready": self.ready,
            "latest": self.latest.as_record(),
        }


def _validate_detection(detection: PoseDetection, keypoint_count: int) -> None:
    if not isinstance(detection, PoseDetection):
        raise ValueError("an observed track must contain a PoseDetection")
    box = detection.box_xyxy
    if (len(box) != 4
            or any(type(value) not in (int, float) or not isfinite(value) for value in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("detection box must contain four finite coordinates with positive size")
    if (type(detection.confidence) not in (int, float)
            or not isfinite(detection.confidence)
            or not 0 <= detection.confidence <= 1):
        raise ValueError("detection confidence must be between 0 and 1")
    if len(detection.keypoints) != keypoint_count:
        raise ValueError(f"expected {keypoint_count} keypoints, received {len(detection.keypoints)}")
    for keypoint in detection.keypoints:
        if (not isinstance(keypoint, Keypoint)
                or any(type(value) not in (int, float) or not isfinite(value)
                       for value in (keypoint.x, keypoint.y, keypoint.confidence))
                or not 0 <= keypoint.confidence <= 1):
            raise ValueError("keypoints must contain finite coordinates and confidence in [0, 1]")


class TemporalFeatureBank:
    """Maintain one fixed-size, timestamped feature history per active track."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._histories: dict[int, deque[PoseFeatureSample]] = {}
        self._last_frame_index = -1
        self._last_timestamp_seconds = -1.0

    def reset(self) -> None:
        self._histories.clear()
        self._last_frame_index = -1
        self._last_timestamp_seconds = -1.0

    def _missing_sample(self, frame: FramePacket, track_id: int) -> PoseFeatureSample:
        empty_keypoints = tuple((None, None) for _ in range(self.config.keypoint_count))
        return PoseFeatureSample(
            track_id=track_id,
            frame_index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            observed=False,
            detection_confidence=None,
            box_center_xy=None,
            box_size_wh=None,
            box_aspect_ratio=None,
            center_velocity_xy_per_second=None,
            keypoints_xy=empty_keypoints,
            keypoint_mask=(False,) * self.config.keypoint_count,
        )

    def _observed_sample(
        self,
        frame: FramePacket,
        snapshot: TrackSnapshot,
        previous: PoseFeatureSample | None,
    ) -> PoseFeatureSample:
        detection = snapshot.detection
        if detection is None:
            raise ValueError("observed feature extraction requires a detection")
        _validate_detection(detection, self.config.keypoint_count)
        left, top, right, bottom = detection.box_xyxy
        width, height = right - left, bottom - top
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        normalized_center = (center_x / frame.width, center_y / frame.height)
        velocity = None
        if previous is not None and previous.observed and previous.box_center_xy is not None:
            elapsed = frame.timestamp_seconds - previous.timestamp_seconds
            if 0 < elapsed <= self.config.max_motion_gap_seconds:
                velocity = (
                    (normalized_center[0] - previous.box_center_xy[0]) / elapsed,
                    (normalized_center[1] - previous.box_center_xy[1]) / elapsed,
                )
        points = []
        mask = []
        for keypoint in detection.keypoints:
            confident = keypoint.confidence >= self.config.min_keypoint_confidence
            mask.append(confident)
            points.append(
                ((keypoint.x - center_x) / width, (keypoint.y - center_y) / height)
                if confident else (None, None)
            )
        return PoseFeatureSample(
            track_id=snapshot.track_id,
            frame_index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            observed=True,
            detection_confidence=detection.confidence,
            box_center_xy=normalized_center,
            box_size_wh=(width / frame.width, height / frame.height),
            box_aspect_ratio=width / height,
            center_velocity_xy_per_second=velocity,
            keypoints_xy=tuple(points),
            keypoint_mask=tuple(mask),
        )

    def update(
        self,
        frame: FramePacket,
        snapshots: tuple[TrackSnapshot, ...] | list[TrackSnapshot],
    ) -> tuple[TemporalFeatureWindow, ...]:
        if not isinstance(frame, FramePacket):
            raise ValueError("frame must be a FramePacket")
        if frame.index <= self._last_frame_index:
            raise ValueError("feature frame indices must increase strictly")
        if frame.timestamp_seconds <= self._last_timestamp_seconds:
            raise ValueError("feature timestamps must increase strictly")
        if not isinstance(snapshots, (tuple, list)):
            raise ValueError("snapshots must be a tuple or list")
        raw_snapshots = tuple(snapshots)
        if any(not isinstance(snapshot, TrackSnapshot) for snapshot in raw_snapshots):
            raise ValueError("snapshots must contain TrackSnapshot values")
        ordered = sorted(raw_snapshots, key=lambda snapshot: snapshot.track_id)
        seen: set[int] = set()
        for snapshot in ordered:
            if type(snapshot.track_id) is not int or snapshot.track_id <= 0:
                raise ValueError("track IDs must be positive integers")
            if snapshot.track_id in seen:
                raise ValueError("snapshots must contain unique track IDs")
            if snapshot.frame_index != frame.index:
                raise ValueError("snapshot and frame indices must match")
            if snapshot.detection is not None:
                _validate_detection(snapshot.detection, self.config.keypoint_count)
            seen.add(snapshot.track_id)

        # PersonTracker emits every active track. Absence therefore means retirement,
        # and dropping the history keeps total memory bounded by active tracks.
        for retired_id in self._histories.keys() - seen:
            del self._histories[retired_id]

        windows = []
        for snapshot in ordered:
            history = self._histories.setdefault(
                snapshot.track_id, deque(maxlen=self.config.window_size)
            )
            previous = history[-1] if history else None
            sample = (
                self._observed_sample(frame, snapshot, previous)
                if snapshot.detection is not None
                else self._missing_sample(frame, snapshot.track_id)
            )
            history.append(sample)
            samples = tuple(history)
            observed_count = sum(item.observed for item in samples)
            windows.append(TemporalFeatureWindow(
                track_id=snapshot.track_id,
                samples=samples,
                observed_count=observed_count,
                ready=(sample.observed and observed_count >= self.config.min_observed_samples),
            ))
        self._last_frame_index = frame.index
        self._last_timestamp_seconds = frame.timestamp_seconds
        return tuple(windows)


class TrackingStage(Protocol):
    def process(self, frame: FramePacket) -> tuple[TrackSnapshot, ...]: ...


class PoseTrackingFeatureStage:
    """Compose tracking and temporal features without assigning a fall state."""

    def __init__(self, tracking_stage: TrackingStage,
                 feature_bank: TemporalFeatureBank | None = None) -> None:
        self.tracking_stage = tracking_stage
        self.feature_bank = feature_bank or TemporalFeatureBank()

    def process(self, frame: FramePacket) -> tuple[TemporalFeatureWindow, ...]:
        return self.feature_bank.update(frame, self.tracking_stage.process(frame))


def _smoke_detection(center_x: float, center_y: float, keypoint_count: int,
                     *, low_confidence_first: bool = False) -> PoseDetection:
    keypoints = tuple(
        Keypoint(
            center_x + (-3 if index % 2 == 0 else 3),
            center_y - 12 + (24 * index / max(1, keypoint_count - 1)),
            0.1 if low_confidence_first and index == 0 else 0.9,
        )
        for index in range(keypoint_count)
    )
    return PoseDetection((center_x - 10, center_y - 20, center_x + 10, center_y + 20),
                         0.9, keypoints)


def run_feature_smoke(config: FeatureConfig | None = None) -> dict[str, object]:
    """Exercise bounded, independent histories with synthetic tracked poses."""
    actual_config = config or FeatureConfig()
    tracker = PersonTracker()
    feature_bank = TemporalFeatureBank(actual_config)
    histories: dict[str, list[bool]] = {}
    low_confidence_masked = False
    final_windows: tuple[TemporalFeatureWindow, ...] = ()
    for frame_index in range(6):
        detections = [_smoke_detection(25, 25 + frame_index * 3,
                                       actual_config.keypoint_count,
                                       low_confidence_first=frame_index == 0)]
        if frame_index != 3:
            detections.append(_smoke_detection(75, 55, actual_config.keypoint_count))
        frame = FramePacket(
            frame_index, frame_index / 10, 100, 100, b"\0" * 30_000,
            source="feature_contract_smoke", timestamp_basis="synthetic_fps",
        )
        windows = feature_bank.update(frame, tracker.update(frame_index, detections))
        final_windows = windows
        for window in windows:
            histories.setdefault(str(window.track_id), []).append(window.latest.observed)
            if window.latest.observed:
                low_confidence_masked = low_confidence_masked or not all(window.latest.keypoint_mask)
    return {
        "event": "feature_smoke",
        "synthetic": True,
        "frames_processed": 6,
        "active_track_ids": [window.track_id for window in final_windows],
        "window_lengths": {str(window.track_id): len(window.samples) for window in final_windows},
        "observation_histories": histories,
        "low_confidence_keypoints_masked": low_confidence_masked,
        "fall_detection_available": False,
    }
