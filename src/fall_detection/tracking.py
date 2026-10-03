"""Deterministic local person-association baseline for pose detections."""

from dataclasses import dataclass, fields
from math import hypot, isfinite
from pathlib import Path
import tomllib
from typing import Protocol

from .contracts import FramePacket
from .pose import Keypoint, PoseDetection


@dataclass(frozen=True)
class TrackerConfig:
    max_missed_frames: int = 2
    max_center_distance: float = 2.5
    max_pose_distance: float = 2.0
    min_iou: float = 0.05
    min_keypoint_confidence: float = 0.30
    velocity_smoothing: float = 0.25
    center_weight: float = 0.55
    iou_weight: float = 0.25
    pose_weight: float = 0.20

    def __post_init__(self) -> None:
        if type(self.max_missed_frames) is not int or not 0 <= self.max_missed_frames <= 1_000:
            raise ValueError("max_missed_frames must be an integer between 0 and 1000")
        for name in ("max_center_distance", "max_pose_distance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than 0")
        for name in ("min_iou", "min_keypoint_confidence", "velocity_smoothing"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        weights = (self.center_weight, self.iou_weight, self.pose_weight)
        if any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in weights):
            raise ValueError("association weights must be finite nonnegative numbers")
        if sum(weights) <= 0:
            raise ValueError("at least one association weight must be positive")


def load_tracker_config(path: Path | None = None, **overrides: object) -> TrackerConfig:
    settings: dict[str, object] = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"tracker"} or not isinstance(document["tracker"], dict):
            raise ValueError("tracker configuration must contain only a [tracker] table")
        settings.update(document["tracker"])
    allowed = {field.name for field in fields(TrackerConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise ValueError("unknown tracker settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return TrackerConfig(**settings)


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    frame_index: int
    age_frames: int
    hits: int
    missed_frames: int
    predicted_box_xyxy: tuple[float, float, float, float]
    detection: PoseDetection | None

    @property
    def observed(self) -> bool:
        return self.detection is not None

    def as_record(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "age_frames": self.age_frames,
            "hits": self.hits,
            "missed_frames": self.missed_frames,
            "observed": self.observed,
            "predicted_box_xyxy": list(self.predicted_box_xyxy),
            "detection_confidence": self.detection.confidence if self.detection else None,
            "keypoint_count": len(self.detection.keypoints) if self.detection else 0,
        }


@dataclass
class _Track:
    track_id: int
    first_frame_index: int
    last_seen_frame_index: int
    hits: int
    detection: PoseDetection
    velocity_x: float = 0.0
    velocity_y: float = 0.0


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    return max(hypot(box[2] - box[0], box[3] - box[1]), 1.0)


def _translate_box(box: tuple[float, float, float, float], dx: float, dy: float):
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _validate_detection(detection: PoseDetection) -> None:
    if not isinstance(detection, PoseDetection):
        raise ValueError("tracker inputs must be PoseDetection values")
    box = detection.box_xyxy
    if len(box) != 4 or any(type(value) not in (int, float) or not isfinite(value) for value in box):
        raise ValueError("detection box must contain four finite coordinates")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("detection box must have positive width and height")
    if type(detection.confidence) not in (int, float) or not isfinite(detection.confidence) or not 0 <= detection.confidence <= 1:
        raise ValueError("detection confidence must be between 0 and 1")
    for keypoint in detection.keypoints:
        if not isinstance(keypoint, Keypoint) or any(
            type(value) not in (int, float) or not isfinite(value)
            for value in (keypoint.x, keypoint.y, keypoint.confidence)
        ) or not 0 <= keypoint.confidence <= 1:
            raise ValueError("keypoints must contain finite coordinates and confidence in [0, 1]")


def _detection_sort_key(detection: PoseDetection):
    keypoints = tuple((point.x, point.y, point.confidence) for point in detection.keypoints)
    return (*detection.box_xyxy, -detection.confidence, keypoints)


class PersonTracker:
    """Greedy motion/IoU/pose association with process-local stable IDs."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_frame_index = -1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_frame_index = -1

    def _predict_box(self, track: _Track, frame_index: int):
        gap = frame_index - track.last_seen_frame_index
        return _translate_box(track.detection.box_xyxy, track.velocity_x * gap, track.velocity_y * gap)

    def _pose_distance(self, track: _Track, detection: PoseDetection, frame_index: int,
                       normalization: float) -> float | None:
        gap = frame_index - track.last_seen_frame_index
        distances = []
        for previous, current in zip(track.detection.keypoints, detection.keypoints):
            if (previous.confidence >= self.config.min_keypoint_confidence
                    and current.confidence >= self.config.min_keypoint_confidence):
                predicted_x = previous.x + track.velocity_x * gap
                predicted_y = previous.y + track.velocity_y * gap
                distances.append(hypot(predicted_x - current.x, predicted_y - current.y) / normalization)
        return sum(distances) / len(distances) if distances else None

    def _association_cost(self, track: _Track, detection: PoseDetection,
                          frame_index: int) -> float | None:
        predicted = self._predict_box(track, frame_index)
        predicted_center = _box_center(predicted)
        detected_center = _box_center(detection.box_xyxy)
        normalization = max(_box_diagonal(predicted), _box_diagonal(detection.box_xyxy))
        center_distance = hypot(predicted_center[0] - detected_center[0],
                                predicted_center[1] - detected_center[1]) / normalization
        overlap = _iou(predicted, detection.box_xyxy)
        pose_distance = self._pose_distance(track, detection, frame_index, normalization)
        motion_ok = center_distance <= self.config.max_center_distance
        pose_ok = pose_distance is not None and pose_distance <= self.config.max_pose_distance
        close_without_overlap = center_distance <= min(1.5, self.config.max_center_distance)
        if not (motion_ok or pose_ok) or not (overlap >= self.config.min_iou or close_without_overlap or pose_ok):
            return None
        pose_component = pose_distance if pose_distance is not None else center_distance
        weight_total = self.config.center_weight + self.config.iou_weight + self.config.pose_weight
        return (
            self.config.center_weight * center_distance
            + self.config.iou_weight * (1.0 - overlap)
            + self.config.pose_weight * pose_component
        ) / weight_total

    def update(self, frame_index: int,
               detections: tuple[PoseDetection, ...] | list[PoseDetection]) -> tuple[TrackSnapshot, ...]:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if frame_index <= self._last_frame_index:
            raise ValueError("tracker frame indices must increase strictly")
        if not isinstance(detections, (tuple, list)):
            raise ValueError("detections must be a tuple or list")
        raw_detections = tuple(detections)
        for detection in raw_detections:
            _validate_detection(detection)
        ordered = tuple(sorted(raw_detections, key=_detection_sort_key))

        # A track may match after at most max_missed intervening frames.
        expired_before_match = [
            track_id for track_id, track in self._tracks.items()
            if frame_index - track.last_seen_frame_index - 1 > self.config.max_missed_frames
        ]
        for track_id in expired_before_match:
            del self._tracks[track_id]

        candidates = []
        for track_id in sorted(self._tracks):
            for detection_index, detection in enumerate(ordered):
                cost = self._association_cost(self._tracks[track_id], detection, frame_index)
                if cost is not None:
                    candidates.append((cost, track_id, detection_index))
        matched_tracks: dict[int, int] = {}
        matched_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidates):
            if track_id not in matched_tracks and detection_index not in matched_detections:
                matched_tracks[track_id] = detection_index
                matched_detections.add(detection_index)

        observed: set[int] = set()
        for track_id, detection_index in matched_tracks.items():
            track, detection = self._tracks[track_id], ordered[detection_index]
            gap = frame_index - track.last_seen_frame_index
            old_center, new_center = _box_center(track.detection.box_xyxy), _box_center(detection.box_xyxy)
            measured_x = (new_center[0] - old_center[0]) / gap
            measured_y = (new_center[1] - old_center[1]) / gap
            if track.hits == 1:
                track.velocity_x, track.velocity_y = measured_x, measured_y
            else:
                keep = self.config.velocity_smoothing
                track.velocity_x = keep * track.velocity_x + (1.0 - keep) * measured_x
                track.velocity_y = keep * track.velocity_y + (1.0 - keep) * measured_y
            track.detection = detection
            track.last_seen_frame_index = frame_index
            track.hits += 1
            observed.add(track_id)

        for detection_index, detection in enumerate(ordered):
            if detection_index not in matched_detections:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _Track(track_id, frame_index, frame_index, 1, detection)
                observed.add(track_id)

        expired_after_update = [
            track_id for track_id, track in self._tracks.items()
            if frame_index - track.last_seen_frame_index > self.config.max_missed_frames
        ]
        for track_id in expired_after_update:
            del self._tracks[track_id]

        snapshots = []
        for track_id in sorted(self._tracks):
            track = self._tracks[track_id]
            is_observed = track_id in observed
            snapshots.append(TrackSnapshot(
                track_id=track_id,
                frame_index=frame_index,
                age_frames=frame_index - track.first_frame_index + 1,
                hits=track.hits,
                missed_frames=0 if is_observed else frame_index - track.last_seen_frame_index,
                predicted_box_xyxy=(track.detection.box_xyxy if is_observed
                                    else self._predict_box(track, frame_index)),
                detection=track.detection if is_observed else None,
            ))
        self._last_frame_index = frame_index
        return tuple(snapshots)


class PoseEstimator(Protocol):
    def estimate(self, frame: FramePacket) -> tuple[PoseDetection, ...]: ...


class PoseTrackingStage:
    """Compose an estimator and tracker without assigning a fall state."""

    def __init__(self, estimator: PoseEstimator, tracker: PersonTracker | None = None) -> None:
        self.estimator = estimator
        self.tracker = tracker or PersonTracker()

    def process(self, frame: FramePacket) -> tuple[TrackSnapshot, ...]:
        return self.tracker.update(frame.index, self.estimator.estimate(frame))


def _smoke_detection(center_x: float, marker: float) -> PoseDetection:
    return PoseDetection(
        (center_x - 8, 10, center_x + 8, 42), 0.9,
        (Keypoint(center_x + marker, 18, 0.9), Keypoint(center_x + marker, 34, 0.9)),
    )


def run_tracking_smoke(config: TrackerConfig | None = None) -> dict[str, object]:
    """Exercise crossing and one missed observation with synthetic detections."""
    tracker = PersonTracker(config)
    sequence = (
        (0, (_smoke_detection(10, -2), _smoke_detection(90, 2))),
        (1, (_smoke_detection(70, 2), _smoke_detection(30, -2))),
        (2, (_smoke_detection(50, 2), _smoke_detection(50, -2))),
        (3, (_smoke_detection(30, 2), _smoke_detection(70, -2))),
        (4, (_smoke_detection(90, -2),)),
        (5, (_smoke_detection(10, 2), _smoke_detection(110, -2))),
    )
    histories: dict[int, list[float | None]] = {}
    final_snapshots: tuple[TrackSnapshot, ...] = ()
    for frame_index, detections in sequence:
        snapshots = tracker.update(frame_index, list(reversed(detections)) if frame_index % 2 else detections)
        final_snapshots = snapshots
        for snapshot in snapshots:
            center = _box_center(snapshot.detection.box_xyxy)[0] if snapshot.detection else None
            histories.setdefault(snapshot.track_id, []).append(center)
    return {
        "event": "tracking_smoke",
        "synthetic": True,
        "frames_processed": len(sequence),
        "active_track_ids": [snapshot.track_id for snapshot in final_snapshots],
        "track_ids_seen": sorted(histories),
        "center_histories": {str(track_id): values for track_id, values in sorted(histories.items())},
        "fall_detection_available": False,
    }
