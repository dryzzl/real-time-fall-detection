"""Explainable temporal baseline for per-person fall states."""

from dataclasses import dataclass, fields, replace
from math import isfinite
from pathlib import Path
import tomllib
from typing import Protocol

from .contracts import FallState, FramePacket
from .features import PoseFeatureSample, TemporalFeatureWindow


@dataclass(frozen=True)
class StateConfig:
    """Thresholds for the deterministic baseline; scores are not probabilities."""

    min_contiguous_observations: int = 3
    min_keypoint_fraction: float = 0.50
    upright_max_aspect_ratio: float = 0.80
    lying_min_aspect_ratio: float = 1.20
    normal_max_vertical_speed: float = 0.12
    falling_min_vertical_speed: float = 0.20
    falling_min_center_drop: float = 0.10
    falling_min_aspect_increase: float = 0.25
    transition_window_seconds: float = 1.50
    fallen_min_duration_seconds: float = 0.50
    settled_max_vertical_speed: float = 0.08

    def __post_init__(self) -> None:
        if (type(self.min_contiguous_observations) is not int
                or not 2 <= self.min_contiguous_observations <= 10_000):
            raise ValueError("min_contiguous_observations must be an integer between 2 and 10000")
        if (type(self.min_keypoint_fraction) not in (int, float)
                or not isfinite(self.min_keypoint_fraction)
                or not 0 <= self.min_keypoint_fraction <= 1):
            raise ValueError("min_keypoint_fraction must be a finite number between 0 and 1")
        positive = (
            "upright_max_aspect_ratio", "lying_min_aspect_ratio",
            "transition_window_seconds", "fallen_min_duration_seconds",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than 0")
        nonnegative = (
            "normal_max_vertical_speed", "falling_min_vertical_speed",
            "falling_min_center_drop", "falling_min_aspect_increase",
            "settled_max_vertical_speed",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.lying_min_aspect_ratio <= self.upright_max_aspect_ratio:
            raise ValueError("lying_min_aspect_ratio must exceed upright_max_aspect_ratio")
        if self.falling_min_vertical_speed <= self.normal_max_vertical_speed:
            raise ValueError("falling_min_vertical_speed must exceed normal_max_vertical_speed")
        if self.settled_max_vertical_speed > self.normal_max_vertical_speed:
            raise ValueError("settled_max_vertical_speed cannot exceed normal_max_vertical_speed")


def load_state_config(path: Path | None = None, **overrides: object) -> StateConfig:
    settings: dict[str, object] = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"state"} or not isinstance(document["state"], dict):
            raise ValueError("state configuration must contain only a [state] table")
        settings.update(document["state"])
    allowed = {field.name for field in fields(StateConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise ValueError("unknown state settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return StateConfig(**settings)


@dataclass(frozen=True)
class StateEvidence:
    contiguous_observations: int
    keypoint_fraction: float | None
    latest_aspect_ratio: float | None
    latest_vertical_speed: float | None
    transition_duration_seconds: float | None = None
    center_drop: float | None = None
    aspect_ratio_increase: float | None = None
    settled_lying_duration_seconds: float | None = None

    def as_record(self) -> dict[str, object]:
        def clean(value: float | None) -> float | None:
            return round(value, 6) if value is not None else None

        return {
            "contiguous_observations": self.contiguous_observations,
            "keypoint_fraction": clean(self.keypoint_fraction),
            "latest_aspect_ratio": clean(self.latest_aspect_ratio),
            "latest_vertical_speed": clean(self.latest_vertical_speed),
            "transition_duration_seconds": clean(self.transition_duration_seconds),
            "center_drop": clean(self.center_drop),
            "aspect_ratio_increase": clean(self.aspect_ratio_increase),
            "settled_lying_duration_seconds": clean(self.settled_lying_duration_seconds),
        }


@dataclass(frozen=True)
class StateDecision:
    track_id: int
    frame_index: int
    state: FallState
    reason: str
    evidence: StateEvidence
    confidence: None = None

    def as_record(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "state": self.state.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": self.evidence.as_record(),
        }


def _keypoint_fraction(sample: PoseFeatureSample) -> float | None:
    if not sample.keypoint_mask:
        return None
    return sum(sample.keypoint_mask) / len(sample.keypoint_mask)


def _vertical_speed(sample: PoseFeatureSample) -> float | None:
    velocity = sample.center_velocity_xy_per_second
    return velocity[1] if velocity is not None else None


def _validate_sample(sample: PoseFeatureSample, track_id: int) -> None:
    if not isinstance(sample, PoseFeatureSample):
        raise ValueError("state windows must contain PoseFeatureSample values")
    if sample.track_id != track_id:
        raise ValueError("all samples must belong to the window track ID")
    if type(sample.frame_index) is not int or sample.frame_index < 0:
        raise ValueError("sample frame indices must be nonnegative integers")
    if (type(sample.timestamp_seconds) not in (int, float)
            or not isfinite(sample.timestamp_seconds) or sample.timestamp_seconds < 0):
        raise ValueError("sample timestamps must be finite and nonnegative")
    if type(sample.observed) is not bool:
        raise ValueError("sample observed must be a boolean")
    if not isinstance(sample.keypoints_xy, tuple) or not isinstance(sample.keypoint_mask, tuple):
        raise ValueError("keypoint coordinates and masks must be tuples")
    if len(sample.keypoints_xy) != len(sample.keypoint_mask):
        raise ValueError("keypoint coordinates and masks must have equal length")
    if any(type(value) is not bool for value in sample.keypoint_mask):
        raise ValueError("keypoint masks must contain booleans")
    for point, available in zip(sample.keypoints_xy, sample.keypoint_mask):
        if not isinstance(point, tuple) or len(point) != 2:
            raise ValueError("normalized keypoints must contain two coordinates")
        if available:
            if any(type(value) not in (int, float) or not isfinite(value) for value in point):
                raise ValueError("available keypoint coordinates must be finite")
        elif point != (None, None):
            raise ValueError("unavailable keypoints must use (None, None)")
    pairs = (
        ("box_center_xy", sample.box_center_xy),
        ("box_size_wh", sample.box_size_wh),
        ("center_velocity_xy_per_second", sample.center_velocity_xy_per_second),
    )
    for name, value in pairs:
        if value is not None and (not isinstance(value, tuple) or len(value) != 2):
            raise ValueError(f"{name} must be an x/y pair when present")
    optional_numbers = [sample.detection_confidence, sample.box_aspect_ratio]
    for _, value in pairs:
        optional_numbers.extend(value or ())
    if any(value is not None and (type(value) not in (int, float) or not isfinite(value))
           for value in optional_numbers):
        raise ValueError("sample features must be finite when present")
    if sample.observed:
        if (sample.detection_confidence is None or sample.box_center_xy is None
                or sample.box_size_wh is None or sample.box_aspect_ratio is None):
            raise ValueError("observed samples require detection and box features")
        if not 0 <= sample.detection_confidence <= 1:
            raise ValueError("detection confidence must be between 0 and 1")
        if sample.box_size_wh[0] <= 0 or sample.box_size_wh[1] <= 0 or sample.box_aspect_ratio <= 0:
            raise ValueError("observed box size and aspect ratio must be positive")
    elif any(value is not None for value in (
            sample.detection_confidence, sample.box_center_xy, sample.box_size_wh,
            sample.box_aspect_ratio, sample.center_velocity_xy_per_second)):
        raise ValueError("missing samples cannot contain current detection features")


def _validate_window(window: TemporalFeatureWindow) -> None:
    if not isinstance(window, TemporalFeatureWindow):
        raise ValueError("classifier input must be a TemporalFeatureWindow")
    if type(window.track_id) is not int or window.track_id <= 0:
        raise ValueError("window track IDs must be positive integers")
    if not isinstance(window.samples, tuple) or not window.samples:
        raise ValueError("state windows must contain at least one sample")
    if type(window.observed_count) is not int or type(window.ready) is not bool:
        raise ValueError("window metadata has invalid types")
    last_index, last_timestamp = -1, -1.0
    for sample in window.samples:
        _validate_sample(sample, window.track_id)
        if sample.frame_index <= last_index or sample.timestamp_seconds <= last_timestamp:
            raise ValueError("window samples must increase strictly in frame index and timestamp")
        last_index, last_timestamp = sample.frame_index, sample.timestamp_seconds
    if window.observed_count != sum(sample.observed for sample in window.samples):
        raise ValueError("window observed_count does not match its samples")


class TemporalStateClassifier:
    """Threshold baseline that returns unknown when its evidence contract is unmet."""

    def __init__(self, config: StateConfig | None = None) -> None:
        self.config = config or StateConfig()

    def _decision(self, window: TemporalFeatureWindow, state: FallState, reason: str,
                  evidence: StateEvidence) -> StateDecision:
        return StateDecision(window.track_id, window.latest.frame_index, state, reason, evidence)

    def classify(self, window: TemporalFeatureWindow) -> StateDecision:
        _validate_window(window)
        latest = window.latest
        latest_fraction = _keypoint_fraction(latest)
        basic_evidence = StateEvidence(
            contiguous_observations=0,
            keypoint_fraction=latest_fraction,
            latest_aspect_ratio=latest.box_aspect_ratio,
            latest_vertical_speed=_vertical_speed(latest),
        )
        if not latest.observed:
            return self._decision(window, FallState.UNKNOWN,
                                  "missing_current_observation", basic_evidence)
        if not window.ready:
            return self._decision(window, FallState.UNKNOWN,
                                  "feature_window_not_ready", basic_evidence)
        if latest_fraction is None or latest_fraction < self.config.min_keypoint_fraction:
            return self._decision(window, FallState.UNKNOWN,
                                  "insufficient_pose_confidence", basic_evidence)

        usable_suffix = []
        for sample in reversed(window.samples):
            fraction = _keypoint_fraction(sample)
            if (not sample.observed or fraction is None
                    or fraction < self.config.min_keypoint_fraction):
                break
            usable_suffix.append(sample)
        usable_suffix.reverse()
        latest_speed = _vertical_speed(latest)
        evidence = StateEvidence(
            contiguous_observations=len(usable_suffix),
            keypoint_fraction=latest_fraction,
            latest_aspect_ratio=latest.box_aspect_ratio,
            latest_vertical_speed=latest_speed,
        )
        if len(usable_suffix) < self.config.min_contiguous_observations:
            return self._decision(window, FallState.UNKNOWN,
                                  "insufficient_contiguous_observations", evidence)

        stable_lying = []
        for sample in reversed(usable_suffix):
            speed = _vertical_speed(sample)
            if (sample.box_aspect_ratio is None
                    or sample.box_aspect_ratio < self.config.lying_min_aspect_ratio
                    or speed is None or abs(speed) > self.config.settled_max_vertical_speed):
                break
            stable_lying.append(sample)
        stable_lying.reverse()
        settled_duration = (
            stable_lying[-1].timestamp_seconds - stable_lying[0].timestamp_seconds
            if len(stable_lying) >= 2 else 0.0
        )
        if settled_duration >= self.config.fallen_min_duration_seconds:
            fallen_evidence = replace(
                evidence, settled_lying_duration_seconds=settled_duration
            )
            return self._decision(window, FallState.FALLEN,
                                  "lying_posture_settled", fallen_evidence)

        transition = None
        for earlier in usable_suffix[:-1]:
            elapsed = latest.timestamp_seconds - earlier.timestamp_seconds
            if elapsed > self.config.transition_window_seconds:
                continue
            if (earlier.box_aspect_ratio is None
                    or earlier.box_aspect_ratio > self.config.upright_max_aspect_ratio
                    or earlier.box_center_xy is None or latest.box_center_xy is None
                    or latest.box_aspect_ratio is None):
                continue
            center_drop = latest.box_center_xy[1] - earlier.box_center_xy[1]
            speed = center_drop / elapsed
            aspect_increase = latest.box_aspect_ratio - earlier.box_aspect_ratio
            transition = (elapsed, center_drop, speed, aspect_increase)
            if (center_drop >= self.config.falling_min_center_drop
                    and speed >= self.config.falling_min_vertical_speed
                    and aspect_increase >= self.config.falling_min_aspect_increase):
                falling_evidence = StateEvidence(
                    contiguous_observations=len(usable_suffix),
                    keypoint_fraction=latest_fraction,
                    latest_aspect_ratio=latest.box_aspect_ratio,
                    latest_vertical_speed=latest_speed,
                    transition_duration_seconds=elapsed,
                    center_drop=center_drop,
                    aspect_ratio_increase=aspect_increase,
                    settled_lying_duration_seconds=settled_duration,
                )
                return self._decision(window, FallState.FALLING,
                                      "rapid_descent_with_posture_change", falling_evidence)

        if (latest.box_aspect_ratio is not None
                and latest.box_aspect_ratio <= self.config.upright_max_aspect_ratio
                and latest_speed is not None
                and abs(latest_speed) <= self.config.normal_max_vertical_speed):
            return self._decision(window, FallState.NORMAL,
                                  "upright_posture_stable", evidence)

        if transition is not None:
            elapsed, center_drop, _, aspect_increase = transition
            evidence = StateEvidence(
                contiguous_observations=len(usable_suffix),
                keypoint_fraction=latest_fraction,
                latest_aspect_ratio=latest.box_aspect_ratio,
                latest_vertical_speed=latest_speed,
                transition_duration_seconds=elapsed,
                center_drop=center_drop,
                aspect_ratio_increase=aspect_increase,
                settled_lying_duration_seconds=settled_duration,
            )
        return self._decision(window, FallState.UNKNOWN,
                              "ambiguous_or_unsettled_pose", evidence)

    def classify_many(
        self, windows: tuple[TemporalFeatureWindow, ...] | list[TemporalFeatureWindow]
    ) -> tuple[StateDecision, ...]:
        if not isinstance(windows, (tuple, list)):
            raise ValueError("windows must be a tuple or list")
        raw = tuple(windows)
        if any(not isinstance(window, TemporalFeatureWindow) for window in raw):
            raise ValueError("windows must contain TemporalFeatureWindow values")
        for window in raw:
            _validate_window(window)
        ordered = sorted(raw, key=lambda window: window.track_id)
        if len({window.track_id for window in ordered}) != len(ordered):
            raise ValueError("windows must contain unique track IDs")
        return tuple(self.classify(window) for window in ordered)


class FeatureStage(Protocol):
    def process(self, frame: FramePacket) -> tuple[TemporalFeatureWindow, ...]: ...


class TemporalStateStage:
    """Compose feature extraction and baseline state decisions per active track."""

    def __init__(self, feature_stage: FeatureStage,
                 classifier: TemporalStateClassifier | None = None) -> None:
        self.feature_stage = feature_stage
        self.classifier = classifier or TemporalStateClassifier()

    def process(self, frame: FramePacket) -> tuple[StateDecision, ...]:
        return self.classifier.classify_many(self.feature_stage.process(frame))


def _smoke_sample(track_id: int, frame_index: int, center_y: float | None,
                  aspect_ratio: float | None, previous_center_y: float | None,
                  *, observed: bool = True) -> PoseFeatureSample:
    timestamp = frame_index * 0.2
    if not observed:
        return PoseFeatureSample(
            track_id, frame_index, timestamp, False, None, None, None, None, None,
            ((None, None),) * 17, (False,) * 17,
        )
    velocity = None if previous_center_y is None else (0.0, (center_y - previous_center_y) / 0.2)
    return PoseFeatureSample(
        track_id, frame_index, timestamp, True, 0.9, (0.5, center_y),
        (0.2, 0.2 / aspect_ratio), aspect_ratio, velocity,
        ((0.0, 0.0),) * 17, (True,) * 17,
    )


def _smoke_window(track_id: int, centers: tuple[float, ...], aspects: tuple[float, ...],
                  *, missing_latest: bool = False) -> TemporalFeatureWindow:
    samples = []
    previous = None
    for index, (center, aspect) in enumerate(zip(centers, aspects)):
        samples.append(_smoke_sample(track_id, index, center, aspect, previous))
        previous = center
    if missing_latest:
        index = len(samples)
        samples.append(_smoke_sample(track_id, index, None, None, previous, observed=False))
    return TemporalFeatureWindow(
        track_id, tuple(samples), sum(sample.observed for sample in samples), not missing_latest
    )


def run_state_smoke(config: StateConfig | None = None) -> dict[str, object]:
    """Classify deterministic standing, lying, descent, and missing sequences."""
    classifier = TemporalStateClassifier(config)
    windows = (
        _smoke_window(1, (0.30,) * 5, (0.45,) * 5),
        _smoke_window(2, (0.70,) * 5, (1.50,) * 5),
        _smoke_window(3, (0.30, 0.34, 0.40, 0.48), (0.45, 0.55, 0.72, 0.95)),
        _smoke_window(4, (0.30, 0.30, 0.30), (0.45, 0.45, 0.45), missing_latest=True),
    )
    decisions = classifier.classify_many(windows)
    return {
        "event": "state_smoke",
        "synthetic": True,
        "scenarios": {
            name: decision.as_record()
            for name, decision in zip(("standing", "lying", "descent", "missing"), decisions)
        },
        "baseline_state_classifier_available": True,
        "fall_detection_available": False,
    }
