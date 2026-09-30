"""Frame and prediction contracts shared by future pipeline stages."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


class FallState(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    FALLING = "falling"
    FALLEN = "fallen"


@dataclass(frozen=True)
class FramePacket:
    index: int
    timestamp_seconds: float
    width: int
    height: int
    bgr: bytes = field(repr=False)
    source: str = "synthetic"

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("frame index must be a nonnegative integer")
        if type(self.timestamp_seconds) not in (int, float) or not isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("frame timestamp must be finite and nonnegative")
        if any(type(value) is not int or value <= 0 for value in (self.width, self.height)):
            raise ValueError("frame dimensions must be positive integers")
        if not isinstance(self.bgr, bytes) or len(self.bgr) != self.width * self.height * 3:
            raise ValueError("frame must contain exactly width * height * 3 BGR bytes")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("frame source must be a nonempty string")


@dataclass(frozen=True)
class Prediction:
    state: FallState = FallState.UNKNOWN
    confidence: float | None = None
    reason: str = "detection_not_implemented"

    def __post_init__(self) -> None:
        if not isinstance(self.state, FallState):
            raise ValueError("prediction state must be a FallState")
        score = self.confidence
        if score is not None and (type(score) not in (int, float) or not isfinite(score) or not 0 <= score <= 1):
            raise ValueError("confidence must be None or a finite number between 0 and 1")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("prediction reason must be a nonempty string")
