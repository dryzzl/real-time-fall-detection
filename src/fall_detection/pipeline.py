"""Streaming pipeline with explicit, honest inference availability."""

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import time
from typing import Protocol

from .contracts import FramePacket, Prediction


class Predictor(Protocol):
    def predict(self, frame: FramePacket) -> Prediction: ...


class UnavailablePredictor:
    def predict(self, frame: FramePacket) -> Prediction:
        return Prediction()


@dataclass(frozen=True)
class RunSummary:
    frames_processed: int
    source_duration_seconds: float
    elapsed_seconds: float
    throughput_fps: float

    def as_record(self) -> dict:
        return {"event": "summary", **asdict(self)}


def run_pipeline(
    frames: Iterable[FramePacket],
    emit: Callable[[dict], None],
    *,
    predictor: Predictor | None = None,
    realtime: bool = False,
    on_frame: Callable[[FramePacket, Prediction], bool] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> RunSummary:
    """Process a stream without accumulating frames in memory.

    Realtime mode paces against source-relative timestamps. Throughput measures
    this run only; it is not a detector benchmark or an accuracy measurement.
    """
    engine = predictor if predictor is not None else UnavailablePredictor()
    start = clock()
    count = 0
    first_timestamp = 0.0
    last_timestamp = -1.0
    last_index = -1
    for frame in frames:
        if frame.index <= last_index or frame.timestamp_seconds <= last_timestamp:
            raise ValueError("frame indices and timestamps must increase strictly")
        if count == 0:
            first_timestamp = frame.timestamp_seconds
        if realtime:
            delay = frame.timestamp_seconds - first_timestamp - (clock() - start)
            if delay > 0:
                sleep(delay)
        result = engine.predict(frame)
        emit({
            "event": "frame",
            "source": frame.source,
            "frame_index": frame.index,
            "timestamp_seconds": frame.timestamp_seconds,
            "timestamp_basis": frame.timestamp_basis,
            "width": frame.width,
            "height": frame.height,
            "state": result.state.value,
            "confidence": result.confidence,
            "reason": result.reason,
        })
        count += 1
        last_index = frame.index
        last_timestamp = frame.timestamp_seconds
        if on_frame is not None and not on_frame(frame, result):
            break
    elapsed = max(0.0, clock() - start)
    return RunSummary(
        frames_processed=count,
        source_duration_seconds=last_timestamp - first_timestamp if count else 0.0,
        elapsed_seconds=elapsed,
        throughput_fps=count / elapsed if elapsed > 0 else 0.0,
    )
