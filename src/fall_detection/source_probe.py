"""Explicit opt-in capture readiness probe for research video sources.

A probe opens a camera or video, decodes requested frames, and closes it.
It does not load model weights or make fall-detection claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


class SourceProbeError(RuntimeError):
    """A capture source could not produce a decoded frame."""


@dataclass(frozen=True)
class SourceProbeResult:
    source: str
    width: int
    height: int
    fps: float | None
    frames_decoded: int = 1
    decode_seconds: float | None = None
    channels: int = 3


def probe_source(source: int | str | Path, *, frames: int = 1, cv2_module: Any = None) -> SourceProbeResult:
    """Decode consecutive BGR frames and release the device even on failures.

    Requires OpenCV only when invoked. A successful probe establishes capture
    readiness at that instant, not inference correctness or future availability.
    """
    if type(frames) is not int or not 1 <= frames <= 120:
        raise ValueError('frames must be an integer from 1 to 120')
    if type(source) is int:
        if source < 0:
            raise ValueError("camera index must be nonnegative")
        capture_arg: int | str = source
    elif isinstance(source, (str, Path)):
        capture_arg = str(source)
        if not Path(capture_arg).is_file():
            raise SourceProbeError(f"video file not found: {capture_arg}")
    else:
        raise TypeError("source must be a camera index or video file path")

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise SourceProbeError("OpenCV is required for --probe-source (install opencv-python)") from exc

    capture = cv2_module.VideoCapture(capture_arg)
    try:
        if not capture.isOpened():
            raise SourceProbeError(f"cannot open capture source: {capture_arg}")
        expected_dimensions = None
        started = perf_counter()
        for index in range(frames):
            ok, frame = capture.read()
            if not ok or frame is None or len(getattr(frame, 'shape', ())) < 2:
                raise SourceProbeError(f'capture source did not decode frame {index + 1}/{frames}: {capture_arg}')
            shape = frame.shape
            if len(shape) != 3 or shape[2] != 3:
                raise SourceProbeError(f'frame {index + 1}/{frames} must be 3-channel BGR; got shape {shape}')
            height, width = shape[:2]
            if width <= 0 or height <= 0:
                raise SourceProbeError(f'frame {index + 1}/{frames} has invalid dimensions')
            if expected_dimensions is None:
                expected_dimensions = (height, width)
            elif (height, width) != expected_dimensions:
                raise SourceProbeError(f'capture dimensions changed at frame {index + 1}/{frames}: '
                                       f'{expected_dimensions[1]}x{expected_dimensions[0]} to {width}x{height}')
        decode_seconds = max(0.0, perf_counter() - started)
        height, width = frame.shape[:2]
        fps_raw = float(capture.get(cv2_module.CAP_PROP_FPS))
        from math import isfinite
        fps = fps_raw if isfinite(fps_raw) and fps_raw > 0 else None
        return SourceProbeResult(str(capture_arg), int(width), int(height), fps, frames, decode_seconds, 3)
    finally:
        capture.release()
