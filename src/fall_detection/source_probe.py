"""Explicit opt-in capture readiness probe for research video sources.

A probe opens a camera or video, decodes exactly one frame, and closes it.
It does not load model weights or make fall-detection claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SourceProbeError(RuntimeError):
    """A capture source could not produce a decoded frame."""


@dataclass(frozen=True)
class SourceProbeResult:
    source: str
    width: int
    height: int
    fps: float | None


def probe_source(source: int | str | Path, *, cv2_module: Any = None) -> SourceProbeResult:
    """Decode one frame and release the device even on read failures.

    Requires OpenCV only when invoked. A successful probe establishes capture
    readiness at that instant, not inference correctness or future availability.
    """
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
        ok, frame = capture.read()
        if not ok or frame is None or len(getattr(frame, "shape", ())) < 2:
            raise SourceProbeError(f"capture source did not decode a frame: {capture_arg}")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise SourceProbeError("decoded frame dimensions must be positive")
        fps_raw = float(capture.get(cv2_module.CAP_PROP_FPS))
        from math import isfinite
        fps = fps_raw if isfinite(fps_raw) and fps_raw > 0 else None
        return SourceProbeResult(str(capture_arg), int(width), int(height), fps)
    finally:
        capture.release()
