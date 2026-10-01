"""Optional OpenCV sources with explicit resource ownership and timestamp policy."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from math import isfinite
from pathlib import Path
import time

from .contracts import FramePacket


class CaptureError(RuntimeError):
    """A capture or optional dependency could not be used."""


def load_opencv():
    try:
        return importlib.import_module("cv2")
    except (ImportError, OSError) as error:
        raise CaptureError(
            'OpenCV is unavailable. Install one extra: pip install -e ".[video]" '
            'for desktop preview, or pip install -e ".[headless]" for no display.'
        ) from error


@dataclass(frozen=True)
class CaptureConfig:
    camera: int | None = None
    video: Path | None = None
    max_frames: int | None = None
    fallback_fps: float = 30.0

    def __post_init__(self):
        if (self.camera is None) == (self.video is None):
            raise ValueError("choose exactly one camera index or local video file")
        if self.camera is not None and (type(self.camera) is not int or self.camera < 0):
            raise ValueError("camera index must be a nonnegative integer")
        if self.video is not None and (not isinstance(self.video, Path) or not self.video.is_file()):
            raise ValueError("video must be an existing local file; stream URLs are not supported")
        if self.max_frames is not None and (type(self.max_frames) is not int or self.max_frames < 1):
            raise ValueError("max_frames must be a positive integer")
        if (type(self.fallback_fps) not in (int, float) or not isfinite(self.fallback_fps)
                or not 0 < self.fallback_fps <= 240):
            raise ValueError("fallback_fps must be finite, greater than 0 and at most 240")


@contextmanager
def open_capture(config: CaptureConfig) -> Iterator[Iterator[FramePacket]]:
    """Use as a context manager, including when stopping iteration early.

    Video timestamps prefer decoder positions and permanently switch to estimated
    FPS timing if those positions become invalid. Camera time is acquisition time
    measured with a monotonic clock, not the sensor's exposure timestamp.
    """
    cv = load_opencv()
    capture = None
    try:
        capture = cv.VideoCapture()
        target = config.camera if config.camera is not None else str(config.video.resolve())
        if not capture.open(target) or not capture.isOpened():
            raise CaptureError("could not open the selected camera or video; check access and codec support")
        yield _frames(capture, cv, config)
    except cv.error as error:
        raise CaptureError(f"OpenCV capture failed: {error}") from error
    finally:
        if capture is not None:
            try:
                capture.release()
            except cv.error as error:
                raise CaptureError(f"could not release capture: {error}") from error


def _frames(capture, cv, config: CaptureConfig) -> Iterator[FramePacket]:
    camera = config.camera is not None
    fps = capture.get(cv.CAP_PROP_FPS)
    valid_fps = isfinite(fps) and 0 < fps <= 240
    fps = fps if valid_fps else config.fallback_fps
    count = capture.get(cv.CAP_PROP_FRAME_COUNT) if not camera else 0.0
    expected_frames = round(count) if isfinite(count) and count > 0 else None
    origin = None
    last = -1.0
    use_position = not camera
    index = 0
    while config.max_frames is None or index < config.max_frames:
        ok, pixels = capture.read()
        if not ok:
            if camera:
                raise CaptureError("camera frame read failed; the device may have disconnected")
            if index == 0:
                raise CaptureError("video contains no decodable frames")
            if expected_frames is not None and index < expected_frames:
                raise CaptureError(f"video stopped early: decoded {index} of {expected_frames} reported frames")
            return
        if (pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3
                or pixels.dtype.name != "uint8" or min(pixels.shape[:2]) <= 0):
            raise CaptureError("capture returned an invalid image; expected nonempty uint8 BGR pixels")
        if camera:
            now = time.monotonic()
            if origin is None:
                origin = now
            timestamp = now - origin
            basis = "camera_monotonic"
            if not isfinite(timestamp) or timestamp <= last:
                raise CaptureError("camera acquisition clock did not increase")
        else:
            position = capture.get(cv.CAP_PROP_POS_MSEC) / 1000.0
            if use_position:
                if origin is None:
                    origin = position if isfinite(position) and position >= 0 else 0.0
                timestamp = position - origin
                use_position = isfinite(position) and position >= 0 and timestamp > last
            if use_position:
                basis = "video_position"
            else:
                timestamp = 0.0 if index == 0 else last + 1.0 / fps
                basis = "video_fps" if valid_fps else "video_fallback_fps"
        height, width = pixels.shape[:2]
        yield FramePacket(index, timestamp, width, height, pixels.tobytes(),
                          source=f"camera:{config.camera}" if camera else "video",
                          timestamp_basis=basis)
        index += 1
        last = timestamp
