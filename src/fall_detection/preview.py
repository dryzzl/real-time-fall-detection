"""Optional local desktop display; no GUI calls in headless mode."""

import importlib
import os
import re
import sys

from .capture import CaptureError, load_opencv
from .contracts import FramePacket, Prediction


class Preview:
    title = "Fall Detection | Q or Esc to quit"

    def __enter__(self):
        self.cv = load_opencv()
        self.np = importlib.import_module("numpy")
        if re.search(r"GUI:\s*(NONE|NO)\b", self.cv.getBuildInformation()):
            raise CaptureError('preview needs the desktop extra: pip install -e ".[video]"; use one OpenCV package')
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise CaptureError("preview needs a local graphical display; omit --preview on a headless machine")
        try:
            self.cv.namedWindow(self.title, self.cv.WINDOW_NORMAL)
        except self.cv.error as error:
            self.close()
            raise CaptureError(f"could not create preview window: {error}") from error
        return self

    def show(self, frame: FramePacket, prediction: Prediction) -> bool:
        pixels = self.np.frombuffer(frame.bgr, dtype=self.np.uint8).reshape(frame.height, frame.width, 3).copy()
        try:
            self.cv.putText(pixels, f"{prediction.state.value} | {prediction.reason}", (12, 28),
                            self.cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2)
            self.cv.imshow(self.title, pixels)
            key = self.cv.waitKey(1) & 0xFF
        except self.cv.error as error:
            raise CaptureError(f"preview failed: {error}") from error
        if key in (ord("q"), ord("Q"), 27):
            return False
        try:
            return self.cv.getWindowProperty(self.title, self.cv.WND_PROP_VISIBLE) >= 1
        except self.cv.error:
            return False  # Some backends raise when the user closes the window.

    def close(self):
        try:
            self.cv.destroyWindow(self.title)
        except self.cv.error:
            pass  # A user may already have closed this window.

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
