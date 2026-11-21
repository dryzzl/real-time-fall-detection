"""Capture readiness probe tests; no physical camera or OpenCV required."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from fall_detection.source_probe import SourceProbeError, probe_source


class FakeCapture:
    def __init__(self, *, opened=True, frame=None, fps=30.0):
        self.opened = opened
        self.frame = frame
        self.fps = fps
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frame is not None, self.frame

    def get(self, key):
        return self.fps

    def release(self):
        self.released = True


class SourceProbeTests(unittest.TestCase):
    def test_camera_probe_reports_decoded_dimensions_and_fps(self):
        capture = FakeCapture(frame=SimpleNamespace(shape=(720, 1280, 3)))
        cv2 = SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_FPS=5)
        result = probe_source(0, cv2_module=cv2)
        self.assertEqual((result.width, result.height, result.fps), (1280, 720, 30.0))
        self.assertTrue(capture.released)

    def test_decode_failure_releases_capture(self):
        capture = FakeCapture()
        cv2 = SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_FPS=5)
        with self.assertRaisesRegex(SourceProbeError, "did not decode"):
            probe_source(0, cv2_module=cv2)
        self.assertTrue(capture.released)

    def test_open_failure_releases_capture(self):
        capture = FakeCapture(opened=False)
        cv2 = SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_FPS=5)
        with self.assertRaisesRegex(SourceProbeError, "cannot open"):
            probe_source(0, cv2_module=cv2)
        self.assertTrue(capture.released)

    def test_video_path_and_invalid_fps(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            path.write_bytes(b"fixture path only")
            capture = FakeCapture(frame=SimpleNamespace(shape=(480, 640, 3)), fps=float("nan"))
            cv2 = SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_FPS=5)
            result = probe_source(path, cv2_module=cv2)
            self.assertEqual(result.source, str(path))
            self.assertIsNone(result.fps)

    def test_multiple_frames_and_boundaries(self):
        capture = FakeCapture(frame=SimpleNamespace(shape=(720, 1280, 3)))
        cv2 = SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_FPS=5)
        self.assertEqual(probe_source(0, frames=3, cv2_module=cv2).frames_decoded, 3)
        self.assertTrue(capture.released)
        for count in (0, 121, True, 1.5):
            with self.assertRaises(ValueError):
                probe_source(0, frames=count)

    def test_missing_file_and_negative_camera_rejected(self):
        with self.assertRaises(SourceProbeError):
            probe_source("/no/such/video.mp4")
        with self.assertRaises(ValueError):
            probe_source(-1)


if __name__ == "__main__":
    unittest.main()
