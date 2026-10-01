import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fall_detection.capture import CaptureConfig, CaptureError, load_opencv, open_capture
from fall_detection.cli import main
from fall_detection.contracts import FramePacket, Prediction
from fall_detection.pipeline import run_pipeline
from fall_detection.preview import Preview


class CvError(Exception):
    pass


class Pixels:
    ndim = 3
    shape = (2, 4, 3)
    dtype = SimpleNamespace(name="uint8")

    def tobytes(self):
        return b"\x10\x20\x30" * 8


class FakeCapture:
    def __init__(self, reads=None, *, opened=True, fps=10.0, count=0.0, positions=None):
        self.reads = iter(reads if reads is not None else [(True, Pixels())])
        self.opened, self.fps, self.count = opened, fps, count
        self.positions = iter(positions if positions is not None else [0.0] * 100)
        self.releases = 0
        self.target = None

    def open(self, target):
        self.target = target
        return self.opened

    def isOpened(self):
        return self.opened

    def read(self):
        result = next(self.reads, (False, None))
        if isinstance(result, BaseException):
            raise result
        return result

    def get(self, key):
        if key == "fps":
            return self.fps
        if key == "count":
            return self.count
        return next(self.positions)

    def release(self):
        self.releases += 1


def fake_cv(capture):
    return SimpleNamespace(VideoCapture=Mock(return_value=capture), error=CvError,
                           CAP_PROP_FPS="fps", CAP_PROP_FRAME_COUNT="count", CAP_PROP_POS_MSEC="position")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.video = Path(self.directory.name) / "fixture.avi"
        self.video.touch()

    def consume(self, capture, **kwargs):
        with patch("fall_detection.capture.load_opencv", return_value=fake_cv(capture)):
            with open_capture(CaptureConfig(video=self.video, **kwargs)) as frames:
                return list(frames)

    def test_configuration_rejects_ambiguous_invalid_or_remote_sources(self):
        for values in ({}, {"camera": 0, "video": self.video}, {"camera": True}, {"camera": -1},
                       {"camera": 0, "max_frames": 0}, {"camera": 0, "max_frames": True},
                       {"camera": 0, "fallback_fps": float("nan")},
                       {"camera": 0, "fallback_fps": True},
                       {"video": Path("https://example.invalid/video.mp4")}, {"video": Path(self.directory.name)}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CaptureConfig(**values)

    def test_missing_dependency_has_install_instructions(self):
        with patch("fall_detection.capture.importlib.import_module", side_effect=ImportError), self.assertRaisesRegex(CaptureError, "headless"):
            load_opencv()

    def test_failed_open_releases_resource(self):
        capture = FakeCapture(opened=False)
        with self.assertRaisesRegex(CaptureError, "could not open"):
            self.consume(capture)
        self.assertEqual(capture.releases, 1)

    def test_first_read_failure_is_not_successful_empty_video(self):
        capture = FakeCapture(reads=[])
        with self.assertRaisesRegex(CaptureError, "no decodable"):
            self.consume(capture)
        self.assertEqual(capture.releases, 1)

    def test_premature_video_end_is_reported(self):
        capture = FakeCapture(count=3)
        with self.assertRaisesRegex(CaptureError, "decoded 1 of 3"):
            self.consume(capture)
        self.assertEqual(capture.releases, 1)

    def test_decoder_exception_is_wrapped_and_released(self):
        capture = FakeCapture(reads=[CvError("bad decode")])
        with self.assertRaisesRegex(CaptureError, "bad decode"):
            self.consume(capture)
        self.assertEqual(capture.releases, 1)

    def test_invalid_decoded_images_fail_and_release(self):
        for pixels in (None, SimpleNamespace(ndim=2),
                       SimpleNamespace(ndim=3, shape=(2, 4, 3), dtype=SimpleNamespace(name="float32"))):
            with self.subTest(pixels=pixels):
                capture = FakeCapture(reads=[(True, pixels)])
                with self.assertRaisesRegex(CaptureError, "invalid image"):
                    self.consume(capture)
                self.assertEqual(capture.releases, 1)

    def test_video_uses_relative_decoder_timestamps_and_native_dimensions(self):
        capture = FakeCapture([(True, Pixels())] * 3, count=3, positions=[1000, 1100, 1250])
        frames = self.consume(capture)
        self.assertEqual(len(frames), 3)
        for actual, expected in zip(frames, [0, 0.1, 0.25]):
            self.assertAlmostEqual(actual.timestamp_seconds, expected)
            self.assertEqual(actual.timestamp_basis, "video_position")
            self.assertEqual((actual.width, actual.height, len(actual.bgr)), (4, 2, 24))
        self.assertEqual(capture.target, str(self.video.resolve()))
        self.assertEqual(capture.releases, 1)

    def test_bad_positions_switch_permanently_to_estimated_fps(self):
        capture = FakeCapture([(True, Pixels())] * 4, positions=[0, 100, 50, 9000])
        frames = self.consume(capture)
        for actual, expected in zip(frames, [0, 0.1, 0.2, 0.3]):
            self.assertAlmostEqual(actual.timestamp_seconds, expected)
        self.assertEqual([f.timestamp_basis for f in frames],
                         ["video_position", "video_position", "video_fps", "video_fps"])

    def test_invalid_fps_and_positions_use_explicit_fallback(self):
        for fps in (0, float("nan"), float("inf"), -1):
            with self.subTest(fps=fps):
                frames = self.consume(FakeCapture([(True, Pixels())] * 2, fps=fps,
                                                  positions=[float("nan"), 0]), fallback_fps=20)
                self.assertEqual([f.timestamp_seconds for f in frames], [0, 0.05])
                self.assertTrue(all(f.timestamp_basis == "video_fallback_fps" for f in frames))

    def test_camera_timing_and_disconnect(self):
        capture = FakeCapture([(True, Pixels())] * 2)
        with patch("fall_detection.capture.load_opencv", return_value=fake_cv(capture)), patch(
                "fall_detection.capture.time.monotonic", side_effect=[42, 42.25]):
            with open_capture(CaptureConfig(camera=0)) as frames:
                self.assertEqual(next(frames).timestamp_seconds, 0)
                second = next(frames)
                self.assertEqual((second.timestamp_seconds, second.timestamp_basis), (0.25, "camera_monotonic"))
                with self.assertRaisesRegex(CaptureError, "disconnected"):
                    next(frames)
        self.assertEqual(capture.target, 0)
        self.assertEqual(capture.releases, 1)

    def test_frame_limit_and_consumer_stop_release_without_extra_reads(self):
        for limit in (None, 1):
            capture = FakeCapture([(True, Pixels()), AssertionError("must not read another frame")])
            with patch("fall_detection.capture.load_opencv", return_value=fake_cv(capture)):
                with open_capture(CaptureConfig(video=self.video, max_frames=limit)) as frames:
                    result = run_pipeline(frames, lambda record: None, on_frame=lambda frame, prediction: False)
            self.assertEqual(result.frames_processed, 1)
            self.assertEqual(capture.releases, 1)
        capture = FakeCapture([(True, Pixels()), AssertionError("limit did not stop")])
        self.assertEqual(len(self.consume(capture, max_frames=1)), 1)
        self.assertEqual(capture.releases, 1)

    def test_consumer_error_releases_capture(self):
        capture = FakeCapture()
        with patch("fall_detection.capture.load_opencv", return_value=fake_cv(capture)):
            with self.assertRaisesRegex(OSError, "disk full"):
                with open_capture(CaptureConfig(video=self.video)) as frames:
                    run_pipeline(frames, Mock(side_effect=OSError("disk full")))
        self.assertEqual(capture.releases, 1)

    def test_interrupt_preserves_records_and_releases_capture(self):
        capture = FakeCapture([(True, Pixels()), KeyboardInterrupt()])
        output = io.StringIO()
        with patch("fall_detection.capture.load_opencv", return_value=fake_cv(capture)), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            result = main(["capture", "--video", str(self.video)])
        self.assertEqual(result, 130)
        self.assertEqual(json.loads(output.getvalue())["state"], "unknown")
        self.assertEqual(capture.releases, 1)

    def test_headless_cli_does_not_create_preview(self):
        output = io.StringIO()
        with patch("fall_detection.capture.load_opencv", return_value=fake_cv(FakeCapture())), patch("fall_detection.cli.Preview") as preview, contextlib.redirect_stdout(output):
            self.assertEqual(main(["capture", "--video", str(self.video), "--headless"]), 0)
        preview.assert_not_called()
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(records[0]["timestamp_basis"], "video_position")
        self.assertEqual(records[-1]["frames_processed"], 1)

    def test_existing_output_prevents_camera_access(self):
        output = Path(self.directory.name) / "existing.jsonl"
        output.write_text("keep", encoding="utf-8")
        with patch("fall_detection.cli.open_capture") as source, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["capture", "--camera", "0", "--output", str(output)]), 2)
        source.assert_not_called()
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_capture_error_returns_nonzero_without_success_summary(self):
        output = io.StringIO()
        with patch("fall_detection.capture.load_opencv", return_value=fake_cv(FakeCapture(reads=[]))), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["capture", "--video", str(self.video)]), 2)
        self.assertEqual(output.getvalue(), "")

    def test_release_error_does_not_emit_success_summary(self):
        capture = FakeCapture()
        capture.release = Mock(side_effect=CvError("release failed"))
        output = io.StringIO()
        with patch("fall_detection.capture.load_opencv", return_value=fake_cv(capture)), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["capture", "--video", str(self.video)]), 2)
        self.assertEqual(json.loads(output.getvalue())["event"], "frame")
        capture.release.assert_called_once()


class PreviewTests(unittest.TestCase):
    def make_cv(self, key=-1, visible=1):
        return SimpleNamespace(error=CvError, WINDOW_NORMAL=0, FONT_HERSHEY_SIMPLEX=0, WND_PROP_VISIBLE=1,
                               getBuildInformation=Mock(return_value="GUI: QT"), namedWindow=Mock(),
                               putText=Mock(), imshow=Mock(), waitKey=Mock(return_value=key),
                               getWindowProperty=Mock(return_value=visible), destroyWindow=Mock())

    def test_quit_keys_and_close_button_stop_preview_and_cleanup(self):
        for key, visible, expected in ((-1, 1, True), (ord("q"), 1, False), (27, 1, False), (-1, 0, False)):
            with self.subTest(key=key, visible=visible):
                cv = self.make_cv(key, visible)
                with patch("fall_detection.preview.load_opencv", return_value=cv), patch.dict(sys.modules, {"numpy": Mock()}), patch.dict("os.environ", {"DISPLAY": ":test"}):
                    with Preview() as preview:
                        self.assertIs(preview.show(FramePacket(0, 0, 4, 2, Pixels().tobytes()), Prediction()), expected)
                cv.destroyWindow.assert_called_once()
                self.assertIn("unknown", cv.putText.call_args.args[1])

    def test_headless_build_and_missing_display_fail_before_gui_calls(self):
        for build in ("GUI: NONE", "GUI: QT"):
            cv = self.make_cv()
            cv.getBuildInformation.return_value = build
            with patch("fall_detection.preview.load_opencv", return_value=cv), patch.dict(sys.modules, {"numpy": Mock()}), patch("fall_detection.preview.sys.platform", "linux"), patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(CaptureError):
                    with Preview():
                        self.fail("preview should not open")
            cv.namedWindow.assert_not_called()

    def test_display_error_closes_window(self):
        cv = self.make_cv()
        cv.imshow.side_effect = CvError("display unavailable")
        with patch("fall_detection.preview.load_opencv", return_value=cv), patch.dict(sys.modules, {"numpy": Mock()}), patch.dict("os.environ", {"DISPLAY": ":test"}):
            with self.assertRaisesRegex(CaptureError, "display unavailable"):
                with Preview() as preview:
                    preview.show(FramePacket(0, 0, 4, 2, Pixels().tobytes()), Prediction())
        cv.destroyWindow.assert_called_once()

    def test_closed_window_property_error_is_a_clean_stop(self):
        cv = self.make_cv()
        cv.getWindowProperty.side_effect = CvError("window was closed")
        with patch("fall_detection.preview.load_opencv", return_value=cv), patch.dict(sys.modules, {"numpy": Mock()}), patch.dict("os.environ", {"DISPLAY": ":test"}):
            with Preview() as preview:
                self.assertFalse(preview.show(FramePacket(0, 0, 4, 2, Pixels().tobytes()), Prediction()))
        cv.destroyWindow.assert_called_once()


try:
    import cv2
    import numpy as np
except (ImportError, OSError):
    cv2 = None


@unittest.skipIf(cv2 is None, "install the headless or video extra for real decoder tests")
class GeneratedVideoTests(unittest.TestCase):
    def test_generated_video_decodes_through_installed_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.avi"
            output = Path(directory) / "records.jsonl"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened(), "MJPG fixture encoder unavailable")
            try:
                for index in range(5):
                    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
                    pixels[:, index * 8:index * 8 + 8] = (40, 150, 240)
                    writer.write(pixels)
            finally:
                writer.release()
            result = subprocess.run([sys.executable, "-m", "fall_detection", "capture", "--video", str(path),
                                     "--headless", "--output", str(output)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["frame_index"] for r in records[:-1]], list(range(5)))
            for index, record in enumerate(records[:-1]):
                self.assertAlmostEqual(record["timestamp_seconds"], index / 10)
                self.assertEqual((record["width"], record["height"], record["state"]), (64, 48, "unknown"))
                self.assertIsNone(record["confidence"])
                self.assertNotIn("bgr", record)
            self.assertEqual(records[-1]["frames_processed"], 5)
            self.assertAlmostEqual(records[-1]["source_duration_seconds"], 0.4)
            self.assertEqual(json.loads(result.stdout)["frames_processed"], 5)
            # Reopening after CLI exit also verifies the native file handle was released.
            with open_capture(CaptureConfig(video=path, max_frames=2)) as frames:
                self.assertEqual(len(list(frames)), 2)


if __name__ == "__main__":
    unittest.main()
