"""Preflight and post-failure browser lifecycle regression tests.

All child processes are mocked; no camera or model weights are required.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fall_detection.monitor_cli import main


class _Child:
    def __init__(self, exit_code):
        self.exit_code = exit_code

    def wait(self, timeout=None):
        return self.exit_code

    def poll(self):
        return self.exit_code


class MonitorPreflightAndFailureHoldTests(unittest.TestCase):
    def test_missing_video_or_model_fails_before_browser_or_child(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "authorized.mp4"
            video.write_bytes(b"not decoded in this test")
            output = str(root / "stream.jsonl")
            cases = (
                ["--video", str(root / "missing.mp4")],
                ["--video", str(video), "--model", str(root / "missing.onnx")],
                ["--camera", "0", "--temporal-model", str(root / "missing.onnx")],
                ["--camera", "0", "--research-alerts", "--alert-config", str(root / "missing.toml")],
            )
            with patch("fall_detection.monitor_cli.create_server") as server, \\
                 patch("fall_detection.monitor_cli.subprocess.Popen") as popen:
                for options in cases:
                    with self.subTest(options=options), self.assertRaises(SystemExit) as error:
                        main([*options, "--output", output])
                    self.assertEqual(error.exception.code, 2)
                server.assert_not_called()
                popen.assert_not_called()

    def test_existing_video_path_is_accepted_for_child_capture(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "authorized.mp4"
            video.write_bytes(b"capture is mocked")
            with patch("fall_detection.monitor_cli.subprocess.Popen", return_value=_Child(0)) as popen:
                self.assertEqual(main(["--video", str(video), "--output",
                                       str(root / "stream.jsonl"), "--port", "0"]), 0)
                self.assertEqual(popen.call_args.args[0][3:5], ["--video", str(video)])

    def test_failed_capture_uses_failure_hold_instead_of_success_hold(self):
        with TemporaryDirectory() as folder:
            output = str(Path(folder) / "stream.jsonl")
            with patch("fall_detection.monitor_cli.subprocess.Popen", return_value=_Child(7)), \\
                 patch("fall_detection.monitor_cli.time.sleep") as sleep:
                result = main(["--camera", "0", "--output", output, "--port", "0",
                               "--hold-seconds", "9", "--failure-hold-seconds", "3"])
            self.assertEqual(result, 7)
            sleep.assert_called_once_with(3.0)

    def test_successful_capture_preserves_normal_hold(self):
        with TemporaryDirectory() as folder:
            output = str(Path(folder) / "stream.jsonl")
            with patch("fall_detection.monitor_cli.subprocess.Popen", return_value=_Child(0)), \\
                 patch("fall_detection.monitor_cli.time.sleep") as sleep:
                result = main(["--camera", "0", "--output", output, "--port", "0",
                               "--hold-seconds", "2", "--failure-hold-seconds", "3"])
            self.assertEqual(result, 0)
            sleep.assert_called_once_with(2.0)

    def test_invalid_failure_hold_rejected_before_launch(self):
        with TemporaryDirectory() as folder:
            output = str(Path(folder) / "stream.jsonl")
            with patch("fall_detection.monitor_cli.subprocess.Popen") as popen:
                for value in ("-1", "3601", "nan", "inf"):
                    with self.subTest(value=value), self.assertRaises(SystemExit) as error:
                        main(["--camera", "0", "--output", output,
                              "--failure-hold-seconds", value])
                    self.assertEqual(error.exception.code, 2)
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
