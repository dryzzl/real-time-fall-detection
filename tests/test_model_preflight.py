import contextlib
import io
import json
import unittest
from unittest.mock import patch

from fall_detection.live_cli import main


class ModelPreflightTests(unittest.TestCase):
    def test_preflight_runs_pose_without_capture(self):
        with patch("fall_detection.live_cli.OnnxPoseEstimator") as factory, \
             patch("fall_detection.live_cli.open_capture") as capture, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            pose = factory.return_value
            pose.config.keypoint_count = 17
            pose.describe.return_value = {"provider": "CPUExecutionProvider"}
            pose.estimate.return_value = ()
            self.assertEqual(main(["--camera", "0", "--check-models"]), 0)
            capture.assert_not_called()
            self.assertEqual(pose.estimate.call_args.args[0].width, 64)
            report = json.loads(output.getvalue())
            self.assertFalse(report["accuracy_validated"])
            self.assertFalse(report["temporal_contract_checked"])

    def test_preflight_requires_no_capture_source(self):
        with patch("fall_detection.live_cli.OnnxPoseEstimator") as factory, \
             patch("fall_detection.live_cli.CaptureConfig") as capture_config, \
             contextlib.redirect_stdout(io.StringIO()):
            pose = factory.return_value
            pose.config.keypoint_count = 17
            pose.describe.return_value = {}
            pose.estimate.return_value = ()
            self.assertEqual(main(["--check-models"]), 0)
            capture_config.assert_not_called()

    def test_live_requires_source(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main([])
        self.assertEqual(error.exception.code, 2)

    def test_preflight_rejects_ignored_capture_options(self):
        for option in (["--output", "unused.jsonl"], ["--latency-report", "unused.json"],
                       ["--preview"], ["--research-alerts"], ["--max-frames", "2"]):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), \
                 patch("fall_detection.live_cli.OnnxPoseEstimator") as factory, \
                 self.assertRaises(SystemExit) as error:
                main(["--check-models", *option])
            self.assertEqual(error.exception.code, 2)
            factory.assert_not_called()
