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
