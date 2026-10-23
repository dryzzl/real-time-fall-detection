"""Live CLI preview selection regression test using mocks."""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from fall_detection.live_cli import main
from fall_detection.pipeline import RunSummary


class PreviewCliTests(unittest.TestCase):
    def test_preview_opt_in(self):
        with patch("fall_detection.live_cli.OnnxPoseEstimator") as pose, \
             patch("fall_detection.live_cli.open_capture") as capture, \
             patch("fall_detection.live_cli.LivePreview") as preview, \
             patch("fall_detection.live_cli.run_live_pipeline") as run, \
             redirect_stdout(io.StringIO()):
            pose.return_value.config.keypoint_count = 17
            capture.return_value.__enter__.return_value = iter(())
            run.return_value = RunSummary(0, 0, 0.1, 0)
            self.assertEqual(main(["--camera", "0", "--preview"]), 0)
            self.assertIsNotNone(run.call_args.kwargs["on_visual_frame"])
            preview.assert_called_once()
            preview.reset_mock()
            self.assertEqual(main(["--camera", "0", "--headless"]), 0)
            self.assertIsNone(run.call_args.kwargs["on_visual_frame"])
            preview.assert_not_called()
