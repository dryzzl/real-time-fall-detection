"""CLI wiring for opt-in research-only alert metadata (no camera or weights)."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fall_detection.alerts import AlertManager
from fall_detection.live_cli import main
from fall_detection.pipeline import RunSummary


class ResearchAlertCliTests(unittest.TestCase):
    def run_mocked(self, args):
        with (patch("fall_detection.live_cli.OnnxPoseEstimator") as pose,
              patch("fall_detection.live_cli.open_capture",
                    return_value=nullcontext(iter(()))),
              patch("fall_detection.live_cli.run_live_pipeline") as pipeline,
              redirect_stdout(io.StringIO())):
            pose.return_value.config.keypoint_count = 17
            pipeline.return_value = RunSummary(0, 0.0, 0.0, 0.0)
            code = main(["--camera", "0", *args])
            self.assertEqual(code, 0)
            return pipeline.call_args.kwargs

    def test_opt_in_supplies_real_alert_manager_to_live_pipeline(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "live.jsonl"
            kwargs = self.run_mocked(["--research-alerts", "--output", str(output)])
            self.assertIsInstance(kwargs["alert_manager"], AlertManager)
            self.assertEqual(kwargs["alert_manager"].config.fallen_persistence_seconds, 1.0)
            self.assertTrue(output.exists())

    def test_default_mode_does_not_create_alert_manager(self):
        self.assertIsNone(self.run_mocked([])["alert_manager"])

    def test_bad_alert_configuration_fails_before_capture_and_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "live.jsonl"
            with (patch("fall_detection.live_cli.OnnxPoseEstimator") as pose,
                  patch("fall_detection.live_cli.open_capture") as capture,
                  redirect_stderr(io.StringIO()) as errors):
                pose.return_value.config.keypoint_count = 17
                code = main(["--camera", "0", "--research-alerts",
                             "--alert-config", str(root / "missing.toml"),
                             "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("missing.toml", errors.getvalue())
            capture.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
