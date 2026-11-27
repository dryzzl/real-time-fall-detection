import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from fall_detection.monitor_cli import main


class MonitorModelPreflightTests(unittest.TestCase):
    def test_failed_model_check_prevents_capture_and_dashboard(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.jsonl"
            with patch("fall_detection.monitor_cli.subprocess.run") as run, \
                 patch("fall_detection.monitor_cli.create_server") as server, \
                 patch("fall_detection.monitor_cli.subprocess.Popen") as child:
                run.return_value = SimpleNamespace(returncode=2)
                self.assertEqual(main(["--camera", "0", "--output", str(output),
                                       "--check-models"]), 2)
                self.assertIn("--check-models", run.call_args.args[0])
                server.assert_not_called()
                child.assert_not_called()
                self.assertFalse(output.exists())
