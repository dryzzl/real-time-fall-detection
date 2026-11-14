"""Synthetic monitor must remain inspectable after its short capture ends."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fall_detection.monitor_cli import main


class _FinishedChild:
    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


class MonitorDemoHoldDefaultsTests(unittest.TestCase):
    def _run(self, *args):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "stream.jsonl"
            with patch("fall_detection.monitor_cli.subprocess.Popen", return_value=_FinishedChild()), \
                 patch("fall_detection.monitor_cli.time.sleep") as sleep, \
                 patch("fall_detection.monitor_cli.webbrowser.open"):
                result = main([*args, "--output", str(output), "--port", "0"])
            self.assertEqual(result, 0)
            return [call.args[0] for call in sleep.call_args_list]

    def test_demo_keeps_dashboard_open_for_thirty_seconds_by_default(self):
        self.assertEqual(self._run("--demo"), [30.0])

    def test_demo_explicit_zero_disables_hold(self):
        self.assertEqual(self._run("--demo", "--hold-seconds", "0"), [])

    def test_live_camera_keeps_previous_immediate_exit_default(self):
        self.assertEqual(self._run("--camera", "0"), [])


if __name__ == "__main__":
    unittest.main()
