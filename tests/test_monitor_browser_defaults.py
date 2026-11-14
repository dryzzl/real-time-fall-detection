"""One-command demo opens a browser; unattended live capture remains headless."""
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


class MonitorBrowserDefaultsTests(unittest.TestCase):
    def _launch(self, *args):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "monitor.jsonl"
            with patch("fall_detection.monitor_cli.subprocess.Popen", return_value=_FinishedChild()), \
                 patch("fall_detection.monitor_cli.time.sleep"), \
                 patch("fall_detection.monitor_cli.webbrowser.open") as browser:
                code = main([*args, "--output", str(output), "--port", "0"])
                calls = browser.call_args_list
            self.assertEqual(code, 0)
            return calls

    def test_synthetic_demo_opens_loopback_dashboard_by_default(self):
        calls = self._launch("--demo")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].args[0].startswith("http://127.0.0.1:"))

    def test_headless_demo_explicitly_suppresses_browser(self):
        self.assertEqual(self._launch("--demo", "--no-open-browser"), [])

    def test_live_capture_does_not_open_browser_unless_requested(self):
        self.assertEqual(self._launch("--camera", "0"), [])
        self.assertEqual(len(self._launch("--camera", "0", "--open-browser")), 1)


if __name__ == "__main__":
    unittest.main()
