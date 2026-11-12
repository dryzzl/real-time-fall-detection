"""Tests for keeping the local dashboard visible after a short capture."""
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch
from contextlib import redirect_stderr

from fall_detection.monitor_cli import main


class MonitorHoldTests(TestCase):
    def test_finished_demo_stays_visible_for_requested_duration(self):
        with TemporaryDirectory() as directory:
            server = MagicMock()
            server.server_port = 8765
            child = MagicMock()
            child.wait.return_value = 0
            child.poll.return_value = 0
            with patch("fall_detection.monitor_cli.create_server", return_value=server), \
                 patch("fall_detection.monitor_cli.subprocess.Popen", return_value=child), \
                 patch("fall_detection.monitor_cli.time.sleep") as sleep:
                result = main(["--demo", "--output", str(Path(directory) / "new.jsonl"),
                               "--hold-seconds", "2.5"])
            self.assertEqual(result, 0)
            sleep.assert_called_once_with(2.5)
            server.shutdown.assert_called_once()
            server.server_close.assert_called_once()

    def test_invalid_hold_seconds_rejected_before_server_start(self):
        with TemporaryDirectory() as directory:
            for value in ("nan", "inf", "-1", "3601"):
                with self.subTest(value=value), \
                     patch("fall_detection.monitor_cli.create_server") as server, \
                     redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        main(["--demo", "--output", str(Path(directory) / "new.jsonl"),
                              "--hold-seconds", value])
                    self.assertEqual(caught.exception.code, 2)
                    server.assert_not_called()
