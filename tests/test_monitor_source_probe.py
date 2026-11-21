"""Monitor source-probe integration tests; no real camera or model required."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fall_detection.monitor_cli import main
from fall_detection.source_probe import SourceProbeError, SourceProbeResult


class MonitorSourceProbeTests(unittest.TestCase):
    def test_dry_run_probe_reports_dimensions_without_starting_capture(self):
        with TemporaryDirectory() as folder:
            output = str(Path(folder) / "capture.jsonl")
            stdout = io.StringIO()
            with patch("fall_detection.monitor_cli.probe_source",
                       return_value=SourceProbeResult("0", 1280, 720, 29.97)) as probe, \
                 patch("fall_detection.monitor_cli.create_server") as server, \
                 patch("fall_detection.monitor_cli.subprocess.Popen") as popen, \
                 contextlib.redirect_stdout(stdout):
                code = main(["--camera", "0", "--output", output, "--probe-source", "--dry-run"])
            self.assertEqual(code, 0)
            probe.assert_called_once_with(0, frames=1)
            server.assert_not_called()
            popen.assert_not_called()
            self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1])["source_probe"]["width"], 1280)
            self.assertFalse(Path(output).exists())

    def test_failed_probe_stops_before_server_and_capture(self):
        with TemporaryDirectory() as folder:
            with patch("fall_detection.monitor_cli.probe_source",
                       side_effect=SourceProbeError("camera busy")), \
                 patch("fall_detection.monitor_cli.create_server") as server, \
                 patch("fall_detection.monitor_cli.subprocess.Popen") as popen, \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = main(["--camera", "0", "--output", str(Path(folder) / "log.jsonl"),
                             "--probe-source"])
            self.assertEqual(code, 2)
            self.assertIn("camera busy", stderr.getvalue())
            server.assert_not_called()
            popen.assert_not_called()

    def test_multiframe_probe_reports_count(self):
        with TemporaryDirectory() as folder:
            output = str(Path(folder) / 'log.jsonl')
            stdout = io.StringIO()
            result = SourceProbeResult('0', 640, 480, 30.0, 5)
            with patch('fall_detection.monitor_cli.probe_source', return_value=result) as probe:
                with contextlib.redirect_stdout(stdout):
                    code = main(['--camera', '0', '--output', output, '--probe-source', '--probe-frames', '5', '--dry-run'])
            self.assertEqual(code, 0)
            probe.assert_called_once_with(0, frames=5)
            self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1])['source_probe']['frames_decoded'], 5)

    def test_multiframe_requires_probe_opt_in(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(SystemExit) as error:
                main(['--camera', '0', '--output', str(Path(folder) / 'log.jsonl'), '--probe-frames', '5', '--dry-run'])
            self.assertEqual(error.exception.code, 2)

    def test_demo_rejects_probe_before_launch(self):
        with TemporaryDirectory() as folder:
            with patch("fall_detection.monitor_cli.probe_source") as probe, \
                 self.assertRaises(SystemExit) as error:
                main(["--demo", "--probe-source", "--output", str(Path(folder) / "log.jsonl")])
            self.assertEqual(error.exception.code, 2)
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
