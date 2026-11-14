"""Integration and lifecycle tests for the one-command live monitor launcher.

All process/camera work is simulated; no trained weights or video are required.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from fall_detection.monitor_cli import build_live_command, main


class MonitorLauncherTests(unittest.TestCase):
    def test_shell_free_command_for_video_and_optional_inference(self):
        from argparse import Namespace
        args = Namespace(
            camera=None, video=Path("video with spaces.mp4"),
            output=Path("out.jsonl"), model=Path("pose.onnx"),
            temporal_model=Path("temporal.onnx"), max_frames=10,
            alert_config=Path("alerts.toml"), research_alerts=True,
        )
        command = build_live_command(args)
        self.assertEqual(command[1:3], ["-m", "fall_detection.live_cli"])
        self.assertEqual(command[3:5], ["--video", "video with spaces.mp4"])
        self.assertIn("--headless", command)
        self.assertIn("--research-alerts", command)
        self.assertEqual(command[command.index("--temporal-model") + 1], "temporal.onnx")

    def test_demo_command_uses_synthetic_producer_without_model_or_camera(self):
        from argparse import Namespace
        args = Namespace(camera=None, video=None, demo=True, output=Path("demo.jsonl"),
                         max_frames=12, model=None, temporal_model=None,
                         alert_config=None, research_alerts=False)
        self.assertEqual(build_live_command(args), [
            __import__("sys").executable, "-m", "fall_detection", "demo",
            "--frames", "12", "--realtime", "--output", "demo.jsonl",
        ])

    def test_demo_browser_stream_integration(self):
        with TemporaryDirectory() as folder:
            from subprocess import run
            from fall_detection.dashboard import read_events
            output = Path(folder) / "demo.jsonl"
            command = build_live_command(__import__("argparse").Namespace(
                camera=None, video=None, demo=True, output=output, max_frames=3,
                model=None, temporal_model=None, alert_config=None, research_alerts=False))
            # Exercise the real synthetic producer, not a mocked frame writer.
            result = run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = read_events(output)
            self.assertEqual(len([e for e in payload["events"] if e["event"] == "frame"]), 3)
            self.assertTrue(all(e.get("state") == "unknown" for e in payload["events"] if e["event"] == "frame"))

    def test_latency_report_is_passed_to_live_child(self):
        from argparse import Namespace
        args = Namespace(camera=0, video=None, demo=False, output=Path("stream.jsonl"),
                         max_frames=8, model=None, temporal_model=None,
                         alert_config=None, research_alerts=False,
                         latency_report=Path("timing.json"))
        command = build_live_command(args)
        self.assertEqual(command[command.index("--latency-report") + 1], "timing.json")

    def test_latency_report_cannot_clobber_stream_or_existing_report(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "stream.jsonl"
            with patch("fall_detection.monitor_cli.subprocess.Popen") as popen:
                for extra in (["--latency-report", str(path)],):
                    with self.assertRaises(SystemExit) as error:
                        main(["--camera", "0", "--output", str(path), *extra])
                    self.assertEqual(error.exception.code, 2)
                report = Path(folder) / "timing.json"
                report.write_text("keep", encoding="utf-8")
                with self.assertRaises(SystemExit) as error:
                    main(["--camera", "0", "--output", str(path), "--latency-report", str(report)])
                self.assertEqual(error.exception.code, 2)
                popen.assert_not_called()
                self.assertEqual(report.read_text(encoding="utf-8"), "keep")

    def test_demo_rejects_inference_flags_that_would_be_ignored(self):
        with TemporaryDirectory() as folder:
            with patch("fall_detection.monitor_cli.subprocess.Popen") as popen:
                with self.assertRaises(SystemExit) as error:
                    main(["--demo", "--output", str(Path(folder) / "out.jsonl"), "--model", "pose.onnx"])
                self.assertEqual(error.exception.code, 2)
                popen.assert_not_called()

    def test_rejects_existing_output_without_launching_child(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "existing.jsonl"
            output.write_text("original", encoding="utf-8")
            with patch("fall_detection.monitor_cli.subprocess.Popen") as popen:
                with self.assertRaises(SystemExit) as error:
                    main(["--camera", "0", "--output", str(output), "--port", "0"])
                self.assertEqual(error.exception.code, 2)
                popen.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "original")

    def test_one_command_exposes_live_jsonl_through_actual_loopback_http(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "live.jsonl"
            observed = {}

            class FakeChild:
                def __init__(self, command):
                    observed["command"] = command

                def wait(self, timeout=None):
                    # A fake producer writes the same metadata contract as live_cli.
                    with output.open("x", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "event": "frame", "frame_index": 0,
                            "state": "unknown", "fall_detection_available": False,
                            "source": "synthetic_test",
                            "bgr": "must_never_reach_browser",
                        }) + "\n")
                    with urlopen(observed["url"] + "/api/snapshot", timeout=3) as response:
                        observed["payload"] = json.load(response)
                    return 0

                def poll(self):
                    return 0

            def fake_launch(command):
                return FakeChild(command)

            # Capture the bound ephemeral port without opening a browser.
            from fall_detection.dashboard import create_server
            def server_factory(path, port):
                server = create_server(path, port)
                observed["url"] = f"http://127.0.0.1:{server.server_port}"
                return server

            with patch("fall_detection.monitor_cli.subprocess.Popen", side_effect=fake_launch), \
                 patch("fall_detection.monitor_cli.create_server", side_effect=server_factory), \
                 patch("fall_detection.monitor_cli.webbrowser.open") as browser:
                self.assertEqual(main(["--camera", "0", "--output", str(output), "--port", "0"]), 0)
                browser.assert_not_called()

            events = observed["payload"]["events"]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["state"], "unknown")
            self.assertFalse(events[0]["fall_detection_available"])
            self.assertNotIn("bgr", json.dumps(observed["payload"]))
            self.assertEqual(observed["command"][3:5], ["--camera", "0"])

    def test_child_exit_code_is_propagated(self):
        with TemporaryDirectory() as folder:
            class FailedChild:
                def wait(self, timeout=None):
                    return 2
                def poll(self):
                    return 2
            with patch("fall_detection.monitor_cli.subprocess.Popen", return_value=FailedChild()):
                self.assertEqual(main(["--camera", "0", "--output",
                                       str(Path(folder) / "new.jsonl"), "--port", "0"]), 2)


if __name__ == "__main__":
    unittest.main()
