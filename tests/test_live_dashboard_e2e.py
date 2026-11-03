"""End-to-end synthetic live-pipeline to local dashboard HTTP contract.

This exercises the actual pose/tracking/feature stages with a fake estimator,
not real video, trained weights, or validated fall detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.parse import quote
from urllib.request import urlopen

from fall_detection.contracts import FramePacket
from fall_detection.dashboard import create_server
from fall_detection.features import FeatureConfig
from fall_detection.live import run_live_pipeline
from fall_detection.pose import Keypoint, PoseDetection


class SyntheticEstimator:
    def estimate(self, frame):
        if frame.index == 2:
            return ()  # Brief occlusion; existing tracks must not disappear.
        return tuple(
            PoseDetection(
                (x, 5.0, x + 15.0, 65.0), 0.95,
                tuple(Keypoint(x + 7.0, 20.0 + i, 0.95) for i in range(17)),
            )
            for x in (12.0, 52.0)
        )


def frames(start, stop):
    for i in range(start, stop):
        yield FramePacket(
            index=i, timestamp_seconds=i * 0.2,
            width=80, height=80, bgr=b"\x7f" * (80 * 80 * 3),
            source="synthetic_integration_fixture",
        )


class LiveDashboardEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "live.jsonl"

    def write_live(self, start, stop):
        with self.path.open("a", encoding="utf-8") as stream:
            def emit(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()

            summary = run_live_pipeline(
                frames(start, stop), emit,
                estimator=SyntheticEstimator(),
                feature_config=FeatureConfig(window_size=5, min_observed_samples=3),
            )
            emit(summary.as_record())

    def start_server(self):
        server = create_server(self.path, 0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    @staticmethod
    def get(base, route):
        with urlopen(base + route, timeout=5) as response:
            return json.loads(response.read())

    def test_live_pose_tracks_are_visible_in_http_snapshot_without_pixels(self):
        self.write_live(0, 6)
        base = self.start_server()
        snapshot = self.get(base, "/api/snapshot?limit=20")
        events = [e for e in snapshot["events"] if e["event"] == "frame"]
        self.assertEqual([e["frame_index"] for e in events], list(range(6)))
        self.assertTrue(all(e["state"] == "unknown" for e in events))
        self.assertTrue(all(e["fall_detection_available"] is False for e in events))
        self.assertTrue(all(e["tracked_person_count"] == 2 for e in events))
        self.assertEqual([p["track_id"] for p in events[0]["people"]], [1, 2])
        self.assertTrue(all(not p["observed"] for p in events[2]["people"]))
        self.assertTrue(all(p["observed"] for p in events[3]["people"]))
        self.assertEqual(events[0]["state_source"], "heuristic_baseline_unvalidated")
        encoded = json.dumps(snapshot)
        for private in ("bgr", "keypoints", "evidence", "image_base64"):
            self.assertNotIn(private, encoded)

    def test_snapshot_cursor_handoff_reads_new_live_frames(self):
        self.write_live(0, 4)
        base = self.start_server()
        snapshot = self.get(base, "/api/snapshot?limit=3")
        self.assertEqual([e["frame_index"] for e in snapshot["events"] if e["event"] == "frame"], [2, 3])
        self.write_live(4, 6)
        delta = self.get(base, "/api/events?cursor=" + quote(snapshot["next_cursor"]) + "&limit=20")
        self.assertFalse(delta["reset"])
        self.assertEqual([e["frame_index"] for e in delta["events"] if e["event"] == "frame"], [4, 5])
        self.assertEqual(len([e for e in delta["events"] if e["event"] == "summary"]), 1)

    def test_opt_in_research_alerts_propagate_to_http_without_private_data(self):
        from fall_detection.alerts import AlertConfig, AlertManager
        from fall_detection.contracts import FallState
        from fall_detection.state import StateDecision, StateEvidence

        class SyntheticStateScript:
            def classify_many(self, windows):
                return tuple(
                    StateDecision(
                        window.track_id, window.latest.frame_index,
                        FallState.FALLEN if window.latest.observed else FallState.UNKNOWN,
                        "synthetic_state_not_validated",
                        StateEvidence(3, 1.0, 1.5, 0.0),
                    )
                    for window in windows
                )

        manager = AlertManager(AlertConfig(fallen_persistence_seconds=0.4))
        with self.path.open("w", encoding="utf-8") as stream:
            def emit(record):
                stream.write(json.dumps(record, allow_nan=False) + "\\n")
                stream.flush()

            run_live_pipeline(
                frames(0, 7), emit, estimator=SyntheticEstimator(),
                feature_config=FeatureConfig(window_size=5, min_observed_samples=3),
                temporal_classifier=SyntheticStateScript(),
                alert_manager=manager,
            )
        base = self.start_server()
        snapshot = self.get(base, "/api/snapshot?limit=20")
        records = [record for record in snapshot["events"] if record["event"] == "frame"]
        self.assertEqual(len(records), 7)
        self.assertTrue(all(record["alerts_research_only"] for record in records))
        self.assertTrue(all(not record["fall_detection_available"] for record in records))
        self.assertEqual(records[-1]["tracked_person_count"], 2)
        self.assertEqual({alert["status"] for alert in records[-1]["alerts"]}, {"active"})
        self.assertEqual(len(records[-1]["alerts"]), 2)
        self.assertEqual({alert["track_id"] for alert in records[-1]["alerts"]}, {1, 2})
        self.assertTrue(all(alert["alert_id"].startswith("alert-")
                            for alert in records[-1]["alerts"]))
        encoded = json.dumps(snapshot)
        for private in ("bgr", "keypoints", "image_base64", "evidence", "color_bgr"):
            self.assertNotIn(private, encoded)

    def test_http_recovery_after_live_log_is_removed(self):
        self.write_live(0, 2)
        base = self.start_server()
        snapshot = self.get(base, "/api/snapshot")
        self.path.unlink()
        missing = self.get(base, "/api/events?cursor=" + quote(snapshot["next_cursor"]))
        self.assertTrue(missing["reset"])
        self.assertFalse(missing["source_exists"])
        self.assertEqual(missing["events"], [])
        self.write_live(0, 2)
        recovered = self.get(base, "/api/snapshot")
        self.assertTrue(recovered["source_exists"])
        self.assertEqual([e["frame_index"] for e in recovered["events"] if e["event"] == "frame"], [0, 1])


if __name__ == "__main__":
    unittest.main()
