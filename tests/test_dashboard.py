"""Standard-library regression tests for the local JSONL dashboard."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from fall_detection.dashboard import DashboardError, create_server, read_events


class DashboardReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "frames.jsonl"

    def write(self, *records):
        with self.log.open("ab") as f:
            for record in records:
                f.write(json.dumps(record).encode() + b"\n")

    def test_missing_log_and_incremental_append(self):
        self.assertFalse(read_events(self.log)["source_exists"])
        self.write({"event": "frame", "frame_index": 0, "state": "unknown"})
        first = read_events(self.log)
        self.assertEqual([r["frame_index"] for r in first["events"]], [0])
        self.write({"event": "frame", "frame_index": 1, "state": "unknown"})
        second = read_events(self.log, first["next_cursor"])
        self.assertEqual([r["frame_index"] for r in second["events"]], [1])
        self.assertEqual(read_events(self.log, second["next_cursor"])["events"], [])

    def test_partial_line_is_not_lost(self):
        self.log.write_bytes(b'{"event":"frame","frame_index":0}')
        first = read_events(self.log)
        self.assertEqual(first["events"], [])
        with self.log.open("ab") as f:
            f.write(b"\n")
        self.assertEqual(read_events(self.log, first["next_cursor"])["events"][0]["frame_index"], 0)

    def test_truncate_and_replace_reset_cursor(self):
        self.write({"event": "frame", "frame_index": 1234})
        cursor = read_events(self.log)["next_cursor"]
        self.log.write_bytes(b'{"event":"frame","frame_index":2}\n')
        result = read_events(self.log, cursor)
        self.assertTrue(result["reset"])
        self.assertEqual(result["events"][0]["frame_index"], 2)

    def test_whitelists_metadata_and_never_serves_raw_frames(self):
        self.write({
            "event": "frame", "frame_index": 1, "state": "unknown",
            "bgr": "PRIVATE_IMAGE_BYTES", "image_base64": "SECRET",
            "people": [{"track_id": 3, "state": "unknown", "keypoints": [1, 2, 3]}],
            "alerts": [{"track_id": 3, "status": "pending", "screenshot": "SECRET"}],
        })
        result = read_events(self.log)["events"][0]
        encoded = json.dumps(result)
        self.assertNotIn("PRIVATE_IMAGE_BYTES", encoded)
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("keypoints", encoded)
        self.assertEqual(result["people"], [{"track_id": 3, "state": "unknown"}])
        self.assertEqual(result["alerts"], [{"track_id": 3, "status": "pending"}])

    def test_live_per_person_heuristic_fields_are_retained_without_pixels(self):
        self.write({
            "event": "frame", "frame_index": 4, "state": "unknown",
            "state_source": "heuristic_baseline_unvalidated",
            "fall_detection_available": False, "tracked_person_count": 1,
            "people": [{"track_id": 7, "state": "falling", "observed": True,
                        "window_ready": True, "window_samples": 30,
                        "reason": "unvalidated", "evidence": {"keypoints": [3, 4]}}],
            "bgr": "PRIVATE_PIXELS", "keypoints": ["PRIVATE_POSE"],
        })
        record = read_events(self.log)["events"][0]
        self.assertEqual(record["state"], "unknown")
        self.assertFalse(record["fall_detection_available"])
        self.assertEqual(record["tracked_person_count"], 1)
        self.assertEqual(record["people"][0]["state"], "falling")
        self.assertTrue(record["people"][0]["observed"])
        self.assertNotIn("PRIVATE", json.dumps(record))
        self.assertNotIn("evidence", json.dumps(record))

    def test_skips_malformed_and_unrelated_lines(self):
        self.log.write_bytes(b'not-json\n[]\n{"event":"not_a_frame"}\n{"event":"summary","frames_processed":3}\n')
        result = read_events(self.log)
        self.assertEqual(result["events"], [{"event": "summary", "frames_processed": 3}])

    def test_limits_and_invalid_cursors(self):
        self.write(*({"event": "frame", "frame_index": i} for i in range(6)))
        first = read_events(self.log, limit=2)
        self.assertEqual(len(first["events"]), 2)
        self.assertEqual(len(read_events(self.log, first["next_cursor"], 2)["events"]), 2)
        for cursor in ("abc", "1:-1", "1:x", "a" * 101):
            with self.subTest(cursor=cursor), self.assertRaises(DashboardError):
                read_events(self.log, cursor)
        for limit in (0, 201, True):
            with self.assertRaises(DashboardError):
                read_events(self.log, limit=limit)

    def test_large_line_is_skipped_without_loading_it(self):
        self.log.write_bytes(b"x" * 1_000_100 + b"\n" + b'{"event":"frame","frame_index":9}\n')
        self.assertEqual(read_events(self.log)["events"][0]["frame_index"], 9)


class DashboardHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "frames.jsonl"
        self.log.write_text('{"event":"frame","frame_index":0,"state":"unknown"}\n')
        self.server = create_server(self.log, 0)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def get(self, path):
        with urlopen(self.base + path, timeout=3) as response:
            return response.status, dict(response.headers), response.read()

    def test_html_and_health_are_local_read_only(self):
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Fall Detection Monitor", body)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(json.loads(self.get("/api/health")[2])["mode"], "read_only_metadata")

    def test_event_api_and_no_raw_frame_data(self):
        status, _, body = self.get("/api/events?cursor=0&limit=10")
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["events"][0]["state"], "unknown")
        self.assertNotIn("bgr", body.decode())
        self.assertEqual(json.loads(self.get("/api/events?cursor=" + quote(result["next_cursor"]))[2])["events"], [])

    def test_rejects_dns_rebinding_host(self):
        request = Request(self.base + "/api/events", headers={"Host": "attacker.example"})
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=3)
        self.assertEqual(context.exception.code, 403)

    def test_rejects_invalid_requests_and_does_not_serve_paths(self):
        for path in ("/secret", "/api/events?limit=999", "/api/events?cursor=bad", "/api/events?limit=1&limit=2"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as ctx:
                self.get(path)
            self.assertIn(ctx.exception.code, (400, 404))


if __name__ == "__main__":
    unittest.main()
