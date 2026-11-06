"""Regression coverage for snapshot cursor and malformed-line recovery.

Uses only synthetic metadata; never reads camera frames or private video.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fall_detection.dashboard_snapshot import read_recent_events


def safe_frame(record):
    if record.get("event") != "frame" or type(record.get("frame_index")) is not int:
        return None
    return {"event": "frame", "frame_index": record["frame_index"]}


class SnapshotRecoveryContractTests(unittest.TestCase):
    def test_corrupt_utf8_and_invalid_json_do_not_hide_valid_history(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "events.jsonl"
            path.write_bytes(
                b'{"event":"frame","frame_index":1}\n'
                + b'\xff\xfe\n'
                + b'{"event":"frame","frame_index":2}\n'
                + b'{"broken":\n'
                + b'{"event":"frame","frame_index":3}\n'
            )
            result = read_recent_events(path, safe_frame, limit=3)
            self.assertEqual([event["frame_index"] for event in result["events"]], [1, 2, 3])
            self.assertTrue(result["source_exists"])

    def test_partial_line_is_not_exposed_until_completed(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "events.jsonl"
            path.write_bytes(b'{"event":"frame","frame_index":1}\n'
                             b'{"event":"frame","frame_index":2}')
            result = read_recent_events(path, safe_frame)
            self.assertEqual([item["frame_index"] for item in result["events"]], [1])
            self.assertTrue(result["partial_trailing_line"])
            with path.open("ab") as stream:
                stream.write(b"\n")
            result = read_recent_events(path, safe_frame)
            self.assertEqual([item["frame_index"] for item in result["events"]], [1, 2])
            self.assertFalse(result["partial_trailing_line"])

    def test_sanitizer_rejects_unrecognized_events_before_limit(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "events.jsonl"
            rows = [
                {"event": "frame", "frame_index": 1},
                {"event": "secret", "frame_index": 99, "raw": "PRIVATE"},
                {"event": "frame", "frame_index": 2},
                {"event": "frame", "frame_index": 3},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = read_recent_events(path, safe_frame, limit=2)
            self.assertEqual([item["frame_index"] for item in result["events"]], [2, 3])
            self.assertNotIn("PRIVATE", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
