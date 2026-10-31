"""Regression tests for bounded dashboard tail snapshots."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fall_detection.dashboard import _sanitize_record, read_events
from fall_detection.dashboard_snapshot import read_recent_events


class DashboardSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "stream.jsonl"

    def write(self, *records):
        with self.path.open("ab") as stream:
            for record in records:
                stream.write(json.dumps(record).encode("utf-8") + b"\n")

    def recent(self, **kwargs):
        return read_recent_events(self.path, _sanitize_record, **kwargs)

    def test_missing_file(self):
        self.assertEqual(self.recent()["events"], [])
        self.assertFalse(self.recent()["source_exists"])
        self.assertEqual(self.recent()["next_cursor"], "0")

    def test_recent_records_and_incremental_handoff(self):
        self.write(*({"event": "frame", "frame_index": i} for i in range(10)))
        result = self.recent(limit=3)
        self.assertEqual([r["frame_index"] for r in result["events"]], [7, 8, 9])
        self.write({"event": "frame", "frame_index": 10})
        following = read_events(self.path, result["next_cursor"])
        self.assertEqual([r["frame_index"] for r in following["events"]], [10])

    def test_incomplete_tail_retried_after_append(self):
        self.write({"event": "frame", "frame_index": 1})
        with self.path.open("ab") as stream:
            stream.write(b'{"event":"frame","frame_index":2}')
        result = self.recent()
        self.assertTrue(result["partial_trailing_line"])
        self.assertEqual([r["frame_index"] for r in result["events"]], [1])
        with self.path.open("ab") as stream:
            stream.write(b"\n")
        self.assertEqual(read_events(self.path, result["next_cursor"])["events"][0]["frame_index"], 2)

    def test_clipped_first_line_is_not_parsed(self):
        self.write({"event": "frame", "frame_index": 1, "reason": "x" * 120},
                   {"event": "frame", "frame_index": 2},
                   {"event": "frame", "frame_index": 3})
        result = self.recent(max_bytes=90)
        self.assertTrue(result["window_truncated"])
        self.assertEqual([r["frame_index"] for r in result["events"]], [2, 3])
        self.assertLessEqual(result["bytes_scanned"], 91)

    def test_invalid_json_oversized_line_and_private_fields(self):
        self.path.write_bytes(b"not-json\n" + b"x" * 1_000_001 + b"\n")
        self.write({"event": "frame", "frame_index": 9, "image_base64": "SECRET"})
        result = self.recent()
        self.assertEqual(result["events"], [{"event": "frame", "frame_index": 9}])
        self.assertNotIn("SECRET", json.dumps(result))

    def test_partial_only_file_preserves_cursor(self):
        self.path.write_bytes(b'{"event":"frame","frame_index":4}')
        result = self.recent()
        self.assertEqual(result["events"], [])
        self.assertTrue(result["partial_trailing_line"])
        with self.path.open("ab") as stream:
            stream.write(b"\n")
        self.assertEqual(read_events(self.path, result["next_cursor"])["events"][0]["frame_index"], 4)

    def test_invalid_arguments(self):
        for limit in (0, 201, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.recent(limit=limit)
        for max_bytes in (0, True, 8 * 1024 * 1024 + 1):
            with self.subTest(max_bytes=max_bytes), self.assertRaises(ValueError):
                self.recent(max_bytes=max_bytes)
        with self.assertRaises(ValueError):
            read_recent_events("not-a-path", _sanitize_record)


if __name__ == "__main__":
    unittest.main()
