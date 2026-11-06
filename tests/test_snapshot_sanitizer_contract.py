"""Fail-closed snapshot reader contract tests for untrusted sanitizer output."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fall_detection.dashboard_snapshot import read_recent_events


class SnapshotSanitizerContractTests(unittest.TestCase):
    def test_non_mapping_sanitizer_output_is_rejected(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "events.jsonl"
            path.write_bytes(b'{"event":"frame","frame_index":1}\n')
            for invalid in ("unsafe", ["unsafe"], 42, False):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "sanitizer must return a dict or None"
                ):
                    read_recent_events(path, lambda record: invalid)

    def test_none_sanitizer_output_skips_event(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "events.jsonl"
            path.write_bytes(b'{"event":"frame","frame_index":1}\n')
            result = read_recent_events(path, lambda record: None)
            self.assertEqual(result["events"], [])
            self.assertTrue(result["source_exists"])

    def test_regular_file_is_accepted(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "events.jsonl"
            path.write_bytes(b'{"event":"frame","frame_index":1}\n')
            result = read_recent_events(path, lambda record: {"frame_index": record["frame_index"]})
            self.assertEqual(result["events"], [{"frame_index": 1}])


if __name__ == "__main__":
    unittest.main()
