"""Multi-person append-only alert journal restart and corruption regression tests.

Events are synthetic metadata, never camera frames or emergency notifications.
"""
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.alerts import AlertConfig, AlertError, AlertManager, JsonlAlertLog
from fall_detection.contracts import FallState
from fall_detection.state import StateDecision, StateEvidence


def decision(track, frame, state):
    return StateDecision(
        track, frame, state, "synthetic_journal_stress",
        StateEvidence(3, 1.0, 1.5 if state is FallState.FALLEN else 0.5, 0.0),
    )


class AlertJournalRestartStressTests(unittest.TestCase):
    def test_reopen_continues_unique_alert_and_event_ids_after_sixteen_tracks(self):
        config = AlertConfig(
            fallen_persistence_seconds=1.0, max_tracks=16, fsync_events=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.jsonl"
            with JsonlAlertLog(path, fsync=False, create_new=True) as log:
                manager = AlertManager(config, log)
                for track in range(1, 17):
                    manager.update(decision(track, 0, FallState.FALLEN), 0.0)
                    manager.update(decision(track, 1, FallState.FALLEN), 1.0)
                self.assertEqual(log.next_event_id, 33)
            first_bytes = path.read_bytes()
            with JsonlAlertLog(path, fsync=False) as log:
                self.assertEqual(len(log.events), 32)
                manager = AlertManager(config, log)
                manager.update(decision(17, 2, FallState.FALLEN), 2.0)
                opened = manager.update(decision(17, 3, FallState.FALLEN), 3.0)
                self.assertEqual(opened.alert_id, "alert-000017")
                self.assertEqual(log.next_event_id, 35)
            self.assertTrue(path.read_bytes().startswith(first_bytes))
            records = [json.loads(line) for line in path.read_text(
                encoding="utf-8"
            ).splitlines()]
            self.assertEqual([item["event_id"] for item in records], list(range(1, 35)))
            self.assertEqual(
                len({item["alert_id"] for item in records if item["transition"] == "alert_opened"}),
                17,
            )
            self.assertTrue(all("bgr" not in item and "keypoints" not in item for item in records))

    def test_truncated_journal_is_rejected_without_appending_or_repairing(self):
        config = AlertConfig(fsync_events=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.jsonl"
            with JsonlAlertLog(path, fsync=False, create_new=True) as log:
                manager = AlertManager(config, log)
                manager.update(decision(1, 0, FallState.FALLEN), 0.0)
            with path.open("ab") as stream:
                stream.write(b'{"event_id":2,"event":')
            before = path.read_bytes()
            with self.assertRaises(AlertError):
                with JsonlAlertLog(path, fsync=False):
                    self.fail("truncated journal must not be accepted")
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
