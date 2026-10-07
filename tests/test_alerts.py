import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fall_detection.alerts import (
    ALERT_EVENT_SCHEMA_VERSION,
    AlertConfig,
    AlertError,
    AlertEvent,
    AlertManager,
    PersistentAlertStage,
    AlertSnapshot,
    AlertStatus,
    JsonlAlertLog,
    load_alert_config,
    run_alert_smoke,
    visible_alert_overlays,
)
from fall_detection.cli import main
from fall_detection.contracts import FallState, FramePacket, Prediction
from fall_detection.preview import Preview
from fall_detection.state import StateDecision, StateEvidence


def decision(track_id, frame_index, state, reason=None):
    return StateDecision(
        track_id,
        frame_index,
        state,
        reason or f"test_{state.value}",
        StateEvidence(3, 1.0, 1.5 if state is FallState.FALLEN else 0.5, 0.0),
    )


def open_alert(manager, track_id=1, *, start=0.0, frame=0):
    pending = manager.update(decision(track_id, frame, FallState.FALLEN), start)
    active = manager.update(
        decision(track_id, frame + 1, FallState.FALLEN),
        start + manager.config.fallen_persistence_seconds,
    )
    return pending, active


class AlertConfigTests(unittest.TestCase):
    def test_file_and_override_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.toml"
            path.write_text(
                "[alerts]\nfallen_persistence_seconds=2.5\n"
                "cooldown_seconds=30\nmax_tracks=40\nfsync_events=false\n",
                encoding="utf-8",
            )
            config = load_alert_config(path, cooldown_seconds=5.0)
            self.assertEqual(config.fallen_persistence_seconds, 2.5)
            self.assertEqual(config.cooldown_seconds, 5.0)
            self.assertEqual(config.max_tracks, 40)
            self.assertFalse(config.fsync_events)

    def test_invalid_values_and_documents_are_rejected(self):
        for values in (
            {"fallen_persistence_seconds": 0},
            {"fallen_persistence_seconds": True},
            {"cooldown_seconds": float("nan")},
            {"max_tracks": 0},
            {"max_tracks": True},
            {"fsync_events": 1},
        ):
            with self.subTest(values=values), self.assertRaises(AlertError):
                AlertConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for content in (
                "[alert]\ncooldown_seconds=3\n",
                "[alerts]\nunknown=1\n",
                "alerts=2\n",
                "[alerts]\n[other]\nx=1\n",
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(AlertError):
                    load_alert_config(path)


class AlertEventTests(unittest.TestCase):
    def sample(self):
        return AlertEvent(
            1, 3.5, 4, 9, "alert-000001", "alert_opened",
            AlertStatus.PENDING, AlertStatus.ACTIVE, FallState.FALLEN,
            "lying_posture_settled",
        )

    def test_record_round_trip_is_finite_metadata_only_json(self):
        event = self.sample()
        record = event.as_record()
        self.assertEqual(AlertEvent.from_record(record), event)
        encoded = json.dumps(record, allow_nan=False)
        self.assertNotIn("bgr", encoded)
        self.assertNotIn("image", encoded)
        self.assertEqual(record["schema_version"], ALERT_EVENT_SCHEMA_VERSION)

    def test_invalid_events_and_records_are_rejected(self):
        event = self.sample()
        invalid_changes = (
            {"event_id": 0},
            {"timestamp_seconds": float("nan")},
            {"track_id": True},
            {"frame_index": -1},
            {"transition": "send_emergency_message"},
            {"transition": []},
            {"from_status": "active"},
            {"from_status": AlertStatus.ACTIVE},
            {"alert_id": None},
            {"observed_state": "fallen"},
            {"reason": ""},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(AlertError):
                replace(event, **changes)
        record = event.as_record()
        for changed in (
            {**record, "extra": 1},
            {**record, "schema_version": 2},
            {**record, "event": "message_sent"},
            {**record, "to_status": "missing"},
        ):
            with self.subTest(changed=changed), self.assertRaises(AlertError):
                AlertEvent.from_record(changed)


class JsonlAlertLogTests(unittest.TestCase):
    def test_log_persists_valid_events_and_continues_ids_without_overwrite(self):
        config = AlertConfig(fallen_persistence_seconds=1, cooldown_seconds=2,
                             fsync_events=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "alerts.jsonl"
            with JsonlAlertLog(path, fsync=False, create_new=True) as event_log:
                manager = AlertManager(config, event_log)
                open_alert(manager)
                self.assertEqual(event_log.next_event_id, 3)
            first_bytes = path.read_bytes()
            with JsonlAlertLog(path, fsync=False) as event_log:
                self.assertEqual(len(event_log.events), 2)
                manager = AlertManager(config, event_log)
                open_alert(manager, track_id=2, start=10, frame=10)
                self.assertEqual(manager.events[0].event_id, 3)
                self.assertEqual(manager.events[1].alert_id, "alert-000002")
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["event_id"] for record in records], [1, 2, 3, 4])
            self.assertTrue(path.read_bytes().startswith(first_bytes))

    def test_create_new_never_overwrites_existing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.jsonl"
            path.write_text("keep\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with JsonlAlertLog(path, create_new=True):
                    self.fail("existing log must not open")
            self.assertEqual(path.read_text(encoding="utf-8"), "keep\n")

    def test_malformed_existing_logs_are_rejected_before_append(self):
        event = AlertEvent(
            2, 1, 1, 1, None, "pending_started",
            AlertStatus.IDLE, AlertStatus.PENDING, FallState.FALLEN, "test",
        )
        bad_values = (
            "\n",
            "{bad json}\n",
            json.dumps(event.as_record()) + "\n",
            json.dumps({**event.as_record(), "timestamp_seconds": float("nan")}) + "\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.jsonl"
            for value in bad_values:
                path.write_text(value, encoding="utf-8")
                original = path.read_bytes()
                with self.subTest(value=value), self.assertRaises(AlertError):
                    with JsonlAlertLog(path, fsync=False):
                        self.fail("invalid log must not open")
                self.assertEqual(path.read_bytes(), original)

    def test_append_requires_open_log_and_contiguous_event_id(self):
        path = Path("unused.jsonl")
        event_log = JsonlAlertLog(path, fsync=False)
        event = AlertEvent(
            1, 0, 1, 0, None, "pending_started",
            AlertStatus.IDLE, AlertStatus.PENDING, FallState.FALLEN, "test",
        )
        with self.assertRaisesRegex(AlertError, "opened"):
            event_log.append(event)
        with tempfile.TemporaryDirectory() as directory:
            actual = Path(directory) / "alerts.jsonl"
            with JsonlAlertLog(actual, fsync=False, create_new=True) as opened:
                with self.assertRaisesRegex(AlertError, "expected"):
                    opened.append(replace(event, event_id=2))
                self.assertEqual(actual.read_bytes(), b"")


class AlertManagerTests(unittest.TestCase):
    def setUp(self):
        self.config = AlertConfig(
            fallen_persistence_seconds=1.0, cooldown_seconds=2.0,
            max_tracks=10, fsync_events=False,
        )
        self.manager = AlertManager(self.config)

    def test_fallen_must_persist_and_interruption_cancels_pending(self):
        first = self.manager.update(decision(1, 0, FallState.FALLEN), 0)
        second = self.manager.update(decision(1, 1, FallState.FALLEN), 0.9)
        cancelled = self.manager.update(decision(1, 2, FallState.UNKNOWN), 1.0)
        self.assertEqual(first.status, AlertStatus.PENDING)
        self.assertEqual(second.status, AlertStatus.PENDING)
        self.assertEqual(cancelled.status, AlertStatus.IDLE)
        self.assertEqual([event.transition for event in self.manager.events],
                         ["pending_started", "pending_cancelled"])

    def test_active_alert_is_latched_and_duplicate_events_are_suppressed(self):
        _, active = open_alert(self.manager)
        alert_id = active.alert_id
        for index, state in enumerate(
                (FallState.FALLEN, FallState.NORMAL, FallState.UNKNOWN), start=2):
            snapshot = self.manager.update(decision(1, index, state), index)
            self.assertEqual(snapshot.status, AlertStatus.ACTIVE)
            self.assertEqual(snapshot.alert_id, alert_id)
        self.assertEqual(
            [event.transition for event in self.manager.events].count("alert_opened"), 1
        )
        self.assertEqual(len(self.manager.events), 2)

    def test_acknowledgment_does_not_clear_and_reset_requires_current_normal(self):
        open_alert(self.manager)
        acknowledged = self.manager.acknowledge(1, 1.1)
        self.assertEqual(acknowledged.status, AlertStatus.ACKNOWLEDGED)
        self.assertTrue(acknowledged.overlay.visible)
        with self.assertRaisesRegex(AlertError, "normal"):
            self.manager.reset(1, 1.2)
        still_latched = self.manager.update(decision(1, 2, FallState.UNKNOWN), 1.3)
        self.assertEqual(still_latched.status, AlertStatus.ACKNOWLEDGED)
        self.manager.update(decision(1, 3, FallState.NORMAL), 1.4)
        reset = self.manager.reset(1, 1.5)
        self.assertEqual(reset.status, AlertStatus.COOLDOWN)
        self.assertFalse(reset.overlay.visible)

    def test_cooldown_suppresses_reopen_and_requires_normal_to_rearm(self):
        open_alert(self.manager)
        self.manager.update(decision(1, 2, FallState.NORMAL), 1.1)
        self.manager.acknowledge(1, 1.2)
        self.manager.reset(1, 1.3)
        during = self.manager.update(decision(1, 3, FallState.FALLEN), 2.0)
        after_but_fallen = self.manager.update(decision(1, 4, FallState.FALLEN), 3.4)
        rearmed = self.manager.update(decision(1, 5, FallState.NORMAL), 3.5)
        self.assertEqual(during.status, AlertStatus.COOLDOWN)
        self.assertEqual(after_but_fallen.status, AlertStatus.COOLDOWN)
        self.assertEqual(rearmed.status, AlertStatus.IDLE)
        self.assertEqual([event.transition for event in self.manager.events].count("alert_opened"), 1)
        pending = self.manager.update(decision(1, 6, FallState.FALLEN), 3.6)
        active = self.manager.update(decision(1, 7, FallState.FALLEN), 4.6)
        self.assertEqual(pending.status, AlertStatus.PENDING)
        self.assertEqual(active.status, AlertStatus.ACTIVE)
        self.assertNotEqual(active.alert_id, "alert-000001")

    def test_tracks_are_independent_and_snapshots_are_sorted(self):
        self.manager.update(decision(2, 0, FallState.NORMAL), 0)
        self.manager.update(decision(1, 0, FallState.FALLEN), 0)
        self.assertEqual([value.track_id for value in self.manager.snapshots()], [1, 2])
        self.assertEqual(self.manager.snapshot(1).status, AlertStatus.PENDING)
        self.assertEqual(self.manager.snapshot(2).status, AlertStatus.IDLE)

    def test_invalid_timeline_is_transactional(self):
        self.manager.update(decision(1, 0, FallState.FALLEN), 1.0)
        before = self.manager.snapshot(1)
        event_count = len(self.manager.events)
        for value, timestamp in (
            (decision(1, 0, FallState.FALLEN), 2.0),
            (decision(1, 1, FallState.FALLEN), 1.0),
            (decision(1, 1, FallState.FALLEN), float("nan")),
        ):
            with self.subTest(timestamp=timestamp), self.assertRaises(AlertError):
                self.manager.update(value, timestamp)
            self.assertEqual(self.manager.snapshot(1), before)
            self.assertEqual(len(self.manager.events), event_count)

    def test_invalid_actions_do_not_change_state(self):
        self.manager.update(decision(1, 0, FallState.NORMAL), 0)
        for action in (
            lambda: self.manager.acknowledge(1, 0),
            lambda: self.manager.reset(1, 0),
            lambda: self.manager.acknowledge(99, 0),
            lambda: self.manager.acknowledge(True, 0),
        ):
            with self.assertRaises(AlertError):
                action()
        self.assertEqual(self.manager.snapshot(1).status, AlertStatus.IDLE)

    def test_max_tracks_and_idle_discard_keep_storage_bounded(self):
        manager = AlertManager(replace(self.config, max_tracks=1))
        manager.update(decision(1, 0, FallState.NORMAL), 0)
        with self.assertRaisesRegex(AlertError, "max_tracks"):
            manager.update(decision(2, 0, FallState.NORMAL), 0)
        manager.discard_idle(1)
        manager.update(decision(2, 0, FallState.NORMAL), 0)
        manager.update(decision(2, 1, FallState.FALLEN), 1)
        with self.assertRaisesRegex(AlertError, "idle"):
            manager.discard_idle(2)

    def test_snapshots_are_finite_json_overlay_ready_and_contain_no_image_data(self):
        _, active = open_alert(self.manager)
        record = active.as_record()
        encoded = json.dumps(record, allow_nan=False)
        self.assertEqual(record["overlay"]["headline"], "FALL ALERT")
        self.assertEqual(record["overlay"]["color_bgr"], [0, 0, 255])
        self.assertNotIn('"bgr":', encoded)
        self.assertEqual(len(visible_alert_overlays(self.manager.snapshots())), 1)
        with self.assertRaises(AlertError):
            visible_alert_overlays(("not-a-snapshot",))


class PersistentAlertStageTests(unittest.TestCase):
    def test_composes_frame_state_decisions_and_sorted_alert_snapshots(self):
        class Stage:
            calls = 0
            def process(self, frame):
                self.calls += 1
                state = FallState.FALLEN
                return (decision(2, frame.index, state), decision(1, frame.index, state))

        manager = AlertManager(AlertConfig(
            fallen_persistence_seconds=1, cooldown_seconds=2, fsync_events=False,
        ))
        stage = PersistentAlertStage(Stage(), manager)
        first = stage.process(FramePacket(0, 0.0, 1, 1, b"\0\0\0"))
        second = stage.process(FramePacket(1, 1.0, 1, 1, b"\0\0\0"))
        self.assertEqual([item.track_id for item in first.alerts], [1, 2])
        self.assertTrue(all(item.status is AlertStatus.PENDING for item in first.alerts))
        self.assertTrue(all(item.status is AlertStatus.ACTIVE for item in second.alerts))
        self.assertEqual(len(second.decisions), 2)

    def test_rejects_malformed_stage_output_before_alert_updates(self):
        class Stage:
            def __init__(self, values):
                self.values = values
            def process(self, frame):
                return self.values

        frame = FramePacket(0, 0.0, 1, 1, b"\0\0\0")
        cases = (
            [decision(1, 0, FallState.FALLEN)],
            (decision(1, 1, FallState.FALLEN),),
            (decision(1, 0, FallState.FALLEN), decision(1, 0, FallState.NORMAL)),
        )
        for values in cases:
            manager = AlertManager(AlertConfig(fsync_events=False))
            with self.subTest(values=values), self.assertRaises(AlertError):
                PersistentAlertStage(Stage(values), manager).process(frame)
            self.assertEqual(manager.snapshots(), ())


class AlertSmokeTests(unittest.TestCase):
    def test_smoke_covers_persistence_duplicate_suppression_reset_and_rearm(self):
        result = run_alert_smoke(AlertConfig(
            fallen_persistence_seconds=1, cooldown_seconds=2, fsync_events=False,
        ))
        self.assertTrue(result["persisted_after_normal"])
        self.assertTrue(result["duplicate_open_suppressed"])
        self.assertEqual(result["final_status"], "idle")
        self.assertEqual(result["transitions"], [
            "pending_started", "alert_opened", "alert_acknowledged",
            "alert_reset", "alert_rearmed",
        ])
        self.assertFalse(result["external_actions_performed"])
        self.assertFalse(result["fall_detection_available"])

    def test_cli_writes_new_local_event_log_and_never_overwrites_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "alerts.jsonl"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([
                    "alert-smoke", "--config", "configs/alerts.toml",
                    "--event-log", str(path),
                ]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["event_count"], 5)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 5)
            self.assertTrue(all(record["event"] == "alert_transition" for record in records))
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "alert-smoke", "--config", "configs/alerts.toml",
                    "--event-log", str(path),
                ]), 2)
            self.assertEqual(path.read_bytes(), original)


class PreviewAlertOverlayTests(unittest.TestCase):
    class CvError(Exception):
        pass

    def make_cv(self):
        return SimpleNamespace(
            error=self.CvError, WINDOW_NORMAL=0, FONT_HERSHEY_SIMPLEX=0,
            WND_PROP_VISIBLE=1, getBuildInformation=Mock(return_value="GUI: QT"),
            namedWindow=Mock(), putText=Mock(), rectangle=Mock(), imshow=Mock(),
            waitKey=Mock(return_value=-1), getWindowProperty=Mock(return_value=1),
            destroyWindow=Mock(),
        )

    def test_preview_draws_visible_alert_banner_without_changing_pipeline_contract(self):
        cv = self.make_cv()
        snapshot = AlertSnapshot(
            1, AlertStatus.ACTIVE, "alert-000001", FallState.FALLEN, "test",
            0.0, 1.0, None, None, 1.0,
        )
        fake_pixels = Mock()
        fake_pixels.reshape.return_value.copy.return_value = Mock()
        fake_numpy = SimpleNamespace(uint8="uint8", frombuffer=Mock(return_value=fake_pixels))
        frame = FramePacket(0, 0, 100, 100, b"\0" * 30_000)
        with patch("fall_detection.preview.load_opencv", return_value=cv), patch.dict(
                sys.modules, {"numpy": fake_numpy}), patch.dict("os.environ", {"DISPLAY": ":test"}):
            with Preview() as preview:
                self.assertTrue(preview.show(frame, Prediction(), (snapshot,)))
        cv.rectangle.assert_called_once()
        rendered_text = [call.args[1] for call in cv.putText.call_args_list]
        self.assertIn("FALL ALERT", rendered_text)
        self.assertTrue(any("Track 1" in value for value in rendered_text))


if __name__ == "__main__":
    unittest.main()
