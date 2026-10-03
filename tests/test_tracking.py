import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.cli import main
from fall_detection.contracts import FramePacket
from fall_detection.pose import Keypoint, PoseDetection
from fall_detection.tracking import (
    PersonTracker,
    PoseTrackingStage,
    TrackerConfig,
    load_tracker_config,
    run_tracking_smoke,
)


def detection(center_x, center_y=25, *, marker=0, confidence=0.9, keypoint_confidence=0.9):
    return PoseDetection(
        (center_x - 8, center_y - 15, center_x + 8, center_y + 15), confidence,
        (
            Keypoint(center_x + marker, center_y - 8, keypoint_confidence),
            Keypoint(center_x + marker, center_y + 8, keypoint_confidence),
        ),
    )


def observed_by_center(snapshots):
    return {
        round((snapshot.detection.box_xyxy[0] + snapshot.detection.box_xyxy[2]) / 2): snapshot.track_id
        for snapshot in snapshots if snapshot.detection is not None
    }


class TrackerConfigTests(unittest.TestCase):
    def test_config_file_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.toml"
            path.write_text("[tracker]\nmax_missed_frames=4\nmax_center_distance=3.0\n", encoding="utf-8")
            config = load_tracker_config(path, max_missed_frames=2, min_iou=None)
            self.assertEqual(config.max_missed_frames, 2)
            self.assertEqual(config.max_center_distance, 3.0)
            self.assertEqual(config.min_iou, 0.05)

    def test_invalid_values_and_documents_are_rejected(self):
        invalid = (
            {"max_missed_frames": -1}, {"max_missed_frames": True},
            {"max_center_distance": 0}, {"max_pose_distance": float("nan")},
            {"min_iou": 1.1}, {"min_keypoint_confidence": True},
            {"velocity_smoothing": -0.1}, {"center_weight": -1},
            {"center_weight": 0, "iou_weight": 0, "pose_weight": 0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrackerConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for content in ("[run]\nmax_missed_frames=2\n", "[tracker]\nunknown=1\n",
                            "tracker=2\n", "[tracker]\n[other]\nx=1\n"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_tracker_config(path)


class PersonTrackerTests(unittest.TestCase):
    def test_initial_ids_are_spatially_deterministic(self):
        for inputs in ([detection(80), detection(20)], [detection(20), detection(80)]):
            with self.subTest(order=[item.box_xyxy for item in inputs]):
                self.assertEqual(observed_by_center(PersonTracker().update(0, inputs)), {20: 1, 80: 2})

    def test_two_people_keep_ids_while_crossing(self):
        tracker = PersonTracker()
        first = tracker.update(0, [detection(10, marker=-2), detection(90, marker=2)])
        second = tracker.update(1, [detection(70, marker=2), detection(30, marker=-2)])
        middle = tracker.update(2, [detection(50, marker=2), detection(50, marker=-2)])
        fourth = tracker.update(3, [detection(30, marker=2), detection(70, marker=-2)])
        self.assertEqual(observed_by_center(first), {10: 1, 90: 2})
        self.assertEqual(observed_by_center(second), {30: 1, 70: 2})
        marker_to_id = {round(item.detection.keypoints[0].x): item.track_id for item in middle}
        self.assertEqual(marker_to_id, {48: 1, 52: 2})
        self.assertEqual(observed_by_center(fourth), {30: 2, 70: 1})
        self.assertEqual([item.hits for item in fourth], [4, 4])

    def test_missed_detection_emits_gap_and_recovers_same_id(self):
        tracker = PersonTracker(TrackerConfig(max_missed_frames=2))
        tracker.update(0, [detection(10)])
        missed = tracker.update(1, [])
        recovered = tracker.update(2, [detection(20)])
        self.assertEqual(len(missed), 1)
        self.assertFalse(missed[0].observed)
        self.assertIsNone(missed[0].detection)
        self.assertEqual((missed[0].track_id, missed[0].missed_frames, missed[0].hits), (1, 1, 1))
        self.assertEqual((recovered[0].track_id, recovered[0].missed_frames, recovered[0].hits), (1, 0, 2))

    def test_track_expires_after_configured_missed_frames(self):
        tracker = PersonTracker(TrackerConfig(max_missed_frames=2))
        tracker.update(0, [detection(10)])
        self.assertEqual([item.missed_frames for item in tracker.update(1, [])], [1])
        self.assertEqual([item.missed_frames for item in tracker.update(2, [])], [2])
        self.assertEqual(tracker.update(3, []), ())
        replacement = tracker.update(4, [detection(10)])
        self.assertEqual(replacement[0].track_id, 2)

    def test_skipped_frame_indices_obey_expiry_boundary(self):
        tracker = PersonTracker(TrackerConfig(max_missed_frames=2))
        tracker.update(0, [detection(10)])
        self.assertEqual(tracker.update(3, [detection(10)])[0].track_id, 1)
        tracker = PersonTracker(TrackerConfig(max_missed_frames=2))
        tracker.update(0, [detection(10)])
        self.assertEqual(tracker.update(4, [detection(10)])[0].track_id, 2)

    def test_far_detection_starts_new_track_and_keeps_old_as_missing(self):
        tracker = PersonTracker(TrackerConfig(max_center_distance=1.0, max_pose_distance=1.0))
        tracker.update(0, [detection(10)])
        snapshots = tracker.update(1, [detection(200)])
        self.assertEqual([(item.track_id, item.observed) for item in snapshots], [(1, False), (2, True)])

    def test_low_confidence_keypoints_fall_back_to_box_motion(self):
        tracker = PersonTracker()
        tracker.update(0, [detection(10, marker=-20, keypoint_confidence=0.1)])
        next_frame = tracker.update(1, [detection(20, marker=20, keypoint_confidence=0.1)])
        self.assertEqual(next_frame[0].track_id, 1)

    def test_snapshots_have_bounded_metadata_without_keypoint_payload(self):
        snapshot = PersonTracker().update(0, [detection(10)])[0]
        record = snapshot.as_record()
        self.assertEqual(record["track_id"], 1)
        self.assertEqual(record["keypoint_count"], 2)
        self.assertNotIn("keypoints", record)
        json.dumps(record, allow_nan=False)

    def test_reset_starts_a_new_process_local_sequence(self):
        tracker = PersonTracker()
        tracker.update(4, [detection(10)])
        tracker.reset()
        snapshot = tracker.update(0, [detection(50)])[0]
        self.assertEqual((snapshot.track_id, snapshot.age_frames), (1, 1))

    def test_rejects_invalid_frame_order_collection_and_detection_values(self):
        tracker = PersonTracker()
        tracker.update(0, [])
        for frame_index in (0, -1, True):
            with self.subTest(frame_index=frame_index), self.assertRaises(ValueError):
                tracker.update(frame_index, [])
        with self.assertRaises(ValueError):
            PersonTracker().update(0, iter([]))
        invalid = (
            "not-a-detection",
            PoseDetection((0, 0, 0, 10), 0.9, ()),
            PoseDetection((0, 0, 10, 10), float("nan"), ()),
            PoseDetection((0, 0, 10, 10), 0.9, (Keypoint(0, 0, 2),)),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PersonTracker().update(0, [value])


class TrackingStageTests(unittest.TestCase):
    def test_composes_pose_estimator_and_tracker_without_fall_state(self):
        class Estimator:
            def __init__(self):
                self.frames = []

            def estimate(self, frame):
                self.frames.append(frame.index)
                return (detection(10 + frame.index * 5),)

        estimator = Estimator()
        stage = PoseTrackingStage(estimator)
        first = stage.process(FramePacket(0, 0, 1, 1, b"\0\0\0"))
        second = stage.process(FramePacket(1, 1, 1, 1, b"\0\0\0"))
        self.assertEqual(estimator.frames, [0, 1])
        self.assertEqual((first[0].track_id, second[0].track_id), (1, 1))
        self.assertFalse(hasattr(second[0], "state"))


class TrackingSmokeTests(unittest.TestCase):
    def test_smoke_crosses_and_recovers_missed_person(self):
        result = run_tracking_smoke()
        self.assertEqual(result["active_track_ids"], [1, 2])
        self.assertEqual(result["center_histories"]["1"], [10, 30, 50, 70, 90, 110])
        self.assertEqual(result["center_histories"]["2"], [90, 70, 50, 30, None, 10])
        self.assertIs(result["fall_detection_available"], False)

    def test_cli_smoke_is_valid_json_and_status_remains_unavailable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["track-smoke", "--config", "configs/tracker.toml"]), 0)
        record = json.loads(output.getvalue())
        self.assertEqual(record["frames_processed"], 6)
        self.assertIs(record["synthetic"], True)
        self.assertIs(record["fall_detection_available"], False)


if __name__ == "__main__":
    unittest.main()
