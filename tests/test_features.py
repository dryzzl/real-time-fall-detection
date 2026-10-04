import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.cli import main
from fall_detection.contracts import FramePacket
from fall_detection.features import (
    FeatureConfig,
    PoseTrackingFeatureStage,
    TemporalFeatureBank,
    load_feature_config,
    run_feature_smoke,
)
from fall_detection.pose import Keypoint, PoseDetection
from fall_detection.tracking import TrackSnapshot


def frame(index, timestamp=None, width=200, height=100):
    timestamp = index / 10 if timestamp is None else timestamp
    return FramePacket(index, timestamp, width, height, b"\0" * (width * height * 3))


def detection(center_x=50, center_y=30, *, width=20, height=40,
              confidences=(0.9, 0.9)):
    left, top = center_x - width / 2, center_y - height / 2
    right, bottom = center_x + width / 2, center_y + height / 2
    points = (
        Keypoint(center_x - width / 4, center_y - height / 4, confidences[0]),
        Keypoint(center_x + width / 4, center_y + height / 4, confidences[1]),
    )
    return PoseDetection((left, top, right, bottom), 0.8, points)


def snapshot(frame_index, track_id=1, pose=None):
    pose = detection() if pose is ... else pose
    predicted = pose.box_xyxy if pose is not None else (40, 10, 60, 50)
    return TrackSnapshot(track_id, frame_index, frame_index + 1, frame_index + 1,
                         0 if pose is not None else 1, predicted, pose)


class FeatureConfigTests(unittest.TestCase):
    def test_config_file_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.toml"
            path.write_text(
                "[features]\nwindow_size=12\nkeypoint_count=2\nmin_observed_samples=4\n",
                encoding="utf-8",
            )
            config = load_feature_config(path, window_size=6, min_keypoint_confidence=None)
            self.assertEqual(config.window_size, 6)
            self.assertEqual(config.keypoint_count, 2)
            self.assertEqual(config.min_observed_samples, 4)
            self.assertEqual(config.min_keypoint_confidence, 0.3)

    def test_invalid_values_and_documents_are_rejected(self):
        invalid = (
            {"window_size": 0}, {"window_size": True}, {"keypoint_count": 0},
            {"min_keypoint_confidence": -0.1}, {"min_keypoint_confidence": float("nan")},
            {"min_observed_samples": 0}, {"window_size": 2, "min_observed_samples": 3},
            {"max_motion_gap_seconds": 0}, {"max_motion_gap_seconds": True},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                FeatureConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for content in (
                "[feature]\nwindow_size=2\n", "[features]\nunknown=1\n",
                "features=2\n", "[features]\n[other]\nx=1\n",
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_feature_config(path)


class TemporalFeatureBankTests(unittest.TestCase):
    def setUp(self):
        self.config = FeatureConfig(window_size=3, keypoint_count=2,
                                    min_observed_samples=2, max_motion_gap_seconds=1.0)

    def test_normalizes_pose_to_box_and_box_to_frame(self):
        bank = TemporalFeatureBank(self.config)
        first = bank.update(frame(0), [snapshot(0, pose=detection())])[0].latest
        scaled = detection(120, 60, width=40, height=80)
        second = bank.update(frame(1), [snapshot(1, pose=scaled)])[0].latest
        self.assertEqual(first.keypoints_xy, ((-0.25, -0.25), (0.25, 0.25)))
        self.assertEqual(second.keypoints_xy, first.keypoints_xy)
        self.assertEqual(first.box_center_xy, (0.25, 0.3))
        self.assertEqual(first.box_size_wh, (0.1, 0.4))
        self.assertEqual(first.box_aspect_ratio, 0.5)
        self.assertEqual(second.box_center_xy, (0.6, 0.6))

    def test_low_confidence_keypoint_is_masked_not_zero_filled(self):
        bank = TemporalFeatureBank(self.config)
        sample = bank.update(
            frame(0), [snapshot(0, pose=detection(confidences=(0.29, 0.3)))]
        )[0].latest
        self.assertEqual(sample.keypoint_mask, (False, True))
        self.assertEqual(sample.keypoints_xy, ((None, None), (0.25, 0.25)))

    def test_missing_observation_has_no_pose_or_motion_values(self):
        bank = TemporalFeatureBank(self.config)
        bank.update(frame(0), [snapshot(0, pose=detection())])
        window = bank.update(frame(1), [snapshot(1, pose=None)])[0]
        sample = window.latest
        self.assertFalse(sample.observed)
        self.assertIsNone(sample.detection_confidence)
        self.assertIsNone(sample.box_center_xy)
        self.assertIsNone(sample.box_size_wh)
        self.assertIsNone(sample.box_aspect_ratio)
        self.assertIsNone(sample.center_velocity_xy_per_second)
        self.assertEqual(sample.keypoint_mask, (False, False))
        self.assertEqual(sample.keypoints_xy, ((None, None), (None, None)))
        self.assertFalse(window.ready)

    def test_recovery_after_missing_does_not_bridge_motion(self):
        bank = TemporalFeatureBank(self.config)
        bank.update(frame(0), [snapshot(0, pose=detection(50))])
        bank.update(frame(1), [snapshot(1, pose=None)])
        recovered = bank.update(frame(2), [snapshot(2, pose=detection(70))])[0].latest
        self.assertIsNone(recovered.center_velocity_xy_per_second)

    def test_motion_uses_normalized_coordinates_and_source_time(self):
        bank = TemporalFeatureBank(self.config)
        bank.update(frame(0, timestamp=0.0), [snapshot(0, pose=detection(50, 30))])
        sample = bank.update(
            frame(1, timestamp=0.5), [snapshot(1, pose=detection(70, 40))]
        )[0].latest
        self.assertAlmostEqual(sample.center_velocity_xy_per_second[0], 0.2)
        self.assertAlmostEqual(sample.center_velocity_xy_per_second[1], 0.2)

    def test_motion_is_unknown_across_excessive_time_gap(self):
        config = FeatureConfig(window_size=3, keypoint_count=2,
                               min_observed_samples=2, max_motion_gap_seconds=0.2)
        bank = TemporalFeatureBank(config)
        bank.update(frame(0, timestamp=0.0), [snapshot(0, pose=detection(50))])
        sample = bank.update(
            frame(1, timestamp=0.5), [snapshot(1, pose=detection(70))]
        )[0].latest
        self.assertIsNone(sample.center_velocity_xy_per_second)

    def test_histories_are_independent_and_ordered_by_track_id(self):
        bank = TemporalFeatureBank(self.config)
        first = bank.update(frame(0), [
            snapshot(0, track_id=2, pose=detection(150)),
            snapshot(0, track_id=1, pose=detection(50)),
        ])
        second = bank.update(frame(1), [
            snapshot(1, track_id=1, pose=detection(60)),
            snapshot(1, track_id=2, pose=None),
        ])
        self.assertEqual([window.track_id for window in first], [1, 2])
        self.assertEqual([window.track_id for window in second], [1, 2])
        self.assertEqual([sample.observed for sample in second[0].samples], [True, True])
        self.assertEqual([sample.observed for sample in second[1].samples], [True, False])
        self.assertNotEqual(second[0].samples[0].box_center_xy,
                            second[1].samples[0].box_center_xy)

    def test_history_is_bounded_and_readiness_uses_current_observation(self):
        bank = TemporalFeatureBank(self.config)
        windows = ()
        for index in range(5):
            windows = bank.update(frame(index), [snapshot(index, pose=detection(50 + index))])
        self.assertEqual([sample.frame_index for sample in windows[0].samples], [2, 3, 4])
        self.assertEqual(windows[0].observed_count, 3)
        self.assertTrue(windows[0].ready)
        missing = bank.update(frame(5), [snapshot(5, pose=None)])[0]
        self.assertEqual(len(missing.samples), 3)
        self.assertEqual(missing.observed_count, 2)
        self.assertFalse(missing.ready)

    def test_retired_tracks_are_removed_instead_of_accumulating(self):
        bank = TemporalFeatureBank(self.config)
        bank.update(frame(0), [snapshot(0, track_id=1, pose=detection())])
        self.assertEqual(bank.update(frame(1), []), ())
        replacement = bank.update(
            frame(2), [snapshot(2, track_id=2, pose=detection())]
        )[0]
        self.assertEqual(len(replacement.samples), 1)
        self.assertEqual(replacement.track_id, 2)

    def test_records_are_finite_json_and_do_not_include_image_bytes(self):
        record = TemporalFeatureBank(self.config).update(
            frame(0), [snapshot(0, pose=detection())]
        )[0].as_record()
        encoded = json.dumps(record, allow_nan=False)
        self.assertNotIn("bgr", encoded)
        self.assertEqual(record["latest"]["keypoint_mask"], [True, True])

    def test_reset_allows_a_new_timeline(self):
        bank = TemporalFeatureBank(self.config)
        bank.update(frame(4), [snapshot(4, pose=detection())])
        bank.reset()
        window = bank.update(frame(0), [snapshot(0, pose=detection())])[0]
        self.assertEqual(len(window.samples), 1)

    def test_rejects_invalid_timeline_and_snapshot_contracts(self):
        bank = TemporalFeatureBank(self.config)
        bank.update(frame(0), [])
        for bad_frame in (frame(0, 0.1), frame(1, 0.0)):
            with self.subTest(bad_frame=bad_frame), self.assertRaises(ValueError):
                bank.update(bad_frame, [])
        invalid_sets = (
            iter([]),
            ["not-a-snapshot"],
            [snapshot(1, track_id=True, pose=detection())],
            [snapshot(0, pose=detection())],
            [snapshot(1, pose=detection()), snapshot(1, pose=detection())],
        )
        for values in invalid_sets:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TemporalFeatureBank(self.config).update(frame(1), values)

    def test_rejects_incompatible_or_invalid_pose_values(self):
        bad_detections = (
            PoseDetection((0, 0, 10, 10), 0.9, (Keypoint(1, 1, 0.9),)),
            PoseDetection((0, 0, 0, 10), 0.9,
                          (Keypoint(1, 1, 0.9), Keypoint(2, 2, 0.9))),
            PoseDetection((0, 0, 10, 10), 0.9,
                          (Keypoint(float("nan"), 1, 0.9), Keypoint(2, 2, 0.9))),
        )
        for pose in bad_detections:
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                TemporalFeatureBank(self.config).update(frame(0), [snapshot(0, pose=pose)])

    def test_invalid_late_snapshot_does_not_partially_advance_histories(self):
        bank = TemporalFeatureBank(self.config)
        bank.update(frame(0), [snapshot(0, track_id=1, pose=detection())])
        incompatible = PoseDetection((0, 0, 10, 10), 0.9, (Keypoint(1, 1, 0.9),))
        with self.assertRaises(ValueError):
            bank.update(frame(1), [
                snapshot(1, track_id=1, pose=detection(60)),
                snapshot(1, track_id=2, pose=incompatible),
            ])
        retry = bank.update(frame(1), [snapshot(1, track_id=1, pose=detection(60))])[0]
        self.assertEqual([sample.frame_index for sample in retry.samples], [0, 1])


class FeatureStageTests(unittest.TestCase):
    def test_composes_tracking_and_features_without_fall_state(self):
        class Stage:
            def process(self, current_frame):
                return (snapshot(current_frame.index, pose=detection(50 + current_frame.index)),)

        stage = PoseTrackingFeatureStage(
            Stage(), TemporalFeatureBank(FeatureConfig(window_size=2, keypoint_count=2,
                                                       min_observed_samples=1))
        )
        first = stage.process(frame(0))[0]
        second = stage.process(frame(1))[0]
        self.assertEqual((len(first.samples), len(second.samples)), (1, 2))
        self.assertFalse(hasattr(second, "state"))


class FeatureSmokeTests(unittest.TestCase):
    def test_smoke_keeps_independent_bounded_histories_and_missing_sample(self):
        config = FeatureConfig(window_size=4, keypoint_count=2, min_observed_samples=2)
        result = run_feature_smoke(config)
        self.assertEqual(result["active_track_ids"], [1, 2])
        self.assertEqual(result["window_lengths"], {"1": 4, "2": 4})
        self.assertIn(False, result["observation_histories"]["2"])
        self.assertIs(result["low_confidence_keypoints_masked"], True)
        self.assertIs(result["fall_detection_available"], False)

    def test_cli_smoke_is_valid_json_and_detection_stays_unavailable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["feature-smoke", "--config", "configs/features.toml"]), 0)
        record = json.loads(output.getvalue())
        self.assertEqual(record["frames_processed"], 6)
        self.assertIs(record["synthetic"], True)
        self.assertIs(record["fall_detection_available"], False)


if __name__ == "__main__":
    unittest.main()
