import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from fall_detection.cli import main
from fall_detection.contracts import FallState, FramePacket
from fall_detection.features import PoseFeatureSample, TemporalFeatureWindow
from fall_detection.state import (
    StateConfig,
    TemporalStateClassifier,
    TemporalStateStage,
    load_state_config,
    run_state_smoke,
)


def sample(track_id, index, center_y, aspect_ratio, previous_center_y=None, *,
           observed=True, confident_keypoints=4, keypoint_count=4):
    timestamp = index * 0.2
    mask = (True,) * confident_keypoints + (False,) * (keypoint_count - confident_keypoints)
    points = tuple((0.0, 0.0) if available else (None, None) for available in mask)
    if not observed:
        return PoseFeatureSample(
            track_id, index, timestamp, False, None, None, None, None, None,
            ((None, None),) * keypoint_count, (False,) * keypoint_count,
        )
    velocity = (
        None if previous_center_y is None
        else (0.0, (center_y - previous_center_y) / 0.2)
    )
    return PoseFeatureSample(
        track_id, index, timestamp, True, 0.9, (0.5, center_y),
        (0.2, 0.2 / aspect_ratio), aspect_ratio, velocity, points, mask,
    )


def window(track_id, centers, aspects, *, ready=True, missing_latest=False,
           confidence_counts=None):
    samples = []
    previous = None
    confidence_counts = confidence_counts or [4] * len(centers)
    for index, (center, aspect, confident) in enumerate(
            zip(centers, aspects, confidence_counts)):
        samples.append(sample(track_id, index, center, aspect, previous,
                              confident_keypoints=confident))
        previous = center
    if missing_latest:
        samples.append(sample(track_id, len(samples), None, None, observed=False))
    return TemporalFeatureWindow(
        track_id, tuple(samples), sum(item.observed for item in samples),
        ready and not missing_latest,
    )


class StateConfigTests(unittest.TestCase):
    def test_config_file_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.toml"
            path.write_text(
                "[state]\nupright_max_aspect_ratio=0.7\n"
                "lying_min_aspect_ratio=1.1\nmin_contiguous_observations=4\n",
                encoding="utf-8",
            )
            config = load_state_config(path, min_contiguous_observations=3,
                                       transition_window_seconds=None)
            self.assertEqual(config.min_contiguous_observations, 3)
            self.assertEqual(config.upright_max_aspect_ratio, 0.7)
            self.assertEqual(config.lying_min_aspect_ratio, 1.1)
            self.assertEqual(config.transition_window_seconds, 1.5)

    def test_invalid_values_and_documents_are_rejected(self):
        invalid = (
            {"min_contiguous_observations": 1},
            {"min_contiguous_observations": True},
            {"min_keypoint_fraction": 1.1},
            {"upright_max_aspect_ratio": 0},
            {"lying_min_aspect_ratio": 0.7},
            {"normal_max_vertical_speed": -0.1},
            {"falling_min_vertical_speed": 0.1},
            {"falling_min_center_drop": float("nan")},
            {"transition_window_seconds": 0},
            {"fallen_min_duration_seconds": True},
            {"settled_max_vertical_speed": 0.13},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                StateConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for content in (
                "[classifier]\nmin_contiguous_observations=3\n",
                "[state]\nunknown=1\n", "state=2\n", "[state]\n[other]\nx=1\n",
            ):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_state_config(path)


class TemporalStateClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = TemporalStateClassifier()

    def test_standing_sequence_is_normal(self):
        decision = self.classifier.classify(
            window(1, [0.30] * 5, [0.45] * 5)
        )
        self.assertEqual(decision.state, FallState.NORMAL)
        self.assertEqual(decision.reason, "upright_posture_stable")
        self.assertEqual(decision.evidence.contiguous_observations, 5)
        self.assertIsNone(decision.confidence)

    def test_settled_lying_sequence_is_fallen(self):
        decision = self.classifier.classify(
            window(1, [0.70] * 5, [1.50] * 5)
        )
        self.assertEqual(decision.state, FallState.FALLEN)
        self.assertEqual(decision.reason, "lying_posture_settled")
        self.assertAlmostEqual(decision.evidence.settled_lying_duration_seconds, 0.6)

    def test_descent_with_posture_change_is_falling(self):
        decision = self.classifier.classify(
            window(1, [0.30, 0.34, 0.40, 0.48], [0.45, 0.55, 0.72, 0.95])
        )
        self.assertEqual(decision.state, FallState.FALLING)
        self.assertEqual(decision.reason, "rapid_descent_with_posture_change")
        self.assertAlmostEqual(decision.evidence.center_drop, 0.18)
        self.assertAlmostEqual(decision.evidence.aspect_ratio_increase, 0.50)
        self.assertAlmostEqual(decision.evidence.transition_duration_seconds, 0.6)

    def test_missing_current_observation_is_unknown(self):
        decision = self.classifier.classify(
            window(1, [0.30] * 3, [0.45] * 3, missing_latest=True)
        )
        self.assertEqual(decision.state, FallState.UNKNOWN)
        self.assertEqual(decision.reason, "missing_current_observation")

    def test_not_ready_window_is_unknown(self):
        decision = self.classifier.classify(
            window(1, [0.30] * 3, [0.45] * 3, ready=False)
        )
        self.assertEqual(decision.state, FallState.UNKNOWN)
        self.assertEqual(decision.reason, "feature_window_not_ready")

    def test_low_confidence_current_pose_is_unknown(self):
        decision = self.classifier.classify(
            window(1, [0.30] * 3, [0.45] * 3, confidence_counts=[4, 4, 1])
        )
        self.assertEqual(decision.state, FallState.UNKNOWN)
        self.assertEqual(decision.reason, "insufficient_pose_confidence")
        self.assertEqual(decision.evidence.keypoint_fraction, 0.25)

    def test_missing_or_low_confidence_history_breaks_contiguous_evidence(self):
        first = sample(1, 0, 0.30, 0.45)
        missing = sample(1, 1, None, None, observed=False)
        recovered = sample(1, 2, 0.42, 0.75, previous_center_y=None)
        latest = sample(1, 3, 0.50, 1.0, previous_center_y=0.42)
        gap_window = TemporalFeatureWindow(1, (first, missing, recovered, latest), 3, True)
        decision = self.classifier.classify(gap_window)
        self.assertEqual(decision.state, FallState.UNKNOWN)
        self.assertEqual(decision.reason, "insufficient_contiguous_observations")
        self.assertEqual(decision.evidence.contiguous_observations, 2)

    def test_ambiguous_static_posture_is_unknown(self):
        decision = self.classifier.classify(
            window(1, [0.40] * 4, [0.95] * 4)
        )
        self.assertEqual(decision.state, FallState.UNKNOWN)
        self.assertEqual(decision.reason, "ambiguous_or_unsettled_pose")

    def test_vertical_motion_without_shape_change_is_not_called_a_fall(self):
        decision = self.classifier.classify(
            window(1, [0.30, 0.36, 0.43, 0.50], [0.45] * 4)
        )
        self.assertEqual(decision.state, FallState.UNKNOWN)

    def test_shape_change_without_center_drop_is_not_called_a_fall(self):
        decision = self.classifier.classify(
            window(1, [0.30] * 4, [0.45, 0.55, 0.75, 1.0])
        )
        self.assertEqual(decision.state, FallState.UNKNOWN)

    def test_thresholds_are_configurable(self):
        sequence = window(1, [0.30, 0.34, 0.40, 0.48], [0.45, 0.55, 0.72, 0.95])
        strict = TemporalStateClassifier(StateConfig(falling_min_center_drop=0.25))
        self.assertEqual(self.classifier.classify(sequence).state, FallState.FALLING)
        self.assertEqual(strict.classify(sequence).state, FallState.UNKNOWN)

    def test_settled_lying_takes_precedence_after_a_descent(self):
        decision = self.classifier.classify(
            window(
                1,
                [0.30, 0.40, 0.55, 0.55, 0.55, 0.55, 0.55],
                [0.45, 0.80, 1.35, 1.50, 1.50, 1.50, 1.50],
            )
        )
        self.assertEqual(decision.state, FallState.FALLEN)

    def test_classify_many_is_deterministic_and_rejects_duplicate_tracks(self):
        decisions = self.classifier.classify_many([
            window(2, [0.70] * 5, [1.50] * 5),
            window(1, [0.30] * 5, [0.45] * 5),
        ])
        self.assertEqual([item.track_id for item in decisions], [1, 2])
        self.assertEqual([item.state for item in decisions], [FallState.NORMAL, FallState.FALLEN])
        with self.assertRaises(ValueError):
            self.classifier.classify_many([window(1, [0.3] * 3, [0.45] * 3),
                                           window(1, [0.3] * 3, [0.45] * 3)])

    def test_decision_record_is_finite_json(self):
        record = self.classifier.classify(
            window(1, [0.30, 0.34, 0.40, 0.48], [0.45, 0.55, 0.72, 0.95])
        ).as_record()
        json.dumps(record, allow_nan=False)
        self.assertEqual(record["state"], "falling")
        self.assertIsNone(record["confidence"])
        self.assertNotIn("bgr", json.dumps(record))

    def test_rejects_malformed_windows_without_partial_inference(self):
        valid = window(1, [0.30] * 3, [0.45] * 3)
        bad_count = replace(valid, observed_count=2)
        bad_track = replace(valid, track_id=2)
        bad_order = replace(valid, samples=(valid.samples[1], valid.samples[0], valid.samples[2]))
        bad_missing = replace(
            valid.samples[-1], observed=False, detection_confidence=0.9,
        )
        bad_missing_window = replace(valid, samples=valid.samples[:-1] + (bad_missing,))
        bad_shape_sample = replace(valid.samples[-1], box_center_xy=(0.5,))
        bad_shape_window = replace(valid, samples=valid.samples[:-1] + (bad_shape_sample,))
        bad_pair_type_sample = replace(valid.samples[-1], box_center_xy="invalid")
        bad_pair_type_window = replace(
            valid, samples=valid.samples[:-1] + (bad_pair_type_sample,)
        )
        for value in (bad_count, bad_track, bad_order, bad_missing_window,
                      bad_shape_window, bad_pair_type_window, "not-a-window"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.classifier.classify(value)


class TemporalStateStageTests(unittest.TestCase):
    def test_composes_feature_stage_and_classifier_per_track(self):
        class Stage:
            def process(self, frame):
                return (window(1, [0.30] * 3, [0.45] * 3),)

        decisions = TemporalStateStage(Stage()).process(
            FramePacket(0, 0.0, 1, 1, b"\0\0\0")
        )
        self.assertEqual(decisions[0].state, FallState.NORMAL)


class StateSmokeTests(unittest.TestCase):
    def test_smoke_distinguishes_all_required_sequences(self):
        result = run_state_smoke()
        states = {name: value["state"] for name, value in result["scenarios"].items()}
        self.assertEqual(states, {
            "standing": "normal", "lying": "fallen",
            "descent": "falling", "missing": "unknown",
        })
        self.assertIs(result["baseline_state_classifier_available"], True)
        self.assertIs(result["fall_detection_available"], False)

    def test_cli_smoke_is_valid_json_and_end_to_end_status_stays_unavailable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["state-smoke", "--config", "configs/state.toml"]), 0)
        record = json.loads(output.getvalue())
        self.assertEqual(record["scenarios"]["descent"]["state"], "falling")
        self.assertEqual(record["scenarios"]["missing"]["state"], "unknown")
        self.assertIs(record["fall_detection_available"], False)


if __name__ == "__main__":
    unittest.main()
