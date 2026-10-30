"""Deterministic bounded latency profiling tests; no model or camera required."""
import json
import unittest

from fall_detection.contracts import FramePacket
from fall_detection.live import run_live_pipeline
from fall_detection.live_latency import FrameLatencyCollector, STAGES


class EmptyEstimator:
    def estimate(self, frame):
        return ()


class FrameLatencyTests(unittest.TestCase):
    def test_empty_profile_is_explicit(self):
        report = FrameLatencyCollector().report()
        self.assertEqual(report["frames_seen"], 0)
        self.assertIsNone(report["latency_ms"]["total_ms"])
        self.assertFalse(report["camera_to_alert_latency_measured"])

    def test_bounded_recent_percentiles(self):
        collector = FrameLatencyCollector(max_samples=4)
        for value in range(1, 7):
            collector.record(dict(pose_tracking_features_ms=value, classification_ms=0,
                                  record_emit_ms=0, total_ms=value))
        report = collector.report()
        self.assertEqual((report["frames_seen"], report["frames_sampled"]), (6, 4))
        self.assertEqual(report["latency_ms"]["total_ms"]["p50"], 4)
        self.assertEqual(report["latency_ms"]["total_ms"]["p95"], 6)
        self.assertEqual(report["latency_ms"]["total_ms"]["mean"], 4.5)

    def test_rejects_bad_samples_without_counting(self):
        collector = FrameLatencyCollector()
        for sample in ({"total_ms": 1},
                       dict.fromkeys(STAGES, float("nan")),
                       dict.fromkeys(STAGES, -1),
                       dict(pose_tracking_features_ms=1, classification_ms=1,
                            record_emit_ms=1, total_ms=2)):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                collector.record(sample)
        self.assertEqual(collector.frames_seen, 0)

    def test_rejects_invalid_capacity(self):
        for value in (0, -1, 100001, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FrameLatencyCollector(value)

    def test_report_json_safe(self):
        collector = FrameLatencyCollector()
        collector.record(dict(pose_tracking_features_ms=1.5, classification_ms=0.5,
                              record_emit_ms=0.25, total_ms=2.25))
        json.dumps(collector.report(), allow_nan=False)

    def test_live_pipeline_stage_timings_are_recorded(self):
        collector = FrameLatencyCollector()
        clock = iter(range(20)).__next__
        frame = FramePacket(index=0, timestamp_seconds=0.0,
                            width=48, height=48, bgr=bytes(48 * 48 * 3),
                            source="synthetic_fixture")
        events = []
        summary = run_live_pipeline(
            [frame], events.append, estimator=EmptyEstimator(),
            clock=clock, on_timing=collector.record,
        )
        self.assertEqual(summary.frames_processed, 1)
        self.assertEqual(len(events), 1)
        report = collector.report()
        self.assertEqual(report["frames_seen"], 1)
        self.assertEqual(report["latency_ms"]["total_ms"]["p50"], 3000.0)
        self.assertEqual(report["latency_ms"]["pose_tracking_features_ms"]["p50"], 1000.0)
        self.assertEqual(report["latency_ms"]["classification_ms"]["p50"], 1000.0)
        self.assertEqual(report["latency_ms"]["record_emit_ms"]["p50"], 1000.0)


if __name__ == "__main__":
    unittest.main()
