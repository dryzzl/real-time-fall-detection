"""Command-line webcam/video runner for the unvalidated per-person baseline."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys

from .alerts import AlertManager, load_alert_config
from .capture import CaptureConfig, CaptureError, open_capture
from .features import load_feature_config
from .live import run_live_pipeline
from .live_latency import FrameLatencyCollector
from .live_preview import LivePreview
from .pose import OnnxPoseEstimator, PoseError, load_pose_config
from .state import load_state_config
from .temporal_inference import OnnxTemporalClassifier
from .training import TrainingError, load_training_config
from .tracking import load_tracker_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ONNX pose + tracked heuristic states (NOT a validated fall detector)"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Local webcam index, usually 0")
    source.add_argument("--video", type=Path, help="Existing local video file")
    parser.add_argument("--model", type=Path, help="Raw YOLO11-Pose ONNX export")
    parser.add_argument("--temporal-model", type=Path, help="Validated-contract temporal ONNX model (research only)")
    parser.add_argument("--research-alerts", action="store_true",
                        help="Opt in to unvalidated local-only alert-state metadata; NOT emergency notifications")
    parser.add_argument("--alert-config", type=Path, default=Path("configs/alerts.toml"),
                        help="Local research alert persistence/cooldown settings")
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.toml"))
    parser.add_argument("--pose-config", type=Path, default=Path("configs/pose.toml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker.toml"))
    parser.add_argument("--feature-config", type=Path, default=Path("configs/features.toml"))
    parser.add_argument("--state-config", type=Path, default=Path("configs/state.toml"))
    parser.add_argument("--check-models", action="store_true", help="Load model contracts and run a blank-frame pose smoke test without opening capture")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="Write new JSONL file; never overwrite")
    parser.add_argument("--latency-report", type=Path, help="Write bounded processing-latency JSON; never overwrite")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--preview", action="store_true", help="Local pose/track overlay (unvalidated); Q/Esc quits")
    display.add_argument("--headless", action="store_true", help="No window (default)")
    args = parser.parse_args(argv)

    if not args.check_models and args.camera is None and args.video is None:
        parser.error("--camera or --video is required for live inference")

    if args.check_models and (args.output is not None or args.latency_report is not None
                              or args.preview or args.research_alerts or args.max_frames is not None):
        parser.error("--check-models cannot be combined with capture output, preview, alerts or frame limits")

    try:
        if (args.output is not None and args.latency_report is not None
                and args.output.resolve() == args.latency_report.resolve()):
            raise ValueError("--output and --latency-report must be distinct paths")
        profiler = FrameLatencyCollector() if args.latency_report is not None else None
        alert_manager = (AlertManager(load_alert_config(args.alert_config))
                         if args.research_alerts else None)
        pose = OnnxPoseEstimator(load_pose_config(args.pose_config, model_path=args.model))
        tracker = load_tracker_config(args.tracker_config)
        features = load_feature_config(args.feature_config)
        state = load_state_config(args.state_config)
        temporal = None
        if args.temporal_model is not None:
            training = load_training_config(args.training_config)
            if (training.sequence_length != features.window_size
                    or training.keypoint_count != features.keypoint_count):
                raise ValueError("training sequence/keypoint dimensions must match feature configuration")
            temporal = OnnxTemporalClassifier(args.temporal_model, training)
        if pose.config.keypoint_count != features.keypoint_count:
            raise ValueError("pose and feature keypoint counts must match")
        if args.check_models:
            from .contracts import FramePacket
            detections = pose.estimate(FramePacket(0, 0.0, 64, 64, bytes(64 * 64 * 3)))
            print(json.dumps({"mode": "model_preflight", "pose": pose.describe(),
                              "blank_frame_detections": len(detections),
                              "temporal_contract_checked": temporal is not None,
                              "accuracy_validated": False}, allow_nan=False))
            return 0
        capture = CaptureConfig(
            camera=args.camera, video=args.video,
            max_frames=args.max_frames, fallback_fps=args.fallback_fps,
        )
        with ExitStack() as stack:
            frames = stack.enter_context(open_capture(capture))
            preview = stack.enter_context(LivePreview(temporal_model=temporal is not None)) if args.preview else None
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(
                args.output.open("x", encoding="utf-8")
                if args.output else nullcontext(sys.stdout)
            )

            def emit(record: dict) -> None:
                print(json.dumps(record, allow_nan=False), file=stream, flush=True)

            summary = run_live_pipeline(
                frames, emit, estimator=pose,
                tracker_config=tracker, feature_config=features, state_config=state,
                temporal_classifier=temporal,
                alert_manager=alert_manager,
                on_visual_frame=preview.show_live if preview is not None else None,
                on_timing=profiler.record if profiler is not None else None,
            )
            emit(summary.as_record())
            if profiler is not None:
                args.latency_report.parent.mkdir(parents=True, exist_ok=True)
                with args.latency_report.open("x", encoding="utf-8") as report_file:
                    json.dump(profiler.report(), report_file, indent=2, allow_nan=False)
                    report_file.write("\n")
        if args.output:
            print(json.dumps(summary.as_record(), allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed output records were preserved.", file=sys.stderr)
        return 130
    except (ValueError, OSError, CaptureError, PoseError, TrainingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
