"""Command-line webcam/video runner for the unvalidated per-person baseline."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys

from .capture import CaptureConfig, CaptureError, open_capture
from .features import load_feature_config
from .live import run_live_pipeline
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="Local webcam index, usually 0")
    source.add_argument("--video", type=Path, help="Existing local video file")
    parser.add_argument("--model", type=Path, help="Raw YOLO11-Pose ONNX export")
    parser.add_argument("--temporal-model", type=Path, help="Validated-contract temporal ONNX model (research only)")
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.toml"))
    parser.add_argument("--pose-config", type=Path, default=Path("configs/pose.toml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker.toml"))
    parser.add_argument("--feature-config", type=Path, default=Path("configs/features.toml"))
    parser.add_argument("--state-config", type=Path, default=Path("configs/state.toml"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="Write new JSONL file; never overwrite")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--preview", action="store_true", help="Local pose/track overlay (unvalidated); Q/Esc quits")
    display.add_argument("--headless", action="store_true", help="No window (default)")
    args = parser.parse_args(argv)

    try:
        capture = CaptureConfig(
            camera=args.camera, video=args.video,
            max_frames=args.max_frames, fallback_fps=args.fallback_fps,
        )
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
        with ExitStack() as stack:
            frames = stack.enter_context(open_capture(capture))
            preview = stack.enter_context(LivePreview()) if args.preview else None
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
                on_visual_frame=preview.show_live if preview is not None else None,
            )
            emit(summary.as_record())
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
