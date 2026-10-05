"""Command-line entry point for synthetic and local video sources."""

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys

from . import __version__
from .capture import CaptureConfig, CaptureError, open_capture
from .config import load_config
from .features import load_feature_config, run_feature_smoke
from .pipeline import run_pipeline
from .pose import OnnxPoseEstimator, PoseError, load_pose_config
from .sources import synthetic_frames
from .preview import Preview
from .state import load_state_config, run_state_smoke
from .tracking import load_tracker_config, run_tracking_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-Time Fall Detection — development prototype")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show implemented and pending capabilities")
    demo = commands.add_parser("demo", help="Run synthetic frames through the pipeline; no fall detection yet")
    demo.add_argument("--config", type=Path)
    demo.add_argument("--frames", type=int)
    demo.add_argument("--fps", type=float)
    demo.add_argument("--width", type=int)
    demo.add_argument("--height", type=int)
    demo.add_argument("--realtime", action="store_true", help="Pace output using source timestamps")
    demo.add_argument("--output", type=Path, help="Write JSONL to a new file; existing files are never overwritten")
    capture = commands.add_parser("capture", help="Read a webcam or local video; no fall detection yet")
    source = capture.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="Local camera index, usually 0")
    source.add_argument("--video", type=Path, help="Existing local video file")
    capture.add_argument("--max-frames", type=int, help="Stop after this many frames; default: EOF or Ctrl+C")
    capture.add_argument("--fallback-fps", type=float, default=30.0, help="Video timing estimate if timestamps and FPS are invalid")
    capture.add_argument("--realtime", action="store_true", help="Pace local-video replay; camera capture is already live")
    display = capture.add_mutually_exclusive_group()
    display.add_argument("--preview", action="store_true", help="Show a local window; Q/Esc or closing it stops capture")
    display.add_argument("--headless", action="store_true", help="No display (the default)")
    capture.add_argument("--output", type=Path, help="Write metadata JSONL to a new file")
    pose = commands.add_parser("pose-check", help="Validate a supported ONNX pose model contract")
    pose.add_argument("--config", type=Path, help="Pose TOML configuration")
    pose.add_argument("--model", type=Path, help="Local YOLO11-pose ONNX model path")
    pose.add_argument("--input-width", type=int)
    pose.add_argument("--input-height", type=int)
    pose.add_argument("--keypoint-count", type=int)
    pose.add_argument("--confidence-threshold", type=float)
    pose.add_argument("--iou-threshold", type=float)
    pose.add_argument("--smoke", action="store_true", help="Run one black frame through the model")
    tracking = commands.add_parser("track-smoke", help="Run deterministic synthetic person-association checks")
    tracking.add_argument("--config", type=Path, help="Tracker TOML configuration")
    features = commands.add_parser("feature-smoke", help="Run deterministic temporal-feature checks")
    features.add_argument("--config", type=Path, help="Feature TOML configuration")
    state = commands.add_parser("state-smoke", help="Run explainable temporal-state sequence checks")
    state.add_argument("--config", type=Path, help="State baseline TOML configuration")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps({
            "version": __version__, "milestone": "6/10",
            "implemented": ["configuration", "frame_contract", "synthetic_stream", "pipeline", "jsonl_output",
                            "webcam_input", "local_video_input", "optional_preview", "headless_capture",
                            "onnx_pose_adapter", "letterbox_preprocessing", "pose_output_decoding",
                            "person_tracking", "stable_track_ids", "stale_track_expiry",
                            "normalized_pose_features", "motion_features", "temporal_feature_windows",
                            "explainable_state_baseline", "per_person_state_decisions"],
            "baseline_state_classifier_available": True,
            "fall_detection_available": False,
            "next": "training_and_export_preparation",
        }))
        return 0
    try:
        if args.command == "state-smoke":
            print(json.dumps(run_state_smoke(load_state_config(args.config)), allow_nan=False))
            return 0
        if args.command == "feature-smoke":
            print(json.dumps(run_feature_smoke(load_feature_config(args.config)), allow_nan=False))
            return 0
        if args.command == "track-smoke":
            print(json.dumps(run_tracking_smoke(load_tracker_config(args.config)), allow_nan=False))
            return 0
        if args.command == "pose-check":
            pose_config = load_pose_config(
                args.config, model_path=args.model, input_width=args.input_width,
                input_height=args.input_height, keypoint_count=args.keypoint_count,
                confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold,
            )
            estimator = OnnxPoseEstimator(pose_config)
            result = {"event": "pose_model_check", **estimator.describe(), "smoke_run": False}
            if args.smoke:
                from .contracts import FramePacket
                frame = FramePacket(0, 0.0, pose_config.input_width, pose_config.input_height,
                                    b"\0" * (pose_config.input_width * pose_config.input_height * 3),
                                    source="pose_contract_smoke")
                result.update(smoke_run=True, smoke_frame="black_test_frame",
                              pose_count=len(estimator.estimate(frame)))
            print(json.dumps(result, allow_nan=False))
            return 0
        if args.command == "demo":
            config = load_config(args.config, frames=args.frames, fps=args.fps, width=args.width, height=args.height)
        else:
            config = CaptureConfig(camera=args.camera, video=args.video,
                                   max_frames=args.max_frames, fallback_fps=args.fallback_fps)
            if args.camera is not None and args.realtime:
                raise ValueError("--realtime is for video replay; cameras already run live")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        context = args.output.open("x", encoding="utf-8") if args.output else nullcontext(sys.stdout)
        with ExitStack() as stack:
            stream = stack.enter_context(context)
            def emit(record: dict) -> None:
                print(json.dumps(record, allow_nan=False), file=stream, flush=True)
            with ExitStack() as resources:
                preview = None
                if args.command == "demo":
                    frames = synthetic_frames(config)
                else:
                    if args.preview:
                        preview = resources.enter_context(Preview())
                    frames = resources.enter_context(open_capture(config))
                summary = run_pipeline(frames, emit, realtime=args.realtime,
                                       on_frame=preview.show if preview else None)
            emit(summary.as_record())
        if args.output:
            print(json.dumps(summary.as_record(), allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; any completed output records have been preserved.", file=sys.stderr)
        return 130
    except (ValueError, OSError, CaptureError, PoseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
