"""Command-line entry point for synthetic and local video sources."""

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys

from . import __version__
from .capture import CaptureConfig, CaptureError, open_capture
from .config import load_config
from .pipeline import run_pipeline
from .sources import synthetic_frames
from .preview import Preview


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
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps({
            "version": __version__, "milestone": "2/10",
            "implemented": ["configuration", "frame_contract", "synthetic_stream", "pipeline", "jsonl_output",
                            "webcam_input", "local_video_input", "optional_preview", "headless_capture"],
            "fall_detection_available": False,
            "next": "onnx_pose_adapter",
        }))
        return 0
    try:
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
    except (ValueError, OSError, CaptureError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
