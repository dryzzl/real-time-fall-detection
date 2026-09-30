"""Command-line entry point for the first development milestone."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from . import __version__
from .config import load_config
from .pipeline import run_pipeline
from .sources import synthetic_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-Time Fall Detection — development foundation")
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
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps({
            "version": __version__, "milestone": "1/10",
            "implemented": ["configuration", "frame_contract", "synthetic_stream", "pipeline", "jsonl_output"],
            "fall_detection_available": False,
            "next": "webcam_and_video_input",
        }))
        return 0
    try:
        config = load_config(args.config, frames=args.frames, fps=args.fps, width=args.width, height=args.height)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        context = args.output.open("x", encoding="utf-8") if args.output else nullcontext(sys.stdout)
        with context as stream:
            def emit(record: dict) -> None:
                print(json.dumps(record, allow_nan=False), file=stream, flush=True)
            summary = run_pipeline(synthetic_frames(config), emit, realtime=args.realtime)
            emit(summary.as_record())
        if args.output:
            print(json.dumps(summary.as_record(), allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; any completed output records have been preserved.", file=sys.stderr)
        return 130
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
