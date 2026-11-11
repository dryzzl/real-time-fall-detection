"""Run the local metadata dashboard alongside the existing live video pipeline.

This launcher does not supply model weights or validate fall predictions. It
starts the existing live CLI as a child process and never serves video pixels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from threading import Thread
import webbrowser

from .dashboard import create_server


def build_live_command(args: argparse.Namespace) -> list[str]:
    """Construct a shell-free command with explicit, supported live options."""
    command = [sys.executable, "-m", "fall_detection.live_cli"]
    if getattr(args, "demo", False):
        return [sys.executable, "-m", "fall_detection", "demo", "--frames",
                str(args.max_frames or 120), "--realtime", "--output", str(args.output)]
    if args.camera is not None:
        command.extend(("--camera", str(args.camera)))
    else:
        command.extend(("--video", str(args.video)))
    command.extend(("--output", str(args.output), "--headless"))
    for flag, value in (
        ("--model", args.model),
        ("--temporal-model", args.temporal_model),
        ("--max-frames", args.max_frames),
        ("--alert-config", args.alert_config),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    if args.research_alerts:
        command.append("--research-alerts")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch local browser metadata monitoring with webcam/video inference (research only)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int)
    source.add_argument("--video", type=Path)
    source.add_argument("--demo", action="store_true", help="Synthetic, dependency-free browser demo; no fall inference")
    parser.add_argument("--model", type=Path, help="Authorized raw YOLO11-Pose ONNX export")
    parser.add_argument("--temporal-model", type=Path, help="Compatible research temporal ONNX model")
    parser.add_argument("--research-alerts", action="store_true")
    parser.add_argument("--alert-config", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL output; never overwritten")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error("output already exists; choose a new JSONL path")
    if args.camera is not None and args.camera < 0:
        parser.error("camera index must be nonnegative")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")

    server = None
    child = None
    try:
        server = create_server(args.output, args.port)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Research metadata dashboard: {url}", flush=True)
        print("Fall states are unvalidated; this is NOT an emergency monitoring system.", flush=True)
        if args.open_browser:
            webbrowser.open(url)
        child = subprocess.Popen(build_live_command(args))
        return child.wait()
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        return 130
    except (OSError, ValueError) as error:
        print(f"monitor error: {error}", file=sys.stderr)
        return 2
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
