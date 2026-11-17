"""Run the local metadata dashboard alongside the existing live video pipeline.

This launcher does not supply model weights or validate fall predictions. It
starts the existing live CLI as a child process and never serves video pixels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from math import isfinite
from threading import Thread
import webbrowser

from .dashboard import create_server


def build_live_command(args: argparse.Namespace) -> list[str]:
    """Construct a shell-free command with explicit, supported live options."""
    command = [sys.executable, "-m", "fall_detection.live_cli"]
    if getattr(args, "demo", False):
        return [sys.executable, "-m", "fall_detection", "demo", "--frames",
                str(args.max_frames or 120),
                *([] if getattr(args, "demo_fast", False) else ["--realtime"]),
                "--output", str(args.output)]
    if args.camera is not None:
        command.extend(("--camera", str(args.camera)))
    else:
        command.extend(("--video", str(args.video)))
    command.extend(("--output", str(args.output), "--headless"))
    for flag, value in (
        ("--model", args.model),
        ("--temporal-model", args.temporal_model),
        ("--max-frames", args.max_frames),
        ("--fallback-fps", getattr(args, "fallback_fps", None)),
        ("--alert-config", args.alert_config),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    if getattr(args, "latency_report", None) is not None:
        command.extend(("--latency-report", str(args.latency_report)))
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
    parser.add_argument("--demo-fast", action="store_true",
                        help="Generate synthetic demo frames without realtime pacing (requires --demo)")
    parser.add_argument("--model", type=Path, help="Authorized raw YOLO11-Pose ONNX export")
    parser.add_argument("--temporal-model", type=Path, help="Compatible research temporal ONNX model")
    parser.add_argument("--research-alerts", action="store_true")
    parser.add_argument("--alert-config", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fallback-fps", type=float, default=None,
                        help="FPS to use when camera/video metadata is unavailable (live only)")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL output; never overwritten")
    parser.add_argument("--latency-report", type=Path, help="New per-stage timing report (live webcam/video only)")
    parser.add_argument("--hold-seconds", type=float, default=None,
                        help="Keep dashboard available after capture ends (0-3600 seconds)")
    parser.add_argument("--failure-hold-seconds", type=float, default=0.0,
                        help="Keep dashboard available after failed capture (0-3600 seconds)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--capture-timeout", type=float, default=None,
                        help="Stop capture after this many seconds (positive, finite; exit 124)")
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--open-browser", dest="open_browser", action="store_true", default=None,
                         help="Open the local dashboard in a browser (default for --demo)")
    browser.add_argument("--no-open-browser", dest="open_browser", action="store_false",
                         help="Do not open a browser (useful for CI and headless demos)")
    args = parser.parse_args(argv)

    # Demo mode is intended to be visible without a second terminal or manual URL.
    # Live capture remains opt-in so unattended camera jobs never launch a browser.
    if args.open_browser is None:
        args.open_browser = args.demo

    # A short synthetic capture otherwise exits before a person can inspect the browser.
    # Explicit --hold-seconds 0 still opts out; live captures retain immediate exit.
    if args.hold_seconds is None:
        args.hold_seconds = 30.0 if args.demo else 0.0

    if args.demo_fast and not args.demo:
        parser.error("--demo-fast requires --demo")
    if args.demo and (args.model or args.temporal_model or args.research_alerts or args.latency_report or args.alert_config or args.fallback_fps is not None):
        parser.error("--demo cannot use live inference, alert, or latency options")
    if args.latency_report is not None and args.output.resolve() == args.latency_report.resolve():
        parser.error("--latency-report must differ from --output")
    if args.latency_report is not None and args.latency_report.exists():
        parser.error("latency report already exists; choose a new path")
    if not isfinite(args.hold_seconds) or not 0 <= args.hold_seconds <= 3600:
        parser.error("--hold-seconds must be finite and between 0 and 3600")
    if not isfinite(args.failure_hold_seconds) or not 0 <= args.failure_hold_seconds <= 3600:
        parser.error("--failure-hold-seconds must be finite and between 0 and 3600")
    if args.capture_timeout is not None and (not isfinite(args.capture_timeout) or args.capture_timeout <= 0):
        parser.error("--capture-timeout must be finite and positive")
    if args.output.exists():
        parser.error("output already exists; choose a new JSONL path")
    if args.fallback_fps is not None and (not isfinite(args.fallback_fps) or args.fallback_fps <= 0):
        parser.error("--fallback-fps must be finite and positive")
    # Fail before opening a browser or starting a child when local inference
    # artifacts cannot possibly be loaded. Camera devices are checked by the
    # live runner because they may be virtual or hot-plugged.
    for label, candidate in (("--video", args.video), ("--model", args.model),
                             ("--temporal-model", args.temporal_model),
                             ("--alert-config", args.alert_config if args.research_alerts else None)):
        if candidate is not None and not candidate.is_file():
            parser.error(f"{label} must refer to an existing regular file: {candidate}")
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
            try:
                opened = webbrowser.open(url)
                if not opened:
                    print(f"Browser did not open automatically; visit {url}", file=sys.stderr)
            except (OSError, webbrowser.Error) as error:
                # An unavailable browser must not prevent video inference or
                # synthetic metadata capture in a headless environment.
                print(f"Browser launch unavailable ({error}); visit {url}", file=sys.stderr)
        child = subprocess.Popen(build_live_command(args))
        try:
            result = child.wait(timeout=args.capture_timeout) if args.capture_timeout is not None else child.wait()
        except subprocess.TimeoutExpired:
            print("Capture exceeded --capture-timeout; terminating child process.", file=sys.stderr)
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            result = 124
        # On failed capture, allow inspecting any partial metadata or the
        # health endpoint before the loopback server shuts down. The explicit
        # failure hold overrides the normal successful-capture hold.
        hold = args.failure_hold_seconds if result != 0 else args.hold_seconds
        if hold:
            print(f"Capture finished (exit {result}); dashboard remains available for {hold:g}s.", flush=True)
            time.sleep(hold)
        return result
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
