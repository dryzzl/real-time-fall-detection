"""Deterministic synthetic input; no camera or model is accessed."""

from collections.abc import Iterator

from .config import RuntimeConfig
from .contracts import FramePacket


def synthetic_frames(config: RuntimeConfig) -> Iterator[FramePacket]:
    """Yield BGR frames with a moving bar and source-relative timestamps.

    This fixture checks data flow. It does not depict a person or a fall.
    Only one frame is allocated at a time.
    """
    for index in range(config.frames):
        row = bytearray(config.width * 3)
        column = index % config.width
        row[column * 3 : column * 3 + 3] = b"\x50\xb0\xff"
        yield FramePacket(
            index=index,
            timestamp_seconds=index / config.fps,
            width=config.width,
            height=config.height,
            bgr=bytes(row) * config.height,
        )
