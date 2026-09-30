"""Small, strict configuration surface for reproducible runs."""

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RuntimeConfig:
    width: int = 320
    height: int = 180
    fps: float = 30.0
    frames: int = 60

    def __post_init__(self) -> None:
        for name, upper in (("width", 1920), ("height", 1080), ("frames", 100_000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")
        if type(self.fps) not in (int, float) or not isfinite(self.fps) or not 0 < self.fps <= 240:
            raise ValueError("fps must be a finite number greater than 0 and at most 240")


def load_config(path: Path | None = None, **overrides: object) -> RuntimeConfig:
    settings: dict = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"runtime"} or not isinstance(document["runtime"], dict):
            raise ValueError("configuration must contain only a [runtime] table")
        settings.update(document["runtime"])
    allowed = {field.name for field in fields(RuntimeConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise ValueError("unknown runtime settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return RuntimeConfig(**settings)
