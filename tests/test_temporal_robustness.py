"""Synthetic temporal alert-policy regression tests (not model validation)."""
from dataclasses import dataclass
import pytest

@dataclass
class AlertPolicy:
    trigger_frames: int = 3
    clear_frames: int = 5
    missing_tolerance: int = 2

    def __post_init__(self):
        if min(self.trigger_frames, self.clear_frames) < 1 or self.missing_tolerance < 0:
            raise ValueError("Invalid thresholds")
        self.active = False
        self.fallen = self.normal = self.missing = 0

    def update(self, state: str | None) -> bool:
        if state is None:
            self.missing += 1
            if self.missing > self.missing_tolerance:
                self.fallen = 0
            return self.active
        if state not in {"Normal", "Falling", "Fallen"}:
            raise ValueError("Unknown state")
        self.missing = 0
        if state == "Fallen":
            self.fallen += 1
            self.normal = 0
            if self.fallen >= self.trigger_frames:
                self.active = True
        elif state == "Normal":
            self.fallen = 0
            self.normal += 1
            if self.normal >= self.clear_frames:
                self.active = False
        else:
            self.fallen = self.normal = 0
        return self.active

def test_single_frame():
    p = AlertPolicy()
    assert [p.update(s) for s in ("Normal", "Fallen", "Normal")] == [False] * 3

def test_persistent():
    p = AlertPolicy()
    for _ in range(3): p.update("Fallen")
    assert p.active
    for _ in range(4): assert p.update("Normal")
    assert not p.update("Normal")

def test_occlusion():
    p = AlertPolicy()
    for _ in range(3): p.update("Fallen")
    for _ in range(4): assert p.update(None)

def test_falling():
    p = AlertPolicy()
    for _ in range(10): assert not p.update("Falling")

def test_invalid():
    with pytest.raises(ValueError): AlertPolicy().update("Unknown")
