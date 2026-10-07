"""Persistent, local-only alert lifecycle for per-person state decisions."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path
import tomllib
from typing import Iterable, Protocol

from .contracts import FallState, FramePacket
from .state import StateDecision, StateEvidence


ALERT_EVENT_SCHEMA_VERSION = 1


class AlertError(ValueError):
    """Raised when alert input, lifecycle actions, or local logs are invalid."""


class AlertStatus(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    COOLDOWN = "cooldown"


_TRANSITIONS = {
    "pending_started",
    "pending_cancelled",
    "alert_opened",
    "alert_acknowledged",
    "alert_reset",
    "alert_rearmed",
}

_TRANSITION_STATUSES = {
    "pending_started": (AlertStatus.IDLE, AlertStatus.PENDING, False),
    "pending_cancelled": (AlertStatus.PENDING, AlertStatus.IDLE, False),
    "alert_opened": (AlertStatus.PENDING, AlertStatus.ACTIVE, True),
    "alert_acknowledged": (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED, True),
    "alert_reset": (AlertStatus.ACKNOWLEDGED, AlertStatus.COOLDOWN, True),
    "alert_rearmed": (AlertStatus.COOLDOWN, AlertStatus.IDLE, True),
}


@dataclass(frozen=True)
class AlertConfig:
    """Validated local alert behavior; no remote action is configured."""

    fallen_persistence_seconds: float = 1.0
    cooldown_seconds: float = 10.0
    max_tracks: int = 1_000
    fsync_events: bool = True

    def __post_init__(self) -> None:
        for name in ("fallen_persistence_seconds", "cooldown_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise AlertError(f"{name} must be a finite number greater than zero")
        if type(self.max_tracks) is not int or not 1 <= self.max_tracks <= 1_000_000:
            raise AlertError("max_tracks must be an integer between 1 and 1000000")
        if type(self.fsync_events) is not bool:
            raise AlertError("fsync_events must be a boolean")


def load_alert_config(path: Path | None = None, **overrides: object) -> AlertConfig:
    settings: dict[str, object] = {}
    if path is not None:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if set(document) != {"alerts"} or not isinstance(document["alerts"], dict):
            raise AlertError("alert configuration must contain only an [alerts] table")
        settings.update(document["alerts"])
    allowed = {field.name for field in fields(AlertConfig)}
    unknown = (settings.keys() | overrides.keys()) - allowed
    if unknown:
        raise AlertError("unknown alert settings: " + ", ".join(sorted(unknown)))
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return AlertConfig(**settings)


def _finite_timestamp(value: object) -> float:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise AlertError("alert timestamps must be finite and nonnegative")
    return float(value)


def _positive_track_id(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise AlertError("track_id must be a positive integer")
    return value


def _nonempty_text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise AlertError(f"{name} must be a nonempty string of at most {maximum} characters")
    if any(ord(character) < 32 for character in value):
        raise AlertError(f"{name} cannot contain control characters")
    return value


@dataclass(frozen=True)
class AlertEvent:
    event_id: int
    timestamp_seconds: float
    track_id: int
    frame_index: int
    alert_id: str | None
    transition: str
    from_status: AlertStatus
    to_status: AlertStatus
    observed_state: FallState
    reason: str

    def __post_init__(self) -> None:
        if type(self.event_id) is not int or self.event_id <= 0:
            raise AlertError("event_id must be a positive integer")
        _finite_timestamp(self.timestamp_seconds)
        _positive_track_id(self.track_id)
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise AlertError("frame_index must be a nonnegative integer")
        if self.alert_id is not None:
            _nonempty_text(self.alert_id, "alert_id", maximum=64)
        if not isinstance(self.transition, str) or self.transition not in _TRANSITIONS:
            raise AlertError("unsupported alert transition")
        if not isinstance(self.from_status, AlertStatus) or not isinstance(self.to_status, AlertStatus):
            raise AlertError("alert event statuses must be AlertStatus values")
        if not isinstance(self.observed_state, FallState):
            raise AlertError("observed_state must be a FallState")
        _nonempty_text(self.reason, "reason")
        expected_from, expected_to, requires_alert_id = _TRANSITION_STATUSES[self.transition]
        if (self.from_status, self.to_status) != (expected_from, expected_to):
            raise AlertError("alert transition statuses do not match the transition")
        if requires_alert_id != (self.alert_id is not None):
            requirement = "requires" if requires_alert_id else "cannot include"
            raise AlertError(f"{self.transition} {requirement} an alert_id")

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": ALERT_EVENT_SCHEMA_VERSION,
            "event": "alert_transition",
            "event_id": self.event_id,
            "timestamp_seconds": self.timestamp_seconds,
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "alert_id": self.alert_id,
            "transition": self.transition,
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "observed_state": self.observed_state.value,
            "reason": self.reason,
        }

    @classmethod
    def from_record(cls, record: object) -> AlertEvent:
        expected = {
            "schema_version", "event", "event_id", "timestamp_seconds", "track_id",
            "frame_index", "alert_id", "transition", "from_status", "to_status",
            "observed_state", "reason",
        }
        if not isinstance(record, dict) or set(record) != expected:
            raise AlertError("alert log record has unknown or missing fields")
        if (type(record["schema_version"]) is not int
                or record["schema_version"] != ALERT_EVENT_SCHEMA_VERSION):
            raise AlertError(f"alert event schema_version must be {ALERT_EVENT_SCHEMA_VERSION}")
        if record["event"] != "alert_transition":
            raise AlertError("alert log event must be alert_transition")
        try:
            return cls(
                event_id=record["event_id"],
                timestamp_seconds=record["timestamp_seconds"],
                track_id=record["track_id"],
                frame_index=record["frame_index"],
                alert_id=record["alert_id"],
                transition=record["transition"],
                from_status=AlertStatus(record["from_status"]),
                to_status=AlertStatus(record["to_status"]),
                observed_state=FallState(record["observed_state"]),
                reason=record["reason"],
            )
        except (TypeError, ValueError) as error:
            raise AlertError(f"invalid alert log record: {error}") from error


class JsonlAlertLog:
    """Append-only local metadata log with strict validation and optional fsync."""

    def __init__(self, path: Path, *, fsync: bool = True, create_new: bool = False) -> None:
        if not isinstance(path, Path):
            raise AlertError("alert log path must be a Path")
        if type(fsync) is not bool or type(create_new) is not bool:
            raise AlertError("fsync and create_new must be booleans")
        self.path = path
        self.fsync = fsync
        self.create_new = create_new
        self._stream = None
        self._events: tuple[AlertEvent, ...] = ()

    @property
    def events(self) -> tuple[AlertEvent, ...]:
        return self._events

    @property
    def next_event_id(self) -> int:
        return len(self._events) + 1

    def _load(self) -> tuple[AlertEvent, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise AlertError(f"alert log is not a file: {self.path}")
        events = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise AlertError(f"alert log line {line_number} is blank")
                if len(line.encode("utf-8")) > 1_000_000:
                    raise AlertError(f"alert log line {line_number} exceeds 1 MB")
                try:
                    record = json.loads(
                        line,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            AlertError(f"non-finite JSON number {value}")
                        ),
                    )
                except (json.JSONDecodeError, AlertError) as error:
                    raise AlertError(f"invalid JSON on alert log line {line_number}: {error}") from error
                event = AlertEvent.from_record(record)
                if event.event_id != line_number:
                    raise AlertError("alert event IDs must be contiguous and start at 1")
                events.append(event)
        return tuple(events)

    def __enter__(self) -> JsonlAlertLog:
        if self._stream is not None:
            raise AlertError("alert log is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.create_new:
            self._events = ()
            self._stream = self.path.open("x", encoding="utf-8")
        else:
            self._events = self._load()
            self._stream = self.path.open("a", encoding="utf-8")
        return self

    def append(self, event: AlertEvent) -> None:
        if self._stream is None:
            raise AlertError("alert log must be opened before append")
        if not isinstance(event, AlertEvent):
            raise AlertError("alert log accepts AlertEvent values")
        if event.event_id != self.next_event_id:
            raise AlertError(f"expected alert event_id {self.next_event_id}")
        line = json.dumps(event.as_record(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        if self.fsync:
            os.fsync(self._stream.fileno())
        self._events = self._events + (event,)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class AlertOverlay:
    visible: bool
    headline: str
    detail: str
    color_bgr: tuple[int, int, int]

    def as_record(self) -> dict[str, object]:
        return {
            "visible": self.visible,
            "headline": self.headline,
            "detail": self.detail,
            "color_bgr": list(self.color_bgr),
        }


@dataclass(frozen=True)
class AlertSnapshot:
    track_id: int
    status: AlertStatus
    alert_id: str | None
    observed_state: FallState
    reason: str
    pending_since_seconds: float | None
    activated_at_seconds: float | None
    acknowledged_at_seconds: float | None
    cooldown_until_seconds: float | None
    updated_at_seconds: float

    @property
    def overlay(self) -> AlertOverlay:
        if self.status is AlertStatus.PENDING:
            return AlertOverlay(
                True, "POSSIBLE FALL", f"Track {self.track_id} | confirmation pending", (0, 210, 255)
            )
        if self.status is AlertStatus.ACTIVE:
            return AlertOverlay(
                True, "FALL ALERT", f"Track {self.track_id} | acknowledgment required", (0, 0, 255)
            )
        if self.status is AlertStatus.ACKNOWLEDGED:
            return AlertOverlay(
                True, "FALL ALERT ACKNOWLEDGED",
                f"Track {self.track_id} | normal observation required before reset",
                (0, 140, 255),
            )
        return AlertOverlay(False, "", "", (0, 0, 0))

    def as_record(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "status": self.status.value,
            "alert_id": self.alert_id,
            "observed_state": self.observed_state.value,
            "reason": self.reason,
            "pending_since_seconds": self.pending_since_seconds,
            "activated_at_seconds": self.activated_at_seconds,
            "acknowledged_at_seconds": self.acknowledged_at_seconds,
            "cooldown_until_seconds": self.cooldown_until_seconds,
            "updated_at_seconds": self.updated_at_seconds,
            "overlay": self.overlay.as_record(),
        }


@dataclass(frozen=True)
class _TrackAlert:
    track_id: int
    status: AlertStatus
    alert_id: str | None
    last_frame_index: int
    last_timestamp_seconds: float
    last_state: FallState
    reason: str
    pending_since_seconds: float | None = None
    activated_at_seconds: float | None = None
    acknowledged_at_seconds: float | None = None
    cooldown_until_seconds: float | None = None

    def snapshot(self) -> AlertSnapshot:
        return AlertSnapshot(
            self.track_id, self.status, self.alert_id, self.last_state, self.reason,
            self.pending_since_seconds, self.activated_at_seconds,
            self.acknowledged_at_seconds, self.cooldown_until_seconds,
            self.last_timestamp_seconds,
        )


def _validate_decision(decision: StateDecision) -> None:
    if not isinstance(decision, StateDecision):
        raise AlertError("alert updates require a StateDecision")
    if type(decision.track_id) is not int or decision.track_id <= 0:
        raise AlertError("decision track_id must be a positive integer")
    if type(decision.frame_index) is not int or decision.frame_index < 0:
        raise AlertError("decision frame_index must be a nonnegative integer")
    if not isinstance(decision.state, FallState):
        raise AlertError("decision state must be a FallState")
    _nonempty_text(decision.reason, "decision reason")


class AlertManager:
    """Latch local alerts until acknowledgment and a guarded reset complete."""

    def __init__(self, config: AlertConfig | None = None,
                 event_log: JsonlAlertLog | None = None) -> None:
        if config is not None and not isinstance(config, AlertConfig):
            raise AlertError("config must be an AlertConfig")
        self.config = config or AlertConfig()
        if event_log is not None and not isinstance(event_log, JsonlAlertLog):
            raise AlertError("event_log must be a JsonlAlertLog")
        if event_log is not None and event_log._stream is None:
            raise AlertError("event_log must be open before constructing AlertManager")
        self.event_log = event_log
        self._tracks: dict[int, _TrackAlert] = {}
        self._events: list[AlertEvent] = []
        self._next_event_id = event_log.next_event_id if event_log is not None else 1
        opened_ids = []
        if event_log is not None:
            for event in event_log.events:
                if event.alert_id and event.alert_id.startswith("alert-"):
                    try:
                        opened_ids.append(int(event.alert_id.removeprefix("alert-")))
                    except ValueError:
                        pass
        self._next_alert_id = max(opened_ids, default=0) + 1

    @property
    def events(self) -> tuple[AlertEvent, ...]:
        return tuple(self._events)

    def _new_track(self, decision: StateDecision, timestamp: float) -> _TrackAlert:
        if len(self._tracks) >= self.config.max_tracks:
            raise AlertError("max_tracks reached; discard idle tracks before adding another")
        return _TrackAlert(
            decision.track_id, AlertStatus.IDLE, None, decision.frame_index,
            timestamp, decision.state, decision.reason,
        )

    def _emit_transition(
        self,
        current: _TrackAlert,
        updated: _TrackAlert,
        transition: str,
    ) -> None:
        event = AlertEvent(
            event_id=self._next_event_id,
            timestamp_seconds=updated.last_timestamp_seconds,
            track_id=updated.track_id,
            frame_index=updated.last_frame_index,
            alert_id=updated.alert_id or current.alert_id,
            transition=transition,
            from_status=current.status,
            to_status=updated.status,
            observed_state=updated.last_state,
            reason=updated.reason,
        )
        if self.event_log is not None:
            self.event_log.append(event)
        self._events.append(event)
        self._tracks[updated.track_id] = updated
        self._next_event_id += 1

    def _set_without_event(self, updated: _TrackAlert) -> None:
        self._tracks[updated.track_id] = updated

    def update(self, decision: StateDecision, timestamp_seconds: float) -> AlertSnapshot:
        _validate_decision(decision)
        timestamp = _finite_timestamp(timestamp_seconds)
        current = self._tracks.get(decision.track_id)
        if current is None:
            current = self._new_track(decision, timestamp)
        elif (decision.frame_index <= current.last_frame_index
              or timestamp <= current.last_timestamp_seconds):
            raise AlertError("alert frame indices and timestamps must increase strictly per track")
        observed = replace(
            current,
            last_frame_index=decision.frame_index,
            last_timestamp_seconds=timestamp,
            last_state=decision.state,
            reason=decision.reason,
        )

        if current.status is AlertStatus.IDLE:
            if decision.state is FallState.FALLEN:
                pending = replace(observed, status=AlertStatus.PENDING,
                                  pending_since_seconds=timestamp)
                self._emit_transition(current, pending, "pending_started")
                return pending.snapshot()
            self._set_without_event(observed)
            return observed.snapshot()

        if current.status is AlertStatus.PENDING:
            if decision.state is not FallState.FALLEN:
                idle = replace(
                    observed, status=AlertStatus.IDLE, pending_since_seconds=None,
                )
                self._emit_transition(current, idle, "pending_cancelled")
                return idle.snapshot()
            assert current.pending_since_seconds is not None
            if (timestamp - current.pending_since_seconds + 1e-9
                    >= self.config.fallen_persistence_seconds):
                alert_id = f"alert-{self._next_alert_id:06d}"
                active = replace(
                    observed, status=AlertStatus.ACTIVE, alert_id=alert_id,
                    activated_at_seconds=timestamp,
                )
                self._emit_transition(current, active, "alert_opened")
                self._next_alert_id += 1
                return active.snapshot()
            self._set_without_event(observed)
            return observed.snapshot()

        if current.status in (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED):
            # Alerts are latched: normal or unknown inference never clears them.
            self._set_without_event(observed)
            return observed.snapshot()

        assert current.status is AlertStatus.COOLDOWN
        assert current.cooldown_until_seconds is not None
        if (timestamp >= current.cooldown_until_seconds
                and decision.state is FallState.NORMAL):
            idle = replace(
                observed, status=AlertStatus.IDLE, alert_id=None,
                pending_since_seconds=None, activated_at_seconds=None,
                acknowledged_at_seconds=None, cooldown_until_seconds=None,
            )
            self._emit_transition(current, idle, "alert_rearmed")
            return idle.snapshot()
        self._set_without_event(observed)
        return observed.snapshot()

    def acknowledge(self, track_id: int, timestamp_seconds: float) -> AlertSnapshot:
        track_id = _positive_track_id(track_id)
        timestamp = _finite_timestamp(timestamp_seconds)
        current = self._tracks.get(track_id)
        if current is None:
            raise AlertError(f"no alert state for track {track_id}")
        if current.status is not AlertStatus.ACTIVE:
            raise AlertError("only an active alert can be acknowledged")
        if timestamp < current.last_timestamp_seconds:
            raise AlertError("acknowledgment timestamp cannot move backward")
        acknowledged = replace(
            current, status=AlertStatus.ACKNOWLEDGED,
            acknowledged_at_seconds=timestamp, last_timestamp_seconds=timestamp,
            reason="alert_acknowledged_locally",
        )
        self._emit_transition(current, acknowledged, "alert_acknowledged")
        return acknowledged.snapshot()

    def reset(self, track_id: int, timestamp_seconds: float) -> AlertSnapshot:
        track_id = _positive_track_id(track_id)
        timestamp = _finite_timestamp(timestamp_seconds)
        current = self._tracks.get(track_id)
        if current is None:
            raise AlertError(f"no alert state for track {track_id}")
        if current.status is not AlertStatus.ACKNOWLEDGED:
            raise AlertError("only an acknowledged alert can be reset")
        if current.last_state is not FallState.NORMAL:
            raise AlertError("reset requires a current normal observation")
        if timestamp < current.last_timestamp_seconds:
            raise AlertError("reset timestamp cannot move backward")
        cooldown = replace(
            current, status=AlertStatus.COOLDOWN,
            last_timestamp_seconds=timestamp,
            cooldown_until_seconds=timestamp + self.config.cooldown_seconds,
            reason="alert_reset_locally",
        )
        self._emit_transition(current, cooldown, "alert_reset")
        return cooldown.snapshot()

    def snapshot(self, track_id: int) -> AlertSnapshot:
        track_id = _positive_track_id(track_id)
        try:
            return self._tracks[track_id].snapshot()
        except KeyError as error:
            raise AlertError(f"no alert state for track {track_id}") from error

    def snapshots(self) -> tuple[AlertSnapshot, ...]:
        return tuple(self._tracks[track_id].snapshot() for track_id in sorted(self._tracks))

    def discard_idle(self, track_id: int) -> None:
        track_id = _positive_track_id(track_id)
        current = self._tracks.get(track_id)
        if current is None:
            return
        if current.status is not AlertStatus.IDLE:
            raise AlertError("only idle track state can be discarded")
        del self._tracks[track_id]


class StateStage(Protocol):
    def process(self, frame: FramePacket) -> tuple[StateDecision, ...]: ...


@dataclass(frozen=True)
class AlertFrameResult:
    decisions: tuple[StateDecision, ...]
    alerts: tuple[AlertSnapshot, ...]


class PersistentAlertStage:
    """Compose state decisions with the local alert lifecycle for one frame."""

    def __init__(self, state_stage: StateStage, manager: AlertManager | None = None) -> None:
        if manager is not None and not isinstance(manager, AlertManager):
            raise AlertError("manager must be an AlertManager")
        self.state_stage = state_stage
        self.manager = manager or AlertManager()

    def process(self, frame: FramePacket) -> AlertFrameResult:
        if not isinstance(frame, FramePacket):
            raise AlertError("alert stage input must be a FramePacket")
        decisions = self.state_stage.process(frame)
        if not isinstance(decisions, tuple):
            raise AlertError("state stage must return a tuple of StateDecision values")
        seen: set[int] = set()
        for decision in decisions:
            _validate_decision(decision)
            if decision.frame_index != frame.index:
                raise AlertError("state decision and frame indices must match")
            if decision.track_id in seen:
                raise AlertError("state stage must return unique track IDs")
            seen.add(decision.track_id)
        for decision in sorted(decisions, key=lambda value: value.track_id):
            self.manager.update(decision, frame.timestamp_seconds)
        return AlertFrameResult(decisions, self.manager.snapshots())


def visible_alert_overlays(
    snapshots: Iterable[AlertSnapshot],
) -> tuple[AlertOverlay, ...]:
    values = tuple(snapshots)
    if any(not isinstance(snapshot, AlertSnapshot) for snapshot in values):
        raise AlertError("overlay input must contain AlertSnapshot values")
    return tuple(snapshot.overlay for snapshot in values if snapshot.overlay.visible)


def _smoke_decision(track_id: int, frame_index: int, state: FallState,
                    reason: str) -> StateDecision:
    return StateDecision(
        track_id, frame_index, state, reason,
        StateEvidence(3, 1.0, 1.5 if state is FallState.FALLEN else 0.5, 0.0),
    )


def run_alert_smoke(
    config: AlertConfig | None = None,
    event_log: JsonlAlertLog | None = None,
) -> dict[str, object]:
    """Exercise latching, acknowledgment, reset, cooldown, and rearming."""
    actual = config or AlertConfig(fallen_persistence_seconds=1.0, cooldown_seconds=2.0)
    manager = AlertManager(actual, event_log)
    manager.update(
        _smoke_decision(1, 0, FallState.FALLEN, "synthetic_fallen"), 0.0
    )
    manager.update(
        _smoke_decision(1, 1, FallState.FALLEN, "synthetic_fallen"),
        actual.fallen_persistence_seconds,
    )
    open_event_count = sum(event.transition == "alert_opened" for event in manager.events)
    manager.update(
        _smoke_decision(1, 2, FallState.FALLEN, "synthetic_fallen"),
        actual.fallen_persistence_seconds + 0.1,
    )
    duplicate_open_suppressed = (
        sum(event.transition == "alert_opened" for event in manager.events) == open_event_count
    )
    manager.update(
        _smoke_decision(1, 3, FallState.NORMAL, "synthetic_recovery"),
        actual.fallen_persistence_seconds + 0.2,
    )
    persisted_after_normal = manager.snapshot(1).status is AlertStatus.ACTIVE
    manager.acknowledge(1, actual.fallen_persistence_seconds + 0.3)
    manager.reset(1, actual.fallen_persistence_seconds + 0.4)
    manager.update(
        _smoke_decision(1, 4, FallState.FALLEN, "synthetic_fallen_during_cooldown"),
        actual.fallen_persistence_seconds + 0.5,
    )
    manager.update(
        _smoke_decision(1, 5, FallState.NORMAL, "synthetic_rearmed"),
        actual.fallen_persistence_seconds + 0.4 + actual.cooldown_seconds,
    )
    return {
        "event": "alert_lifecycle_smoke",
        "synthetic": True,
        "transitions": [event.transition for event in manager.events],
        "final_status": manager.snapshot(1).status.value,
        "persisted_after_normal": persisted_after_normal,
        "duplicate_open_suppressed": duplicate_open_suppressed,
        "event_count": len(manager.events),
        "local_event_log": str(event_log.path) if event_log is not None else None,
        "external_actions_performed": False,
        "fall_detection_available": False,
    }
