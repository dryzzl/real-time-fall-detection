# Persistent local alert lifecycle

Milestone 8 adds a local-only, per-track alert state machine over `StateDecision` values. It latches a confirmed alert until a local acknowledgment and guarded reset occur, writes metadata transitions to an optional JSONL journal, and exposes high-contrast overlay data. It does not send messages, call emergency services, upload footage, or make the default capture pipeline claim working detection.

## Lifecycle

| Status | Entry | Exit |
| --- | --- | --- |
| `idle` | New track or completed rearm | A `fallen` decision starts `pending` |
| `pending` | First uninterrupted `fallen` decision | Persistent `fallen` evidence opens `active`; any other state cancels to `idle` |
| `active` | `fallen` persisted for `fallen_persistence_seconds` | Explicit local acknowledgment only |
| `acknowledged` | Active alert acknowledged locally | Explicit reset, allowed only after the latest decision is `normal` |
| `cooldown` | Guarded reset | After `cooldown_seconds`, a new `normal` decision rearms `idle` |

An active or acknowledged alert is latched. Later `normal`, `falling`, or `unknown` decisions update the observation shown to the operator but never clear the alert. `unknown` is not treated as recovery. Repeated decisions cannot emit a second `alert_opened` event for the same incident.

Reset requires both acknowledgment and a current `normal` observation. Cooldown suppresses a new incident even if `fallen` reappears, and it does not expire into an armed state while the person still appears `fallen` or inference is `unknown`. This provides two distinct safeguards: recognition by an operator and a verified recovery state.

Configuration is stored in `configs/alerts.toml`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `fallen_persistence_seconds` | 1.0 | Continuous source-time `fallen` duration before opening |
| `cooldown_seconds` | 10.0 | Minimum delay after reset before rearming |
| `max_tracks` | 1000 | Bound on retained process-local track states |
| `fsync_events` | true | Flush each event to local durable storage |

All per-track frame indices and source timestamps must increase strictly. Different tracks remain independent and snapshots are sorted by track ID. Idle state can be discarded when the tracker retires an ID; pending, active, acknowledged, and cooldown state cannot be silently discarded.

## Local JSONL events

`JsonlAlertLog` records only lifecycle metadata:

- contiguous event ID and schema version;
- source timestamp, track ID, and frame index;
- process-local alert ID;
- transition and before/after status;
- observed state and machine-readable reason.

It never stores frames, keypoints, credentials, recipients, network destinations, or notification payloads. Existing logs are fully validated before append, event IDs must remain contiguous, and a malformed/truncated log is rejected without adding data. The smoke CLI uses exclusive creation so it never overwrites a prior log. Library callers can reopen a valid journal and continue event/alert numbering.

The journal is a persistent audit record, not a restart checkpoint. Runtime alert state is intentionally process-local in this milestone; reopening a log does not recreate an active alert. End-to-end recovery and supported runtime integration remain part of final hardening.

Example transition:

```json
{"schema_version":1,"event":"alert_transition","event_id":2,"timestamp_seconds":1.0,"track_id":1,"frame_index":1,"alert_id":"alert-000001","transition":"alert_opened","from_status":"pending","to_status":"active","observed_state":"fallen","reason":"lying_posture_settled"}
```

## Overlay

Every `AlertSnapshot` contains an `AlertOverlay` with visibility, headline, detail, and BGR color. Pending evidence renders amber, an active alert renders red, and an acknowledged alert renders orange. Idle and cooldown snapshots are hidden. `Preview.show` accepts an optional tuple of alert snapshots and draws a local banner; its existing two-argument pipeline callback remains compatible.

`PersistentAlertStage` composes any state-decision stage with `AlertManager`, validates all per-frame decisions before processing, and returns both the decisions and sorted alert snapshots. The default `demo` and `capture` commands are still connected to `UnavailablePredictor`, so they do not fabricate alerts or a safe state.

## Smoke check

```bash
fall-detection alert-smoke --config configs/alerts.toml

# Also create a new local metadata journal
fall-detection alert-smoke --config configs/alerts.toml \
  --event-log outputs/alert-smoke.jsonl
```

The synthetic sequence starts pending, opens one alert, repeats `fallen` without duplicating it, proves that `normal` does not clear the active alert, acknowledges and resets locally, suppresses a `fallen` decision during cooldown, and rearms on a later `normal` decision. It performs no external action and is not real-world validation.

## Limitations

- Alert behavior is only as reliable as upstream pose, tracking, and state decisions. None have been evaluated on authorized real recordings in this repository.
- Track IDs are process-local association handles. Identity switches or tracker expiry can separate alert state from a real person.
- Source timestamps are suitable within one ordered stream, not as wall-clock incident times across processes.
- A local acknowledgment means only that software recorded the action. It is not proof that a person was checked or that assistance was provided.
- This prototype is not a medical device or monitored emergency service. No external notifications or emergency calls exist, and users must not rely on it as their only safety mechanism.
