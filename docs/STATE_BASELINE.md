# Explainable temporal state baseline

Milestone 6 adds a deterministic, per-person state baseline over `TemporalFeatureWindow` values. It makes every decision from documented thresholds and returns the evidence used. It is not a trained model, a calibrated risk score, or a validated medical or emergency detector.

## States and decision order

`TemporalStateClassifier` evaluates one active track at a time in this order:

1. **Unknown input:** the current observation is missing, the feature window is not ready, pose confidence is insufficient, or too few usable observations are contiguous.
2. **Fallen:** the person has remained above the lying aspect-ratio threshold with low vertical motion for the configured settled duration.
3. **Falling:** a recent contiguous sequence begins upright and shows all three required signals within the transition window: downward center displacement, downward speed, and increasing box aspect ratio.
4. **Normal:** the current posture is below the upright aspect-ratio threshold and vertical motion is within the stable limit.
5. **Unknown ambiguity:** every other pose, including intermediate or unsettled posture, rapid vertical motion without shape change, and shape change without downward displacement.

Image coordinates increase downward, so a positive normalized vertical speed represents descent. Box centers and displacements are measured in frame-height units; speeds are measured in frame heights per source second. Aspect ratio is box width divided by box height.

The baseline deliberately does not infer across a missing or low-confidence observation. Those samples break the contiguous evidence sequence. A settled lying decision takes precedence after a completed descent so a track can move from `falling` to `fallen` once its motion remains low for long enough.

## Configuration

`configs/state.toml` defines:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `min_contiguous_observations` | 3 | Minimum uninterrupted usable samples |
| `min_keypoint_fraction` | 0.50 | Minimum fraction of confident joints per used sample |
| `upright_max_aspect_ratio` | 0.80 | Largest width/height ratio treated as upright |
| `lying_min_aspect_ratio` | 1.20 | Smallest width/height ratio treated as lying |
| `normal_max_vertical_speed` | 0.12 | Maximum absolute vertical speed for `normal` |
| `falling_min_vertical_speed` | 0.20 | Minimum average downward transition speed |
| `falling_min_center_drop` | 0.10 | Minimum downward center displacement |
| `falling_min_aspect_increase` | 0.25 | Minimum width/height increase during descent |
| `transition_window_seconds` | 1.50 | Longest upright-to-current transition considered |
| `fallen_min_duration_seconds` | 0.50 | Required settled lying duration |
| `settled_max_vertical_speed` | 0.08 | Maximum absolute speed in a settled lying suffix |

Configuration validation rejects non-finite values, incompatible posture thresholds, a falling-speed threshold that does not exceed the normal limit, and a settled-speed limit above the normal limit.

## Output contract

Every `StateDecision` contains the track ID, frame index, `FallState`, machine-readable reason, and `StateEvidence`. Evidence includes the number of contiguous observations, current keypoint fraction, aspect ratio, vertical speed, and applicable transition or settled-lying measurements.

`confidence` is intentionally `null`. Threshold margins are not probabilities, and assigning a numeric confidence would suggest calibration that has not been performed. Multi-person decisions are ordered by track ID. `TemporalStateStage` composes any feature stage with the classifier but does not connect the default capture CLI to a model.

## Synthetic smoke check

```bash
fall-detection state-smoke --config configs/state.toml
```

The command evaluates four handcrafted feature sequences and must finish with:

- standing: `normal`
- lying and settled: `fallen`
- rapid descent with posture change: `falling`
- missing current observation: `unknown`

This proves deterministic branch behavior only. The sequences are not video, model output, a benchmark, or evidence of real-world accuracy.

## Limitations

- Box aspect ratio is sensitive to camera angle, cropping, detector jitter, furniture, crouching, and unusual body positions.
- Apparent center motion can result from camera motion or tracking identity switches rather than a fall.
- Fixed thresholds have not been tuned or evaluated on authorized recordings. No precision, recall, accuracy, or latency claim is available.
- The default `demo` and `capture` commands still use the unavailable predictor. End-to-end fall detection remains false until model selection, full pipeline integration, and real-input evaluation are completed.
- A `normal` baseline decision is only a threshold result for one observed track. It is not proof that the scene is safe and must not be used as a medical or emergency-service determination.
