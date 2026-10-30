"""Fail-closed ONNX temporal classifier for per-person feature windows.

The model interface is verified on disk before inference. Predictions are
research outputs and do not imply clinical or safety validation.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path

from .contracts import FallState
from .features import TemporalFeatureWindow
from .state import StateDecision, StateEvidence
from .training import (
    CLASS_NAMES, TrainingConfig, TrainingError, encode_feature_window,
    verify_temporal_onnx_contract,
)


class OnnxTemporalClassifier:
    """Predict from exact-length windows; reject missing or malformed evidence."""

    def __init__(
        self,
        model_path: Path,
        config: TrainingConfig | None = None,
        *,
        session: object | None = None,
        min_observed_fraction: float = 0.8,
    ) -> None:
        if not isinstance(model_path, Path):
            raise ValueError("model_path must be a Path")
        if (type(min_observed_fraction) not in (int, float)
                or not isfinite(min_observed_fraction)
                or not 0 < min_observed_fraction <= 1):
            raise ValueError("min_observed_fraction must be in (0, 1]")
        self.config = config or TrainingConfig()
        self.min_observed_fraction = float(min_observed_fraction)
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise TrainingError("ONNX inference requires pip install -e '.[model]'") from error
        self._np = np
        if session is None:
            verify_temporal_onnx_contract(model_path, self.config)
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        else:
            # Dependency injection for deterministic tests only. The real path
            # always validates graph shape, metadata, and smoke inference.
            self.session = session

    def _unknown(self, window: TemporalFeatureWindow, reason: str) -> StateDecision:
        return StateDecision(
            window.track_id, window.latest.frame_index, FallState.UNKNOWN, reason,
            StateEvidence(0, None, None, None),
        )

    def classify(self, window: TemporalFeatureWindow) -> StateDecision:
        if not isinstance(window, TemporalFeatureWindow) or not window.samples:
            raise TrainingError("temporal inference requires a nonempty feature window")
        if not window.latest.observed:
            return self._unknown(window, "missing_current_observation")
        if (not window.ready or len(window.samples) != self.config.sequence_length
                or window.observed_count / len(window.samples) < self.min_observed_fraction):
            return self._unknown(window, "insufficient_temporal_evidence")
        matrix = encode_feature_window(window, self.config)
        tensor = self._np.asarray([matrix], dtype=self._np.float32)
        try:
            outputs = self.session.run(["logits"], {"features": tensor})
        except Exception as error:
            raise TrainingError(f"temporal ONNX inference failed: {error}") from error
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise TrainingError("temporal model must return one logits tensor")
        logits = self._np.asarray(outputs[0])
        if logits.shape != (1, len(CLASS_NAMES)) or not self._np.isfinite(logits).all():
            raise TrainingError("temporal logits must be finite with shape [1, 3]")
        # Stable softmax avoids overflow on large finite logits.
        values = logits[0].astype(self._np.float64)
        shifted = values - values.max()
        probabilities = self._np.exp(shifted)
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        return StateDecision(
            window.track_id, window.latest.frame_index,
            FallState(CLASS_NAMES[index]), "onnx_temporal_unvalidated",
            StateEvidence(0, None, None, None),
            float(probabilities[index]),
        )

    def classify_many(
        self, windows: tuple[TemporalFeatureWindow, ...]
    ) -> tuple[StateDecision, ...]:
        return tuple(self.classify(window) for window in windows)
