"""
src/models/inference.py
────────────────────────
Frame-level pose inference pipeline:
  raw features  →  impute  →  scale (RF only needs impute)  →  predict  →  decode

Also implements a lightweight rolling majority-vote smoothing buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.models.model_loader import YogaArtifacts


@dataclass
class PosePrediction:
    pose_name: str               # decoded class label
    confidence: float            # probability of the top class  [0, 1]
    probabilities: np.ndarray    # full class probability vector
    raw_prediction: str          # top class before smoothing
    smoothed: bool               # whether smoothing changed the result
    status: str                  # "detected" | "low_confidence" | "no_person"
    top3: list[tuple[str, float]] = field(default_factory=list)


class PoseInference:
    """
    Wraps preprocessing + classifier into a single predict() call.
    Maintains a rolling prediction history for majority-vote smoothing.
    """

    def __init__(
        self,
        artifacts: YogaArtifacts,
        active_classifier: str = "rf",
        confidence_threshold: float = 0.45,
        smoothing_window: int = 5,
    ):
        self._art = artifacts
        self._clf = active_classifier.lower()
        self._threshold = confidence_threshold
        self._history: deque[str] = deque(maxlen=smoothing_window)

        if self._clf not in ("rf", "mlp"):
            raise ValueError(f"active_classifier must be 'rf' or 'mlp', got '{self._clf}'")
        if self._clf == "mlp" and artifacts.mlp_model is None:
            print("[Inference] MLP not available, falling back to RF.")
            self._clf = "rf"

    def predict(self, feature_series: pd.Series) -> PosePrediction:
        """
        Run the full inference pipeline on a single feature Series.

        feature_series must have the same index/column names as feature_cols.
        Returns a PosePrediction even when confidence is low.
        """
        art = self._art

        # ── Align to training column order, fill missing with NaN ─────────────
        series_aligned = feature_series.reindex(art.feature_cols)
        # Replace ±inf with NaN before passing to imputer
        values = series_aligned.values.astype(float)
        values[~np.isfinite(values)] = np.nan
        X_df = pd.DataFrame([values], columns=art.feature_cols)

        # ── Impute (always required) ──────────────────────────────────────────
        X_imp = art.imputer.transform(X_df)

        # ── Predict ───────────────────────────────────────────────────────────
        if self._clf == "rf":
            proba = art.rf_model.predict_proba(X_imp)[0]
        else:
            X_scaled = art.scaler.transform(X_imp)
            proba = art.mlp_model.predict(X_scaled, verbose=0)[0]

        raw_idx = int(np.argmax(proba))
        confidence = float(proba[raw_idx])
        raw_name = art.class_names[raw_idx]

        # ── Rolling majority-vote smoothing ───────────────────────────────────
        self._history.append(raw_name)
        counts: dict[str, int] = {}
        for name in self._history:
            counts[name] = counts.get(name, 0) + 1
        smoothed_name = max(counts, key=lambda k: counts[k])
        was_smoothed = smoothed_name != raw_name

        # ── Status ────────────────────────────────────────────────────────────
        if confidence < self._threshold:
            status = "low_confidence"
        else:
            status = "detected"

        top3_indices = np.argsort(proba)[::-1][:3]
        top3 = [(art.class_names[i], float(proba[i])) for i in top3_indices]

        return PosePrediction(
            pose_name=smoothed_name,
            confidence=confidence,
            probabilities=proba,
            raw_prediction=raw_name,
            smoothed=was_smoothed,
            status=status,
            top3=top3,
        )

    def reset_history(self) -> None:
        self._history.clear()
