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
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.model_loader import YogaArtifacts, YogaArtifactsZenodo
from src.models.sequence_tracker import YogaSequenceTracker, SequenceState


@dataclass
class PosePrediction:
    pose_name: str               # decoded class label
    confidence: float            # probability of the top class  [0, 1]
    probabilities: np.ndarray    # full class probability vector
    raw_prediction: str          # top class before smoothing
    smoothed: bool               # whether smoothing changed the result
    status: str                  # "detected" | "low_confidence" | "no_person"
    top3: list[tuple[str, float]] = field(default_factory=list)
    sequence_state: Optional[SequenceState] = None


@dataclass
class DualTaskPrediction(PosePrediction):
    """
    Extends PosePrediction with binary form correctness verification
    (Right vs Wrong form prediction from Model B).
    """
    is_form_correct: bool = True
    correctness_label: str = "Right"       # "Right" | "Wrong" | "Unknown"
    correctness_confidence: float = 1.0     # Probability of predicted correctness [0, 1]
    correctness_probabilities: Optional[np.ndarray] = None


def compute_gravitational_posture_prior(
    landmarks: Optional[list[dict]],
    class_names: list[str],
    user_profile: Optional[Any] = None,
) -> np.ndarray:
    """
    Computes an anatomical gravitational likelihood mask based on physical
    keypoint spatial hierarchy and calibrated user anthropometric baselines:
      • INVERTED/SUPINE: Feet/knees above hips or shoulders below hips
        -> Activates Viparita Karani, Sarvangasana, Anantasana
      • PRONE: Shoulders and hips horizontal on the floor
        -> Activates Bhujangasana
      • UPRIGHT SEATED/KNEELING: Vertical torso, but thighs/legs foreshortened/compressed
        -> Activates Vajrasana, Parvatasana (distinguishing arms overhead in Step 3/4)
        -> Strongly suppresses standing poses (Tadasana, Kati Chakrasana, Ardhakati Chakrasana)
      • UPRIGHT STANDING: Vertical torso with full leg extension matching calibrated baseline
        -> Activates Tadasana, Kati Chakrasana, Ardhakati Chakrasana
        -> Suppresses ground, prone, seated, and inverted poses
    """
    n_classes = len(class_names)
    if not landmarks or len(landmarks) < 33:
        w_vec = np.ones(n_classes, dtype=float)
        return w_vec / np.sum(w_vec)

    try:
        sy = (float(landmarks[11].get("y", 0.5)) + float(landmarks[12].get("y", 0.5))) / 2.0
        hy = (float(landmarks[23].get("y", 0.5)) + float(landmarks[24].get("y", 0.5))) / 2.0
        ky = (float(landmarks[25].get("y", 0.5)) + float(landmarks[26].get("y", 0.5))) / 2.0
        ay = (float(landmarks[27].get("y", 0.5)) + float(landmarks[28].get("y", 0.5))) / 2.0
        sx = (float(landmarks[11].get("x", 0.5)) + float(landmarks[12].get("x", 0.5))) / 2.0
        hx = (float(landmarks[23].get("x", 0.5)) + float(landmarks[24].get("x", 0.5))) / 2.0
        wy = (float(landmarks[15].get("y", 0.5)) + float(landmarks[16].get("y", 0.5))) / 2.0

        ankles_vis = max(float(landmarks[27].get("visibility", 0.0)), float(landmarks[28].get("visibility", 0.0)))

        dy_torso = hy - sy
        dx_torso = abs(hx - sx)
        leg_dy = ay - hy
        shin_dy = ay - ky
        ratio_leg = leg_dy / max(0.01, dy_torso)

        # Calibrated user baseline integration
        is_calibrated = False
        baseline_thigh_ratio = 0.85
        baseline_leg_ratio = 1.25

        if user_profile is not None:
            if hasattr(user_profile, "calibrated") and user_profile.calibrated:
                is_calibrated = True
                baseline_thigh_ratio = float(getattr(user_profile, "thigh_to_torso_ratio", 0.85))
                baseline_leg_ratio = float(getattr(user_profile, "leg_to_torso_ratio", 1.25))
            elif isinstance(user_profile, dict) and user_profile.get("calibrated", False):
                is_calibrated = True
                baseline_thigh_ratio = float(user_profile.get("thigh_to_torso_ratio", 0.85))
                baseline_leg_ratio = float(user_profile.get("leg_to_torso_ratio", 1.25))

        weights = {c: 1.0 for c in class_names}

        # 1. INVERTED / SUPINE (Feet above hips or shoulders below hips)
        is_inverted = (ay < hy - 0.08 and ky < hy - 0.04) or (sy > hy + 0.06)
        if is_inverted:
            for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana", "Bhujangasana", "Marjariasana", "Vajrasana", "Parvatasana"):
                if pose in weights:
                    weights[pose] = 0.005
            for pose in ("Viparita Karani", "Sarvangasana", "Anantasana"):
                if pose in weights:
                    weights[pose] = 1.0
            w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
            return w_vec / np.sum(w_vec)

        # 2. PRONE (Horizontal on stomach, e.g. Cobra Pose Bhujangasana)
        is_horizontal_torso = abs(dy_torso) < 0.10 and dx_torso > 0.08
        if is_horizontal_torso and (ay > sy - 0.15) and (ky > sy - 0.15):
            for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana", "Vajrasana", "Parvatasana", "Sarvangasana", "Viparita Karani"):
                if pose in weights:
                    weights[pose] = 0.005
            if "Bhujangasana" in weights:
                weights["Bhujangasana"] = 1.0
            if "Marjariasana" in weights:
                weights["Marjariasana"] = 0.50
            w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
            return w_vec / np.sum(w_vec)

        # 3. UPRIGHT TORSO (Head/shoulders at top, hips below shoulders, vertical spine)
        is_upright = (dy_torso > 0.07) and (dx_torso / max(0.01, dy_torso) < 0.95)
        if is_upright:
            # Live vertical body ratios relative to torso
            live_thigh_dy = max(0.0, ky - hy)
            live_thigh_ratio = live_thigh_dy / max(0.01, dy_torso)
            live_leg_ratio = ratio_leg

            if is_calibrated:
                # Calibrated Anthropometric Comparison:
                # When seated/kneeling, 2D vertical thigh drops to < 55% of standing baseline
                is_seated_or_kneeling = (
                    (live_thigh_ratio < 0.55 * baseline_thigh_ratio)
                    or (live_leg_ratio < 0.55 * baseline_leg_ratio)
                    or (shin_dy < 0.045)
                    or (ay <= ky + 0.04)
                    or (abs(ky - hy) < 0.08 and abs(ay - hy) < 0.12)
                    or (ankles_vis < 0.30 and ky >= 0.60 and hy >= 0.48)
                )
            else:
                # Fallback geometric check when uncalibrated
                is_seated_or_kneeling = (
                    (ratio_leg < 0.60)
                    or (shin_dy < 0.045)
                    or (ay <= ky + 0.04)
                    or (abs(ky - hy) < 0.08 and abs(ay - hy) < 0.12)
                    or (ankles_vis < 0.30 and ky >= 0.60 and hy >= 0.48)
                )

            # Suppress prone and inverted poses whenever torso is upright
            for pose in ("Bhujangasana", "Viparita Karani", "Sarvangasana", "Anantasana", "Marjariasana"):
                if pose in weights:
                    weights[pose] = 0.005

            if is_seated_or_kneeling:
                # SEATED / KNEELING: Suppress standing postures
                for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana"):
                    if pose in weights:
                        weights[pose] = 0.005

                # Distinguish Step 1 & 2 vs Step 3 & 4 (Parvatasana arms overhead)
                arms_overhead = (wy < sy - 0.02)
                if arms_overhead:
                    if "Parvatasana" in weights:
                        weights["Parvatasana"] = 1.0
                    if "Vajrasana" in weights:
                        weights["Vajrasana"] = 0.005
                else:
                    if "Vajrasana" in weights:
                        weights["Vajrasana"] = 1.0
                    if "Parvatasana" in weights:
                        weights["Parvatasana"] = 1.0
            else:
                # STANDING: Suppress seated postures
                for pose in ("Vajrasana", "Parvatasana"):
                    if pose in weights:
                        weights[pose] = 0.005
                for pose in ("Tadasana", "Kati Chakrasana", "Ardhakati Chakrasana"):
                    if pose in weights:
                        weights[pose] = 1.0

            w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
            return w_vec / np.sum(w_vec)

        w_vec = np.array([weights.get(c, 1.0) for c in class_names], dtype=float)
        return w_vec / np.sum(w_vec)
    except Exception:
        w_vec = np.ones(n_classes, dtype=float)
        return w_vec / np.sum(w_vec)


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
        self._ema_proba: Optional[np.ndarray] = None
        self._alpha = 0.55
        self._seq_tracker = YogaSequenceTracker(debounce_frames=2)

        if self._clf not in ("rf", "mlp"):
            raise ValueError(f"active_classifier must be 'rf' or 'mlp', got '{self._clf}'")
        if self._clf == "mlp" and artifacts.mlp_model is None:
            print("[Inference] MLP not available, falling back to RF.")
            self._clf = "rf"

    def predict(
        self,
        feature_series: pd.Series,
        landmarks: Optional[list[dict]] = None,
        user_profile: Optional[Any] = None,
    ) -> PosePrediction:
        """
        Run the full inference pipeline on a single feature Series.

        feature_series must have the same index/column names as feature_cols.
        Returns a PosePrediction even when confidence is low.
        """
        art = self._art

        if isinstance(feature_series, dict):
            feature_series = pd.Series(feature_series)
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

        # ── Gravitational Orientation Prior Filter & Sequence Pruning ─────────
        seq_state: Optional[SequenceState] = None
        if landmarks:
            feat_dict = feature_series.to_dict() if hasattr(feature_series, "to_dict") else None
            seq_state = self._seq_tracker.update(
                landmarks,
                feature_dict=feat_dict,
                user_profile=user_profile,
                available_classes=art.class_names,
            )
            proba = self._seq_tracker.filter_probabilities(proba, art.class_names, seq_state)

            prior = compute_gravitational_posture_prior(landmarks, art.class_names, user_profile=user_profile)
            proba = proba * prior
            p_sum = float(np.sum(proba))
            if p_sum > 1e-6:
                proba = proba / p_sum

        raw_idx = int(np.argmax(proba))
        raw_name = art.class_names[raw_idx]

        # ── Responsive Probability Smoothing (EMA) ───────────────────────────
        if self._ema_proba is None:
            self._ema_proba = proba.copy()
        else:
            self._ema_proba = self._alpha * proba + (1.0 - self._alpha) * self._ema_proba

        smoothed_idx = int(np.argmax(self._ema_proba))
        smoothed_conf = float(self._ema_proba[smoothed_idx])
        smoothed_name = art.class_names[smoothed_idx]
        was_smoothed = smoothed_name != raw_name

        self._history.append(smoothed_name)

        # ── Status ────────────────────────────────────────────────────────────
        if smoothed_conf < self._threshold:
            status = "low_confidence"
        else:
            status = "detected"

        top3_indices = np.argsort(self._ema_proba)[::-1][:3]
        top3 = [(art.class_names[i], float(self._ema_proba[i])) for i in top3_indices]

        return PosePrediction(
            pose_name=smoothed_name,
            confidence=smoothed_conf,
            probabilities=self._ema_proba,
            raw_prediction=raw_name,
            smoothed=was_smoothed,
            status=status,
            top3=top3,
            sequence_state=seq_state,
        )

    def reset_history(self) -> None:
        self._history.clear()
        self._ema_proba = None
        self._seq_tracker.reset()


class DualTaskInference:
    """
    Inference engine for Zenodo dual-task models:
      • Model A: 10-Class Asana Classifier
      • Model B: Binary Form Verifier (Right vs Wrong)

    Pipeline per feature vector:
      features → align columns → impute → [scale if MLP] → predict asana & correctness
    """

    def __init__(
        self,
        artifacts: YogaArtifactsZenodo,
        confidence_threshold: float = 0.30,
        verifier_threshold: float = 0.50,
        smoothing_window: int = 3,
        alpha: float = 0.55,
    ):
        self._art = artifacts
        self._threshold = confidence_threshold
        self._verifier_threshold = verifier_threshold
        self._alpha = alpha
        self._ema_proba: Optional[np.ndarray] = None
        self._history: deque[str] = deque(maxlen=smoothing_window)
        self._seq_tracker = YogaSequenceTracker(debounce_frames=2)

    def predict(
        self,
        feature_series: pd.Series,
        landmarks: Optional[list[dict]] = None,
        user_profile: Optional[Any] = None,
    ) -> DualTaskPrediction:
        """
        Run inference on feature vector, producing both Asana classification
        and Form verification predictions with responsive probability smoothing.
        """
        art = self._art

        if isinstance(feature_series, dict):
            feature_series = pd.Series(feature_series)
        series_aligned = feature_series.reindex(art.feature_cols)
        values = series_aligned.values.astype(float)
        values[~np.isfinite(values)] = np.nan
        X_df = pd.DataFrame([values], columns=art.feature_cols)

        # ── Impute ────────────────────────────────────────────────────────────
        X_imp = art.imputer.transform(X_df)
        X_eval = art.scaler.transform(X_imp) if getattr(art, "model_type", "rf") == "mlp" else X_imp

        # ── Model A: Asana Prediction ─────────────────────────────────────────
        proba_asana = art.asana_model.predict_proba(X_eval)[0]

        # ── Gravitational Orientation Prior Filter & Sequence Pruning ─────────
        seq_state: Optional[SequenceState] = None
        if landmarks:
            feat_dict = feature_series.to_dict() if hasattr(feature_series, "to_dict") else None
            seq_state = self._seq_tracker.update(
                landmarks,
                feature_dict=feat_dict,
                user_profile=user_profile,
                available_classes=art.class_names,
            )
            proba_asana = self._seq_tracker.filter_probabilities(proba_asana, art.class_names, seq_state)

            prior = compute_gravitational_posture_prior(landmarks, art.class_names, user_profile=user_profile)
            proba_asana = proba_asana * prior
            p_sum = float(np.sum(proba_asana))
            if p_sum > 1e-6:
                proba_asana = proba_asana / p_sum

        raw_idx = int(np.argmax(proba_asana))
        raw_name = art.class_names[raw_idx]

        # ── Responsive Probability Smoothing (EMA) ───────────────────────────
        if self._ema_proba is None:
            self._ema_proba = proba_asana.copy()
        else:
            self._ema_proba = self._alpha * proba_asana + (1.0 - self._alpha) * self._ema_proba

        smoothed_idx = int(np.argmax(self._ema_proba))
        smoothed_conf = float(self._ema_proba[smoothed_idx])
        smoothed_name = art.class_names[smoothed_idx]
        was_smoothed = smoothed_name != raw_name

        self._history.append(smoothed_name)

        status = "detected" if smoothed_conf >= self._threshold else "low_confidence"
        top3_indices = np.argsort(self._ema_proba)[::-1][:3]
        top3 = [(art.class_names[i], float(self._ema_proba[i])) for i in top3_indices]

        # ── Model B: Form Verifier Prediction ─────────────────────────────────
        proba_corr = art.verifier_model.predict_proba(X_eval)[0]
        corr_classes = list(art.correctness_label_encoder.classes_)
        right_idx = corr_classes.index("Right") if "Right" in corr_classes else 1
        wrong_idx = corr_classes.index("Wrong") if "Wrong" in corr_classes else 0

        p_right = float(proba_corr[right_idx])
        p_wrong = float(proba_corr[wrong_idx])
        is_right = p_right >= self._verifier_threshold
        corr_label = "Right" if is_right else "Wrong"
        corr_conf = p_right if is_right else p_wrong

        return DualTaskPrediction(
            pose_name=smoothed_name,
            confidence=smoothed_conf,
            probabilities=self._ema_proba,
            raw_prediction=raw_name,
            smoothed=was_smoothed,
            status=status,
            top3=top3,
            is_form_correct=is_right,
            correctness_label=corr_label,
            correctness_confidence=corr_conf,
            correctness_probabilities=proba_corr,
            sequence_state=seq_state,
        )

    def reset_history(self) -> None:
        self._history.clear()
        self._ema_proba = None
        self._seq_tracker.reset()

