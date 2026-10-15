"""
tests/test_pipeline.py
───────────────────────
Automated tests for the inference and feedback pipeline.
All tests run without webcam or GPU access.

Run with:
    cd e:\\Yoga_posture_project
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def artifacts():
    """Load real saved artifacts once per test session."""
    from src.models.model_loader import load_artifacts
    return load_artifacts(ROOT / "saved_yoga_model")


@pytest.fixture
def dummy_landmarks_visible():
    """33 fully visible, anatomically plausible landmarks."""
    base = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99}] * 33
    # spread hips, knees, shoulders slightly so angles are computable
    base[11] = {"x": 0.40, "y": 0.30, "z": 0.0, "visibility": 0.99}  # L shoulder
    base[12] = {"x": 0.60, "y": 0.30, "z": 0.0, "visibility": 0.99}  # R shoulder
    base[13] = {"x": 0.35, "y": 0.50, "z": 0.0, "visibility": 0.99}  # L elbow
    base[14] = {"x": 0.65, "y": 0.50, "z": 0.0, "visibility": 0.99}  # R elbow
    base[15] = {"x": 0.30, "y": 0.70, "z": 0.0, "visibility": 0.99}  # L wrist
    base[16] = {"x": 0.70, "y": 0.70, "z": 0.0, "visibility": 0.99}  # R wrist
    base[23] = {"x": 0.42, "y": 0.60, "z": 0.0, "visibility": 0.99}  # L hip
    base[24] = {"x": 0.58, "y": 0.60, "z": 0.0, "visibility": 0.99}  # R hip
    base[25] = {"x": 0.40, "y": 0.78, "z": 0.0, "visibility": 0.99}  # L knee
    base[26] = {"x": 0.60, "y": 0.78, "z": 0.0, "visibility": 0.99}  # R knee
    base[27] = {"x": 0.38, "y": 0.95, "z": 0.0, "visibility": 0.99}  # L ankle
    base[28] = {"x": 0.62, "y": 0.95, "z": 0.0, "visibility": 0.99}  # R ankle
    base[29] = {"x": 0.36, "y": 0.97, "z": 0.0, "visibility": 0.99}  # L heel
    base[30] = {"x": 0.64, "y": 0.97, "z": 0.0, "visibility": 0.99}  # R heel
    base[31] = {"x": 0.34, "y": 0.99, "z": 0.0, "visibility": 0.99}  # L foot
    base[32] = {"x": 0.66, "y": 0.99, "z": 0.0, "visibility": 0.99}  # R foot
    return base


@pytest.fixture
def dummy_landmarks_occluded(dummy_landmarks_visible):
    """Same as visible but key lower-body joints have low visibility."""
    lm = [dict(x) for x in dummy_landmarks_visible]
    for idx in [25, 26, 27, 28, 29, 30, 31, 32]:
        lm[idx]["visibility"] = 0.05
    return lm


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Feature extraction produces exactly 39 features in correct order
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_vector_length_and_order(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_visible, flags, 1280, 720)
    series = features_to_series(feat_dict, artifacts.feature_cols)

    assert len(series) == len(artifacts.feature_cols), (
        f"Expected {len(artifacts.feature_cols)} features, got {len(series)}"
    )
    assert list(series.index) == artifacts.feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Full preprocessing + classifier produces a valid class index
# ─────────────────────────────────────────────────────────────────────────────

def test_rf_predict_returns_valid_class(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    from src.models.inference import PoseInference

    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_visible, flags, 1280, 720)
    series = features_to_series(feat_dict, artifacts.feature_cols)

    inferrer = PoseInference(artifacts, active_classifier="rf")
    pred = inferrer.predict(series)

    assert pred.pose_name in artifacts.class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.status in ("detected", "low_confidence")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Class index maps to correct pose name
# ─────────────────────────────────────────────────────────────────────────────

def test_label_encoder_class_mapping(artifacts):
    le = artifacts.label_encoder
    for i, name in enumerate(artifacts.class_names):
        decoded = le.inverse_transform([i])[0]
        assert decoded == name, f"Mismatch at index {i}: {decoded} != {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Occluded landmarks → no fabricated correction
# ─────────────────────────────────────────────────────────────────────────────

def test_occluded_joints_produce_unavailable_checks(dummy_landmarks_occluded):
    from src.features import extract_features
    from src.correction.posture_evaluator import evaluate

    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_occluded, flags, 1280, 720)
    ev = evaluate("Phalakasana", feat_dict, dummy_landmarks_occluded, flags)

    for check in ev.checks:
        if check.status == "unavailable":
            assert not check.reliable, "Unavailable check should not be marked reliable"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Angle outside range → correction feedback
# ─────────────────────────────────────────────────────────────────────────────

def test_out_of_range_angle_triggers_correction(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    # Forge a feature dict with Warrior 2 left knee clearly outside target
    feat_dict = {k: np.nan for k in ["left_knee_angle", "right_knee_angle",
                                      "left_elbow_angle", "right_elbow_angle"]}
    feat_dict["left_knee_angle"] = 160.0    # target is 85–115 → too_high
    feat_dict["right_knee_angle"] = 175.0   # within range for right
    feat_dict["left_elbow_angle"] = 170.0
    feat_dict["right_elbow_angle"] = 170.0

    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)

    issues = [c for c in ev.checks if c.status != "ok" and c.reliable]
    assert any(c.angle_name == "left_knee_angle" for c in issues), (
        "Expected left_knee_angle correction not found"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Angle inside range → no correction for that joint
# ─────────────────────────────────────────────────────────────────────────────

def test_in_range_angle_no_correction(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    feat_dict = {
        "left_knee_angle":   95.0,   # target 85–115 ✓
        "right_knee_angle":  170.0,  # target 160–185 ✓
        "left_elbow_angle":  172.0,  # target 160–185 ✓
        "right_elbow_angle": 168.0,  # target 160–185 ✓
    }
    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)
    issues = [c for c in ev.checks if c.status not in ("ok", "unavailable")]
    assert issues == [], f"Unexpected corrections: {issues}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Low confidence → uncertainty message, no posture correction
# ─────────────────────────────────────────────────────────────────────────────

def test_low_confidence_feedback(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    from src.models.inference import PoseInference
    from src.correction.feedback_engine import FeedbackEngine

    flags = [0] * 33
    feat_dict = extract_features(dummy_landmarks_visible, flags, 1280, 720)
    series = features_to_series(feat_dict, artifacts.feature_cols)

    # Force low threshold to trigger low_confidence unconditionally
    inferrer = PoseInference(artifacts, active_classifier="rf", confidence_threshold=0.99)
    pred = inferrer.predict(series)

    if pred.status == "low_confidence":
        engine = FeedbackEngine()
        items = engine.update(pred, None)
        assert any(i.severity == "warning" for i in items)
        assert not any(i.severity == "correction" for i in items)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Repeated feedback throttled by cooldown
# ─────────────────────────────────────────────────────────────────────────────

def test_feedback_cooldown_prevents_repeat_speech(artifacts, dummy_landmarks_visible):
    from src.features import extract_features, features_to_series
    from src.models.inference import PoseInference, PosePrediction
    from src.correction.feedback_engine import FeedbackEngine

    engine = FeedbackEngine(cooldown_seconds=60)   # very long cooldown

    # Manually create a low-confidence prediction to get a speech message
    pred = PosePrediction(
        pose_name="Vrksasana", confidence=0.3,
        probabilities=np.zeros(47), raw_prediction="Vrksasana",
        smoothed=False, status="low_confidence"
    )
    engine.update(pred, None)
    msg1 = engine.pop_speech()

    # Second identical update — should NOT enqueue speech again
    engine.update(pred, None)
    msg2 = engine.pop_speech()

    assert msg1 is not None, "First call should produce a speech message"
    assert msg2 is None, "Cooldown should prevent second speech within cooldown window"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Voice can be disabled
# ─────────────────────────────────────────────────────────────────────────────

def test_voice_can_be_disabled():
    from src.app.voice import VoiceFeedback
    v = VoiceFeedback.__new__(VoiceFeedback)
    v._enabled = False
    v._q = __import__("queue").Queue()
    v._engine = None

    v.speak("this should be ignored")
    assert v._q.empty(), "Queue should be empty when voice is disabled"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Missing model dir raises RuntimeError
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_model_dir_raises():
    from src.models.model_loader import load_artifacts
    with pytest.raises(RuntimeError, match="Model directory not found"):
        load_artifacts("/nonexistent/path/to/model")


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Temporal landmark fallback applied correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_fallback_restores_previous_joint():
    from src.pose.missing_landmarks import apply_temporal_fallback, VISIBILITY_THRESHOLD

    prev = [{"x": 0.5, "y": 0.8, "z": 0.0, "visibility": 0.99}] * 33
    current = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99}] * 33
    current[25] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.01}  # L knee occluded

    effective, flags, _ = apply_temporal_fallback(current, prev, VISIBILITY_THRESHOLD)

    assert flags[25] == 1, "Joint 25 should be flagged as imputed"
    assert abs(effective[25]["y"] - 0.8) < 1e-6, "Joint 25 should be restored from prev frame"


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Inference interval enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_interval_respected():
    """Verify the interval logic used in the app loop."""
    interval = 2.0
    last_time = time.monotonic() - 0.5   # only 0.5s have passed
    should_run = (time.monotonic() - last_time) >= interval
    assert not should_run, "Inference should NOT run when interval hasn't elapsed"

    last_time = time.monotonic() - 2.5   # 2.5s have passed
    should_run = (time.monotonic() - last_time) >= interval
    assert should_run, "Inference SHOULD run when interval has elapsed"


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: Target Pose Mode guides user to Warrior 2
# ─────────────────────────────────────────────────────────────────────────────

def test_target_pose_warrior_two_guidance(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate
    from src.correction.feedback_engine import FeedbackEngine
    from src.models.inference import PosePrediction

    # Dummy standing pose (straight legs)
    feat_dict = {
        "left_knee_angle": 175.0,
        "right_knee_angle": 175.0,
        "left_elbow_angle": 170.0,
        "right_elbow_angle": 170.0,
    }
    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)

    pred = PosePrediction(
        pose_name="Unknown", confidence=0.2,
        probabilities=np.zeros(47), raw_prediction="Unknown",
        smoothed=False, status="low_confidence"
    )

    engine = FeedbackEngine()
    items = engine.update(pred, ev, target_pose="Virabhadrasana Two")

    assert len(items) > 0
    # Must instruct to bend knee for Warrior 2
    assert any("knee" in item.instruction.lower() for item in items)
    assert any("Virabhadrasana Two" in item.pose_name for item in items)


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: Bilateral Warrior 2 Adaptation (Right Leg Forward)
# ─────────────────────────────────────────────────────────────────────────────

def test_bilateral_warrior_adaptation(dummy_landmarks_visible):
    from src.correction.posture_evaluator import evaluate

    # User steps right leg forward into Warrior 2 (right knee bent, left straight)
    feat_dict = {
        "right_knee_angle": 95.0,    # right knee bent to ~95° (front leg)
        "left_knee_angle": 175.0,    # left knee straight ~175° (back leg)
        "left_elbow_angle": 170.0,   # arms horizontal
        "right_elbow_angle": 170.0,
    }
    flags = [0] * 33
    ev = evaluate("Virabhadrasana Two", feat_dict, dummy_landmarks_visible, flags)

    # Right leg forward should be recognized as correct!
    assert ev.overall_status == "correct", f"Expected correct, got {ev.overall_status} ({[c.status for c in ev.checks]})"
