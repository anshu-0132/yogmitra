"""
src/app/webcam_app.py
──────────────────────
Main application entry-point.

Pipeline per frame:
  camera frame
    → MediaPipe detection
    → temporal landmark fallback
    → feature extraction  (geometry.py — exact training parity)
    → impute + [scale] + RF/MLP predict   (inference.py)
    → posture evaluate   (posture_evaluator.py)
    → feedback build     (feedback_engine.py)
    → OpenCV overlay     (overlay.py)
    → async TTS          (voice.py)

Inference only runs at the configured interval; the camera feed renders
at full frame-rate in between.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ── Locate project root so relative imports work when launched from any cwd ───
_ROOT = Path(__file__).resolve().parent.parent.parent   # …/Yoga_posture_project
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pose import PoseLandmarkerWrapper, SKELETON_CONNECTIONS
from src.pose.missing_landmarks import apply_temporal_fallback
from src.features import extract_features, features_to_series, ANGLE_TRIPLETS
from src.models.model_loader import load_artifacts
from src.models.inference import PoseInference, PosePrediction
from src.correction.posture_evaluator import evaluate as evaluate_posture
from src.correction.feedback_engine import FeedbackEngine, FeedbackItem
from src.app.voice import VoiceFeedback
from src.app.overlay import (
    draw_skeleton, draw_hud, draw_key_hints, draw_angle_values
)


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
def run(config_input: str | Path | dict = "config.yaml") -> None:
    if isinstance(config_input, dict):
        cfg = config_input
    else:
        cfg = load_config(config_input)

    model_cfg     = cfg["model"]
    infer_cfg     = cfg["inference"]
    camera_cfg    = cfg["camera"]
    display_cfg   = cfg["display"]
    voice_cfg     = cfg["voice"]

    # ── Load artifacts ────────────────────────────────────────────────────────
    model_dir = Path(model_cfg["dir"])
    task_path = str(model_cfg["mediapipe_task"])
    print(f"[App] Loading model artifacts from: {model_dir.resolve()}")
    try:
        artifacts = load_artifacts(model_dir)
    except RuntimeError as exc:
        sys.exit(f"[App] FATAL: {exc}")

    print(f"[App] Classifier: {model_cfg['active_classifier'].upper()}, "
          f"classes: {artifacts.n_classes}, features: {artifacts.n_features}")

    # ── Inference engine ──────────────────────────────────────────────────────
    inferrer = PoseInference(
        artifacts=artifacts,
        active_classifier=model_cfg["active_classifier"],
        confidence_threshold=infer_cfg["confidence_threshold"],
        smoothing_window=infer_cfg["smoothing_window"],
    )

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    if not Path(task_path).exists():
        sys.exit(f"[App] MediaPipe task file not found: {task_path}")

    landmarker = PoseLandmarkerWrapper(
        model_path=task_path,
        visibility_threshold=infer_cfg["visibility_threshold"],
    )

    # ── Voice ─────────────────────────────────────────────────────────────────
    voice = VoiceFeedback(
        rate=voice_cfg["rate"],
        volume=voice_cfg["volume"],
    )
    if not voice_cfg["enabled"]:
        voice._enabled = False

    # ── Feedback engine ───────────────────────────────────────────────────────
    feedback_engine = FeedbackEngine(
        cooldown_seconds=voice_cfg["cooldown_seconds"],
        max_display=display_cfg["max_feedback_lines"],
    )

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_cfg["device_index"])
    if not cap.isOpened():
        sys.exit("[App] Could not open webcam. Check device_index in config.yaml.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])

    # ── Practice / Target Mode ────────────────────────────────────────────────
    practice_cfg = cfg.get("practice", {})
    target_mode: bool = practice_cfg.get("enabled", True)
    initial_pose: str = practice_cfg.get("target_pose", "Virabhadrasana Two")

    practice_poses = [
        "Virabhadrasana Two",     # Warrior 2 (easy standing test pose, arms wide)
        "Virabhadrasana One",     # Warrior 1 (lunge, arms overhead)
        "Utkatasana",             # Chair Pose (knees bent, arms up)
        "Trikonasana",            # Triangle Pose (straight legs, side bend)
        "Phalakasana",            # High Plank
        "Adho Mukha Svanasana",   # Downward-Facing Dog
        "Vrksasana",              # Tree Pose
        "Uttanasana",             # Standing Forward Bend
        "Anjaneyasana",           # Low Lunge
    ]
    if initial_pose not in practice_poses:
        practice_poses.insert(0, initial_pose)
    target_idx: int = practice_poses.index(initial_pose)

    # ── State ─────────────────────────────────────────────────────────────────
    interval: float = infer_cfg["interval_seconds"]
    last_inference_time: float = 0.0
    last_landmarks: Optional[list[dict]] = None
    prev_raw_landmarks: Optional[list[dict]] = None

    # Persistent results (shown between inference runs)
    last_prediction: Optional[PosePrediction] = None
    last_features: dict = {}
    last_feedback: list[FeedbackItem] = []
    last_lm_effective: Optional[list[dict]] = None
    last_imputed: list[int] = []

    show_angles: bool = display_cfg["show_angles"]
    screenshot_dir = _ROOT / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    mode_label = f"Guided Practice [{practice_poses[target_idx]}]" if target_mode else "Auto-Detect"
    print(f"[App] Starting in {mode_label}.")
    print("[App] Press Q to quit. Press T/N to change pose, M to toggle Free/Guided mode.")
    print("[App] Starting main loop…")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("[App] Camera read failed. Retrying…")
            time.sleep(0.05)
            continue

        now = time.monotonic()
        run_inference = (now - last_inference_time) >= interval
        active_target = practice_poses[target_idx] if target_mode else None

        # ── Landmark detection (runs on unmirrored frame for anatomical accuracy) ──
        raw_lm = landmarker.detect(frame_raw)

        if raw_lm is None:
            last_lm_effective = None
            prev_raw_landmarks = None
            if run_inference:
                last_inference_time = now
                dummy_pred = PosePrediction(
                    pose_name="—",
                    confidence=0.0,
                    probabilities=np.zeros(artifacts.n_classes),
                    raw_prediction="—",
                    smoothed=False,
                    status="no_person",
                    top3=[],
                )
                last_prediction = dummy_pred
                last_feedback = feedback_engine.update(dummy_pred, None, target_pose=active_target)
        else:
            # Temporal fallback for missing joints
            effective_lm, flags, raw_for_history = apply_temporal_fallback(
                raw_lm, prev_raw_landmarks,
                threshold=infer_cfg["visibility_threshold"],
            )
            prev_raw_landmarks = raw_for_history
            last_lm_effective  = effective_lm
            last_imputed       = flags

            h, w = frame_raw.shape[:2]
            feat_dict = extract_features(effective_lm, flags, w, h)
            last_features = feat_dict

            # ── Periodic classifier & posture evaluation ───────────────────────
            if run_inference:
                last_inference_time = now
                feat_series = features_to_series(feat_dict, artifacts.feature_cols)

                # Predict
                prediction = inferrer.predict(feat_series)
                last_prediction = prediction

                # Evaluate posture: in target mode, evaluate against active target pose
                eval_pose = active_target if target_mode else prediction.pose_name
                eval_result = evaluate_posture(
                    pose_name=eval_pose,
                    feature_dict=feat_dict,
                    landmarks=effective_lm,
                    imputed_flags=flags,
                )

                # Generate feedback
                last_feedback = feedback_engine.update(
                    prediction,
                    eval_result,
                    target_pose=active_target,
                )

                # Dispatch voice
                msg = feedback_engine.pop_speech()
                if msg:
                    voice.speak(msg)

        # ── Prepare display frame (mirrored horizontally for natural view) ──────
        if camera_cfg["mirror"]:
            frame_display = cv2.flip(frame_raw, 1)
            display_landmarks = [
                {
                    "x": 1.0 - lm["x"],
                    "y": lm["y"],
                    "z": lm["z"],
                    "visibility": lm.get("visibility", 0.0),
                }
                for lm in last_lm_effective
            ] if last_lm_effective else None
        else:
            frame_display = frame_raw.copy()
            display_landmarks = last_lm_effective

        # ── Draw skeleton ─────────────────────────────────────────────────────
        if display_landmarks and display_cfg["show_landmarks"]:
            draw_skeleton(
                frame_display,
                display_landmarks,
                show_landmarks=display_cfg["show_landmarks"],
                show_skeleton=display_cfg["show_skeleton"],
            )

        # ── Draw angle values ─────────────────────────────────────────────────
        if show_angles and display_landmarks and last_features:
            draw_angle_values(
                frame_display, display_landmarks, last_features,
                list(ANGLE_TRIPLETS.keys())
            )

        # ── HUD ───────────────────────────────────────────────────────────────
        if last_prediction:
            draw_hud(
                frame_display,
                prediction=last_prediction,
                feedback_items=last_feedback,
                voice_enabled=voice.enabled,
                inference_interval=interval,
                last_inference_age=now - last_inference_time,
                show_angles=show_angles,
                font_scale=display_cfg["font_scale"],
                max_lines=display_cfg["max_feedback_lines"],
                target_pose=active_target,
                landmarks=last_lm_effective,
            )

        draw_key_hints(frame_display)

        cv2.imshow("Yoga Pose Correction — Press Q to quit", frame_display)

        # ── Key controls ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:                # Q / Esc → quit
            break
        elif key == ord("t") or key == ord("n"):         # T / N → cycle target pose
            target_idx = (target_idx + 1) % len(practice_poses)
            target_mode = True
            feedback_engine.clear()
            print(f"[App] Target pose changed to: {practice_poses[target_idx]}")
        elif key == ord("m"):                            # M → toggle Guided vs Free mode
            target_mode = not target_mode
            feedback_engine.clear()
            curr_mode = f"Guided [{practice_poses[target_idx]}]" if target_mode else "Free Auto-Detect"
            print(f"[App] Mode switched to: {curr_mode}")
        elif key == ord("v"):                            # V → toggle voice
            state = voice.toggle()
            print(f"[App] Voice {'ON' if state else 'OFF'}")
        elif key == ord("a"):                            # A → toggle angles
            show_angles = not show_angles
            print(f"[App] Angle overlay {'ON' if show_angles else 'OFF'}")
        elif key == ord("r"):                            # R → reset history
            inferrer.reset_history()
            feedback_engine.clear()
            print("[App] Prediction history reset.")
        elif key == ord("s"):                            # S → screenshot
            fname = screenshot_dir / f"yoga_{int(time.time())}.jpg"
            cv2.imwrite(str(fname), frame_display)
            print(f"[App] Screenshot saved: {fname}")
        elif key == ord("+") or key == ord("="):         # + → increase interval
            interval = min(interval + 0.5, 10.0)
            print(f"[App] Inference interval: {interval:.1f}s")
        elif key == ord("-"):                            # - → decrease interval
            interval = max(interval - 0.5, 0.5)
            print(f"[App] Inference interval: {interval:.1f}s")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    voice.shutdown()
    print("[App] Exited cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve config path relative to project root
    cfg_path = _ROOT / "config.yaml"
    run(str(cfg_path))
