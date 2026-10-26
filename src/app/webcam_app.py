"""
src/app/webcam_app.py
──────────────────────
Main application entry-point for Yoga Mitra: AI-Powered Personalized
Yoga Posture Detection and Correction System.

Pipeline per frame:
  camera frame
    → MediaPipe detection (unmirrored raw frame for anatomical accuracy)
    → temporal landmark fallback
    → feature extraction (geometry.py — exact 39-feature training schema)
    → impute + [scale] + RF/MLP predict (inference.py)
    → posture evaluate (posture_evaluator.py — severity, direction, score)
    → feedback engine build (feedback_engine.py — confidence-aware cues)
    → biomechanical calibration (calibration.py — Mountain Pose baseline)
    → session progress tracker (progress_tracker.py — SQLite logging)
    → two-panel layout:
        Left panel: Live webcam feed + skeleton HUD + deviant joint highlight
        Right panel: Persistent Reference Asana card + alignment cues
    → async TTS voice engine (voice.py — non-blocking thread-local SAPI5)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
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
from src.correction.posture_evaluator import evaluate as evaluate_posture, PostureEvaluation
from src.correction.feedback_engine import FeedbackEngine, FeedbackItem
from src.correction.calibration import UserProfile, load_profile, save_profile, CalibrationSession
from src.correction.progress_tracker import ProgressTracker
from src.app.voice import VoiceFeedback
from src.app.reference_display import ReferenceDisplay
from src.app.overlay import (
    draw_skeleton, draw_hud, draw_key_hints, draw_angle_values, draw_calibration_overlay
)

# Landmark ID mapping for visual joint error highlighting
JOINT_LM_MAP: dict[str, int] = {
    "left_knee_angle": 25,
    "right_knee_angle": 26,
    "left_elbow_angle": 13,
    "right_elbow_angle": 14,
    "left_shoulder_angle": 11,
    "right_shoulder_angle": 12,
    "left_hip_angle": 23,
    "right_hip_angle": 24,
    "left_ankle_angle": 27,
    "right_ankle_angle": 28,
}


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

    # ── Reference Display Panel (Right side) ──────────────────────────────────
    ref_display = ReferenceDisplay(
        assets_dir=_ROOT / "assets" / "pose_references",
        panel_width=380,
    )

    # ── User Biomechanical Calibration ───────────────────────────────────────
    user_profile = load_profile(_ROOT / "data" / "user_profile.json")
    calib_session = CalibrationSession(save_path=_ROOT / "data" / "user_profile.json")
    if user_profile.calibrated:
        print(f"[App] Loaded user profile: {user_profile.user_name} (Calibrated: {user_profile.calibration_date})")
    else:
        print("[App] Default user profile loaded. Press C anytime to run Mountain Pose calibration.")

    # ── Session Progress Tracker ──────────────────────────────────────────────
    progress_tracker = ProgressTracker(db_path=_ROOT / "data" / "yoga_sessions.db")
    progress_tracker.start_session(calibrated=user_profile.calibrated)

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
        "Virabhadrasana Two",     # Warrior 2
        "Virabhadrasana One",     # Warrior 1
        "Utkatasana",             # Chair Pose
        "Trikonasana",            # Triangle Pose
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

    # Persistent results
    last_prediction: Optional[PosePrediction] = None
    last_features: dict = {}
    last_feedback: list[FeedbackItem] = []
    last_lm_effective: Optional[list[dict]] = None
    last_eval_result: Optional[PostureEvaluation] = None
    highlight_joint_ids: set[int] = set()

    show_angles: bool = display_cfg["show_angles"]
    screenshot_dir = _ROOT / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    mode_label = f"Guided Practice [{practice_poses[target_idx]}]" if target_mode else "Auto-Detect"
    print(f"[App] Starting in {mode_label}.")
    print("[App] Controls: Q=Quit, T/N=Change Pose, M=Guided/Free, C=Calibrate, V=Voice, A=Angles")
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

        # ── Biomechanical Calibration step (if triggered by user) ─────────────
        if calib_session.is_active and raw_lm is not None:
            h_raw, w_raw = frame_raw.shape[:2]
            calib_done, calib_msg, new_prof = calib_session.update(raw_lm, w_raw, h_raw)
            if calib_done and new_prof is not None:
                user_profile = new_prof
                progress_tracker.calibrated = True
                voice.speak("Calibration complete! Your biomechanical baseline is saved.")
                print(f"[App] {calib_msg}")

        if raw_lm is None:
            last_lm_effective = None
            prev_raw_landmarks = None
            highlight_joint_ids.clear()
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
                last_eval_result = None
                last_feedback = feedback_engine.update(dummy_pred, None, target_pose=active_target)
        else:
            # Temporal fallback for missing joints
            effective_lm, flags, raw_for_history = apply_temporal_fallback(
                raw_lm, prev_raw_landmarks,
                threshold=infer_cfg["visibility_threshold"],
            )
            prev_raw_landmarks = raw_for_history
            last_lm_effective  = effective_lm

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
                    user_profile=asdict(user_profile) if user_profile.calibrated else None,
                )
                last_eval_result = eval_result

                # Update joints to highlight on skeleton
                highlight_joint_ids.clear()
                if eval_result and eval_result.checks:
                    for chk in eval_result.checks:
                        if chk.status != "ok" and chk.reliable and chk.severity in ("moderate", "major"):
                            lm_id = JOINT_LM_MAP.get(chk.angle_name)
                            if lm_id is not None:
                                highlight_joint_ids.add(lm_id)

                # Log to session progress tracker
                if eval_result:
                    is_rel = (eval_result.overall_status != "low_visibility")
                    progress_tracker.log_frame(
                        eval_pose,
                        eval_result.alignment_score,
                        eval_result.checks,
                        is_reliable=is_rel,
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

        # ── Draw skeleton with deviant joint highlights ───────────────────────
        if display_landmarks and display_cfg["show_landmarks"]:
            draw_skeleton(
                frame_display,
                display_landmarks,
                show_landmarks=display_cfg["show_landmarks"],
                show_skeleton=display_cfg["show_skeleton"],
                highlight_joint_ids=highlight_joint_ids,
            )

        # ── Draw angle values ─────────────────────────────────────────────────
        if show_angles and display_landmarks and last_features:
            draw_angle_values(
                frame_display, display_landmarks, last_features,
                list(ANGLE_TRIPLETS.keys())
            )

        # ── HUD with alignment score ──────────────────────────────────────────
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
                alignment_score=last_eval_result.alignment_score if last_eval_result else None,
            )

        # ── Biomechanical Calibration Overlay ─────────────────────────────────
        if calib_session.is_active:
            pct = int((calib_session.frame_count / calib_session.target_frames) * 100)
            draw_calibration_overlay(frame_display, True, pct, calib_session.status_message)

        draw_key_hints(frame_display)

        # ── Render Right Reference Asana Panel ────────────────────────────────
        if target_mode:
            current_ref_pose = practice_poses[target_idx]
        else:
            current_ref_pose = ref_display.update_auto_detected_pose(
                last_prediction.pose_name if last_prediction else None,
                last_prediction.confidence if last_prediction else 0.0,
            )

        primary_joint = (
            last_eval_result.primary_check.joint_name
            if (last_eval_result and last_eval_result.primary_check and last_eval_result.primary_check.status != "ok")
            else None
        )

        ref_panel = ref_display.render_panel(
            pose_name=current_ref_pose,
            height=frame_display.shape[0],
            focus_joint=primary_joint,
            is_guided=target_mode,
            confidence=last_prediction.confidence if last_prediction else None,
        )

        # ── Combine Left Live Panel and Right Reference Panel ─────────────────
        combined_display = cv2.hconcat([frame_display, ref_panel])

        cv2.imshow("Yoga Mitra: AI-Powered Yoga Posture Detection & Correction", combined_display)

        # ── Key controls ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:                # Q / Esc → quit
            break
        elif key == ord("t") or key == ord("n"):         # T / N → cycle target pose
            target_idx = (target_idx + 1) % len(practice_poses)
            target_mode = True
            feedback_engine.clear()
            ref_display.reset_stability()
            print(f"[App] Target pose changed to: {practice_poses[target_idx]}")
        elif key == ord("m"):                            # M → toggle Guided vs Free mode
            target_mode = not target_mode
            feedback_engine.clear()
            ref_display.reset_stability()
            curr_mode = f"Guided [{practice_poses[target_idx]}]" if target_mode else "Free Auto-Detect"
            print(f"[App] Mode switched to: {curr_mode}")
        elif key == ord("c"):                            # C → start/cancel calibration
            if not calib_session.is_active:
                calib_session.start()
                voice.speak("Stand tall in Mountain Pose for calibration.")
                print("[App] Calibration started. Stand tall in Mountain Pose (Tadasana)...")
            else:
                calib_session.cancel()
                print("[App] Calibration cancelled.")
        elif key == ord("v"):                            # V → toggle voice
            state = voice.toggle()
            print(f"[App] Voice {'ON' if state else 'OFF'}")
        elif key == ord("a"):                            # A → toggle angles
            show_angles = not show_angles
            print(f"[App] Angle overlay {'ON' if show_angles else 'OFF'}")
        elif key == ord("r"):                            # R → reset history
            inferrer.reset_history()
            feedback_engine.clear()
            ref_display.reset_stability()
            print("[App] Prediction history reset.")
        elif key == ord("s"):                            # S → screenshot
            fname = screenshot_dir / f"yoga_{int(time.time())}.jpg"
            cv2.imwrite(str(fname), combined_display)
            print(f"[App] Screenshot saved: {fname}")
        elif key == ord("+") or key == ord("="):         # + → increase interval
            interval = min(interval + 0.5, 10.0)
            print(f"[App] Inference interval: {interval:.1f}s")
        elif key == ord("-"):                            # - → decrease interval
            interval = max(interval - 0.5, 0.5)
            print(f"[App] Inference interval: {interval:.1f}s")

    # ── Cleanup & Session Summary ─────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    voice.shutdown()

    summary = progress_tracker.end_session()
    if summary and summary.get("frames_evaluated", 0) > 0:
        print("\n" + "=" * 65)
        print(f"  YOGA MITRA — PRACTICE SESSION SUMMARY")
        print("=" * 65)
        print(f"  Session Duration:  {summary['duration_sec']:.1f}s")
        print(f"  Frames Evaluated:  {summary['frames_evaluated']}")
        print(f"  Poses Practiced:   {', '.join(summary['poses_practiced']) or 'None'}")
        print(f"  Avg Alignment:     {summary['avg_alignment_score']:.1f}%")
        print(f"  Peak Alignment:    {summary['peak_alignment_score']:.1f}%")
        print(f"  User Calibrated:   {'Yes' if summary['calibrated'] else 'No'}")
        if summary.get("top_joint_faults"):
            print("  Top Recurring Alignment Issues:")
            for fault in summary["top_joint_faults"][:3]:
                print(f"    - {fault['pose']} {fault['joint']}: {fault['count']}x (avg dev: {fault['avg_dev_deg']}°)")
        print("=" * 65 + "\n")

    print("[App] Exited cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve config path relative to project root
    cfg_path = _ROOT / "config.yaml"
    run(str(cfg_path))
