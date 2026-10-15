"""
src/app/overlay.py
────────────────────
Draws MediaPipe skeleton, pose label, confidence, correction messages, and
optional joint-angle values directly onto an OpenCV BGR frame.

All drawing helpers return the modified frame for convenience.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.pose import SKELETON_CONNECTIONS
from src.correction.feedback_engine import FeedbackItem
from src.models.inference import PosePrediction


# ── Colour palette ─────────────────────────────────────────────────────────────
CLR_GOOD    = (80,  200, 80)   # BGR green  — correct alignment
CLR_WARN    = (60,  160, 255)  # BGR orange — low confidence / occluded
CLR_CORRECT = (60,   80, 255)  # BGR red    — correction needed
CLR_INFO    = (200, 200, 200)  # BGR grey   — info text
CLR_PANEL   = (20,   20,  20)  # near-black panel background
CLR_BONE    = (180, 120,  60)  # skeleton connection colour
CLR_JOINT   = (60,  200, 255)  # joint dot colour
CLR_WHITE   = (255, 255, 255)


def draw_skeleton(
    frame: np.ndarray,
    landmarks: list[dict],
    show_landmarks: bool = True,
    show_skeleton: bool  = True,
) -> np.ndarray:
    """Draw joints and connections onto the frame (in-place, also returns frame)."""
    h, w = frame.shape[:2]
    if not landmarks:
        return frame

    pts: dict[int, tuple[int, int]] = {}
    for i, lm in enumerate(landmarks):
        x = int(lm["x"] * w)
        y = int(lm["y"] * h)
        pts[i] = (x, y)
        if show_landmarks:
            vis = float(lm.get("visibility", 0.0))
            colour = CLR_JOINT if vis >= 0.30 else (80, 80, 80)
            cv2.circle(frame, (x, y), 5, colour, -1)

    if show_skeleton:
        for (a, b) in SKELETON_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], CLR_BONE, 2)

    return frame


def draw_angle_values(
    frame: np.ndarray,
    landmarks: list[dict],
    feature_dict: dict,
    angle_names: list[str],
) -> np.ndarray:
    """Render a small angle value next to each joint midpoint."""
    h, w = frame.shape[:2]
    from src.features import ANGLE_TRIPLETS

    for name in angle_names:
        val = feature_dict.get(name, np.nan)
        if not isinstance(val, float) or not np.isfinite(val):
            continue
        triplet = ANGLE_TRIPLETS.get(name)
        if triplet is None:
            continue
        _, b_id, _ = triplet
        if b_id >= len(landmarks):
            continue
        lm = landmarks[b_id]
        x = int(lm["x"] * w) + 8
        y = int(lm["y"] * h) - 8
        label = f"{val:.0f}"
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, CLR_WHITE, 1, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    prediction: PosePrediction,
    feedback_items: list[FeedbackItem],
    voice_enabled: bool,
    inference_interval: float,
    last_inference_age: float,          # seconds since last inference
    show_angles: bool = False,
    font_scale: float = 0.65,
    max_lines: int = 4,
    target_pose: Optional[str] = None,
    landmarks: Optional[list[dict]] = None,
) -> np.ndarray:
    """Draw the full HUD panel at the top of the frame."""
    h, w = frame.shape[:2]
    panel_h = 42 + max_lines * 26 + 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), CLR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lh = 26   # line height in pixels
    x0 = 12

    # ── Check Framing / Full Body Visibility ──────────────────────────────────
    framing_warning: Optional[str] = None
    if landmarks and len(landmarks) >= 29:
        knees_ok = (landmarks[25].get("visibility", 0) >= 0.25) or (landmarks[26].get("visibility", 0) >= 0.25)
        ankles_ok = (landmarks[27].get("visibility", 0) >= 0.25) or (landmarks[28].get("visibility", 0) >= 0.25)
        if not knees_ok and not ankles_ok:
            framing_warning = "CAMERA WARNING: Knees & feet cut off! Step 2-3 meters back"
        elif not ankles_ok:
            framing_warning = "CAMERA WARNING: Feet not visible! Step back so full legs are seen"

    # ── Row 0: target / pose name + confidence + interval ─────────────────────
    interval_str = f"  |  Interval: {inference_interval:.1f}s"
    voice_str = "  |  Voice: ON" if voice_enabled else "  |  Voice: OFF"

    if target_pose:
        if prediction.pose_name == target_pose and prediction.status == "detected":
            header_str = f"PRACTICE: {target_pose} [Match {prediction.confidence*100:.0f}%]"
            status_colour = CLR_GOOD
        else:
            header_str = f"PRACTICE: {target_pose} (Aligning...)"
            status_colour = CLR_WARN
    else:
        pose_str = prediction.pose_name or "—"
        conf_str = f"{prediction.confidence * 100:.0f}%"
        header_str = f"AUTO-DETECT: {pose_str} [{conf_str}]"
        status_colour = {
            "detected":      CLR_GOOD,
            "low_confidence":CLR_WARN,
            "no_person":     CLR_CORRECT,
        }.get(prediction.status, CLR_INFO)

    cv2.putText(frame, f"{header_str}{interval_str}{voice_str}",
                (x0, lh), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, status_colour, 1, cv2.LINE_AA)

    # ── Row 1: Top 3 candidate predictions ────────────────────────────────────
    if prediction.top3:
        top_items = [f"{name} ({prob*100:.0f}%)" for name, prob in prediction.top3[:3]]
        top_str = "Candidates: " + "  |  ".join(top_items)
        cv2.putText(frame, top_str,
                    (x0, 2 * lh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.70, (180, 220, 180), 1, cv2.LINE_AA)

    # ── Framing Warning Banner if user is too close ───────────────────────────
    row_offset = 3 if prediction.top3 else 2
    if framing_warning:
        cv2.putText(frame, framing_warning,
                    (x0, row_offset * lh),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.80, (0, 140, 255), 2, cv2.LINE_AA)
        row_offset += 1

    # ── Rows: correction messages ─────────────────────────────────────────────
    for i, item in enumerate(feedback_items[:max_lines]):
        colour = {
            "correction": CLR_CORRECT,
            "warning":    CLR_WARN,
            "info":       CLR_GOOD,
        }.get(item.severity, CLR_INFO)
        cv2.putText(frame, item.instruction,
                    (x0, (row_offset + i) * lh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.80, colour, 1, cv2.LINE_AA)

    return frame


def draw_key_hints(frame: np.ndarray) -> np.ndarray:
    """Tiny key-binding reminder at the bottom of the frame."""
    h, w = frame.shape[:2]
    hints = (
        "Q=Quit  T/N=Change Pose  M=Toggle Guided/Free  "
        "V=Voice  A=Angles  +/-=Interval  S=Screenshot"
    )
    cv2.putText(frame, hints, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, CLR_INFO, 1, cv2.LINE_AA)
    return frame
