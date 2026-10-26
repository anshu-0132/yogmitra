"""
src/correction/calibration.py
──────────────────────────────
User-specific biomechanical calibration module.

Measures baseline body proportions (arm span to torso, leg to torso, shoulder width)
across 25 stable frames in Mountain Pose (Tadasana) and persists the profile to
data/user_profile.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.features.geometry import LANDMARK_IDS, VISIBILITY_THRESHOLD

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_PATH = Path("data/user_profile.json")


@dataclass
class UserProfile:
    calibrated: bool = False
    user_name: str = "Default Yogi"
    tolerance_multiplier: float = 1.0  # Range tolerance modifier (e.g. 1.1 for beginner flexibility)
    arm_span_ratio: float = 1.80       # Arm span relative to torso length
    leg_to_torso_ratio: float = 1.65   # Leg length relative to torso length
    shoulder_width_ratio: float = 0.65 # Shoulder width relative to torso
    baseline_torso_px: float = 240.0   # Measured reference pixel scale
    calibration_date: Optional[str] = None


def load_profile(path: Path | str = DEFAULT_PROFILE_PATH) -> UserProfile:
    """Load user profile from disk, returning default if not found."""
    p = Path(path)
    if not p.exists():
        return UserProfile()

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return UserProfile(
            calibrated=data.get("calibrated", True),
            user_name=data.get("user_name", "Default Yogi"),
            tolerance_multiplier=float(data.get("tolerance_multiplier", 1.0)),
            arm_span_ratio=float(data.get("arm_span_ratio", 1.80)),
            leg_to_torso_ratio=float(data.get("leg_to_torso_ratio", 1.65)),
            shoulder_width_ratio=float(data.get("shoulder_width_ratio", 0.65)),
            baseline_torso_px=float(data.get("baseline_torso_px", 240.0)),
            calibration_date=data.get("calibration_date"),
        )
    except Exception as exc:
        logger.warning("[Calibration] Could not parse profile: %s. Using default.", exc)
        return UserProfile()


def save_profile(profile: UserProfile, path: Path | str = DEFAULT_PROFILE_PATH) -> bool:
    """Save user profile to disk."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2)
        logger.info("[Calibration] Profile saved to %s", p)
        return True
    except Exception as exc:
        logger.error("[Calibration] Failed to save profile: %s", exc)
        return False


class CalibrationSession:
    """
    Guides the user through a 25-frame calibration in Mountain Pose (Tadasana).
    """

    def __init__(self, target_frames: int = 25, save_path: Path | str = DEFAULT_PROFILE_PATH):
        self.target_frames = target_frames
        self.save_path = Path(save_path)
        self.is_active = False
        self.frame_count = 0
        self._arm_spans: list[float] = []
        self._leg_ratios: list[float] = []
        self._torso_lengths: list[float] = []
        self._shoulder_widths: list[float] = []
        self.status_message = "Press C to start calibration."

    def start(self) -> None:
        """Start a new calibration session."""
        self.is_active = True
        self.frame_count = 0
        self._arm_spans.clear()
        self._leg_ratios.clear()
        self._torso_lengths.clear()
        self._shoulder_widths.clear()
        self.status_message = "Stand tall in Mountain Pose (Tadasana). Hold steady."

    def cancel(self) -> None:
        self.is_active = False
        self.status_message = "Calibration cancelled."

    def update(self, landmarks: list[dict], width: int, height: int) -> tuple[bool, str, Optional[UserProfile]]:
        """
        Process a single frame during calibration.
        Returns (is_complete, status_message, Optional[UserProfile]).
        """
        if not self.is_active:
            return False, self.status_message, None

        if not landmarks or len(landmarks) < 33:
            return False, "Searching for full body in camera...", None

        # Verify key landmarks visibility: shoulders, hips, knees, ankles
        required_ids = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28]
        if any(landmarks[i].get("visibility", 0.0) < VISIBILITY_THRESHOLD for i in required_ids):
            return False, "Step back: Ensure head, shoulders, hips, and feet are visible.", None

        # Extract 2D pixel coordinates
        pts = {}
        for name, idx in LANDMARK_IDS.items():
            lm = landmarks[idx]
            pts[name] = np.array([lm["x"] * width, lm["y"] * height], dtype=float)

        shoulder_mid = (pts["left_shoulder"] + pts["right_shoulder"]) / 2.0
        hip_mid = (pts["left_hip"] + pts["right_hip"]) / 2.0
        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        if torso_len < 20.0:
            return False, "Torso scale too small. Stand 2 to 3 meters from camera.", None

        # Measurements
        arm_span = float(np.linalg.norm(pts["left_wrist"] - pts["right_wrist"]))
        left_leg = float(np.linalg.norm(pts["left_hip"] - pts["left_knee"]) + np.linalg.norm(pts["left_knee"] - pts["left_ankle"]))
        right_leg = float(np.linalg.norm(pts["right_hip"] - pts["right_knee"]) + np.linalg.norm(pts["right_knee"] - pts["right_ankle"]))
        avg_leg = (left_leg + right_leg) / 2.0
        shoulder_w = float(np.linalg.norm(pts["left_shoulder"] - pts["right_shoulder"]))

        self._arm_spans.append(arm_span / torso_len)
        self._leg_ratios.append(avg_leg / torso_len)
        self._shoulder_widths.append(shoulder_w / torso_len)
        self._torso_lengths.append(torso_len)
        self.frame_count += 1

        pct = int((self.frame_count / self.target_frames) * 100)
        self.status_message = f"Calibrating: {pct}% — Hold steady in Mountain Pose..."

        if self.frame_count >= self.target_frames:
            # Build and persist profile
            profile = UserProfile(
                calibrated=True,
                user_name="Calibrated Yogi",
                tolerance_multiplier=1.0,
                arm_span_ratio=round(float(np.median(self._arm_spans)), 3),
                leg_to_torso_ratio=round(float(np.median(self._leg_ratios)), 3),
                shoulder_width_ratio=round(float(np.median(self._shoulder_widths)), 3),
                baseline_torso_px=round(float(np.median(self._torso_lengths)), 1),
                calibration_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            save_profile(profile, self.save_path)
            self.is_active = False
            self.status_message = "Calibration complete! Profile saved."
            return True, self.status_message, profile

        return False, self.status_message, None
