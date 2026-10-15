"""
src/pose/landmark_extraction.py
────────────────────────────────
Runs MediaPipe PoseLandmarker on a single BGR frame and returns raw landmark
dicts.  Mirrors the extraction logic used in the training notebook.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional


# ── MediaPipe connection indices used for skeleton drawing ────────────────────
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # L arm
    (12, 14), (14, 16),  # R arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # L leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # R leg
]


class PoseLandmarkerWrapper:
    """Wraps MediaPipe IMAGE-mode PoseLandmarker for frame-by-frame detection."""

    def __init__(self, model_path: str, visibility_threshold: float = 0.30):
        self.visibility_threshold = visibility_threshold
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.50,
            min_pose_presence_confidence=0.50,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr) -> Optional[list[dict]]:
        """
        Run detection on a BGR frame.

        Returns:
            List of 33 landmark dicts {x, y, z, visibility} in normalised coords,
            or None when no person is detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.pose_landmarks:
            return None

        detected = result.pose_landmarks[0]
        return [
            {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 0.0)),
            }
            for lm in detected
        ]

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
