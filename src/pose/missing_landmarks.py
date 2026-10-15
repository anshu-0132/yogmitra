"""
src/pose/missing_landmarks.py
──────────────────────────────
Handles low-visibility landmarks for live video inference.

Policy (matches corrected notebook):
  • Temporal fallback only: if a joint was clearly visible in the previous
    frame, carry that position forward.
  • No symmetry mirroring across the body centerline – yoga poses are
    asymmetric and mirroring distorts genuine posture geometry.
  • Missing joints that cannot be estimated are left with their original
    (low-visibility) values and flagged; downstream features become NaN.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

VISIBILITY_THRESHOLD = 0.30
IMPUTED_VISIBILITY = 0.35  # conservative value assigned to estimated joints

# Bilateral landmark pairs (left_id, right_id) – kept for reference only.
OPPOSITE_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def _is_reliable(point: dict, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    return point is not None and float(point.get("visibility", 0.0)) >= threshold


def apply_temporal_fallback(
    raw_landmarks: list[dict],
    previous_raw: Optional[list[dict]],
    threshold: float = VISIBILITY_THRESHOLD,
) -> tuple[list[dict], list[int], list[dict]]:
    """
    For each low-visibility joint, fall back to the previous directly-observed
    position from the preceding frame (video-only; skip for static images).

    Returns:
        effective  – 33-element list of landmark dicts used for feature extraction
        flags      – 1 where a joint was estimated, 0 where directly observed
        raw        – original detections (for the caller to store as next prev_raw)
    """
    effective = [dict(p) for p in raw_landmarks]
    flags = [0] * len(raw_landmarks)

    for joint_id, point in enumerate(raw_landmarks):
        if _is_reliable(point, threshold):
            continue

        # Temporal fallback: use the previous frame's position if it was reliable
        if previous_raw is not None and joint_id < len(previous_raw):
            prev = previous_raw[joint_id]
            if _is_reliable(prev, threshold):
                effective[joint_id] = {
                    "x": prev["x"],
                    "y": prev["y"],
                    "z": prev["z"],
                    "visibility": IMPUTED_VISIBILITY,
                }
                flags[joint_id] = 1

    return effective, flags, raw_landmarks
