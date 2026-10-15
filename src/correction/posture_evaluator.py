"""
src/correction/posture_evaluator.py
─────────────────────────────────────
Compares measured joint angles against the pose's reference ranges and
returns structured evaluation results.

Rules:
  • Never use an angle that was computed from unreliable landmarks.
  • Compute visibility of each joint in an angle triplet before trusting it.
  • Report separately when a joint is simply not visible.
  • Use the ANGLE_TRIPLETS map to know which landmark IDs underpin each angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.correction.reference_ranges import REFERENCE_RANGES, JOINT_DISPLAY_NAMES
from src.features import ANGLE_TRIPLETS
from src.pose.missing_landmarks import VISIBILITY_THRESHOLD


@dataclass
class AngleCheck:
    angle_name: str
    display_name: str
    value: Optional[float]         # None = not computable
    target_range: tuple[float, float]
    status: str                    # "ok" | "too_low" | "too_high" | "unavailable"
    deviation: float               # degrees outside range (0 if ok)
    reliable: bool                 # True if underlying landmarks were visible


@dataclass
class PostureEvaluation:
    pose_name: str
    checks: list[AngleCheck]
    imputed_fraction: float
    has_reference: bool
    overall_status: str            # "correct" | "needs_work" | "no_reference" | "low_visibility"


def get_adapted_ranges(pose_name: str, feature_dict: dict) -> dict[str, tuple[float, float]]:
    """
    Return reference angle ranges adapted to the user's active side.
    For asymmetric poses (Warrior 1/2, lunges), detects whether left or right
    leg is forward so the user can perform the pose on either side.
    """
    base = REFERENCE_RANGES.get(pose_name)
    if not base:
        return {}
    ranges = dict(base)

    if pose_name in ("Virabhadrasana One", "Virabhadrasana Two", "Utthita Parsvakonasana", "Anjaneyasana", "Parsva Virabhadrasana"):
        lk = feature_dict.get("left_knee_angle", np.nan)
        rk = feature_dict.get("right_knee_angle", np.nan)
        # If right knee is bent deeper than left knee, right is the leading front leg
        if np.isfinite(lk) and np.isfinite(rk) and rk < lk:
            ranges["right_knee_angle"] = (85, 115)
            ranges["left_knee_angle"] = (160, 185)
        else:
            ranges["left_knee_angle"] = (85, 115)
            ranges["right_knee_angle"] = (160, 185)

    elif pose_name == "Vrksasana":
        lk = feature_dict.get("left_knee_angle", np.nan)
        rk = feature_dict.get("right_knee_angle", np.nan)
        if np.isfinite(lk) and np.isfinite(rk) and rk < lk:
            ranges["right_knee_angle"] = (35, 75)
            ranges["left_knee_angle"] = (160, 185)
        else:
            ranges["left_knee_angle"] = (35, 75)
            ranges["right_knee_angle"] = (160, 185)

    return ranges


def evaluate(
    pose_name: str,
    feature_dict: dict,
    landmarks: list[dict],
    imputed_flags: list[int],
    imputed_fraction_limit: float = 0.25,
) -> PostureEvaluation:
    """
    Evaluate posture alignment for the predicted pose.

    pose_name     – classifier output (must match REFERENCE_RANGES key)
    feature_dict  – 39-key dict from geometry.extract_features()
    landmarks     – 33-element list of {x,y,z,visibility} dicts
    imputed_flags – 33-element list of ints (1 = estimated)
    """
    imputed_fraction = float(np.mean(imputed_flags)) if imputed_flags else 0.0

    if imputed_fraction > imputed_fraction_limit:
        return PostureEvaluation(
            pose_name=pose_name,
            checks=[],
            imputed_fraction=imputed_fraction,
            has_reference=pose_name in REFERENCE_RANGES,
            overall_status="low_visibility",
        )

    ranges = get_adapted_ranges(pose_name, feature_dict)
    if not ranges:
        return PostureEvaluation(
            pose_name=pose_name,
            checks=[],
            imputed_fraction=imputed_fraction,
            has_reference=False,
            overall_status="no_reference",
        )

    checks: list[AngleCheck] = []
    any_issues = False

    for angle_name, (low, high) in ranges.items():
        display = JOINT_DISPLAY_NAMES.get(angle_name, angle_name.replace("_", " "))

        # ── Landmark reliability ───────────────────────────────────────────
        triplet_ids = ANGLE_TRIPLETS.get(angle_name)
        reliable = True
        if triplet_ids:
            for lm_id in triplet_ids:
                if lm_id >= len(landmarks):
                    reliable = False
                    break
                vis = float(landmarks[lm_id].get("visibility", 0.0))
                if vis < VISIBILITY_THRESHOLD:
                    reliable = False
                    break

        # ── Angle value ────────────────────────────────────────────────────
        value_raw = feature_dict.get(angle_name, np.nan)
        try:
            value = float(value_raw)
        except (TypeError, ValueError):
            value = np.nan

        if not reliable or not np.isfinite(value):
            checks.append(AngleCheck(
                angle_name=angle_name,
                display_name=display,
                value=None,
                target_range=(low, high),
                status="unavailable",
                deviation=0.0,
                reliable=False,
            ))
            continue

        # ── Compare against range ──────────────────────────────────────────
        if value < low:
            status = "too_low"
            deviation = low - value
            any_issues = True
        elif value > high:
            status = "too_high"
            deviation = value - high
            any_issues = True
        else:
            status = "ok"
            deviation = 0.0

        checks.append(AngleCheck(
            angle_name=angle_name,
            display_name=display,
            value=round(value, 1),
            target_range=(low, high),
            status=status,
            deviation=round(deviation, 1),
            reliable=True,
        ))

    overall = "needs_work" if any_issues else "correct"
    return PostureEvaluation(
        pose_name=pose_name,
        checks=checks,
        imputed_fraction=imputed_fraction,
        has_reference=True,
        overall_status=overall,
    )
