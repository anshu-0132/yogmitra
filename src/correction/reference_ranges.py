"""
src/correction/reference_ranges.py
────────────────────────────────────
Pose-specific joint-angle reference ranges and their directional correction
messages.

⚠  These values are provisional and intended as illustrative starting points.
   They are NOT clinically validated biomechanical standards.
   Edit REFERENCE_RANGES freely to match your yoga protocol.

Each entry format:
  "Pose Name": {
      "feature_col_name": (min_degrees, max_degrees),
      ...
  }

feature_col_name must match an angle key in ANGLE_TRIPLETS (geometry.py):
  left_elbow_angle, right_elbow_angle,
  left_shoulder_angle, right_shoulder_angle,
  left_hip_angle, right_hip_angle,
  left_knee_angle, right_knee_angle,
  left_ankle_angle, right_ankle_angle
"""

from __future__ import annotations

# ── Reference angle ranges (provisional) ─────────────────────────────────────
REFERENCE_RANGES: dict[str, dict[str, tuple[float, float]]] = {

    "Virabhadrasana One": {          # Warrior 1
        "left_knee_angle":   (85, 115),
        "right_knee_angle":  (160, 185),
        "left_elbow_angle":  (165, 185),
        "right_elbow_angle": (165, 185),
        "left_shoulder_angle": (150, 185),
        "right_shoulder_angle": (150, 185),
    },

    "Virabhadrasana Two": {          # Warrior 2
        "left_knee_angle":   (85, 115),
        "right_knee_angle":  (160, 185),
        "left_elbow_angle":  (160, 185),
        "right_elbow_angle": (160, 185),
    },

    "Virabhadrasana Three": {        # Warrior 3 / T-balance
        "left_knee_angle":   (165, 185),
        "right_hip_angle":   (85, 100),
        "left_elbow_angle":  (160, 185),
        "right_elbow_angle": (160, 185),
    },

    "Utkatasana": {                  # Chair Pose
        "left_knee_angle":  (70, 110),
        "right_knee_angle": (70, 110),
        "left_hip_angle":   (60, 100),
        "right_hip_angle":  (60, 100),
        "left_elbow_angle": (165, 185),
        "right_elbow_angle": (165, 185),
    },

    "Phalakasana": {                 # High Plank
        "left_elbow_angle":  (165, 185),
        "right_elbow_angle": (165, 185),
        "left_knee_angle":   (165, 185),
        "right_knee_angle":  (165, 185),
        "left_hip_angle":    (160, 195),
        "right_hip_angle":   (160, 195),
    },

    "Adho Mukha Svanasana": {        # Downward-Facing Dog
        "left_knee_angle":   (160, 185),
        "right_knee_angle":  (160, 185),
        "left_shoulder_angle": (145, 185),
        "right_shoulder_angle": (145, 185),
    },

    "Trikonasana": {                 # Triangle
        "left_knee_angle":  (165, 185),
        "right_knee_angle": (165, 185),
        "left_hip_angle":   (60, 100),
        "right_hip_angle":  (60, 100),
    },

    "Vrksasana": {                   # Tree Pose
        "left_knee_angle":  (165, 185),
        "right_knee_angle": (35, 65),
    },

    "Uttanasana": {                  # Standing Forward Bend
        "left_knee_angle":  (160, 185),
        "right_knee_angle": (160, 185),
    },

    "Utthita Parsvakonasana": {      # Extended Side Angle
        "left_knee_angle":   (85, 115),
        "right_knee_angle":  (160, 185),
        "left_elbow_angle":  (155, 185),
        "right_elbow_angle": (155, 185),
    },

    "Parsva Virabhadrasana": {       # Reverse / Peaceful Warrior
        "left_knee_angle":   (85, 115),
        "right_knee_angle":  (160, 185),
    },

    "Balasana": {                    # Child's Pose
        "left_knee_angle":   (0, 50),
        "right_knee_angle":  (0, 50),
        "left_hip_angle":    (0, 60),
        "right_hip_angle":   (0, 60),
    },

    "Garudasana": {                  # Eagle Pose
        "left_knee_angle":   (70, 130),
        "right_knee_angle":  (70, 130),
    },

    "Malasana": {                    # Yogi Squat
        "left_knee_angle":   (40, 90),
        "right_knee_angle":  (40, 90),
    },

    "Anjaneyasana": {                # Low Lunge
        "left_knee_angle":   (85, 115),
        "right_knee_angle":  (160, 195),
    },

    "Ardha Chandrasana": {           # Half Moon
        "left_knee_angle":   (165, 185),
        "right_hip_angle":   (80, 100),
    },
}

# ── Readable joint names for feedback messages ────────────────────────────────
JOINT_DISPLAY_NAMES: dict[str, str] = {
    "left_knee_angle":    "left knee",
    "right_knee_angle":   "right knee",
    "left_elbow_angle":   "left elbow",
    "right_elbow_angle":  "right elbow",
    "left_shoulder_angle": "left shoulder",
    "right_shoulder_angle": "right shoulder",
    "left_hip_angle":     "left hip",
    "right_hip_angle":    "right hip",
    "left_ankle_angle":   "left ankle",
    "right_ankle_angle":  "right ankle",
}
