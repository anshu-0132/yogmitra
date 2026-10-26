"""
src/correction/reference_ranges.py
────────────────────────────────────
Pose-specific joint-angle reference ranges, descriptive metadata,
and key alignment cues for real-time guidance.
"""

from __future__ import annotations

# ── Reference angle ranges ───────────────────────────────────────────────────
REFERENCE_RANGES: dict[str, dict[str, tuple[float, float]]] = {

    "Virabhadrasana One": {          # Warrior 1
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
        "left_shoulder_angle":  (150, 185),
        "right_shoulder_angle": (150, 185),
    },

    "Virabhadrasana Two": {          # Warrior 2
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
    },

    "Virabhadrasana Three": {        # Warrior 3 / T-balance
        "left_knee_angle":      (165, 185),
        "right_hip_angle":      (85, 105),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
    },

    "Utkatasana": {                  # Chair Pose
        "left_knee_angle":      (70, 115),
        "right_knee_angle":     (70, 115),
        "left_hip_angle":       (60, 105),
        "right_hip_angle":      (60, 105),
        "left_elbow_angle":     (160, 185),
        "right_elbow_angle":    (160, 185),
    },

    "Phalakasana": {                 # High Plank
        "left_elbow_angle":     (165, 185),
        "right_elbow_angle":    (165, 185),
        "left_knee_angle":      (165, 185),
        "right_knee_angle":     (165, 185),
        "left_hip_angle":       (160, 195),
        "right_hip_angle":      (160, 195),
    },

    "Adho Mukha Svanasana": {        # Downward-Facing Dog
        "left_knee_angle":      (155, 185),
        "right_knee_angle":     (155, 185),
        "left_shoulder_angle":  (145, 185),
        "right_shoulder_angle": (145, 185),
    },

    "Trikonasana": {                 # Triangle Pose
        "left_knee_angle":      (160, 185),
        "right_knee_angle":     (160, 185),
        "left_hip_angle":       (60, 105),
        "right_hip_angle":      (60, 105),
    },

    "Vrksasana": {                   # Tree Pose
        "left_knee_angle":      (165, 185),
        "right_knee_angle":     (35, 75),
    },

    "Uttanasana": {                  # Standing Forward Bend
        "left_knee_angle":      (155, 185),
        "right_knee_angle":     (155, 185),
    },

    "Utthita Parsvakonasana": {      # Extended Side Angle
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
        "left_elbow_angle":     (155, 185),
        "right_elbow_angle":    (155, 185),
    },

    "Parsva Virabhadrasana": {       # Reverse Warrior
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 185),
    },

    "Balasana": {                    # Child's Pose
        "left_knee_angle":      (0, 50),
        "right_knee_angle":     (0, 50),
        "left_hip_angle":       (0, 60),
        "right_hip_angle":      (0, 60),
    },

    "Anjaneyasana": {                # Low Lunge
        "left_knee_angle":      (85, 115),
        "right_knee_angle":     (160, 195),
    },
}

# ── Joint Display Names ───────────────────────────────────────────────────────
JOINT_DISPLAY_NAMES: dict[str, str] = {
    "left_knee_angle":      "left knee",
    "right_knee_angle":     "right knee",
    "left_elbow_angle":     "left arm",
    "right_elbow_angle":    "right arm",
    "left_shoulder_angle":  "left shoulder",
    "right_shoulder_angle": "right shoulder",
    "left_hip_angle":       "left hip",
    "right_hip_angle":      "right hip",
    "left_ankle_angle":     "left ankle",
    "right_ankle_angle":    "right ankle",
}

# ── Pose Metadata & Reference Cues ───────────────────────────────────────────
POSE_METADATA: dict[str, dict] = {
    "Virabhadrasana Two": {
        "english_name": "Warrior II",
        "sanskrit_name": "Virabhadrasana II",
        "asset_name": "virabhadrasana_two.png",
        "cues": [
            "Bend front knee to 90 deg directly over ankle",
            "Keep back leg straight and grounded",
            "Extend arms parallel to floor at shoulder height",
            "Keep torso centered and upright over pelvis",
        ],
        "key_joints": ["front knee", "back leg", "arms", "torso"],
    },
    "Virabhadrasana One": {
        "english_name": "Warrior I",
        "sanskrit_name": "Virabhadrasana I",
        "asset_name": "virabhadrasana_one.png",
        "cues": [
            "Deep front knee bend at 90 deg",
            "Extend back leg straight with foot anchored",
            "Reach both arms high overhead with palms facing",
            "Square chest and hips toward the front",
        ],
        "key_joints": ["front knee", "back leg", "arms overhead"],
    },
    "Utkatasana": {
        "english_name": "Chair Pose",
        "sanskrit_name": "Utkatasana",
        "asset_name": "utkatasana.png",
        "cues": [
            "Bend knees deeply and sit hips back",
            "Keep weight centered in heels",
            "Extend arms overhead alongside ears",
            "Lengthen spine and lift chest forward",
        ],
        "key_joints": ["knees", "hips", "arms"],
    },
    "Trikonasana": {
        "english_name": "Triangle Pose",
        "sanskrit_name": "Trikonasana",
        "asset_name": "trikonasana.png",
        "cues": [
            "Both legs straight without locking knees",
            "Reach bottom hand down toward shin or floor",
            "Extend top arm straight up toward sky",
            "Keep torso in line with front leg, chest open",
        ],
        "key_joints": ["legs", "arms", "side body"],
    },
    "Phalakasana": {
        "english_name": "Plank Pose",
        "sanskrit_name": "Phalakasana",
        "asset_name": "phalakasana.png",
        "cues": [
            "Arms straight with wrists under shoulders",
            "Body forms a straight diagonal line from head to heels",
            "Engage core without letting hips sag or pike",
            "Keep legs firm and heels pressing back",
        ],
        "key_joints": ["arms", "hips", "legs"],
    },
    "Adho Mukha Svanasana": {
        "english_name": "Downward Dog",
        "sanskrit_name": "Adho Mukha Svanasana",
        "asset_name": "adho_mukha_svanasana.png",
        "cues": [
            "Hands shoulder-width apart, fingers spread wide",
            "Press hips high creating an inverted V-shape",
            "Straighten arms and press chest toward thighs",
            "Lengthen legs and draw heels toward floor",
        ],
        "key_joints": ["shoulders", "spine", "legs"],
    },
    "Vrksasana": {
        "english_name": "Tree Pose",
        "sanskrit_name": "Vrksasana",
        "asset_name": "vrksasana.png",
        "cues": [
            "Stand tall on standing leg, root into ground",
            "Place other foot on inner calf or thigh (avoid knee)",
            "Hands at chest in prayer or extended overhead",
            "Find a steady gaze point for balance",
        ],
        "key_joints": ["standing leg", "bent knee", "balance"],
    },
    "Uttanasana": {
        "english_name": "Forward Fold",
        "sanskrit_name": "Uttanasana",
        "asset_name": "uttanasana.png",
        "cues": [
            "Hinge forward at hip joints, not waist",
            "Keep spine long and fold torso over legs",
            "Legs straight with gentle micro-bend if tight",
            "Let head and neck hang completely relaxed",
        ],
        "key_joints": ["hips", "hamstrings", "spine"],
    },
    "Anjaneyasana": {
        "english_name": "Low Lunge",
        "sanskrit_name": "Anjaneyasana",
        "asset_name": "anjaneyasana.png",
        "cues": [
            "Front knee stacked directly above ankle at 90 deg",
            "Back knee rested gently on mat, leg extended",
            "Sweep arms overhead and lift chest",
            "Sink hips forward and down comfortably",
        ],
        "key_joints": ["front knee", "hips", "arms"],
    },
}
