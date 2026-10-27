"""
run_app.py  –  launcher script (project root)

Usage (from e:\\Yoga_posture_project):
    python run_app.py
    python run_app.py --interval 1 --classifier rf
    python run_app.py --no-voice
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when launched from project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml


def _patch_config(cfg: dict, args: argparse.Namespace) -> dict:
    if args.interval is not None:
        cfg["inference"]["interval_seconds"] = args.interval
    if args.classifier is not None:
        cfg["model"]["active_classifier"] = args.classifier
    if args.profile is not None:
        cfg["model"]["profile"] = args.profile
    if args.no_voice:
        cfg["voice"]["enabled"] = False
    if args.camera is not None:
        cfg["camera"]["device_index"] = args.camera
    if "practice" not in cfg:
        cfg["practice"] = {}
    if args.target_pose is not None:
        cfg["practice"]["target_pose"] = args.target_pose
        cfg["practice"]["enabled"] = True
    if args.free:
        cfg["practice"]["enabled"] = False
    if getattr(args, "calibrate", False):
        if "calibration" not in cfg:
            cfg["calibration"] = {}
        cfg["calibration"]["force_startup"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yoga Pose Correction — Real-Time Webcam App"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--profile", choices=["kaggle", "zenodo"], default=None,
                        help="Active model profile: 'kaggle' (47-class default) or 'zenodo' (10-class dual-task)")
    parser.add_argument("--interval", type=float, default=None,
                        help="Inference interval in seconds (overrides config)")
    parser.add_argument("--classifier", choices=["rf", "mlp"], default=None,
                        help="Active classifier (overrides config)")
    parser.add_argument("--target-pose", default=None,
                        help="Target pose to practice (default: Virabhadrasana Two or Tadasana)")
    parser.add_argument("--free", action="store_true",
                        help="Start in free auto-detect mode")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice feedback")
    parser.add_argument("--test-voice", action="store_true",
                        help="Run audio speaker diagnostic test and exit")
    parser.add_argument("--history", action="store_true",
                        help="Display practice session history table from SQLite and exit")
    parser.add_argument("--progress", action="store_true",
                        help="Display comprehensive longitudinal progress and personalization report and exit")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device index (overrides config)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Force 5-second body baseline calibration on startup")
    args = parser.parse_args()

    if args.test_voice:
        from src.app.voice import test_diagnostic
        success = test_diagnostic()
        sys.exit(0 if success else 1)

    if args.history:
        from src.correction.progress_tracker import ProgressTracker
        tracker = ProgressTracker()
        print("\n" + tracker.format_history_table(limit=15) + "\n")
        sys.exit(0)

    if args.progress:
        from src.correction.progress_tracker import ProgressTracker
        tracker = ProgressTracker()
        print("\n" + tracker.format_progress_report(limit=15) + "\n")
        sys.exit(0)


    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg = _patch_config(cfg, args)

    from src.app.webcam_app import run
    run(cfg)


if __name__ == "__main__":
    main()
