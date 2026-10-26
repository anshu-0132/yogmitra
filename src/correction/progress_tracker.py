"""
src/correction/progress_tracker.py
───────────────────────────────────
Session history and progress tracking module using SQLite.

Tracks:
- Session start/end timestamp and duration
- Poses practiced during the session
- Frame counts and alignment scores (average & peak)
- Recurring joint-level errors and average deviation magnitudes
- Biomechanical calibration status

Provides lightweight, zero-dependency persistence to data/yoga_sessions.db.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/yoga_sessions.db")


@dataclass
class FrameRecord:
    pose_name: str
    alignment_score: float
    deviations: list[dict[str, Any]]
    is_reliable: bool
    timestamp: float = field(default_factory=time.time)


class ProgressTracker:
    """
    Tracks real-time session metrics and persists session summaries to SQLite.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.session_active = False
        self.start_timestamp: float = 0.0
        self.start_iso: str = ""
        self.calibrated: bool = False

        # Session accumulators
        self.poses_seen: set[str] = set()
        self.frame_count: int = 0
        self.scores: list[float] = []
        self.joint_fault_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.joint_deviation_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        duration_sec REAL NOT NULL,
                        poses_practiced TEXT NOT NULL,
                        frames_evaluated INTEGER NOT NULL,
                        avg_alignment_score REAL NOT NULL,
                        peak_alignment_score REAL NOT NULL,
                        calibrated INTEGER NOT NULL DEFAULT 0
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        pose_name TEXT NOT NULL,
                        joint_name TEXT NOT NULL,
                        error_count INTEGER NOT NULL,
                        avg_deviation REAL NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                conn.commit()
        except Exception as exc:
            logger.error("[ProgressTracker] DB init error: %s", exc)

    def start_session(self, calibrated: bool = False) -> None:
        """Start tracking a new yoga session."""
        self.session_active = True
        self.start_timestamp = time.time()
        self.start_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        self.calibrated = calibrated
        self.poses_seen.clear()
        self.frame_count = 0
        self.scores.clear()
        self.joint_fault_counts.clear()
        self.joint_deviation_sums.clear()
        logger.info("[ProgressTracker] Session started at %s (calibrated=%s)", self.start_iso, calibrated)

    def log_frame(
        self,
        pose_name: str,
        alignment_score: float,
        deviations: list[Any],
        is_reliable: bool = True,
    ) -> None:
        """
        Record a single evaluated frame during practice.
        Accepts PostureDeviation objects or dicts.
        """
        if not self.session_active or not pose_name or pose_name == "—":
            return

        self.poses_seen.add(pose_name)
        self.frame_count += 1

        if is_reliable and alignment_score > 0.0:
            self.scores.append(alignment_score)

        for dev in deviations:
            if isinstance(dev, dict):
                joint = dev.get("joint_name") or dev.get("display_name") or "unknown"
                status = dev.get("status", "needs_work")
                mag = float(dev.get("deviation_magnitude", dev.get("deviation", 0.0)))
            else:
                joint = getattr(dev, "joint_name", None) or getattr(dev, "display_name", "unknown")
                status = getattr(dev, "status", "needs_work")
                mag = float(getattr(dev, "deviation_magnitude", getattr(dev, "deviation", 0.0)))

            # Only record as a fault if status is not 'ok' and there is deviation
            if status != "ok" and mag > 0.0:
                self.joint_fault_counts[pose_name][joint] += 1
                self.joint_deviation_sums[pose_name][joint] += abs(mag)

    def end_session(self) -> Optional[dict[str, Any]]:
        """
        End the active session, persist summary to SQLite, and return summary dict.
        """
        if not self.session_active:
            return None

        self.session_active = False
        duration = round(max(0.0, time.time() - self.start_timestamp), 1)
        end_iso = time.strftime("%Y-%m-%d %H:%M:%S")

        avg_score = round(float(sum(self.scores) / len(self.scores)), 1) if self.scores else 0.0
        peak_score = round(float(max(self.scores)), 1) if self.scores else 0.0
        poses_str = ", ".join(sorted(self.poses_seen)) if self.poses_seen else "None"

        summary = {
            "start_time": self.start_iso,
            "end_time": end_iso,
            "duration_sec": duration,
            "poses_practiced": sorted(self.poses_seen),
            "frames_evaluated": self.frame_count,
            "avg_alignment_score": avg_score,
            "peak_alignment_score": peak_score,
            "calibrated": self.calibrated,
            "top_joint_faults": [],
        }

        # If session had negligible activity, skip DB write
        if self.frame_count == 0 and duration < 2.0:
            return summary

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sessions (
                        start_time, end_time, duration_sec, poses_practiced,
                        frames_evaluated, avg_alignment_score, peak_alignment_score, calibrated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.start_iso,
                        end_iso,
                        duration,
                        poses_str,
                        self.frame_count,
                        avg_score,
                        peak_score,
                        1 if self.calibrated else 0,
                    ),
                )
                session_id = cursor.lastrowid

                # Record joint errors
                fault_records = []
                for p_name, joints in self.joint_fault_counts.items():
                    for j_name, cnt in joints.items():
                        total_deg = self.joint_deviation_sums[p_name][j_name]
                        avg_deg = round(total_deg / max(1, cnt), 1)
                        cursor.execute(
                            """
                            INSERT INTO session_errors (
                                session_id, pose_name, joint_name, error_count, avg_deviation
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (session_id, p_name, j_name, cnt, avg_deg),
                        )
                        fault_records.append({
                            "pose": p_name,
                            "joint": j_name,
                            "count": cnt,
                            "avg_dev_deg": avg_deg,
                        })

                conn.commit()
                # Sort top faults by count descending
                fault_records.sort(key=lambda x: x["count"], reverse=True)
                summary["top_joint_faults"] = fault_records[:5]
                logger.info("[ProgressTracker] Saved session #%s (duration %.1fs, avg %.1f%%)",
                            session_id, duration, avg_score)
        except Exception as exc:
            logger.error("[ProgressTracker] Failed to persist session: %s", exc)

        return summary

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent session records from database."""
        if not self.db_path.exists():
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, start_time, duration_sec, poses_practiced,
                           frames_evaluated, avg_alignment_score, peak_alignment_score, calibrated
                    FROM sessions
                    ORDER BY session_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()

                results = []
                for r in rows:
                    s_id = r["session_id"]
                    cursor.execute(
                        """
                        SELECT pose_name, joint_name, error_count, avg_deviation
                        FROM session_errors
                        WHERE session_id = ?
                        ORDER BY error_count DESC
                        LIMIT 3
                        """,
                        (s_id,),
                    )
                    err_rows = cursor.fetchall()
                    top_errors = [
                        f"{er['joint_name']} ({er['error_count']}x, ~{er['avg_deviation']}°)"
                        for er in err_rows
                    ]
                    results.append({
                        "session_id": s_id,
                        "date": r["start_time"],
                        "duration_sec": r["duration_sec"],
                        "poses": r["poses_practiced"],
                        "frames": r["frames_evaluated"],
                        "avg_score": r["avg_alignment_score"],
                        "peak_score": r["peak_alignment_score"],
                        "calibrated": bool(r["calibrated"]),
                        "top_faults": top_errors,
                    })
                return results
        except Exception as exc:
            logger.error("[ProgressTracker] History fetch error: %s", exc)
            return []

    def format_history_table(self, limit: int = 10) -> str:
        """Return a formatted text table of past sessions."""
        records = self.get_history(limit=limit)
        if not records:
            return "No practice sessions recorded yet. Start practicing with Yoga Mitra to build history!"

        lines = [
            "=" * 78,
            f"{'ID':<4} | {'Date & Time':<19} | {'Dur(s)':<6} | {'Avg%':<5} | {'Peak%':<5} | {'Poses':<24}",
            "-" * 78,
        ]
        for r in records:
            poses = r["poses"]
            if len(poses) > 24:
                poses = poses[:21] + "..."
            lines.append(
                f"{r['session_id']:<4} | {r['date']:<19} | {r['duration_sec']:<6.0f} | "
                f"{r['avg_score']:<5.1f} | {r['peak_score']:<5.1f} | {poses:<24}"
            )
            if r["top_faults"]:
                fault_str = "       Top recurring issues: " + "; ".join(r["top_faults"])
                lines.append(fault_str)
        lines.append("=" * 78)
        return "\n".join(lines)
