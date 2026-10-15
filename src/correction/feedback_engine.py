"""
src/correction/feedback_engine.py
───────────────────────────────────
Converts PostureEvaluation + PosePrediction into structured FeedbackItem
objects, deduplicates repeated messages, and controls the voice cooldown.

Feedback priority:
  1. "No person detected" / "Low confidence"
  2. Low-visibility warning
  3. Most-deviant angle correction  (only when landmark reliable)
  4. "Looks good" acknowledgement
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.correction.posture_evaluator import PostureEvaluation, AngleCheck
from src.models.inference import PosePrediction


@dataclass
class FeedbackItem:
    pose_name: str
    body_part: str                # e.g. "left knee"
    issue: str                    # e.g. "too low"
    instruction: str              # human-readable correction sentence
    severity: str                 # "info" | "warning" | "correction"
    reliable: bool                # based on visible landmarks
    confidence_ok: bool           # prediction confidence above threshold


class FeedbackEngine:
    """
    Stateful engine that:
      • converts evaluation + prediction → FeedbackItems
      • deduplicates: does not enqueue voice speech for a repeated message
        until cooldown_seconds have passed
      • exposes feedback_text() for HUD overlay and speech_queue for TTS
    """

    def __init__(self, cooldown_seconds: float = 12.0, max_display: int = 4):
        self._cooldown = cooldown_seconds
        self._max_display = max_display
        self._last_spoken: dict[str, float] = {}   # msg → timestamp
        self._speech_queue: list[str] = []
        self._current_items: list[FeedbackItem] = []

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        prediction: PosePrediction,
        evaluation: Optional[PostureEvaluation],
        target_pose: Optional[str] = None,
    ) -> list[FeedbackItem]:
        """
        Process latest prediction + evaluation.
        If target_pose is specified, guides the user toward that specific pose.
        Returns a list of FeedbackItems for the current frame.
        """
        items: list[FeedbackItem] = []

        # ── No person ─────────────────────────────────────────────────────────
        if prediction.status == "no_person":
            items.append(FeedbackItem(
                pose_name="",
                body_part="",
                issue="no person",
                instruction="No person detected. Please stand in front of the camera.",
                severity="info",
                reliable=False,
                confidence_ok=False,
            ))
            self._enqueue_speech(items[-1].instruction)
            self._current_items = items
            return items

        # ── Low visibility ────────────────────────────────────────────────────
        if evaluation and evaluation.overall_status == "low_visibility":
            msg = "Some joints are occluded. Adjust position so your full body is visible."
            items.append(FeedbackItem(
                pose_name=target_pose or prediction.pose_name,
                body_part="",
                issue="low visibility",
                instruction=msg,
                severity="warning",
                reliable=False,
                confidence_ok=True,
            ))
            self._enqueue_speech(msg)
            self._current_items = items
            return items

        # ── If in Target Pose Mode, evaluate directly toward target pose ─────
        if target_pose and evaluation and evaluation.has_reference:
            active_pose = target_pose
            corrections = _build_corrections(evaluation)
            if not corrections:
                items.append(FeedbackItem(
                    pose_name=active_pose,
                    body_part="",
                    issue="",
                    instruction=f"Great form! {active_pose} alignment looks correct.",
                    severity="info",
                    reliable=True,
                    confidence_ok=True,
                ))
                self._enqueue_speech(items[-1].instruction)
            else:
                corrections.sort(key=lambda c: (-c.deviation, c.status))
                for check in corrections[: self._max_display]:
                    msg = _check_to_sentence(check)
                    items.append(FeedbackItem(
                        pose_name=active_pose,
                        body_part=check.display_name,
                        issue=check.status,
                        instruction=msg,
                        severity="correction" if check.reliable else "warning",
                        reliable=check.reliable,
                        confidence_ok=True,
                    ))
                primary = items[0].instruction
                self._enqueue_speech(primary)

            self._current_items = items
            return items

        # ── Free Auto-Detect Mode: check low confidence ───────────────────────
        if prediction.status == "low_confidence":
            msg = (
                f"Prediction uncertain ({prediction.confidence*100:.0f}%). "
                "Ensure your full body is visible."
            )
            items.append(FeedbackItem(
                pose_name=prediction.pose_name,
                body_part="",
                issue="low confidence",
                instruction=msg,
                severity="warning",
                reliable=False,
                confidence_ok=False,
            ))
            self._enqueue_speech(msg)
            self._current_items = items
            return items

        # ── No reference ranges ───────────────────────────────────────────────
        if evaluation is None or not evaluation.has_reference:
            items.append(FeedbackItem(
                pose_name=prediction.pose_name,
                body_part="",
                issue="no reference",
                instruction=f"Pose recognised: {prediction.pose_name}. No correction rules configured yet.",
                severity="info",
                reliable=True,
                confidence_ok=True,
            ))
            self._current_items = items
            return items

        # ── Generate per-angle corrections ────────────────────────────────────
        corrections = _build_corrections(evaluation)

        if not corrections:
            items.append(FeedbackItem(
                pose_name=prediction.pose_name,
                body_part="",
                issue="",
                instruction=f"Good form! {prediction.pose_name} alignment looks correct.",
                severity="info",
                reliable=True,
                confidence_ok=True,
            ))
            self._enqueue_speech(items[-1].instruction)
        else:
            # Sort: most deviant first, then unavailable joints
            corrections.sort(key=lambda c: (-c.deviation, c.status))
            for check in corrections[: self._max_display]:
                msg = _check_to_sentence(check)
                items.append(FeedbackItem(
                    pose_name=prediction.pose_name,
                    body_part=check.display_name,
                    issue=check.status,
                    instruction=msg,
                    severity="correction" if check.reliable else "warning",
                    reliable=check.reliable,
                    confidence_ok=True,
                ))
            # Speak only the top-priority correction
            primary = items[0].instruction
            self._enqueue_speech(primary)

        self._current_items = items
        return items

    # ── Speech queue access ───────────────────────────────────────────────────

    def pop_speech(self) -> Optional[str]:
        """Pop and return the next message to speak, or None."""
        return self._speech_queue.pop(0) if self._speech_queue else None

    @property
    def current_items(self) -> list[FeedbackItem]:
        return self._current_items

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enqueue_speech(self, msg: str) -> None:
        now = time.monotonic()
        last = self._last_spoken.get(msg, 0.0)
        if now - last >= self._cooldown:
            self._speech_queue.append(msg)
            self._last_spoken[msg] = now

    def clear(self) -> None:
        self._current_items = []
        self._speech_queue = []
        self._last_spoken = {}


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_corrections(ev: PostureEvaluation) -> list[AngleCheck]:
    return [c for c in ev.checks if c.status != "ok"]


def _check_to_sentence(check: AngleCheck) -> str:
    if check.status == "unavailable":
        return f"Your {check.display_name} is not clearly visible; please adjust position."
    low, high = check.target_range
    val_str = f"{check.value:.1f}°" if check.value is not None else "?"
    if check.status == "too_low":
        return (
            f"Your {check.display_name} angle is {val_str} — straighten it "
            f"by about {check.deviation:.0f}° (target {low:.0f}°–{high:.0f}°)."
        )
    return (
        f"Your {check.display_name} angle is {val_str} — bend it "
        f"by about {check.deviation:.0f}° more (target {low:.0f}°–{high:.0f}°)."
    )
