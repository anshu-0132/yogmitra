"""
src/app/voice.py
──────────────────
Asynchronous text-to-speech using pyttsx3.

Runs in a daemon thread so it never blocks webcam capture or inference.
Messages are dequeued one at a time; overlapping speech is avoided by
finishing the current sentence before starting the next.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional


class VoiceFeedback:
    """Thread-safe wrapper around pyttsx3."""

    def __init__(self, rate: int = 155, volume: float = 0.9):
        self._enabled = False
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._engine = None

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
            self._enabled = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        except Exception as exc:
            print(f"[Voice] pyttsx3 not available ({exc}). Voice feedback disabled.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> bool:
        """Toggle voice on/off. Returns new state."""
        self._enabled = not self._enabled
        if not self._enabled:
            # Drain queue so pending messages don't play after re-enable
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
        return self._enabled

    def speak(self, message: str) -> None:
        """Queue a message. Drops silently if voice is disabled."""
        if self._enabled and self._engine is not None:
            self._q.put(message)

    def shutdown(self) -> None:
        """Signal the worker thread to exit."""
        self._q.put(None)  # sentinel

    def _worker(self) -> None:
        while True:
            msg = self._q.get()
            if msg is None:
                break
            if self._enabled and self._engine is not None:
                try:
                    self._engine.say(msg)
                    self._engine.runAndWait()
                except Exception as exc:
                    print(f"[Voice] TTS error: {exc}")
