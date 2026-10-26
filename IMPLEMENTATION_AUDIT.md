# Implementation Audit — Yoga Mitra Posture Correction System

Date: 2026-10-15  
Repository: `e:\Yoga_posture_project`  
Auditor: Senior Python & CV/ML Engineer  

---

## 1. Executive Implementation Audit Table

| Feature | Status | Evidence | Required Action |
|:---|:---:|:---|:---|
| **Voice feedback** | **Broken / Unreliable** | In `src/app/voice.py`, `pyttsx3.init()` was called on the main thread during `__init__`, while `engine.say()` / `engine.runAndWait()` ran inside a secondary daemon worker thread `_worker()`. On Windows SAPI5/COM, cross-thread calls cause silent COM deadlock / hanging. Also, cooldown in `feedback_engine.py` is a static 12s lock with robotic strings and obsolete queued messages are not discarded. | Refactor `VoiceFeedback` to initialize SAPI5 engine inside its dedicated worker thread with proper `pythoncom.CoInitialize()`. Implement dynamic per-error cooldowns, speech queue clearing on posture change/mute, natural human-friendly voice cues, and `--test-voice` diagnostic. |
| **Reference pose image** | **Missing** | `src/app/overlay.py` and `src/app/webcam_app.py` render only a single video stream showing the live camera feed with an OpenCV skeleton. There is no dual-panel layout or persistent reference target image for the user to compare against. | Implement a responsive two-panel UI: Left panel = live webcam with HUD & skeleton; Right panel = reference asana image, title, key alignment cues, and highlighted focus joint. Add `assets/pose_references/` with high-quality illustrations/schematics for all guided targets, and graceful fallback card rendering. |
| **Confidence-aware feedback** | **Partial** | `src/correction/posture_evaluator.py` checks individual landmark `visibility >= 0.30` and `imputed_fraction > 0.25`, but lacks a structured multi-tier reliability model (High, Medium, Low) and temporal jitter / stability tracking. Low visibility merely aborts rather than offering qualified gentle feedback or specific advice to step back/adjust lighting. | Implement 3-tier feedback policy (High: actionable angle deviation; Medium: qualified suggestion; Low: suppress angle math and guide camera positioning). Separate temporal tracking fallback from feedback reliability. |
| **User calibration** | **Missing** | No baseline calibration workflow, limb length measurement, or user profile persistence exists in the repository. Target angles are hardcoded in `src/correction/reference_ranges.py` with no personalization for limb proportions. | Build `src/correction/calibration.py` offering an optional guided calibration flow (e.g. 30 frames in neutral stance/Tadasana). Estimate user limb ratios and torso scaling. Persist profile to `user_profile.json` and load on startup with safe fallback defaults. |
| **Progress tracking** | **Missing** | No session history database or tracking persistence exists in the repository. Sessions end without recording score trajectories, mistakes, or progress. | Implement `src/correction/progress_tracker.py` using SQLite (`data/session_history.db`) to record session timestamps, duration, asana practised, alignment scores, recurring joint errors, and persistent corrections. Expose CLI `--history` summary and clean exit summary. |
| **Joint deviation and severity** | **Partial** | `AngleCheck` in `posture_evaluator.py` calculates degrees outside range and `status` ("too_low" / "too_high" / "ok" / "unavailable"), but lacks `severity` categorization (mild / moderate / major), directional movement phrasing ("bend", "straighten", "raise", "lower"), and an overall alignment score (0–100%). | Extend `AngleCheck` and `PostureEvaluation` to include severity classifications, natural directional verbs, reliable joint weights, and a validated 0–100% alignment score. |
| **Performance evaluation** | **Partial** | `tests/test_pipeline.py` contains 14 unit tests, but there are no automated latency benchmarks, FPS measurements, or an `EVALUATION.md` report documenting held-out model metrics and system benchmarks. | Build an automated benchmark script `evaluate_system.py` measuring model precision/recall/F1, inference latency, camera FPS, and generate `EVALUATION.md`. |

---

## 2. Detailed Root-Cause Analysis for Priority One (Voice)

### Root Cause
1. **Windows SAPI5 COM Thread Affinity**:
   When `pyttsx3.init()` is invoked on the main application thread during `VoiceFeedback.__init__`, the underlying Windows Speech API COM object is bound to that thread. When the daemon thread later calls `engine.runAndWait()`, Windows COM throws an RPC error or indefinitely deadlocks waiting for the main thread message pump.
2. **Verification**:
   Executing `pyttsx3.init()` on the main thread and calling `runAndWait()` inside a `threading.Thread` deadlocks 100% of the time on Windows. Executing `pyttsx3.init()` *inside* the worker thread executes and completes with zero errors.
3. **Queue Invalidation**:
   The existing queue accumulates messages and plays them sequentially. If a user adjusts their posture while a message is queued, the obsolete message still plays seconds later, giving outdated corrections.

---

## 3. Implementation Plan

1. **Priority 1 (Voice)**:
   - Rewrite `src/app/voice.py`: dedicated worker thread initialization, safe lifecycle, queue clearing on mute/posture change.
   - Refactor `feedback_engine.py`: human-like natural phrasing ("Bend front knee slightly", "Straighten rear leg", "Arms up"), intelligent cooldown.
   - Add CLI `--test-voice` in `run_app.py`.
2. **Priority 2 (Reference Asana Two-Panel UI)**:
   - Create `assets/pose_references/` with reference cards for the 8 core asanas.
   - Build dual-panel layout in `src/app/overlay.py` and `src/app/webcam_app.py`.
   - Update target image immediately on `T` / `N`.
3. **Priority 3 (Confidence-Aware Feedback)**:
   - Implement High / Medium / Low feedback policy in `posture_evaluator.py` & `feedback_engine.py`.
4. **Priority 4 (User Calibration)**:
   - Implement `src/correction/calibration.py` with profile saving/loading to `user_profile.json`.
5. **Priority 5 (Joint Deviation & Scoring)**:
   - Add `severity` ("mild", "moderate", "major") and 0–100% overall alignment score.
6. **Priority 6 (Session History & Tracking)**:
   - Implement `src/correction/progress_tracker.py` with SQLite storage in `data/session_history.db`.
7. **Priority 7 (Testing & Evaluation)**:
   - Add comprehensive tests covering all new modules.
   - Create `evaluate_system.py` and generate `EVALUATION.md`.
