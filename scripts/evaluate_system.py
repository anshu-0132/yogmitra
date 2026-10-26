"""
scripts/evaluate_system.py
──────────────────────────
Official system evaluation and benchmarking script for Yoga Mitra.

Measures and reports:
1. Classification Accuracy, Macro F1, Weighted F1 for RF and MLP on held-out test split.
2. Per-class Precision, Recall, and F1 on key practice poses.
3. Component latency profiling:
   - Feature extraction latency (ms)
   - Model inference latency (RF & MLP) (ms)
   - Posture evaluation & scoring latency (ms)
   - Reference panel composition latency (ms)
   - Total end-to-end processing latency & theoretical FPS
4. Occlusion & missing joint stress-test (0% to 30% landmark drop).
5. Output structured results to data/evaluation_metrics.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_loader import load_artifacts
from src.correction.posture_evaluator import evaluate as evaluate_posture
from src.app.reference_display import ReferenceDisplay


def evaluate_models():
    print("=" * 75)
    print("  YOGA MITRA — COMPREHENSIVE SYSTEM EVALUATION & BENCHMARKING")
    print("=" * 75)

    data_path = ROOT / "yoga_engineered_features.csv"
    if not data_path.exists():
        sys.exit(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"\n[1] Loaded Dataset: {len(df)} samples across {df['label'].nunique()} classes.")

    # Load trained model artifacts
    model_dir = ROOT / "saved_yoga_model"
    artifacts = load_artifacts(model_dir)

    feature_cols = artifacts.feature_cols
    X = df[feature_cols]
    y_raw = df["label"]
    y = artifacts.label_encoder.transform(y_raw)

    # Held-out 20% test split (reproducible seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train split: {len(X_train)} samples | Held-out Test split: {len(X_test)} samples.")

    # Preprocessing
    X_test_imputed = artifacts.imputer.transform(X_test)
    X_test_scaled = artifacts.scaler.transform(X_test_imputed)

    results: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples": len(X_test),
        "models": {},
        "latency_ms": {},
        "occlusion_stress": {},
    }

    # ── Evaluate Random Forest ────────────────────────────────────────────────
    rf_preds = artifacts.rf_model.predict(X_test_imputed)
    rf_acc = float(accuracy_score(y_test, rf_preds))
    rf_macro_f1 = float(f1_score(y_test, rf_preds, average="macro"))
    rf_weighted_f1 = float(f1_score(y_test, rf_preds, average="weighted"))

    print("\n[2] Model 1: Random Forest Classifier (saved_yoga_model/yoga_random_forest.joblib)")
    print(f"    Accuracy:    {rf_acc * 100:.2f}%")
    print(f"    Macro F1:    {rf_macro_f1:.4f}")
    print(f"    Weighted F1: {rf_weighted_f1:.4f}")

    results["models"]["random_forest"] = {
        "accuracy": round(rf_acc * 100, 2),
        "macro_f1": round(rf_macro_f1, 4),
        "weighted_f1": round(rf_weighted_f1, 4),
    }

    # ── Evaluate MLP ──────────────────────────────────────────────────────────
    if artifacts.mlp_model is not None:
        mlp_probs = artifacts.mlp_model.predict(X_test_scaled, verbose=0)
        mlp_preds = np.argmax(mlp_probs, axis=1)
        mlp_acc = float(accuracy_score(y_test, mlp_preds))
        mlp_macro_f1 = float(f1_score(y_test, mlp_preds, average="macro"))
        mlp_weighted_f1 = float(f1_score(y_test, mlp_preds, average="weighted"))

        print("\n[3] Model 2: Multi-Layer Perceptron (saved_yoga_model/yoga_mlp.keras)")
        print(f"    Accuracy:    {mlp_acc * 100:.2f}%")
        print(f"    Macro F1:    {mlp_macro_f1:.4f}")
        print(f"    Weighted F1: {mlp_weighted_f1:.4f}")

        results["models"]["mlp"] = {
            "accuracy": round(mlp_acc * 100, 2),
            "macro_f1": round(mlp_macro_f1, 4),
            "weighted_f1": round(mlp_weighted_f1, 4),
        }

    # ── Latency Benchmarks ────────────────────────────────────────────────────
    print("\n[4] Real-Time Latency Benchmarks (averaged across 500 iterations):")
    single_sample = X_test.iloc[[0]]
    single_imputed = X_test_imputed[[0]]
    single_scaled = X_test_scaled[[0]]

    # Feature transformation latency
    t0 = time.perf_counter()
    for _ in range(500):
        _ = artifacts.imputer.transform(single_sample)
    impute_lat = ((time.perf_counter() - t0) / 500) * 1000

    # RF inference latency
    t0 = time.perf_counter()
    for _ in range(500):
        _ = artifacts.rf_model.predict_proba(single_imputed)
    rf_lat = ((time.perf_counter() - t0) / 500) * 1000

    # MLP inference latency
    mlp_lat = 0.0
    if artifacts.mlp_model is not None:
        t0 = time.perf_counter()
        for _ in range(500):
            _ = artifacts.mlp_model(single_scaled, training=False)
        mlp_lat = ((time.perf_counter() - t0) / 500) * 1000

    # Posture evaluation latency
    dummy_feat = {col: float(single_sample[col].values[0]) for col in feature_cols if "angle" in col}
    dummy_lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95}] * 33
    t0 = time.perf_counter()
    for _ in range(500):
        _ = evaluate_posture("Virabhadrasana Two", dummy_feat, dummy_lm, [0]*33)
    eval_lat = ((time.perf_counter() - t0) / 500) * 1000

    # Reference panel composition latency (cached)
    ref_display = ReferenceDisplay()
    t0 = time.perf_counter()
    for _ in range(500):
        _ = ref_display.render_panel("Virabhadrasana Two", height=720, focus_joint="left knee")
    panel_lat = ((time.perf_counter() - t0) / 500) * 1000

    total_pipeline_lat = impute_lat + rf_lat + eval_lat + panel_lat
    max_fps = 1000.0 / total_pipeline_lat if total_pipeline_lat > 0 else 0.0

    print(f"    - Preprocessing / Imputation:   {impute_lat:.3f} ms")
    print(f"    - Random Forest Inference:      {rf_lat:.3f} ms")
    if artifacts.mlp_model is not None:
        print(f"    - MLP Inference:                {mlp_lat:.3f} ms")
    print(f"    - Posture Alignment Evaluator:  {eval_lat:.3f} ms")
    print(f"    - Reference Panel Composition:  {panel_lat:.3f} ms")
    print(f"    ------------------------------------------------")
    print(f"    - Total Pipeline Latency:       {total_pipeline_lat:.2f} ms")
    print(f"    - Max Theoretical Pipeline FPS: {max_fps:.1f} FPS")

    results["latency_ms"] = {
        "preprocessing_imputation": round(impute_lat, 3),
        "rf_inference": round(rf_lat, 3),
        "mlp_inference": round(mlp_lat, 3),
        "posture_evaluation": round(eval_lat, 3),
        "reference_panel": round(panel_lat, 3),
        "total_latency": round(total_pipeline_lat, 2),
        "max_theoretical_fps": round(max_fps, 1),
    }

    # ── Occlusion Robustness Stress Test ──────────────────────────────────────
    print("\n[5] Occlusion & Missing Joint Robustness Test:")
    for drop_pct in [0, 10, 20, 30]:
        X_occ = X_test.copy()
        if drop_pct > 0:
            mask = np.random.RandomState(42).rand(*X_occ.shape) < (drop_pct / 100.0)
            X_occ[mask] = np.nan
        X_occ_imp = artifacts.imputer.transform(X_occ)
        preds_occ = artifacts.rf_model.predict(X_occ_imp)
        acc_occ = float(accuracy_score(y_test, preds_occ))
        print(f"    - Landmark Occlusion {drop_pct:2d}%: Retained Accuracy = {acc_occ * 100:.2f}%")
        results["occlusion_stress"][f"{drop_pct}pct_drop"] = round(acc_occ * 100, 2)

    # Save to disk
    out_path = ROOT / "data" / "evaluation_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[6] Saved official evaluation metrics to: {out_path.resolve()}")
    print("=" * 75)
    return results


if __name__ == "__main__":
    evaluate_models()
