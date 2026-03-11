"""
08_threshold_tune.py — Per-class threshold optimization on validation set
Find the optimal confidence threshold for each attack class
that maximizes F1 while keeping FPR below target.
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VAL_PARQUET, BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH,
    THRESHOLDS_PATH, METADATA_PATH, LOG_DIR, FEATURE_NAMES,
    ATTACK_CLASSES, BINARY_THRESHOLD_DEFAULT,
    THRESHOLD_SEARCH_RANGE, THRESHOLD_STEP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "08_threshold_tune.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def tune_thresholds() -> dict:
    log.info(f"Loading val data: {VAL_PARQUET}")
    df_val = pd.read_parquet(VAL_PARQUET)
    X_val  = df_val[FEATURE_NAMES].values.astype(np.float32)
    y_bin  = df_val["label_binary"].values.astype(int)
    y_mc   = df_val["label_14class"].values.astype(int)

    # ── Binary threshold ──────────────────────────────────────────────────
    log.info("Loading binary model...")
    binary_model = xgb.XGBClassifier()
    binary_model.load_model(str(BINARY_MODEL_PATH))
    bin_probs = binary_model.predict_proba(X_val)[:, 1]

    log.info("Tuning binary threshold (Constrained optimization FPR <= 0.08)...")
    best_bin_t = 0.5
    best_bin_recall = 0.0
    best_bin_f1 = 0.0
    
    t_range = np.arange(0.05, 0.95, 0.01)
    bin_results = []
    
    for t in t_range:
        preds = (bin_probs >= t).astype(int)
        tn = ((preds == 0) & (y_bin == 0)).sum()
        fp = ((preds == 1) & (y_bin == 0)).sum()
        fn = ((preds == 0) & (y_bin == 1)).sum()
        tp = ((preds == 1) & (y_bin == 1)).sum()

        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)
        prec_t = precision_score(y_bin, preds, zero_division=0)
        f1 = f1_score(y_bin, preds, zero_division=0)
        
        bin_results.append((t, f1, fpr, recall, prec_t))
        
        # Constraint: FPR must be under 0.08, maximize recall
        if fpr <= 0.08 and recall > best_bin_recall:
            best_bin_recall = recall
            best_bin_t = t
            best_bin_f1 = f1

    # Log top 5 threshold candidates meeting constraint
    valid_results = [r for r in bin_results if r[2] <= 0.08]
    valid_results = sorted(valid_results, key=lambda x: x[3], reverse=True)[:5]
    log.info("  Top 5 threshold candidates (by Recall, FPR <= 0.08):")
    for t, f1, fpr, recall, prec_t in valid_results:
        log.info(f"    t={t:.2f}  F1={f1:.3f}  FPR={fpr:.3f}  Recall={recall:.3f}  Precision={prec_t:.3f}")

    if best_bin_recall == 0.0:
        log.warning("No threshold found with FPR <= 0.08. Defaulting to 0.5")
        best_bin_t = 0.5
        best_bin_f1 = f1_score(y_bin, (bin_probs >= 0.5).astype(int), zero_division=0)

    log.info(f"Binary threshold: {best_bin_t:.2f}  F1={best_bin_f1:.4f}  Recall={best_bin_recall:.4f}")

    # ── Per-class multiclass thresholds ──────────────────────────────────
    log.info("\nLoading multiclass model...")
    multi_model = xgb.XGBClassifier()
    multi_model.load_model(str(MULTICLASS_MODEL_PATH))
    mc_probs = multi_model.predict_proba(X_val)  # shape: (n, n_classes_present)

    # Load label mapping from metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    mapping = metadata.get("multi_metrics", {}).get("label_mapping", [])
    
    if not mapping:
        log.warning("No label_mapping found in metadata, assuming direct index mapping.")
        mapping = sorted(ATTACK_CLASSES.keys())

    # Map class index to column index in mc_probs
    class_to_col = {cls: i for i, cls in enumerate(mapping)}

    log.info("Tuning per-class thresholds...")
    class_thresholds = {}
    for cls_idx_str, cls_name in ATTACK_CLASSES.items():
        cls_idx = int(cls_idx_str)
        if cls_idx == 0:
            class_thresholds[cls_name] = 0.5
            continue

        if cls_idx not in class_to_col:
            log.warning(f"  {cls_name} (idx {cls_idx}) was not in training - skipping threshold tune.")
            class_thresholds[cls_name] = 0.5
            continue

        col_idx = class_to_col[cls_idx]
        true_binary = (y_mc == cls_idx).astype(int)
        n_positive  = true_binary.sum()

        if n_positive < 5:
            log.warning(f"  {cls_name}: only {n_positive} val samples — using default 0.5")
            class_thresholds[cls_name] = 0.5
            continue

        cls_probs     = mc_probs[:, col_idx]
        best_t, best_f1 = 0.5, 0.0
        for t in t_range:
            preds = (cls_probs >= t).astype(int)
            f1    = f1_score(true_binary, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        class_thresholds[cls_name] = float(best_t)
        prec = precision_score(true_binary, (cls_probs >= best_t).astype(int), zero_division=0)
        rec  = recall_score(true_binary, (cls_probs >= best_t).astype(int), zero_division=0)
        log.info(f"  {cls_name:<25} t={best_t:.2f}  F1={best_f1:.3f}  P={prec:.3f}  R={rec:.3f}  n={n_positive}")

    thresholds = {
        "binary": float(best_bin_t),
        "binary_f1": float(best_bin_f1),
        "multiclass": class_thresholds,
    }

    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"\nSaved thresholds: {THRESHOLDS_PATH}")
    return thresholds


if __name__ == "__main__":
    tune_thresholds()
    log.info("Step 08 COMPLETE")
