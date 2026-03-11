"""
09_evaluate.py — Full evaluation on LOCKED test set
Only run this once — after training is complete and frozen.
Produces final metrics report with confusion matrix.
Uses LabelEncoder mapping from metadata to properly decode predictions.
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_score,
    recall_score, log_loss, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEST_PARQUET, BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH,
    THRESHOLDS_PATH, METADATA_PATH, REPORT_DIR, LOG_DIR,
    FEATURE_NAMES, ATTACK_CLASSES, N_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "09_evaluate.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def evaluate():
    log.info(f"Loading LOCKED test set: {TEST_PARQUET}")
    log.info("WARNING: This consumes the locked test set — only run after training is final")
    df_test = pd.read_parquet(TEST_PARQUET)
    X_test  = df_test[FEATURE_NAMES].values.astype(np.float32)
    y_bin   = df_test["label_binary"].values.astype(int)
    y_mc    = df_test["label_14class"].values.astype(int)

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    # ── Load label mapping from metadata ──────────────────────────────────
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    label_mapping = metadata.get("multi_metrics", {}).get("label_mapping", [])
    if label_mapping:
        col_to_class = {i: cls for i, cls in enumerate(label_mapping)}
        log.info(f"Loaded label mapping: {col_to_class}")
    else:
        col_to_class = None
        log.warning("No label_mapping in metadata — using raw model predictions")

    # ── Binary evaluation ─────────────────────────────────────────────────
    log.info("\nBINARY CLASSIFIER EVALUATION")
    binary_model = xgb.XGBClassifier()
    binary_model.load_model(str(BINARY_MODEL_PATH))
    bin_probs = binary_model.predict_proba(X_test)[:, 1]
    bin_thresh = thresholds["binary"]
    y_bin_pred = (bin_probs >= bin_thresh).astype(int)

    tp = ((y_bin_pred == 1) & (y_bin == 1)).sum()
    fp = ((y_bin_pred == 1) & (y_bin == 0)).sum()
    tn = ((y_bin_pred == 0) & (y_bin == 0)).sum()
    fn = ((y_bin_pred == 0) & (y_bin == 1)).sum()

    fpr  = fp / max(fp + tn, 1)
    fnr  = fn / max(fn + tp, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    auc  = roc_auc_score(y_bin, bin_probs)

    log.info(f"  Threshold:  {bin_thresh:.2f}")
    log.info(f"  F1:         {f1:.4f}")
    log.info(f"  Precision:  {prec:.4f}")
    log.info(f"  Recall:     {rec:.4f}")
    log.info(f"  ROC-AUC:    {auc:.4f}")
    log.info(f"  FPR:        {fpr:.4f}  {'[PASS]' if fpr < 0.08 else '[FAIL] (>0.08)'}")
    log.info(f"  FNR:        {fnr:.4f}")
    log.info(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    # ── Multiclass evaluation ─────────────────────────────────────────────
    log.info("\nMULTICLASS CLASSIFIER EVALUATION")
    multi_model = xgb.XGBClassifier()
    multi_model.load_model(str(MULTICLASS_MODEL_PATH))
    mc_probs_raw = multi_model.predict_proba(X_test)
    y_mc_pred_encoded = multi_model.predict(X_test)

    # Decode predictions back to original class indices using label mapping
    if col_to_class is not None:
        y_mc_pred = np.array([col_to_class[int(p)] for p in y_mc_pred_encoded])
        log.info(f"  Decoded predictions: encoded {sorted(set(y_mc_pred_encoded))} -> original {sorted(set(y_mc_pred))}")
        
        # Also remap probabilities to align columns with original class indices
        # mc_probs_raw columns correspond to encoded indices [0, 1, 2, ...] 
        # which map to original classes via label_mapping
        n_samples = len(X_test)
        max_class_idx = max(label_mapping) + 1
        mc_probs = np.zeros((n_samples, max_class_idx), dtype=np.float32)
        for encoded_idx, original_cls in col_to_class.items():
            mc_probs[:, original_cls] = mc_probs_raw[:, encoded_idx]
    else:
        y_mc_pred = y_mc_pred_encoded
        mc_probs = mc_probs_raw

    f1_macro  = f1_score(y_mc, y_mc_pred, average="macro",    zero_division=0)
    f1_wtd    = f1_score(y_mc, y_mc_pred, average="weighted", zero_division=0)

    # For logloss, use only columns/labels present in both truth and mapping
    classes_in_test = sorted(set(y_mc))
    if col_to_class is not None:
        # Compute logloss using encoded space for consistency
        classes_in_mapping = set(label_mapping)
        classes_for_ll = sorted(set(classes_in_test) & classes_in_mapping)
        if classes_for_ll:
            mask = np.isin(y_mc, classes_for_ll)
            y_mc_masked = y_mc[mask]
            mc_probs_masked = mc_probs_raw[mask]
            from sklearn.preprocessing import LabelEncoder
            le_ll = LabelEncoder()
            le_ll.classes_ = np.array(label_mapping)
            y_mc_encoded_for_ll = le_ll.transform(y_mc_masked)
            ll = log_loss(y_mc_encoded_for_ll, mc_probs_masked)
        else:
            ll = float('nan')
    else:
        ll = log_loss(y_mc, mc_probs)

    try:
        auc_mc = roc_auc_score(y_mc, mc_probs[:, classes_in_test],
                               multi_class="ovr", average="macro",
                               labels=classes_in_test)
    except Exception:
        auc_mc = 0.0

    log.info(f"  F1 macro:    {f1_macro:.4f}")
    log.info(f"  F1 weighted: {f1_wtd:.4f}")
    log.info(f"  LogLoss:     {ll:.4f}")
    log.info(f"  ROC-AUC:     {auc_mc:.4f}")

    class_names = [ATTACK_CLASSES.get(c, f"c{c}") for c in classes_in_test]
    log.info(f"\nPer-class metrics (test set):")
    f1_per = f1_score(y_mc, y_mc_pred, average=None, zero_division=0,
                      labels=classes_in_test)
    prec_per = precision_score(y_mc, y_mc_pred, average=None, zero_division=0,
                               labels=classes_in_test)
    rec_per = recall_score(y_mc, y_mc_pred, average=None, zero_division=0,
                           labels=classes_in_test)
    for i, cls in enumerate(classes_in_test):
        name  = ATTACK_CLASSES.get(cls, f"class_{cls}")
        count = (y_mc == cls).sum()
        log.info(f"  {cls:2d} {name:<25} F1={f1_per[i]:.3f}  P={prec_per[i]:.3f}  R={rec_per[i]:.3f}  n={count:>4}")

    # ── Confusion matrix plot ─────────────────────────────────────────────
    cm = confusion_matrix(y_mc, y_mc_pred, labels=classes_in_test)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(f"CSIC Expert  - Confusion Matrix (Test Set)\nF1 macro={f1_macro:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"\nSaved confusion matrix: {cm_path}")

    # ── Save evaluation report ────────────────────────────────────────────
    report = {
        "binary": {
            "threshold": float(bin_thresh),
            "f1": float(f1), "precision": float(prec), "recall": float(rec),
            "roc_auc": float(auc), "fpr": float(fpr), "fnr": float(fnr),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        },
        "multiclass": {
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_wtd),
            "logloss": float(ll),
            "roc_auc": float(auc_mc),
            "per_class_f1": {
                ATTACK_CLASSES.get(cls, f"c{cls}"): float(f1_per[i])
                for i, cls in enumerate(classes_in_test)
            },
            "per_class_precision": {
                ATTACK_CLASSES.get(cls, f"c{cls}"): float(prec_per[i])
                for i, cls in enumerate(classes_in_test)
            },
            "per_class_recall": {
                ATTACK_CLASSES.get(cls, f"c{cls}"): float(rec_per[i])
                for i, cls in enumerate(classes_in_test)
            },
        },
    }
    report_path = REPORT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved evaluation report: {report_path}")

    # ── Print classification report ───────────────────────────────────────
    log.info("\nClassification report (multiclass):")
    log.info(classification_report(
        y_mc, y_mc_pred,
        labels=classes_in_test,
        target_names=class_names,
        zero_division=0,
    ))

    return report


if __name__ == "__main__":
    evaluate()
    log.info("Step 09 COMPLETE")
