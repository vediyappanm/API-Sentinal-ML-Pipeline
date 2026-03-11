"""
07_train.py — XGBoost training: binary + multiclass models
5-fold stratified cross-validation with early stopping.
Saves both models in XGBoost JSON format (portable + ONNX-compatible).
"""

import sys
import json
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, log_loss
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_BAL_PARQUET, VAL_PARQUET,
    BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH,
    FEATURE_NAMES_PATH, METADATA_PATH,
    LOG_DIR, FEATURE_NAMES,
    BINARY_XGB_PARAMS, MULTICLASS_XGB_PARAMS,
    CV_FOLDS, EARLY_STOPPING, RANDOM_STATE,
    ATTACK_CLASSES, N_CLASSES,
    EXPERT_ID, EXPERT_NAME, EXPERT_VERSION, DATASET_URL,
    STRONG_CLASSES, WEAK_CLASSES, TRUST_WEIGHT,
)

from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "07_train.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _get_Xy(df: pd.DataFrame, label_col: str):
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[label_col].values.astype(int)
    return X, y


def cross_validate(X: np.ndarray, y: np.ndarray,
                   params: dict, label_col_name: str) -> dict:
    """Run CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = {"f1_macro": [], "auc": [], "logloss": []}

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )

        y_pred = model.predict(X_vl)
        y_prob = model.predict_proba(X_vl)
        
        # Use macro for multi, binary for binary
        avg_mode = "macro" if params.get("objective") != "binary:logistic" else "binary"
        f1 = f1_score(y_vl, y_pred, average=avg_mode, zero_division=0)
        ll = log_loss(y_vl, y_prob)

        if params.get("objective") == "binary:logistic":
            auc = roc_auc_score(y_vl, y_prob[:, 1])
        else:
            try:
                auc = roc_auc_score(y_vl, y_prob, multi_class="ovr", average="macro")
            except Exception:
                auc = 0.0

        scores["f1_macro"].append(f1)
        scores["auc"].append(auc)
        scores["logloss"].append(ll)

        log.info(f"  Fold {fold}/{CV_FOLDS}: F1={f1:.4f} AUC={auc:.4f}")

    means = {k: float(np.mean(v)) for k, v in scores.items()}
    stds = {k: float(np.std(v)) for k, v in scores.items()}
    log.info(f"  CV Mean: F1={means['f1_macro']:.4f} AUC={means['auc']:.4f}")
    return means


def train_binary(X_train, y_train_bin, X_val, y_val_bin) -> xgb.XGBClassifier:
    log.info(f"\n{'='*60}")
    log.info("TRAINING BINARY CLASSIFIER")
    log.info(f"{'='*60}")
    log.info(f"Train: {len(X_train):,}  (normal={( y_train_bin==0).sum():,} attack={(y_train_bin==1).sum():,})")
    log.info(f"Val:   {len(X_val):,}")
    log.info(f"Params: {BINARY_XGB_PARAMS}")

    log.info("\nRunning 5-fold CV...")
    cv_scores = cross_validate(X_train, y_train_bin, BINARY_XGB_PARAMS, "binary")

    log.info("\nTraining final binary model on full training set...")
    t0    = time.time()
    model = xgb.XGBClassifier(**BINARY_XGB_PARAMS)
    model.fit(
        X_train, y_train_bin,
        eval_set=[(X_val, y_val_bin)],
        verbose=50,
    )
    elapsed = time.time() - t0
    log.info(f"Training time: {elapsed:.1f}s")

    # Final val metrics
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    f1     = f1_score(y_val_bin, y_pred, average="binary", zero_division=0)
    auc    = roc_auc_score(y_val_bin, y_prob)
    fpr    = ((y_pred == 1) & (y_val_bin == 0)).sum() / (y_val_bin == 0).sum()

    log.info(f"\nVal metrics (binary):")
    log.info(f"  F1:         {f1:.4f}")
    log.info(f"  ROC-AUC:    {auc:.4f}")
    log.info(f"  FPR:        {fpr:.4f}  <- target <0.08")
    log.info(f"  CV F1:      {cv_scores['f1_macro']:.4f}")

    if fpr > 0.15:
        log.warning(f"  HIGH FPR: {fpr:.3f} - consider scaling")

    model.save_model(str(BINARY_MODEL_PATH))
    log.info(f"Saved binary model: {BINARY_MODEL_PATH}")
    return model, {"f1": f1, "auc": auc, "fpr": fpr, "cv_f1": cv_scores["f1_macro"]}


def train_multiclass(X_train, y_train_mc, X_val, y_val_mc) -> xgb.XGBClassifier:
    log.info(f"\n{'='*60}")
    log.info("TRAINING MULTICLASS CLASSIFIER (14 classes)")
    log.info(f"{'='*60}")

    # Log class distribution
    from collections import Counter
    dist = Counter(y_train_mc)
    log.info("Train class distribution:")
    for cls in sorted(dist.keys()):
        name = ATTACK_CLASSES.get(cls, f"class_{cls}")
        log.info(f"  {cls:2d} {name:<25} {dist[cls]:>6,}")

    # Handle non-contiguous labels with LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_mc)
    y_val_encoded = le.transform(y_val_mc)
    
    mc_params = MULTICLASS_XGB_PARAMS.copy()
    mc_params["num_class"] = len(le.classes_)
    
    log.info(f"Using LabelEncoder: {len(le.classes_)} classes present in training.")
    log.info(f"Mapping: {dict(enumerate(le.classes_))}")

    log.info("\nRunning 5-fold CV...")
    cv_scores = cross_validate(X_train, y_train_encoded, mc_params, "multiclass")

    log.info("\nTraining final multiclass model...")
    t0 = time.time()
    model = xgb.XGBClassifier(**mc_params)
    model.fit(
        X_train, y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=100,
    )
    elapsed = time.time() - t0
    log.info(f"Training time: {elapsed:.1f}s")

    # Final val metrics
    y_pred_encoded = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    y_pred = le.inverse_transform(y_pred_encoded)

    f1_mac = f1_score(y_val_encoded, y_pred_encoded, average="macro", zero_division=0)
    f1_wtd = f1_score(y_val_mc, y_pred, average="weighted", zero_division=0)
    ll     = log_loss(y_val_encoded, y_prob)

    log.info(f"\nVal metrics (multiclass):")
    log.info(f"  F1 macro:    {f1_mac:.4f}")
    log.info(f"  F1 weighted: {f1_wtd:.4f}")
    log.info(f"  LogLoss:     {ll:.4f}")
    log.info(f"  CV F1:       {cv_scores['f1_macro']:.4f}")

    # Per-class F1
    log.info("\nPer-class F1 (val set):")
    f1_per = f1_score(y_val_encoded, y_pred_encoded, average=None, zero_division=0)
    
    for i, cls in enumerate(le.classes_):
        name = ATTACK_CLASSES.get(cls, f"class_{cls}")
        count = (y_val_mc == cls).sum()
        f1_val = f1_per[i] if i < len(f1_per) else 0.0
        log.info(f"  {cls:2d} {name:<25} F1={f1_val:.3f}  (n={count})")

    model.save_model(str(MULTICLASS_MODEL_PATH))
    log.info(f"Saved multiclass model: {MULTICLASS_MODEL_PATH}")
    
    metrics = {
        "f1_macro": f1_mac, 
        "f1_weighted": f1_wtd, 
        "logloss": ll,
        "cv_f1": cv_scores["f1_macro"],
        "label_mapping": le.classes_.tolist()
    }
    return model, metrics


def save_metadata(binary_metrics: dict, multi_metrics: dict):
    import datetime
    metadata = {
        "expert_id":       EXPERT_ID,
        "expert_name":     EXPERT_NAME,
        "version":         EXPERT_VERSION,
        "dataset_url":     DATASET_URL,
        "trained_at":      datetime.datetime.utcnow().isoformat() + "Z",
        "feature_names":   FEATURE_NAMES,
        "n_features":      len(FEATURE_NAMES),
        "attack_classes":  ATTACK_CLASSES,
        "n_classes":       N_CLASSES,
        "strong_classes":  STRONG_CLASSES,
        "weak_classes":    WEAK_CLASSES,
        "trust_weight":    TRUST_WEIGHT,
        "binary_params":   BINARY_XGB_PARAMS,
        "multi_params":    MULTICLASS_XGB_PARAMS,
        "binary_metrics":  binary_metrics,
        "multi_metrics":   multi_metrics,
        "notes": [
            "Groups 6+7 (behavioral/time) set to training defaults.",
            "Populate from Redis at production inference time.",
            "CSIC is lab data — real-world FPR may differ.",
            "Sub-class labels are keyword-heuristic derived.",
        ],
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata: {METADATA_PATH}")

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    log.info(f"Saved feature names: {FEATURE_NAMES_PATH}")


def train():
    log.info(f"Loading training data: {TRAIN_BAL_PARQUET}")
    df_train = pd.read_parquet(TRAIN_BAL_PARQUET)
    log.info(f"Loading val data:      {VAL_PARQUET}")
    df_val   = pd.read_parquet(VAL_PARQUET)

    X_train, y_train_bin = _get_Xy(df_train, "label_binary")
    X_val,   y_val_bin   = _get_Xy(df_val,   "label_binary")
    X_train, y_train_mc  = _get_Xy(df_train, "label_14class")
    X_val,   y_val_mc    = _get_Xy(df_val,   "label_14class")

    # ── Binary model ──────────────────────────────────────────────────────
    binary_model, binary_metrics = train_binary(
        X_train, y_train_bin, X_val, y_val_bin
    )

    # ── Multiclass model ──────────────────────────────────────────────────
    multi_model, multi_metrics = train_multiclass(
        X_train, y_train_mc, X_val, y_val_mc
    )

    # ── Save metadata ─────────────────────────────────────────────────────
    save_metadata(binary_metrics, multi_metrics)

    log.info(f"\n{'='*60}")
    log.info("TRAINING COMPLETE SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Binary  — F1: {binary_metrics['f1']:.4f}  AUC: {binary_metrics['auc']:.4f}  FPR: {binary_metrics['fpr']:.4f}")
    log.info(f"Multiclass — F1 macro: {multi_metrics['f1_macro']:.4f}  LogLoss: {multi_metrics['logloss']:.4f}")

    return binary_model, multi_model


if __name__ == "__main__":
    train()
    log.info("Step 07 COMPLETE")
