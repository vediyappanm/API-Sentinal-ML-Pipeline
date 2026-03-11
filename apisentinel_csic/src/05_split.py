"""
05_split.py — Stratified 70/15/15 train/val/test split
Test set is LOCKED and never used until final evaluation.
Stratification ensures all 14 classes present in all splits.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LABELED_PARQUET, TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET,
    LOG_DIR, TEST_SIZE, VAL_SIZE, RANDOM_STATE, FEATURE_NAMES, ATTACK_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "05_split.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def split(df: pd.DataFrame = None) -> tuple:
    if df is None:
        log.info(f"Loading: {LABELED_PARQUET}")
        df = pd.read_parquet(LABELED_PARQUET)

    y = df["label_14class"].values
    log.info(f"Total samples: {len(df):,}")

    # ── Step 1: Split off test set (15%) ──────────────────────────────────
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    trainval_idx, test_idx = next(sss1.split(df, y))
    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    # ── Step 2: Split train/val from trainval (val = 15% of total) ────────
    # val_size_adjusted: 15% of total = 15/85 of trainval
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    y_trainval = df_trainval["label_14class"].values
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_adjusted, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(sss2.split(df_trainval, y_trainval))
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    # ── Verify stratification ─────────────────────────────────────────────
    log.info(f"\nSplit sizes:")
    log.info(f"  Train: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    log.info(f"  Val:   {len(df_val):,}   ({len(df_val)/len(df)*100:.1f}%)")
    log.info(f"  Test:  {len(df_test):,}  ({len(df_test)/len(df)*100:.1f}%)")

    log.info(f"\nClass distribution across splits:")
    for cls_idx in sorted(df["label_14class"].unique()):
        name  = ATTACK_CLASSES.get(cls_idx, f"class_{cls_idx}")
        tr    = (df_train["label_14class"] == cls_idx).sum()
        vl    = (df_val["label_14class"] == cls_idx).sum()
        te    = (df_test["label_14class"] == cls_idx).sum()
        total = (df["label_14class"] == cls_idx).sum()
        log.info(f"  {name:<25} total={total:>5} | train={tr:>4} val={vl:>3} test={te:>3}")
        if te == 0 and total > 0:
            log.warning(f"    CLASS {name} HAS ZERO TEST SAMPLES — consider merging rare classes")

    # ── Save splits ────────────────────────────────────────────────────────
    df_train.to_parquet(TRAIN_PARQUET, index=False, engine="pyarrow")
    df_val.to_parquet(VAL_PARQUET,   index=False, engine="pyarrow")
    df_test.to_parquet(TEST_PARQUET,  index=False, engine="pyarrow")

    log.info(f"\nSaved splits:")
    log.info(f"  {TRAIN_PARQUET}")
    log.info(f"  {VAL_PARQUET}")
    log.info(f"  {TEST_PARQUET}  ← LOCKED — do not use until final eval")

    return df_train, df_val, df_test


if __name__ == "__main__":
    split()
    log.info("Step 05 COMPLETE")
