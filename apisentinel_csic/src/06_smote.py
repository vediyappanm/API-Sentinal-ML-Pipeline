"""
06_smote.py — SMOTE class balancing on training set ONLY
CRITICAL: Never apply SMOTE to validation or test sets.
Only upsample classes below SMOTE_MIN_SAMPLES threshold.
"""

import sys
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_PARQUET, TRAIN_BAL_PARQUET, LOG_DIR,
    FEATURE_NAMES, SMOTE_MIN_SAMPLES, SMOTE_TARGET, RANDOM_STATE, ATTACK_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "06_smote.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def apply_smote(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        log.info(f"Loading: {TRAIN_PARQUET}")
        df = pd.read_parquet(TRAIN_PARQUET)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label_14class"].values.astype(int)

    class_counts = Counter(y)
    log.info("Class counts before SMOTE:")
    for cls, count in sorted(class_counts.items()):
        name = ATTACK_CLASSES.get(cls, f"class_{cls}")
        log.info(f"  {cls:2d} {name:<25} {count:>6,}")

    # Build SMOTE sampling strategy — only upsample rare classes
    sampling_strategy = {}
    for cls, count in class_counts.items():
        if count < SMOTE_MIN_SAMPLES:
            target = min(SMOTE_TARGET, count * 5)  # cap at 5x original
            sampling_strategy[cls] = target
            name = ATTACK_CLASSES.get(cls, f"class_{cls}")
            log.info(f"  SMOTE: {name} {count} -> {target}")

    if not sampling_strategy:
        log.info("No classes below threshold — skipping SMOTE")
        df.to_parquet(TRAIN_BAL_PARQUET, index=False, engine="pyarrow")
        return df

    # k_neighbors must be < smallest minority class count
    min_count  = min(count for cls, count in class_counts.items()
                     if cls in sampling_strategy)
    k_neighbors = min(5, min_count - 1)

    if k_neighbors < 1:
        log.warning(f"Smallest class has only {min_count} samples — skipping SMOTE for very rare classes")
        # Remove classes too small for SMOTE
        sampling_strategy = {
            cls: target for cls, target in sampling_strategy.items()
            if class_counts[cls] >= 2
        }
        if not sampling_strategy:
            log.warning("No classes eligible for SMOTE after safety check")
            df.to_parquet(TRAIN_BAL_PARQUET, index=False, engine="pyarrow")
            return df
        k_neighbors = 1

    log.info(f"Applying SMOTE with k_neighbors={k_neighbors}")
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE,
    )

    X_bal, y_bal = smote.fit_resample(X, y)

    # Reconstruct dataframe
    # Separate original from synthetic rows to preserve metadata
    n_original = len(df)
    n_synthetic = len(X_bal) - n_original

    # Final combined dataframe
    df_synthetic = pd.DataFrame(X_bal[n_original:], columns=FEATURE_NAMES)
    df_synthetic["label_14class"] = y_bal[n_original:]
    df_synthetic["label_binary"]  = (y_bal[n_original:] != 0).astype(int)
    df_synthetic["label_raw"]     = "SYNTHETIC"
    df_synthetic["Method"]        = "SYNTHETIC"
    df_synthetic["URL"]           = ""

    df_bal = pd.concat([df, df_synthetic], ignore_index=True)

    # Ensure metadata columns are strings to avoid Arrow type conversion errors
    for col in ["label_raw", "Method", "URL"]:
        if col in df_bal.columns:
            df_bal[col] = df_bal[col].astype(str)

    log.info(f"\nSMOTE results:")
    log.info(f"  Original samples:  {n_original:,}")
    log.info(f"  Synthetic samples: {n_synthetic:,}")
    log.info(f"  Total:             {len(df_bal):,}")

    after_counts = Counter(y_bal)
    log.info("\nClass counts after SMOTE:")
    for cls, count in sorted(after_counts.items()):
        name = ATTACK_CLASSES.get(cls, f"class_{cls}")
        before = class_counts.get(cls, 0)
        log.info(f"  {cls:2d} {name:<25} {before:>6,} -> {count:>6,}")

    df_bal.to_parquet(TRAIN_BAL_PARQUET, index=False, engine="pyarrow")
    log.info(f"\nSaved balanced training data: {TRAIN_BAL_PARQUET}")
    return df_bal


if __name__ == "__main__":
    apply_smote()
    log.info("Step 06 COMPLETE")
