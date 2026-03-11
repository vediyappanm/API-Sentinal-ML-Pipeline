"""
04_label_map.py — Map CSIC binary labels to APISentinel 14-class taxonomy
CSIC is binary (normal/anomalous). We sub-classify anomalous rows
using keyword heuristics to assign fine-grained attack type labels.
This is imperfect but produces a usable multi-class dataset.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_PARQUET, LABELED_PARQUET, LOG_DIR,
    ATTACK_RULES, ATTACK_CLASSES, ATTACK_CLASS_INDEX, FEATURE_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "04_label_map.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def assign_attack_type(row: pd.Series) -> int:
    """
    For anomalous rows: determine specific attack type from keyword scores.
    Uses pre-computed keyword scores from feature extraction step.
    Returns APISentinel 14-class index.
    """
    if row["label_binary"] == 0:
        return ATTACK_CLASS_INDEX["benign"]  # 0

    # Check each attack rule in priority order
    for class_idx, score_col in ATTACK_RULES:
        if score_col in row.index and row[score_col] >= 1:
            return class_idx

    # Fallback: if anomalous but no keyword matched → unknown_threat
    return ATTACK_CLASS_INDEX["unknown_threat"]  # 13


def label_map(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        log.info(f"Loading: {FEATURES_PARQUET}")
        df = pd.read_parquet(FEATURES_PARQUET)

    log.info(f"Mapping labels for {len(df):,} rows...")

    df["label_14class"] = df.apply(assign_attack_type, axis=1)

    # ACTION 2: Drop unlearnable unknown_threats
    unknown_idx = ATTACK_CLASS_INDEX["unknown_threat"]
    unknown_count = (df["label_14class"] == unknown_idx).sum()
    log.info(f"Dropping {unknown_count:,} 'unknown_threat' samples (CSIC parameter tampering).")
    log.info("These lack distinguishing static features in our 50-feature space.")
    log.info("Keeping them would corrupt the binary classifier and cause massive FPR.")
    df = df[df["label_14class"] != unknown_idx].copy()

    # Distribution
    dist = df["label_14class"].value_counts().sort_index()
    log.info("\n14-class label distribution (after dropping unknown_threat):")
    for idx, count in dist.items():
        name = ATTACK_CLASSES.get(idx, f"class_{idx}")
        pct  = count / len(df) * 100
        log.info(f"  {idx:2d} {name:<25} {count:>6,} ({pct:5.1f}%)")

    # Warn about classes with very few samples
    for idx, count in dist.items():
        if count < 100 and idx != 0:
            name = ATTACK_CLASSES.get(idx, f"class_{idx}")
            log.warning(f"  LOW SAMPLE CLASS: {name} only has {count} samples")

    df.to_parquet(LABELED_PARQUET, index=False, engine="pyarrow")
    log.info(f"\nSaved labeled parquet: {LABELED_PARQUET}")
    return df


if __name__ == "__main__":
    label_map()
    log.info("Step 04 COMPLETE")
