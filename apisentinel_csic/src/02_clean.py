"""
02_clean.py — Data cleaning, deduplication, label normalization
Removes duplicates, fixes nulls, normalizes labels, removes noise.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_PARQUET, CLEAN_PARQUET, LOG_DIR,
    CSIC_BINARY_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "02_clean.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def clean(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        log.info(f"Loading: {RAW_PARQUET}")
        df = pd.read_parquet(RAW_PARQUET)

    original_len = len(df)
    log.info(f"Starting clean. Rows: {original_len:,}")

    # ── Step 1: String normalization ───────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "None": "", "null": ""})

    # ── Step 2: Label normalization ────────────────────────────────────────
    log.info("Normalizing labels...")
    # Handle numeric or string labels safely
    if df["label"].dtype == object:
        raw_labels = df["label"].astype(str).str.lower().str.strip()
    else:
        raw_labels = df["label"]

    df["label_raw"] = df["label"].copy()   # preserve original
    df["label_binary"] = raw_labels.map(CSIC_BINARY_LABELS)

    unknown_labels = df[df["label_binary"].isna()]["label"].unique()
    if len(unknown_labels) > 0:
        log.warning(f"Unknown labels found (will be dropped): {unknown_labels}")
        df = df.dropna(subset=["label_binary"])
        log.info(f"Dropped {original_len - len(df):,} rows with unknown labels")

    df["label_binary"] = df["label_binary"].astype(int)

    label_dist = df["label_binary"].value_counts()
    log.info(f"Label distribution after normalization:\n  0=normal: {label_dist.get(0,0):,}\n  1=attack: {label_dist.get(1,0):,}")

    # ── Step 3: Fill nulls with safe defaults ─────────────────────────────
    defaults = {
        "Method":         "GET",
        "URL":            "/",
        "Payload":        "",
        "User-Agent":     "",
        "content-type":   "",
        "content-length": "0",
    }
    for col, default in defaults.items():
        if col in df.columns:
            null_count = (df[col] == "").sum() + df[col].isna().sum()
            if null_count > 0:
                df[col] = df[col].fillna(default).replace("", default)
                log.info(f"  Filled {null_count:,} nulls in '{col}' with '{default}'")

    # ── Step 4: Normalize HTTP methods ────────────────────────────────────
    df["Method"] = df["Method"].str.upper().str.strip()
    valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
    invalid_methods = df[~df["Method"].isin(valid_methods)]["Method"].unique()
    if len(invalid_methods) > 0:
        log.warning(f"Non-standard HTTP methods (kept): {invalid_methods[:10]}")

    # ── Step 5: content-length → numeric ──────────────────────────────────
    df["content_length_num"] = (
        pd.to_numeric(df["content-length"], errors="coerce").fillna(0).astype(int)
    )
    df.loc[df["content_length_num"] < 0, "content_length_num"] = 0

    # ── Step 6: Deduplicate ────────────────────────────────────────────────
    # CSIC known to have duplicate entries across normal/anomalous files
    dedup_cols = ["Method", "URL", "Payload", "label_binary"]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    after_dedup  = len(df)
    log.info(f"Removed {before_dedup - after_dedup:,} duplicates")

    # ── Step 7: Remove noise — rows with both normal+anomalous ────────────
    # CSIC v01 known bug: some requests appear in BOTH files with diff labels
    # Detect by URL+Payload hash and drop both conflicting rows
    df["_hash"] = df["URL"].fillna("") + "|" + df["Payload"].fillna("")
    conflict_hashes = (
        df.groupby("_hash")["label_binary"]
        .nunique()
        .where(lambda x: x > 1)
        .dropna()
        .index
    )
    conflict_count = len(df[df["_hash"].isin(conflict_hashes)])
    if conflict_count > 0:
        log.warning(f"Dropping {conflict_count:,} conflicting label rows (CSIC v01 noise)")
        df = df[~df["_hash"].isin(conflict_hashes)]
    df = df.drop(columns=["_hash"])

    # ── Step 8: URL sanity ─────────────────────────────────────────────────
    # Drop rows where URL is completely empty after cleaning
    empty_url = df["URL"].str.strip().eq("") | df["URL"].str.strip().eq("/")
    # Keep empty URLs — they can be valid (root requests)
    log.info(f"Root/empty URL requests: {empty_url.sum():,} (kept)")

    # ── Step 9: Final stats ────────────────────────────────────────────────
    final_len = len(df)
    log.info(f"\n{'='*60}")
    log.info("CLEAN STATS")
    log.info(f"{'='*60}")
    log.info(f"Original rows:  {original_len:,}")
    log.info(f"Final rows:     {final_len:,}")
    log.info(f"Removed total:  {original_len - final_len:,} ({(original_len-final_len)/original_len*100:.1f}%)")
    log.info(f"Normal:   {(df['label_binary']==0).sum():,}")
    log.info(f"Anomalous:{(df['label_binary']==1).sum():,}")
    log.info(f"Method breakdown:\n{df['Method'].value_counts().to_string()}")

    df.to_parquet(CLEAN_PARQUET, index=False, engine="pyarrow")
    log.info(f"\nSaved clean parquet: {CLEAN_PARQUET}")
    return df


if __name__ == "__main__":
    clean()
    log.info("Step 02 COMPLETE")
