"""
01_ingest.py — Raw data ingestion and schema validation
Loads CSIC 2010 CSV, validates all required columns exist,
handles multiple column name variants, saves clean parquet.
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_PARQUET, LOG_DIR,
    POSSIBLE_METHOD_COLS, POSSIBLE_URL_COLS, POSSIBLE_PAYLOAD_COLS,
    POSSIBLE_UA_COLS, POSSIBLE_LABEL_COLS, POSSIBLE_CT_COLS, POSSIBLE_CL_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "01_ingest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def resolve_column(df: pd.DataFrame, candidates: list, canonical: str) -> str:
    """Find whichever candidate column name exists in df. Rename to canonical."""
    for col in candidates:
        if col in df.columns:
            if col != canonical:
                df.rename(columns={col: canonical}, inplace=True)
                log.info(f"  Mapped column '{col}' -> '{canonical}'")
            return canonical
    # Not found — create empty column with warning
    log.warning(f"  Column not found (tried {candidates}). Creating empty '{canonical}'.")
    df[canonical] = ""
    return canonical


def ingest(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    log.info(f"Loading dataset: {path}")
    log.info(f"File size: {path.stat().st_size / 1024 / 1024:.1f} MB")

    # Try multiple encodings — CSIC has Latin characters
    df = None
    for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="warn",
            )
            log.info(f"Loaded with encoding: {encoding}")
            break
        except (UnicodeDecodeError, Exception) as e:
            log.debug(f"Encoding {encoding} failed: {e}")
            continue

    if df is None:
        raise RuntimeError("Failed to load CSV with any supported encoding.")

    log.info(f"Raw shape: {df.shape}")
    log.info(f"Raw columns: {list(df.columns)}")
    log.info(f"Raw dtypes:\n{df.dtypes}")

    # ── Normalize column names ──────────────────────────────────────────────
    log.info("Resolving column names...")
    resolve_column(df, POSSIBLE_METHOD_COLS,  "Method")
    resolve_column(df, POSSIBLE_URL_COLS,     "URL")
    resolve_column(df, POSSIBLE_PAYLOAD_COLS, "Payload")
    resolve_column(df, POSSIBLE_UA_COLS,      "User-Agent")
    resolve_column(df, POSSIBLE_LABEL_COLS,   "label")
    resolve_column(df, POSSIBLE_CT_COLS,      "content-type")
    resolve_column(df, POSSIBLE_CL_COLS,      "content-length")

    # ── Basic stats ────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("DATASET STATS AFTER COLUMN RESOLUTION")
    log.info(f"{'='*60}")
    log.info(f"Total rows:  {len(df):,}")
    log.info(f"Total cols:  {len(df.columns)}")

    if "label" in df.columns:
        log.info(f"Label distribution:\n{df['label'].value_counts()}")
    else:
        raise RuntimeError("Could not find label column. Ingestion failed.")

    if "Method" in df.columns:
        log.info(f"Method distribution:\n{df['Method'].value_counts()}")

    # ── Null check ─────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        log.info(f"Null values per column:\n{null_cols}")
    else:
        log.info("No null values found in raw data.")

    # ── Sample rows ────────────────────────────────────────────────────────
    log.info(f"\nSample rows:\n{df.head(3).to_string()}")

    # ── Save raw parquet ───────────────────────────────────────────────────
    df.to_parquet(RAW_PARQUET, index=False, engine="pyarrow")
    log.info(f"\nSaved raw parquet: {RAW_PARQUET}")
    log.info(f"File size: {RAW_PARQUET.stat().st_size / 1024:.1f} KB")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSIC 2010 ingestion")
    parser.add_argument("--data_path", required=True, help="Path to CSIC 2010 CSV file")
    args = parser.parse_args()
    ingest(args.data_path)
    log.info("Step 01 COMPLETE")
