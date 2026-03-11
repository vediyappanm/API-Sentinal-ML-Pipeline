"""
12_validate_onnx.py — Verify ONNX predictions match XGBoost predictions exactly
Critical production safety check before enabling ONNX inference.
Max allowed difference: 1e-4 (floating point rounding only).
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VAL_PARQUET,
    BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH,
    BINARY_ONNX_PATH, MULTICLASS_ONNX_PATH,
    LOG_DIR, FEATURE_NAMES, N_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "12_validate_onnx.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MAX_DIFF          = 1e-4
VALIDATE_N_ROWS   = 1000


def validate_model(xgb_path: Path, onnx_path: Path,
                   X: np.ndarray, model_name: str) -> bool:
    if not onnx_path.exists():
        log.warning(f"{model_name}: ONNX file not found at {onnx_path} — skipping")
        return False

    log.info(f"\nValidating {model_name}...")
    try:
        import onnxruntime as ort
    except ImportError:
        log.error("onnxruntime not installed — cannot validate")
        return False

    # XGBoost predictions
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))
    xgb_probs  = xgb_model.predict_proba(X).astype(np.float32)
    xgb_labels = xgb_model.predict(X)

    # ONNX predictions
    sess   = ort.InferenceSession(str(onnx_path))
    inp    = sess.get_inputs()[0].name
    output = sess.run(None, {inp: X})

    # output[0] = class labels, output[1] = probabilities dict or array
    onnx_labels = output[0]
    if len(output) > 1:
        raw = output[1]
        if isinstance(raw, list):
            # dict format — convert to array
            onnx_probs = np.array([[d.get(i, 0.0) for i in range(xgb_probs.shape[1])]
                                    for d in raw], dtype=np.float32)
        else:
            onnx_probs = np.array(raw, dtype=np.float32)
    else:
        log.warning(f"  {model_name}: ONNX output missing probability — label check only")
        onnx_probs = None

    # Label agreement
    label_match = np.mean(xgb_labels == onnx_labels)
    log.info(f"  Label agreement: {label_match*100:.3f}%")
    if label_match < 0.999:
        log.error(f"  FAIL: Label agreement {label_match:.4f} < 0.999")
        return False

    # Probability agreement
    if onnx_probs is not None:
        max_diff  = np.max(np.abs(xgb_probs - onnx_probs))
        mean_diff = np.mean(np.abs(xgb_probs - onnx_probs))
        log.info(f"  Max prob diff:  {max_diff:.2e}  (limit: {MAX_DIFF:.0e})")
        log.info(f"  Mean prob diff: {mean_diff:.2e}")
        if max_diff > MAX_DIFF:
            log.error(f"  FAIL: Max diff {max_diff:.2e} exceeds {MAX_DIFF:.0e}")
            return False

    # Latency benchmark
    import time
    n_bench = 100
    t0 = time.perf_counter()
    for _ in range(n_bench):
        sess.run(None, {inp: X[:1]})
    t_onnx = (time.perf_counter() - t0) / n_bench * 1000

    t0 = time.perf_counter()
    for _ in range(n_bench):
        xgb_model.predict_proba(X[:1])
    t_xgb = (time.perf_counter() - t0) / n_bench * 1000

    speedup = t_xgb / max(t_onnx, 0.001)
    log.info(f"  ONNX latency:   {t_onnx:.3f}ms/req")
    log.info(f"  XGBoost latency:{t_xgb:.3f}ms/req")
    log.info(f"  Speedup:        {speedup:.1f}x")

    log.info(f"  {model_name}: [PASS]")
    return True


def validate():
    log.info(f"Loading val samples: {VAL_PARQUET}")
    df_val = pd.read_parquet(VAL_PARQUET)
    n      = min(VALIDATE_N_ROWS, len(df_val))
    X      = df_val[FEATURE_NAMES].values[:n].astype(np.float32)
    log.info(f"Validating on {n} samples")

    bin_ok   = validate_model(BINARY_MODEL_PATH,     BINARY_ONNX_PATH,
                               X, "Binary Classifier")
    multi_ok = validate_model(MULTICLASS_MODEL_PATH, MULTICLASS_ONNX_PATH,
                               X, "Multiclass Classifier")

    results = {"binary_pass": bin_ok, "multiclass_pass": multi_ok,
               "all_pass": bin_ok and multi_ok}

    with open(LOG_DIR / "onnx_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n{'='*40}")
    log.info(f"ONNX VALIDATION RESULTS")
    log.info(f"Binary:     {'[PASS]' if bin_ok   else '[FAIL]'}")
    log.info(f"Multiclass: {'[PASS]' if multi_ok else '[FAIL]'}")
    if bin_ok and multi_ok:
        log.info("All models validated — safe to use ONNX in production")
    else:
        log.warning("Some validations failed — use XGBoost fallback in production")

    return results


if __name__ == "__main__":
    validate()
    log.info("Step 12 COMPLETE")
