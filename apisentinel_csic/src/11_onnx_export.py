"""
11_onnx_export.py — Convert XGBoost models to ONNX for production inference
ONNX gives 2-3x speedup vs raw XGBoost at inference time.
Target: <2ms per request after ONNX conversion.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH,
    BINARY_ONNX_PATH, MULTICLASS_ONNX_PATH,
    LOG_DIR, FEATURE_NAMES, N_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "11_onnx_export.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

N_FEATURES = len(FEATURE_NAMES)


def export_to_onnx(xgb_path: Path, onnx_path: Path,
                   model_name: str, n_classes: int) -> bool:
    log.info(f"\nConverting {model_name}...")
    log.info(f"  Input:  {xgb_path}")
    log.info(f"  Output: {onnx_path}")

    # Load XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(str(xgb_path))
    booster = model.get_booster()

    # Define ONNX input type
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]

    # Convert
    try:
        onnx_model = convert_xgboost(booster, initial_types=initial_type)
        onnx.save_model(onnx_model, str(onnx_path))
        log.info(f"  Saved ONNX model: {onnx_path}")
        log.info(f"  ONNX file size: {onnx_path.stat().st_size / 1024:.1f} KB")
        return True
    except Exception as e:
        log.error(f"  ONNX conversion failed: {e}")
        log.info("  Trying alternative skl2onnx approach...")
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType
            initial_type_sk = [("float_input", SkFloatTensorType([None, N_FEATURES]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type_sk)
            onnx.save_model(onnx_model, str(onnx_path))
            log.info(f"  Saved via skl2onnx: {onnx_path}")
            return True
        except Exception as e2:
            log.error(f"  skl2onnx also failed: {e2}")
            log.warning(f"  ONNX export failed for {model_name} — model still usable via XGBoost directly")
            return False


def export():
    success_bin  = export_to_onnx(BINARY_MODEL_PATH, BINARY_ONNX_PATH,
                                   "Binary Classifier", 2)
    success_multi = export_to_onnx(MULTICLASS_MODEL_PATH, MULTICLASS_ONNX_PATH,
                                    "Multiclass Classifier", N_CLASSES)

    results = {
        "binary_onnx":     str(BINARY_ONNX_PATH) if success_bin  else None,
        "multiclass_onnx": str(MULTICLASS_ONNX_PATH) if success_multi else None,
    }
    with open(LOG_DIR / "onnx_export_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if success_bin and success_multi:
        log.info("\nAll ONNX exports successful")
    else:
        log.warning("\nSome ONNX exports failed — check logs")

    return results


if __name__ == "__main__":
    export()
    log.info("Step 11 COMPLETE")
