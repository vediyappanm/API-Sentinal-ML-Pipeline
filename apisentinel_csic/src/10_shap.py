"""
10_shap.py — SHAP explainability for both models
Generates feature importance plots and saves SHAP explainer.
SHAP is used in production to explain predictions to SOC analysts.
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VAL_PARQUET, BINARY_MODEL_PATH, MULTICLASS_MODEL_PATH,
    SHAP_PATH, REPORT_DIR, LOG_DIR, FEATURE_NAMES, ATTACK_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "10_shap.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MAX_SHAP_SAMPLES = 500  # SHAP is slow — use subset for plots


def run_shap():
    log.info(f"Loading val data: {VAL_PARQUET}")
    df_val = pd.read_parquet(VAL_PARQUET)
    X_val  = df_val[FEATURE_NAMES].values.astype(np.float32)

    # Subsample for SHAP computation
    np.random.seed(42)
    idx     = np.random.choice(len(X_val), min(MAX_SHAP_SAMPLES, len(X_val)), replace=False)
    X_shap  = X_val[idx]
    X_shap_df = pd.DataFrame(X_shap, columns=FEATURE_NAMES)

    # ── Binary SHAP ───────────────────────────────────────────────────────
    log.info("Computing SHAP for binary model...")
    binary_model = xgb.XGBClassifier()
    binary_model.load_model(str(BINARY_MODEL_PATH))

    explainer_bin  = shap.TreeExplainer(binary_model)
    shap_vals_bin  = explainer_bin.shap_values(X_shap_df)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals_bin, X_shap_df, show=False, max_display=20)
    plt.title("CSIC Expert Binary — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_binary_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved binary SHAP plot")

    # Top features by mean |SHAP|
    mean_shap    = np.abs(shap_vals_bin).mean(axis=0)
    top_features = sorted(zip(FEATURE_NAMES, mean_shap), key=lambda x: x[1], reverse=True)
    log.info("\nTop 15 features by mean |SHAP| (binary):")
    for feat, score in top_features[:15]:
        log.info(f"  {feat:<35} {score:.4f}")

    # ── Multiclass SHAP ───────────────────────────────────────────────────
    log.info("\nComputing SHAP for multiclass model...")
    multi_model = xgb.XGBClassifier()
    multi_model.load_model(str(MULTICLASS_MODEL_PATH))

    explainer_mc  = shap.TreeExplainer(multi_model)
    shap_vals_mc  = explainer_mc.shap_values(X_shap_df)  # list of arrays per class

    # Summary for most important attack class (sql_injection = class 1)
    if isinstance(shap_vals_mc, list) and len(shap_vals_mc) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals_mc[1], X_shap_df, show=False, max_display=15)
        plt.title("CSIC Expert Multiclass — SHAP for sql_injection")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_multiclass_sqli.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved multiclass SHAP plot (sql_injection class)")

    # Save feature importance JSON for production use
    top_feature_names = [f for f, _ in top_features[:20]]
    top_feature_scores = [float(s) for _, s in top_features[:20]]
    importance_data = {
        "binary_top20": [
            {"feature": name, "mean_abs_shap": score}
            for name, score in zip(top_feature_names, top_feature_scores)
        ]
    }
    with open(REPORT_DIR / "feature_importance.json", "w") as f:
        json.dump(importance_data, f, indent=2)
    log.info(f"Saved feature importance: {REPORT_DIR / 'feature_importance.json'}")

    # Save SHAP values for offline analysis
    np.savez_compressed(
        SHAP_PATH,
        binary=shap_vals_bin,
        feature_names=np.array(FEATURE_NAMES),
        sample_indices=idx,
    )
    log.info(f"Saved SHAP values: {SHAP_PATH}")


def explain_single_prediction(features: dict,
                               binary_model_path: str = None,
                               multi_model_path: str  = None) -> dict:
    """
    Production function: explain a single prediction.
    Called by APISentinel stream.py for each ML-flagged request.
    Returns top 5 contributing features with direction.
    """
    binary_model_path = binary_model_path or str(BINARY_MODEL_PATH)
    multi_model_path  = multi_model_path  or str(MULTICLASS_MODEL_PATH)

    binary_model = xgb.XGBClassifier()
    binary_model.load_model(binary_model_path)
    multi_model  = xgb.XGBClassifier()
    multi_model.load_model(multi_model_path)

    X = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]], dtype=np.float32)
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    explainer = shap.TreeExplainer(binary_model)
    shap_vals  = explainer.shap_values(X_df)[0]

    top5 = sorted(zip(FEATURE_NAMES, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
    return {
        "top_features": [
            {
                "feature":      name,
                "shap_value":   float(val),
                "direction":    "attack" if val > 0 else "benign",
                "feature_value": float(features.get(name, 0.0)),
            }
            for name, val in top5
        ]
    }


if __name__ == "__main__":
    run_shap()
    log.info("Step 10 COMPLETE")
