"""
pipeline.py — Master runner for CSIC 2010 Expert Training Pipeline
Runs all 12 steps end-to-end with error handling and progress tracking.
Each step can also be run individually from its own file.

Usage:
    python pipeline.py --data_path /path/to/csic_2010.csv
    python pipeline.py --data_path /path/to/csic_2010.csv --start_step 7
    python pipeline.py --data_path /path/to/csic_2010.csv --skip_eval
"""

import sys
import time
import logging
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import LOG_DIR, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     APISentinel — CSIC 2010 Expert Training Pipeline         ║
║     Expert 1 of 9 — HTTP Attack Detection                    ║
║     Production-grade XGBoost training + ONNX export          ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_step(step_num: int, step_name: str, fn, *args, **kwargs):
    log.info(f"\n{'━'*64}")
    log.info(f"STEP {step_num:02d} — {step_name}")
    log.info(f"{'━'*64}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        log.info(f"✓ Step {step_num:02d} complete in {elapsed:.1f}s")
        return result, True
    except Exception as e:
        elapsed = time.time() - t0
        log.error(f"✗ Step {step_num:02d} FAILED after {elapsed:.1f}s: {e}")
        log.error(traceback.format_exc())
        return None, False


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="APISentinel CSIC Expert Pipeline")
    parser.add_argument("--data_path",   required=True, help="Path to CSIC 2010 CSV file")
    parser.add_argument("--start_step",  type=int, default=1, help="Start from step N (1-12)")
    parser.add_argument("--skip_eval",   action="store_true", help="Skip locked test set evaluation")
    parser.add_argument("--skip_shap",   action="store_true", help="Skip SHAP (saves time)")
    parser.add_argument("--skip_onnx",   action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        log.error(f"Dataset not found: {data_path}")
        sys.exit(1)

    log.info(f"Dataset:    {data_path}")
    log.info(f"Output dir: {OUTPUT_DIR}")
    log.info(f"Start step: {args.start_step}")

    pipeline_start = time.time()
    step_results   = {}

    # ── Import all step modules via importlib (numeric prefixes) ──────────
    import importlib.util, types

    def load_step(filename, attr):
        path = Path(__file__).parent / "src" / filename
        spec = importlib.util.spec_from_file_location(filename, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, attr)

    ingest          = load_step("01_ingest.py",         "ingest")
    clean           = load_step("02_clean.py",          "clean")
    extract_all     = load_step("03_features.py",       "extract_all")
    label_map       = load_step("04_label_map.py",      "label_map")
    split           = load_step("05_split.py",          "split")
    apply_smote     = load_step("06_smote.py",          "apply_smote")
    train           = load_step("07_train.py",          "train")
    tune_thresholds = load_step("08_threshold_tune.py", "tune_thresholds")
    evaluate        = load_step("09_evaluate.py",       "evaluate")
    run_shap        = load_step("10_shap.py",           "run_shap")
    export          = load_step("11_onnx_export.py",    "export")
    validate        = load_step("12_validate_onnx.py",  "validate")

    steps = [
        (1,  "Raw Data Ingestion",         lambda: ingest(str(data_path))),
        (2,  "Data Cleaning",              clean),
        (3,  "Feature Extraction (50)",    extract_all),
        (4,  "Label Mapping (14-class)",   label_map),
        (5,  "Stratified Split 70/15/15",  split),
        (6,  "SMOTE Class Balancing",      apply_smote),
        (7,  "XGBoost Training (CV+ES)",   train),
        (8,  "Threshold Tuning",           tune_thresholds),
        (9,  "Evaluation (Locked Test)",   evaluate   if not args.skip_eval else lambda: log.info("Skipped")),
        (10, "SHAP Explainability",        run_shap   if not args.skip_shap else lambda: log.info("Skipped")),
        (11, "ONNX Export",                export     if not args.skip_onnx else lambda: log.info("Skipped")),
        (12, "ONNX Validation",            validate   if not args.skip_onnx else lambda: log.info("Skipped")),
    ]

    failed_steps = []
    for step_num, step_name, fn in steps:
        if step_num < args.start_step:
            log.info(f"Skipping step {step_num:02d} (start_step={args.start_step})")
            continue

        result, success = run_step(step_num, step_name, fn)
        step_results[step_num] = {"name": step_name, "success": success}

        if not success:
            failed_steps.append(step_num)
            log.error(f"Pipeline stopped at step {step_num} — fix error and rerun with --start_step {step_num}")
            break

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    log.info(f"\n{'═'*64}")
    log.info("PIPELINE SUMMARY")
    log.info(f"{'═'*64}")
    log.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    for step_num, info in step_results.items():
        status = "✓" if info["success"] else "✗ FAILED"
        log.info(f"  Step {step_num:02d} {info['name']:<35} {status}")

    if not failed_steps:
        log.info(f"\n{'='*64}")
        log.info("ALL STEPS COMPLETE — CSIC Expert ready for APISentinel")
        log.info(f"{'='*64}")
        log.info(f"Models saved to:    outputs/models/")
        log.info(f"Reports saved to:   outputs/reports/")
        log.info(f"Logs saved to:      outputs/logs/")
        log.info(f"\nNext step: Run expert_2_kaggle pipeline")
        log.info(f"After all 9 experts: Train meta-learner on OOF predictions")
    else:
        log.error(f"\nPipeline FAILED at steps: {failed_steps}")
        sys.exit(1)


if __name__ == "__main__":
    main()
