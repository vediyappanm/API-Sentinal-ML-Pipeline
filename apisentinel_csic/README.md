# APISentinel — CSIC 2010 Expert Training Pipeline

## File Structure
```
apisentinel_csic/
├── README.md
├── requirements.txt
├── config.py                  # All constants, paths, hyperparameters
├── src/
│   ├── 01_ingest.py           # Load raw CSV, validate schema, save parquet
│   ├── 02_clean.py            # Dedup, nulls, noise removal, label fix
│   ├── 03_features.py         # Extract all 50 features from HTTP fields
│   ├── 04_label_map.py        # Map CSIC labels → APISentinel 14-class taxonomy
│   ├── 05_split.py            # Stratified 70/15/15 split, save locked test set
│   ├── 06_smote.py            # Class balancing (SMOTE on train only)
│   ├── 07_train.py            # XGBoost training with 5-fold CV + early stopping
│   ├── 08_threshold_tune.py   # Per-class threshold optimization on val set
│   ├── 09_evaluate.py         # Full evaluation: F1, AUC, FPR, confusion matrix
│   ├── 10_shap.py             # SHAP explainability + feature importance
│   ├── 11_onnx_export.py      # Convert to ONNX for production inference
│   └── 12_validate_onnx.py    # Verify ONNX output matches XGBoost output
├── pipeline.py                # Master runner — runs all 12 steps end to end
└── outputs/                   # Created at runtime
    ├── data/                  # Processed parquet files
    ├── models/                # Trained model files
    ├── reports/               # Evaluation reports
    └── logs/                  # Training logs
```

## Usage
```bash
pip install -r requirements.txt

# Full pipeline
python pipeline.py --data_path /path/to/csic_2010.csv

# Individual steps
python src/01_ingest.py --data_path /path/to/csic_2010.csv
python src/07_train.py
python src/09_evaluate.py
```

## Dataset
- Source: https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks
- Format: CSV with columns: Method, URL, Payload, User-Agent, Pragma,
          Cache-Control, Accept, Accept-Encoding, Accept-Charset,
          language, content-type, connection, content-length, label
- Labels: "Normal" / "Anomalous"  (binary in source, expanded to 14-class)
- Size: ~61,000 rows
