APISentinel Expert 2 Handoff Guide (Kaggle Dataset)
Welcome! You are building Expert 2 for the APISentinel 9-Expert ML pipeline.

We just finished perfecting Expert 1 (CSIC dataset). We have built a highly robust, production-ready pipeline that extracts 50 HTTP features, trains XGBoost models (binary and 14-class multiclass), optimizes thresholds for high-recall security detection, and exports sub-millisecond ONNX models.

All the hard engineering is done. Your job is to adapt this pipeline for the Kaggle Dataset.

You can copy almost our entire codebase, but you MUST change the ingestion and labeling logic to avoid a critical machine learning trap.

📂 Step 1: Copy the Architecture
Copy the apisentinel_csic folder and rename it to apisentinel_kaggle.

✅ What stays EXACTLY the same (Do Not Change!):
These scripts represent the core APISentinel standard. They must remain identical so the downstream Meta-Learner can interpret Expert 1 and Expert 2 uniformly.

03_features.py
: Computes the exact same 50 features.
05_split.py
: Strict 70/15/15 stratified split.
06_smote.py
: Balances rare classes (Train set only).
07_train.py
: XGBoost 5-fold CV architecture.
08_threshold_tune.py
: F2-weighted (recall priority) optimization.
09_evaluate.py
: locked test set validation.
10_shap.py
: Explainability logic.
11_onnx_export.py
 & 
12_validate_onnx.py
: Production inference prep.
🛠️ Step 2: What You MUST Change
1. 
config.py
Update all paths to point to your new apisentinel_kaggle/outputs/... directories.
Update the DATA_PATH to point to your new Kaggle CSV file.
2. 
01_ingest.py
The Kaggle CSV will have completely different column names than the CSIC dataset.

Write standard pandas code to load the Kaggle CSV.
Standardize the column names so they output: Method, URL, Payload, User-Agent. (If the dataset doesn't have a payload/body column, just fill it with empty strings).
Save the output as 01_raw.parquet.
3. 
02_clean.py
Handle nulls and duplicates specific to the Kaggle dataset.
4. 
04_label_map.py
 (🚨 CRITICAL CHANGE 🚨)
Do NOT copy Expert 1's keyword labeling heuristic.

In Expert 1, the CSIC dataset only had binary labels ("Anomalous" vs "Normal"). We forced it into a multiclass model by counting keywords in the payload to guess the attack type. Because the model learned from the exact same keywords we used to label it, it achieved an artificial F1 score of 1.000 (Circular Learning).

The Kaggle dataset should have human-annotated true attack labels (e.g., marking a row explicitly as "SQL Injection").

You must write a dictionary mapper that translates the Kaggle dataset's specific label strings into our official APISentinel 14-class integers defined in 
config.py
 (e.g., "SQLi" -> 1, "XSS" -> 3).
Do not parse the payload to determine the label. Trust the dataset's ground truth column. Because the features are now independent of the labeling mechanism, the XGBoost model will learn real, generalizable patterns!
🎯 Step 3: Realistic Targets
Because you are using independent, ground-truth labels for Expert 2, you will not get an F1 score of 1.000. If you do, you have label leakage.

What to aim for on the Locked Test Set:

Binary Model F1: 0.85 - 0.92
Binary Model FPR: strictly < 0.08 (8%)
Multiclass Model Macro F1: 0.75 - 0.85
If some attack classes have very few samples (e.g., under 50), it is perfectly fine if the model performs poorly on them. This is why we have 9 Experts! If Expert 2 is weak at detecting Command Injection, maybe Expert 3 (Synthetic Data) will specialize in it.

The goal of Expert 2 is to be a broader, more generalized detector than Expert 1.

Good luck! Once your 
pipeline.py
 runs green end-to-end and spits out 
.onnx
 models, Expert 2 is complete!