"""
config.py — APISentinel CSIC 2010 Expert Training Pipeline
All constants, paths, and hyperparameters in one place.
Never hardcode values in individual pipeline steps.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "outputs"
DATA_DIR    = OUTPUT_DIR / "data"
MODEL_DIR   = OUTPUT_DIR / "models"
REPORT_DIR  = OUTPUT_DIR / "reports"
LOG_DIR     = OUTPUT_DIR / "logs"

# Create all output directories on import
for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data file paths ───────────────────────────────────────────────────────────
RAW_PARQUET       = DATA_DIR / "01_raw.parquet"
CLEAN_PARQUET     = DATA_DIR / "02_clean.parquet"
FEATURES_PARQUET  = DATA_DIR / "03_features.parquet"
LABELED_PARQUET   = DATA_DIR / "04_labeled.parquet"
TRAIN_PARQUET     = DATA_DIR / "05_train.parquet"
VAL_PARQUET       = DATA_DIR / "05_val.parquet"
TEST_PARQUET      = DATA_DIR / "05_test.parquet"   # LOCKED — never touch until final eval
TRAIN_BAL_PARQUET = DATA_DIR / "06_train_balanced.parquet"

# ── Model file paths ──────────────────────────────────────────────────────────
BINARY_MODEL_PATH     = MODEL_DIR / "csic_expert_binary.json"
MULTICLASS_MODEL_PATH = MODEL_DIR / "csic_expert_multiclass.json"
BINARY_ONNX_PATH      = MODEL_DIR / "csic_expert_binary.onnx"
MULTICLASS_ONNX_PATH  = MODEL_DIR / "csic_expert_multiclass.onnx"
THRESHOLDS_PATH       = MODEL_DIR / "csic_expert_thresholds.json"
METADATA_PATH         = MODEL_DIR / "csic_expert_metadata.json"
SCALER_PATH           = MODEL_DIR / "csic_expert_scaler.joblib"
FEATURE_NAMES_PATH    = MODEL_DIR / "csic_expert_feature_names.json"
SHAP_PATH             = REPORT_DIR / "shap_values.npz"

# ── Dataset schema ────────────────────────────────────────────────────────────
# Kaggle ispangler CSIC 2010 CSV columns
# Handles multiple possible column name variants found in community versions
POSSIBLE_METHOD_COLS  = ["Method", "method", "METHOD", "http_method"]
POSSIBLE_URL_COLS     = ["URL", "url", "Uri", "uri", "URI", "path"]
POSSIBLE_PAYLOAD_COLS = ["Payload", "payload", "Body", "body", "content"]
POSSIBLE_UA_COLS      = ["User-Agent", "user-agent", "UserAgent", "useragent"]
POSSIBLE_LABEL_COLS   = ["label", "Label", "LABEL", "class", "Class", "target", "classification"]
POSSIBLE_CT_COLS      = ["content-type", "Content-Type", "content_type"]
POSSIBLE_CL_COLS      = ["content-length", "Content-Length", "content_length", "lenght"]

# ── Label mapping ─────────────────────────────────────────────────────────────
# CSIC binary labels → APISentinel universal 14-class taxonomy
# CSIC is binary (Normal/Anomalous) — we do attack-type detection
# on the anomalous subset using keyword-based sub-labeling

CSIC_BINARY_LABELS = {
    "normal":    0,
    "norm":      0,
    "anomalous": 1,
    "anom":      1,
    "anomaly":   1,
    "malicious": 1,
    "attack":    1,
    0:           0,
    1:           1,
}

# APISentinel 14-class universal taxonomy
ATTACK_CLASSES = {
    0:  "benign",
    1:  "sql_injection",
    2:  "blind_sqli",
    3:  "xss",
    4:  "path_traversal",
    5:  "command_injection",
    6:  "code_injection",
    7:  "ssrf",
    8:  "ldap_injection",
    9:  "scanning_tool",
    10: "sensitive_file_access",
    11: "credential_stuffing",
    12: "api_enumeration",
    13: "unknown_threat",
}
N_CLASSES = len(ATTACK_CLASSES)

# Reverse mapping: name → index
ATTACK_CLASS_INDEX = {v: k for k, v in ATTACK_CLASSES.items()}

# ── Unified Keyword Definitions ──────────────────────────────────────────────
# Used by both Step 03 (features) and Step 04 (labeling)
KEYWORDS = {
    "sqli": [
        "select ", "union ", "insert ", "update ", "delete ", "drop table",
        "drop database", "exec(", "execute(", "xp_cmdshell", "sp_executesql",
        "char(", "concat(", "information_schema", "sys.tables", "@@version",
        "' or '", "' or 1", "or 1=1", "1=1", "-- ", "# ", "/*", "*/"
    ],
    "blind_sqli": [
        "sleep(", "waitfor delay", "benchmark(", "pg_sleep(", "1=1 --",
        "1=2 --", "' and 1=1", "' and 1=2", "and sleep", "or sleep"
    ],
    "xss": [
        "<script", "</script>", "onerror=", "onload=", "onclick=",
        "onmouseover=", "javascript:", "alert(", "document.cookie",
        "document.write", "eval(", "<img ", "<svg ", "<iframe",
        "src=javascript", "&#x", "&#60", "%3cscript"
    ],
    "path_traversal": [
        "../", "..\\", "%2e%2e/", "%2e%2e\\", "%252e%252e",
        "/etc/passwd", "/etc/shadow", "/etc/hosts", "boot.ini",
        "win.ini", "system32", "/proc/self", "....//", "..%2f", "..%5c"
    ],
    "code_injection": [
        "eval(", "base64_decode(", "system(", "exec(", "passthru(",
        "shell_exec(", "popen(", "__import__", "compile(", "getruntime",
        "{{", "${", "<%=", "#{", "phpinfo(", "file_get_contents("
    ],
    "command_injection": [
        ";cat ", "; cat", "|cat", "&&cat", ";id", "; id", "|id",
        "whoami", "/bin/bash", "/bin/sh", "cmd.exe", "& echo",
        "; wget", "; curl", "| wget", "ping -c", "nslookup ",
        "$(", "`id`", "$(id)"
    ],
    "ldap_injection": [
        "cn=", "dc=", "ou=", "objectclass=", "ldap://", "*(|",
        "*(objectclass", "))(", "|(cn=", ")(uid=", "\\00", "\\2a", "\\28"
    ],
    "ssrf": [
        "file://", "gopher://", "dict://", "169.254.169.254",
        "metadata.google.internal", "instance-data",
        "http://[::1]", "2130706433",
        # NOTE: removed "http://localhost", "http://127.0.0.1", "http://0.0.0.0", "ftp://"
        # because CSIC dataset uses http://localhost:8080 as the base URL for ALL requests,
        # causing 100% false SSRF labeling. In production, these are checked by the nginx
        # layer and the behavioral features (ip_request_rate, etc.) handle redirect-based SSRF.
    ],
    "scanning_tool": [
        "nikto", "sqlmap", "nmap", "dirbuster", "burpsuite", "w3af",
        "paros", "acunetix", "nessus", "openvas", "masscan", "wfuzz",
        "gobuster", "dirb ", "hydra", "metasploit", "zap", "arachni"
    ],
    "sensitive_file": [
        ".env", ".git/", ".htaccess", "web.config", "phpinfo", ".sql",
        ".bak", "backup.", "config.php", "wp-config.php", "database.yml",
        ".aws/", ".ssh/", "id_rsa", "authorized_keys", "shadow"
    ]
}

# ── Attack Detection Rules (Step 04) ──────────────────────────────────────────
# Each entry: (class_index, score_col_name)
# Order defines priority if multiple keywords match
ATTACK_RULES = [
    (2,  "blind_sqli_score"),
    (1,  "sqli_keyword_score"),
    (3,  "xss_keyword_score"),
    (4,  "path_traversal_score"),
    (6,  "code_injection_score"),
    (5,  "cmd_injection_score"),
    (8,  "ldap_injection_score"),
    (7,  "ssrf_score"),
    (10, "is_sensitive_file"),
    (9,  "ua_is_known_scanner"),
]

# ── Feature configuration ─────────────────────────────────────────────────────
# 50 features matching APISentinel nginx production feature space

FEATURE_NAMES = [
    # Group 1: URL features (13)
    "url_length", "path_length", "path_depth", "query_string_length",
    "param_count", "url_entropy", "path_entropy", "query_entropy",
    "special_char_count", "encoded_char_count", "double_encoded",
    "null_byte_present", "path_traversal_depth",
    # Group 2: File & extension (5)
    "has_file_extension", "extension_type", "is_sensitive_file",
    "is_backup_file", "is_executable",
    # Group 3: Method & response (5)
    "method_encoded", "status_code", "status_class",
    "response_bytes", "is_zero_response",
    # Group 4: User-Agent (7)
    "ua_length", "ua_entropy", "ua_is_empty", "ua_is_known_scanner",
    "ua_is_curl_wget", "ua_is_common_browser", "ua_has_version_number",
    # Group 5: Payload keyword scores (8)
    "sqli_keyword_score", "blind_sqli_score", "xss_keyword_score",
    "cmd_injection_score", "code_injection_score", "ldap_injection_score",
    "ssrf_score", "path_traversal_score",
    # Group 6: Behavioral — set to 0 for CSIC (no Redis in training) (7)
    "ip_request_rate_60s", "ip_request_rate_300s", "ip_error_rate_60s",
    "ip_unique_paths_60s", "ip_unique_paths_300s",
    "ip_prior_alert_count", "ip_is_known_bad",
    # Group 7: Time features — set to defaults for CSIC (no timestamps) (5)
    "hour_of_day", "day_of_week", "is_business_hours",
    "is_weekend", "seconds_since_last_req",
]
assert len(FEATURE_NAMES) == 50, f"Expected 50 features, got {len(FEATURE_NAMES)}"

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
BINARY_XGB_PARAMS = {
    "n_estimators":      400,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "scale_pos_weight":  1.5,   # CSIC: 36K normal vs 25K anomalous ≈ 1.44:1
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "tree_method":       "hist",
    "random_state":      42,
    "n_jobs":            -1,
    "use_label_encoder": False,
}

MULTICLASS_XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         7,
    "learning_rate":     0.04,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "multi:softprob",
    "num_class":         N_CLASSES,
    "eval_metric":       "mlogloss",
    "tree_method":       "hist",
    "random_state":      42,
    "n_jobs":            -1,
    "use_label_encoder": False,
}

# ── Training config ───────────────────────────────────────────────────────────
CV_FOLDS           = 5
EARLY_STOPPING     = 50
TEST_SIZE          = 0.15
VAL_SIZE           = 0.15   # of remaining after test split
RANDOM_STATE       = 42

# SMOTE: only apply to classes with fewer than this many samples
SMOTE_MIN_SAMPLES  = 500
SMOTE_TARGET       = 2000   # upsample rare classes to this count

# ── Threshold tuning ──────────────────────────────────────────────────────────
THRESHOLD_SEARCH_RANGE = (0.10, 0.90)
THRESHOLD_STEP         = 0.05

# Default binary threshold — tune per deployment
BINARY_THRESHOLD_DEFAULT = 0.50

# ── Expert metadata ───────────────────────────────────────────────────────────
EXPERT_ID      = "expert_1_csic"
EXPERT_NAME    = "CSIC 2010 HTTP Dataset Expert"
EXPERT_VERSION = "1.0.0"
DATASET_URL    = "https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks"
STRONG_CLASSES = ["sql_injection", "xss", "path_traversal",
                  "ldap_injection", "code_injection"]
WEAK_CLASSES   = ["ssrf", "scanning_tool", "credential_stuffing",
                  "blind_sqli"]
TRUST_WEIGHT   = 0.85   # used by meta-learner to weight this expert
