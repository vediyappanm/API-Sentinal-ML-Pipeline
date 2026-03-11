"""
03_features.py — Extract all 50 APISentinel features from HTTP fields
Every feature here must be computable from nginx access logs in production.
Groups 6+7 (behavioral/time) are set to safe defaults for CSIC training —
they will be populated from Redis in production inference.
"""

import sys
import re
import math
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote, unquote_plus

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLEAN_PARQUET, FEATURES_PARQUET, LOG_DIR, FEATURE_NAMES,
    KEYWORDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "03_features.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Keyword lists ─────────────────────────────────────────────────────────────
_SQLI         = KEYWORDS["sqli"]
_BLIND_SQLI   = KEYWORDS["blind_sqli"]
_XSS          = KEYWORDS["xss"]
_CMD          = KEYWORDS["command_injection"]
_CODE         = KEYWORDS["code_injection"]
_LDAP         = KEYWORDS["ldap_injection"]
_SSRF         = KEYWORDS["ssrf"]
_PATH_TRAV    = KEYWORDS["path_traversal"]
_SCANNERS     = KEYWORDS["scanning_tool"]
_SENSITIVE    = KEYWORDS["sensitive_file"]

_SCRIPT_EXTS  = {"php", "asp", "aspx", "jsp", "cgi", "pl", "py", "rb", "sh", "cfm"}
_CONFIG_EXTS  = {"env", "config", "conf", "ini", "cfg", "yml", "yaml", "xml", "json"}
_DATA_EXTS    = {"sql", "db", "sqlite", "bak", "backup", "dump", "log", "old", "orig"}
_BROWSER_SIGS = ["mozilla", "chrome", "safari", "firefox", "edge", "opera", "msie"]
_AUTO_SIGS    = ["curl", "wget", "python", "java", "go-http", "libwww",
                 "requests", "httpie", "axios", "okhttp"]

_VERSION_RE = re.compile(r"\d+\.\d+")


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip()


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


def keyword_score(text: str, keywords: list) -> float:
    if not text:
        return 0.0
    t = text.lower()
    return float(sum(1 for kw in keywords if kw in t))


def extract_features(row: pd.Series) -> dict:
    method   = _safe_str(row.get("Method", "GET")).upper()
    raw_url  = _safe_str(row.get("URL", "/"))
    payload  = _safe_str(row.get("Payload", ""))
    ua       = _safe_str(row.get("User-Agent", ""))
    cl       = int(row.get("content_length_num", 0) or 0)

    # Combine URL and payload for keyword scoring
    # CSIC stores POST body in Payload column
    full_text = (raw_url + " " + payload).lower()

    # Decode URL encoding for analysis (but keep raw for entropy)
    try:
        decoded_url = unquote(unquote(raw_url))  # double-decode
    except Exception:
        decoded_url = raw_url
    decoded_full = (decoded_url + " " + payload).lower()

    # Parse URL components
    try:
        parsed = urlparse(raw_url if raw_url.startswith("http") else f"http://x{raw_url}")
        path   = parsed.path or "/"
        query  = parsed.query or ""
        params = parse_qs(query, keep_blank_values=True)
    except Exception:
        path, query, params = "/", "", {}

    # Extension analysis
    path_lower = path.lower()
    ext = ""
    if "." in Path(path).name:
        ext = Path(path).suffix.lstrip(".").lower()

    ua_lower = ua.lower()

    # ── Group 1: URL features ──────────────────────────────────────────────
    url_length           = len(raw_url)
    path_length          = len(path)
    path_depth           = path.count("/")
    query_string_length  = len(query)
    param_count          = len(params)
    url_entropy          = shannon_entropy(raw_url)
    path_entropy         = shannon_entropy(path)
    query_entropy        = shannon_entropy(query)
    special_char_count   = sum(raw_url.count(c) for c in ";:<>'\"()[]{}|\\")
    encoded_char_count   = raw_url.count("%")
    double_encoded       = int("%25" in raw_url.lower() or "%2525" in raw_url.lower())
    null_byte_present    = int("%00" in raw_url.lower() or "\x00" in raw_url)
    path_trav_depth      = (decoded_full.count("../") + decoded_full.count("..\\"))

    # ── Group 2: File & extension ─────────────────────────────────────────
    has_file_ext    = int(bool(ext))
    ext_type        = (
        1 if ext in _SCRIPT_EXTS  else
        2 if ext in _CONFIG_EXTS  else
        3 if ext in _DATA_EXTS    else
        4 if ext in {"jpg","png","gif","ico","jpeg","webp","svg"} else
        0
    )
    is_sensitive    = int(any(s in decoded_full for s in _SENSITIVE))
    is_backup       = int(ext in _DATA_EXTS or any(
        b in decoded_full for b in [".bak", ".old", ".backup", ".orig"]))
    is_executable   = int(ext in _SCRIPT_EXTS)

    # ── Group 3: Method & response ────────────────────────────────────────
    method_map    = {"GET":0,"POST":1,"PUT":2,"DELETE":3,"PATCH":4,"HEAD":5,"OPTIONS":6}
    method_enc    = method_map.get(method, 7)
    # CSIC has no response codes — use content-length as proxy
    # 0 = no content (typical normal), >0 = has body
    status_code   = 200  # CSIC default — all are requests, no responses
    status_class  = 2    # all 2xx (requests)
    resp_bytes    = cl   # use content-length as proxy
    is_zero_resp  = int(cl == 0)

    # ── Group 4: User-Agent ───────────────────────────────────────────────
    ua_length       = len(ua)
    ua_entropy_val  = shannon_entropy(ua)
    ua_is_empty     = int(not ua.strip())
    ua_is_scanner   = int(any(s in ua_lower for s in _SCANNERS))
    ua_is_curl_wget = int(any(s in ua_lower for s in _AUTO_SIGS))
    ua_is_browser   = int(any(s in ua_lower for s in _BROWSER_SIGS))
    ua_has_version  = int(bool(_VERSION_RE.search(ua)))

    # ── Group 5: Payload keyword scores ──────────────────────────────────
    sqli_score     = keyword_score(decoded_full, _SQLI)
    blind_score    = keyword_score(decoded_full, _BLIND_SQLI)
    xss_score      = keyword_score(decoded_full, _XSS)
    cmd_score      = keyword_score(decoded_full, _CMD)
    code_score     = keyword_score(decoded_full, _CODE)
    ldap_score     = keyword_score(decoded_full, _LDAP)
    ssrf_score_val = keyword_score(decoded_full, _SSRF)
    pt_score       = keyword_score(decoded_full, _PATH_TRAV)

    # ── Group 6: Behavioral (zero for training — Redis in production) ─────
    # These will be populated at inference time from Redis sliding window.
    # Training on zeros is correct — the meta-learner learns that CSIC expert
    # has no behavioral signal and weights URL/payload features more heavily.
    ip_req_60s   = 0.0
    ip_req_300s  = 0.0
    ip_err_60s   = 0.0
    ip_paths_60s = 0.0
    ip_paths_300s= 0.0
    ip_alerts    = 0.0
    ip_known_bad = 0.0

    # ── Group 7: Time (defaults for CSIC — no timestamps) ────────────────
    hour_of_day   = 12.0  # midday default
    day_of_week   = 1.0   # Tuesday default (mid-week, business hours)
    is_biz_hours  = 1.0
    is_weekend    = 0.0
    secs_since    = 60.0  # 1 minute gap default

    return {
        # Group 1
        "url_length":            url_length,
        "path_length":           path_length,
        "path_depth":            path_depth,
        "query_string_length":   query_string_length,
        "param_count":           param_count,
        "url_entropy":           url_entropy,
        "path_entropy":          path_entropy,
        "query_entropy":         query_entropy,
        "special_char_count":    special_char_count,
        "encoded_char_count":    encoded_char_count,
        "double_encoded":        double_encoded,
        "null_byte_present":     null_byte_present,
        "path_traversal_depth":  path_trav_depth,
        # Group 2
        "has_file_extension":    has_file_ext,
        "extension_type":        ext_type,
        "is_sensitive_file":     is_sensitive,
        "is_backup_file":        is_backup,
        "is_executable":         is_executable,
        # Group 3
        "method_encoded":        method_enc,
        "status_code":           status_code,
        "status_class":          status_class,
        "response_bytes":        resp_bytes,
        "is_zero_response":      is_zero_resp,
        # Group 4
        "ua_length":             ua_length,
        "ua_entropy":            ua_entropy_val,
        "ua_is_empty":           ua_is_empty,
        "ua_is_known_scanner":   ua_is_scanner,
        "ua_is_curl_wget":       ua_is_curl_wget,
        "ua_is_common_browser":  ua_is_browser,
        "ua_has_version_number": ua_has_version,
        # Group 5
        "sqli_keyword_score":    sqli_score,
        "blind_sqli_score":      blind_score,
        "xss_keyword_score":     xss_score,
        "cmd_injection_score":   cmd_score,
        "code_injection_score":  code_score,
        "ldap_injection_score":  ldap_score,
        "ssrf_score":            ssrf_score_val,
        "path_traversal_score":  pt_score,
        # Group 6 (behavioral — zeros in training)
        "ip_request_rate_60s":   ip_req_60s,
        "ip_request_rate_300s":  ip_req_300s,
        "ip_error_rate_60s":     ip_err_60s,
        "ip_unique_paths_60s":   ip_paths_60s,
        "ip_unique_paths_300s":  ip_paths_300s,
        "ip_prior_alert_count":  ip_alerts,
        "ip_is_known_bad":       ip_known_bad,
        # Group 7 (time — defaults in training)
        "hour_of_day":           hour_of_day,
        "day_of_week":           day_of_week,
        "is_business_hours":     is_biz_hours,
        "is_weekend":            is_weekend,
        "seconds_since_last_req":secs_since,
    }


def extract_all(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        log.info(f"Loading: {CLEAN_PARQUET}")
        df = pd.read_parquet(CLEAN_PARQUET)

    log.info(f"Extracting features for {len(df):,} rows...")
    log.info("Note: Groups 6+7 set to training defaults (zeros/medians)")
    log.info("      These are populated from Redis at production inference time")

    tqdm.pandas(desc="Extracting features")
    feature_rows = df.progress_apply(extract_features, axis=1)
    feat_df = pd.DataFrame(list(feature_rows))

    # Validate all 50 features present
    missing = [f for f in FEATURE_NAMES if f not in feat_df.columns]
    if missing:
        raise RuntimeError(f"Missing features: {missing}")

    extra = [f for f in feat_df.columns if f not in FEATURE_NAMES]
    if extra:
        log.warning(f"Extra feature columns (dropping): {extra}")
        feat_df = feat_df[FEATURE_NAMES]

    # Reorder to canonical order
    feat_df = feat_df[FEATURE_NAMES]

    # Attach labels and metadata
    feat_df["label_binary"]  = df["label_binary"].values
    feat_df["label_raw"]     = df["label_raw"].values
    feat_df["Method"]        = df["Method"].values
    feat_df["URL"]           = df["URL"].values

    # Feature stats
    log.info(f"\nFeature stats (first 10):\n{feat_df[FEATURE_NAMES[:10]].describe().round(3).to_string()}")
    log.info(f"\nFeature null counts: {feat_df[FEATURE_NAMES].isnull().sum().sum()}")

    # Sanity: at least some attacks should have high keyword scores
    attack_rows = feat_df[feat_df["label_binary"] == 1]
    normal_rows = feat_df[feat_df["label_binary"] == 0]
    log.info(f"\nKeyword score comparison (attack vs normal):")
    for feat in ["sqli_keyword_score", "xss_keyword_score", "path_traversal_score"]:
        atk_mean = attack_rows[feat].mean()
        nrm_mean = normal_rows[feat].mean()
        log.info(f"  {feat}: attack={atk_mean:.3f}, normal={nrm_mean:.3f}")
        if atk_mean <= nrm_mean:
            log.warning(f"  UNEXPECTED: attack mean <= normal mean for {feat}")

    feat_df.to_parquet(FEATURES_PARQUET, index=False, engine="pyarrow")
    log.info(f"\nSaved features parquet: {FEATURES_PARQUET}")
    log.info(f"Shape: {feat_df.shape}")
    return feat_df


if __name__ == "__main__":
    extract_all()
    log.info("Step 03 COMPLETE")
