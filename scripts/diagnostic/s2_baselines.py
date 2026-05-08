"""
S2 - Rigorous baselines to contextualize TwoStageTCN v1.

Objective: train and evaluate a battery of classical baselines and tabular ML models
(persistence, AR(k), physical regression, XGBoost, RandomForest) on the same
chronological 70/15/15 split and the same test indices that the TCN would use with
`seq_length=72` and horizon `h` in `{1, 3, 6}`.

Writes two artifacts:
  - outputs/diagnostic/S2_baselines.json
  - outputs/diagnostic/S2_baselines.md

Only uses: pandas, numpy, sklearn, xgboost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path("C:/Dev/TFM")
PARQUET_PATH = ROOT / "outputs" / "cache" / "df_with_features.parquet"
OUT_DIR = ROOT / "outputs" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chronological 70/15/15 split (indices already defined by the official pipeline).
IDX_TRAIN_END = 771374
IDX_VAL_END = 936669

SEQ_LENGTH = 72  # same temporal context as the TCN
HORIZONS = [1, 3, 6]

TARGET_COL = "stormflow_mgd"

FEATURES_22 = [
    "rain_in", "temp_daily_f", "api_dynamic",
    "rain_sum_10m", "rain_sum_15m", "rain_sum_30m", "rain_sum_60m",
    "rain_sum_120m", "rain_sum_180m", "rain_sum_360m",
    "rain_max_10m", "rain_max_30m", "rain_max_60m",
    "minutes_since_last_rain",
    "delta_flow_5m", "delta_flow_15m",
    "delta_rain_10m", "delta_rain_30m",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
FEATURES_20 = [c for c in FEATURES_22 if not c.startswith("delta_flow_")]

PHYSICAL_FEATURES = ["rain_sum_60m", "api_dynamic"]

BUCKETS = [
    ("Base",     -np.inf, 0.5),
    ("Leve",     0.5,     5.0),
    ("Moderado", 5.0,     25.0),
    ("Alto",     25.0,    50.0),
    ("Extremo",  50.0,    np.inf),
]

# Fixed hyperparameters (no tuning).
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42,
)
RF_TRAIN_SUBSAMPLE = 300_000  # limite para que quepa en RAM / tiempo razonable


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def peak_err_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    peak_true = float(np.max(y_true))
    peak_pred = float(np.max(y_pred))
    if peak_true == 0:
        return float("nan")
    return (peak_pred - peak_true) / peak_true * 100.0


def bucket_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in BUCKETS:
        mask = (y_true >= lo) & (y_true < hi)
        n = int(np.sum(mask))
        if n == 0:
            out[name] = {"n": 0, "bias": float("nan"), "nse": float("nan"),
                         "rmse": float("nan"), "mae": float("nan")}
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        out[name] = {
            "n": n,
            "bias": float(np.mean(yp - yt)),
            "nse": float(nse(yt, yp)),
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
        }
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        "nse": float(nse(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "peak_real": float(np.max(y_true)),
        "peak_pred": float(np.max(y_pred)),
        "peak_err_pct": peak_err_pct(y_true, y_pred),
        "n": int(len(y_true)),
        "buckets": bucket_stats(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_target(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """`y_target(t) = stormflow_mgd(t + horizon)`. NaN in the last `horizon` rows."""
    return df[TARGET_COL].shift(-horizon).to_numpy()


def aligned_indices(
    split_start: int, split_end: int, horizon: int, total_len: int
) -> np.ndarray:
    """Absolute indices `t` such that:
      - the full previous `SEQ_LENGTH`-step window exists (`t >= split_start + SEQ_LENGTH`)
      - the origin `t` falls inside split `[split_start, split_end)`
      - the target `y(t+h)` exists inside the full dataframe (`t+h <= total_len - 1`)
    Replicates the TCN convention: the target may fall outside the split
    (at the split boundary) as long as it still exists in the dataframe.
    Expected test counts (split `[936669, 1101964)`, `N=1101964`):
      `H=1 -> 165223 ; H=3 -> 165221 ; H=6 -> 165218` (same as `local_eval_metrics`)."""
    first = split_start + SEQ_LENGTH
    last_in_split = split_end - 1
    last_by_target = total_len - 1 - horizon
    last = min(last_in_split, last_by_target)
    return np.arange(first, last + 1)


def lag_matrix(series: np.ndarray, lags: int) -> np.ndarray:
    """Returns a matrix `(n, lags)` with columns `[y(t), y(t-1), ..., y(t-lags+1)]`.
    The first `lags-1` rows remain NaN."""
    n = len(series)
    out = np.full((n, lags), np.nan, dtype=float)
    for k in range(lags):
        out[k:, k] = series[: n - k]
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_naive(df: pd.DataFrame, idx_test: np.ndarray, horizon: int) -> np.ndarray:
    """y_hat(t+h) = y(t)."""
    return df[TARGET_COL].to_numpy()[idx_test]


def baseline_ar1_analytic(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Analytical AR(1): `rho` from Pearson correlation on train."""
    y_full = df[TARGET_COL].to_numpy()
    y_train_t = y_full[idx_train]
    y_train_th = y_full[idx_train + horizon]
    rho = float(np.corrcoef(y_train_t, y_train_th)[0, 1])
    mean_train = float(np.mean(y_train_t))

    y_test_t = y_full[idx_test]
    y_hat_const = rho * y_test_t + (1.0 - rho) * mean_train
    y_hat_noconst = rho * y_test_t
    return y_hat_const, y_hat_noconst, rho, mean_train


def baseline_ar_k(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
    k: int,
) -> np.ndarray:
    """AR(k) via linear regression."""
    y_full = df[TARGET_COL].to_numpy()
    # Build the lag matrix over the whole series, then select valid rows.
    lags = lag_matrix(y_full, k)
    # Filter train/test to rows without NaN in lags (the first `k-1` rows of the full dataset).
    idx_train_f = idx_train[idx_train >= k - 1]
    idx_test_f = idx_test[idx_test >= k - 1]
    X_train = lags[idx_train_f]
    y_train = y_full[idx_train_f + horizon]
    X_test = lags[idx_test_f]
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_hat_test_f = reg.predict(X_test)
    # Reindex to the full length of `idx_test` (it should match because `idx_test`
    # starts at `split_start + SEQ_LENGTH + horizon - 1 >> k`).
    y_hat = np.full(len(idx_test), np.nan)
    # map idx_test -> position
    pos = {v: i for i, v in enumerate(idx_test)}
    for i, ix in enumerate(idx_test_f):
        y_hat[pos[ix]] = y_hat_test_f[i]
    return y_hat


def baseline_physical(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
) -> np.ndarray:
    X = df[PHYSICAL_FEATURES].to_numpy()
    y = df[TARGET_COL].to_numpy()
    reg = LinearRegression()
    reg.fit(X[idx_train], y[idx_train + horizon])
    return reg.predict(X[idx_test])


def baseline_xgb(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
    features: List[str],
    params: Dict,
) -> Tuple[np.ndarray, float]:
    X = df[features].to_numpy()
    y = df[TARGET_COL].to_numpy()
    model = XGBRegressor(**params)
    t0 = time.time()
    model.fit(X[idx_train], y[idx_train + horizon])
    t_fit = time.time() - t0
    y_hat = model.predict(X[idx_test])
    return y_hat, t_fit


def baseline_rf(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
    features: List[str],
    params: Dict,
    subsample: int,
) -> Tuple[np.ndarray, float, int]:
    X = df[features].to_numpy()
    y = df[TARGET_COL].to_numpy()
    # Stratified chronological subsample: take `subsample` rows evenly spaced
    # across train to preserve temporal coverage.
    if len(idx_train) > subsample:
        sel = np.linspace(0, len(idx_train) - 1, subsample).astype(int)
        idx_train_sub = idx_train[sel]
    else:
        idx_train_sub = idx_train
    model = RandomForestRegressor(**params)
    t0 = time.time()
    model.fit(X[idx_train_sub], y[idx_train_sub + horizon])
    t_fit = time.time() - t0
    y_hat = model.predict(X[idx_test])
    return y_hat, t_fit, len(idx_train_sub)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_horizon(df: pd.DataFrame, horizon: int) -> Dict[str, object]:
    """Evaluates all baselines for a specific horizon."""
    print(f"\n=== Horizon h={horizon} ===")
    idx_train = aligned_indices(0, IDX_TRAIN_END, horizon, len(df))
    idx_val = aligned_indices(IDX_TRAIN_END, IDX_VAL_END, horizon, len(df))
    idx_test = aligned_indices(IDX_VAL_END, len(df), horizon, len(df))
    print(f"n_train={len(idx_train):,}  n_val={len(idx_val):,}  n_test={len(idx_test):,}")

    y_full = df[TARGET_COL].to_numpy()
    y_true_test = y_full[idx_test + horizon]

    results: Dict[str, object] = {
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "test_index_first": int(idx_test[0]),
        "test_index_last": int(idx_test[-1]),
        "test_timestamp_first": str(df.iloc[idx_test[0]]["timestamp"]),
        "test_timestamp_last": str(df.iloc[idx_test[-1]]["timestamp"]),
        "baselines": {},
    }

    # (a) Naive
    print("  [a] naive...")
    y_hat = baseline_naive(df, idx_test, horizon)
    results["baselines"]["naive"] = compute_metrics(y_true_test, y_hat)

    # (b) Analytical AR(1) with and without constant term
    print("  [b] AR(1) analytic...")
    yh_c, yh_nc, rho, mean_tr = baseline_ar1_analytic(df, idx_train, idx_test, horizon)
    m = compute_metrics(y_true_test, yh_c)
    m["rho"] = rho
    m["mean_train"] = mean_tr
    results["baselines"]["ar1_analytic"] = m
    results["baselines"]["ar1_noconst"] = compute_metrics(y_true_test, yh_nc)

    # (c) AR(5)
    print("  [c] AR(5)...")
    y_hat = baseline_ar_k(df, idx_train, idx_test, horizon, k=5)
    results["baselines"]["ar5"] = compute_metrics(y_true_test, y_hat)

    # (d) AR(12)
    print("  [d] AR(12)...")
    y_hat = baseline_ar_k(df, idx_train, idx_test, horizon, k=12)
    results["baselines"]["ar12"] = compute_metrics(y_true_test, y_hat)

    # (e) Physical predictor
    print("  [e] Physical (rain_sum_60m + api_dynamic)...")
    y_hat = baseline_physical(df, idx_train, idx_test, horizon)
    results["baselines"]["physical_linear"] = compute_metrics(y_true_test, y_hat)

    # (f) XGBoost 20 features (without delta_flow)
    print("  [f] XGBoost 20 features (without delta_flow)...")
    y_hat, t_fit = baseline_xgb(df, idx_train, idx_test, horizon, FEATURES_20, XGB_PARAMS)
    m = compute_metrics(y_true_test, y_hat)
    m["fit_seconds"] = t_fit
    m["n_features"] = len(FEATURES_20)
    results["baselines"]["xgb_20"] = m
    print(f"     NSE={m['nse']:.4f}  t_fit={t_fit:.1f}s")

    # (g) Random Forest 20 features
    print(f"  [g] RandomForest 20 features (subsample={RF_TRAIN_SUBSAMPLE})...")
    y_hat, t_fit, n_used = baseline_rf(
        df, idx_train, idx_test, horizon, FEATURES_20, RF_PARAMS, RF_TRAIN_SUBSAMPLE
    )
    m = compute_metrics(y_true_test, y_hat)
    m["fit_seconds"] = t_fit
    m["n_features"] = len(FEATURES_20)
    m["n_train_subsample"] = n_used
    results["baselines"]["rf_20"] = m
    print(f"     NSE={m['nse']:.4f}  t_fit={t_fit:.1f}s")

    # (h) XGBoost 22 features (with delta_flow) - CRITICAL
    print("  [h] XGBoost 22 features (with delta_flow)...")
    y_hat, t_fit = baseline_xgb(df, idx_train, idx_test, horizon, FEATURES_22, XGB_PARAMS)
    m = compute_metrics(y_true_test, y_hat)
    m["fit_seconds"] = t_fit
    m["n_features"] = len(FEATURES_22)
    results["baselines"]["xgb_22"] = m
    print(f"     NSE={m['nse']:.4f}  t_fit={t_fit:.1f}s")

    return results


def format_md_table(all_results: Dict[int, Dict[str, object]], tcn_ref: Dict) -> str:
    """Comparative table by horizon."""
    lines: List[str] = []
    lines.append("# S2 - Rigorous baselines\n")
    lines.append(
        "Battery of classical baselines / tabular ML models trained on the same "
        "chronological 70/15/15 split and the same test indices as the TCN "
        "(`seq_length=72`, `horizon h`). Objective: contextualize the real gain "
        "of TwoStageTCN v1 and diagnose the weight of the `delta_flow` shortcut.\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        "- **Split**: train `iloc[:771374]` (hasta 2022-12-11), val `[771374:936669]`, "
        "test `[936669:]` (hasta 2026-01-31).\n"
        "- **Evaluation window**: indices with the full previous 72-step window "
        "y `y(t+h)` disponible (mismos indices que consumiria la TCN).\n"
        "- **Target**: `stormflow_mgd(t+h)`.\n"
        "- **No tuning**: fixed hyperparameters defined in `s2_baselines.py`.\n"
    )
    lines.append("## Dimensions by horizon\n")
    lines.append("| h | n_train | n_val | n_test | primer ts test | ultimo ts test |")
    lines.append("|---|--------:|------:|-------:|----------------|----------------|")
    for h in HORIZONS:
        r = all_results[h]
        lines.append(
            f"| {h} | {r['n_train']:,} | {r['n_val']:,} | {r['n_test']:,} | "
            f"{r['test_timestamp_first']} | {r['test_timestamp_last']} |"
        )
    lines.append("")

    # Main table by horizon
    for h in HORIZONS:
        lines.append(f"## Horizon h={h}\n")
        lines.append("| Baseline | NSE | RMSE | MAE | ErrPico % | Peak pred | N features | Notas |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        # TCN v1 (reference)
        key = f"H{h}_sinSF"
        if key in tcn_ref:
            g = tcn_ref[key]["global"]
            lines.append(
                f"| **TCN v1 sinSF** (ref) | {g['nse']:.4f} | {g['rmse']:.3f} | "
                f"{g['mae']:.3f} | {g['peak_err_pct']:+.1f} | {g['peak_pred']:.1f} | 22 | "
                f"n_test={g['n_total']:,} |"
            )
        key2 = f"H{h}_conSF"
        if key2 in tcn_ref:
            g = tcn_ref[key2]["global"]
            lines.append(
                f"| TCN v1 conSF (ref) | {g['nse']:.4f} | {g['rmse']:.3f} | "
                f"{g['mae']:.3f} | {g['peak_err_pct']:+.1f} | {g['peak_pred']:.1f} | 22+sf | - |"
            )
        # Baselines S2
        order = [
            ("naive",           "Naive persistence y(t)"),
            ("ar1_analytic",    "AR(1) analytic (rho + const)"),
            ("ar1_noconst",     "AR(1) analytic (without const)"),
            ("ar5",             "AR(5) linear"),
            ("ar12",            "AR(12) linear"),
            ("physical_linear", "Physical linear (rain_60m+API)"),
            ("xgb_20",          "XGBoost 20 feats (without delta_flow)"),
            ("rf_20",           "RandomForest 20 feats"),
            ("xgb_22",          "XGBoost 22 feats (with delta_flow)"),
        ]
        bl = all_results[h]["baselines"]
        for k, label in order:
            m = bl[k]
            nf = m.get("n_features", "-")
            extra = ""
            if k == "ar1_analytic":
                extra = f"rho={m['rho']:.4f}"
            elif k == "rf_20":
                extra = f"sub={m.get('n_train_subsample', '?'):,}"
            elif k in ("xgb_20", "xgb_22", "rf_20"):
                extra = f"t={m.get('fit_seconds', 0):.0f}s"
            lines.append(
                f"| {label} | {m['nse']:.4f} | {m['rmse']:.3f} | {m['mae']:.3f} | "
                f"{m['peak_err_pct']:+.1f} | {m['peak_pred']:.1f} | {nf} | {extra} |"
            )
        lines.append("")

    # Bias by bucket (H=1 only)
    lines.append("## Bias by bucket (H=1)\n")
    lines.append(
        "`bias = mean(y_pred - y_true)` inside each `y_true` range (MGD). "
        "Negative = underestimation, positive = overestimation.\n"
    )
    buckets_names = [b[0] for b in BUCKETS]
    lines.append("| Baseline | " + " | ".join(buckets_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(buckets_names)) + "|")
    if "H1_sinSF" in tcn_ref:
        r = tcn_ref["H1_sinSF"]["ranges"]
        row = ["**TCN v1 sinSF**"]
        for bn in buckets_names:
            row.append(f"{r[bn]['bias']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    bl = all_results[1]["baselines"]
    for k, label in [
        ("naive", "Naive"),
        ("ar1_analytic", "AR(1) analytic"),
        ("ar5", "AR(5)"),
        ("ar12", "AR(12)"),
        ("physical_linear", "Physical linear"),
        ("xgb_20", "XGB-20"),
        ("rf_20", "RF-20"),
        ("xgb_22", "XGB-22"),
    ]:
        b = bl[k]["buckets"]
        row = [label]
        for bn in buckets_names:
            bv = b[bn].get("bias", float("nan"))
            row.append(f"{bv:+.3f}" if bv == bv else "n/a")  # nan check
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # NSE by bucket (H=1)
    lines.append("## NSE by bucket (H=1)\n")
    lines.append(
        "Local NSE inside the bucket. `NSE<0` indicates that the model is worse than "
        "predicting the bucket mean.\n"
    )
    lines.append("| Baseline | " + " | ".join(buckets_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(buckets_names)) + "|")
    if "H1_sinSF" in tcn_ref:
        r = tcn_ref["H1_sinSF"]["ranges"]
        row = ["**TCN v1 sinSF**"]
        for bn in buckets_names:
            row.append(f"{r[bn]['nse']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    for k, label in [
        ("naive", "Naive"),
        ("ar1_analytic", "AR(1)"),
        ("ar5", "AR(5)"),
        ("ar12", "AR(12)"),
        ("physical_linear", "Physical"),
        ("xgb_20", "XGB-20"),
        ("rf_20", "RF-20"),
        ("xgb_22", "XGB-22"),
    ]:
        b = bl[k]["buckets"]
        row = [label]
        for bn in buckets_names:
            bv = b[bn].get("nse", float("nan"))
            row.append(f"{bv:+.3f}" if bv == bv else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


def build_verdict(all_results: Dict[int, Dict[str, object]], tcn_ref: Dict) -> str:
    r1 = all_results[1]["baselines"]
    r3 = all_results[3]["baselines"]
    r6 = all_results[6]["baselines"]

    nse_tcn_h1 = tcn_ref["H1_sinSF"]["global"]["nse"]
    nse_xgb20_h1 = r1["xgb_20"]["nse"]
    nse_xgb22_h1 = r1["xgb_22"]["nse"]
    nse_naive_h1 = r1["naive"]["nse"]
    nse_ar12_h1 = r1["ar12"]["nse"]
    nse_phys_h1 = r1["physical_linear"]["nse"]

    delta_shortcut = nse_xgb22_h1 - nse_xgb20_h1
    delta_xgb_vs_naive = nse_xgb20_h1 - nse_naive_h1
    delta_xgb_vs_ar12 = nse_xgb20_h1 - nse_ar12_h1
    delta_tcn_vs_xgb20 = nse_tcn_h1 - nse_xgb20_h1
    delta_tcn_vs_xgb22 = nse_tcn_h1 - nse_xgb22_h1

    lines: List[str] = []
    lines.append("## Verdict\n")
    lines.append(
        "Direct answers to the key diagnostic questions. "
        "All numbers refer to the aligned test set (72-step window), "
        "without tuning or early stopping, and to the target `stormflow_mgd(t+h)`.\n"
    )

    # P1
    lines.append(
        f"### 1. XGBoost-20 vs TCN v1 (H=1)\n"
        f"- NSE XGB-20 = **{nse_xgb20_h1:.4f}**\n"
        f"- NSE TCN v1 sinSF = **{nse_tcn_h1:.4f}**\n"
        f"- Difference TCN - XGB-20 = **{delta_tcn_vs_xgb20:+.4f}** NSE.\n"
    )
    if delta_tcn_vs_xgb20 >= 0.02:
        lines.append(
            "Interpretation: the TCN **does add** some architectural value over "
            "XGBoost with the same 20 features (without the `delta_flow` shortcut), although "
            "the gain is modest. We still need to assess whether it justifies the added complexity.\n"
        )
    elif abs(delta_tcn_vs_xgb20) < 0.02:
        lines.append(
            "Interpretation: XGBoost-20 **ties** TCN v1. The temporal architecture "
            "(causal convolutions, shared backbone, two-stage) "
            "is not adding measurable value relative to a tabular GBM with the "
            "same features and without the shortcut.\n"
        )
    else:
        lines.append(
            "Interpretation: XGBoost-20 **outperforms** TCN v1. The TCN is not "
            "extracting useful information from the temporal dynamics beyond what "
            "XGBoost captures with the aggregated features (`rain_sum_*`, `api_dynamic`, "
            "`delta_rain_*`). All apparent TCN gains were coming from the shortcut.\n"
        )

    # P2
    lines.append(
        f"### 2. Real weight of the `delta_flow` shortcut\n"
        f"- NSE XGB-22 (with `delta_flow`) = **{nse_xgb22_h1:.4f}**\n"
        f"- NSE XGB-20 (without `delta_flow`) = **{nse_xgb20_h1:.4f}**\n"
        f"- Shortcut contribution = **{delta_shortcut:+.4f}** NSE.\n"
        f"- NSE XGB-22 - TCN v1 = **{delta_tcn_vs_xgb22:+.4f}** (negative = XGB-22 wins).\n"
    )
    if delta_shortcut >= 0.03:
        lines.append(
            "Interpretation: the `delta_flow_*` shortcut contributes a clearly "
            "measurable gain in XGBoost as well. This confirms the iter16 finding: the "
            "TCN jump over the naive baseline came largely from those two features.\n"
        )
    else:
        lines.append(
            "Interpretation: `delta_flow_*` adds little in XGBoost (<0.03 NSE). "
            "The shortcut may be more useful to the TCN because of how it combines it internally.\n"
        )

    # P3
    lines.append(
        f"### 3. XGBoost-20 vs baselines triviales\n"
        f"- XGB-20 - naive = **{delta_xgb_vs_naive:+.4f}** NSE\n"
        f"- XGB-20 - AR(12) = **{delta_xgb_vs_ar12:+.4f}** NSE\n"
    )
    if delta_xgb_vs_naive < 0.02:
        lines.append(
            "Interpretation: XGBoost-20 **barely improves** over persistence. The "
            "rainfall features are being ignored (or they do not add signal at H=1).\n"
        )
    else:
        lines.append(
            "Interpretation: XGBoost-20 **does use** the rainfall/API features. "
            "The exogenous signal has predictive value.\n"
        )

    # P4
    lines.append(
        f"### 4. Physical predictor (2 features) as sanity check (H=1)\n"
        f"- NSE linear(`rain_sum_60m + api_dynamic`) = **{nse_phys_h1:.4f}**\n"
    )
    if nse_phys_h1 >= 0.3:
        lines.append(
            "Interpretation: with only two physical features, NSE reaches >=0.3. "
            "There is learnable rainfall->stormflow signal, and a simple model already captures "
            "part of it.\n"
        )
    elif nse_phys_h1 > 0:
        lines.append(
            "Interpretation: NSE is positive but modest. The rainfall->stormflow relationship "
            "is nonlinear and needs more features / a more expressive model to "
            "capture it well.\n"
        )
    else:
        lines.append(
            "Interpretation: NSE is negative or zero. The linear combination of "
            "`rain_sum_60m + api_dynamic` is not enough; the problem has strong nonlinearities "
            "or the horizon confounds the scales.\n"
        )

    # P5: which horizon makes sense
    def best_nse(res):
        return max((v["nse"] for v in res.values() if isinstance(v, dict) and "nse" in v))
    best_h1 = best_nse(r1)
    best_h3 = best_nse(r3)
    best_h6 = best_nse(r6)
    lines.append(
        f"### 5. Operational horizon\n"
        f"- Best baseline NSE at H=1: **{best_h1:.4f}**\n"
        f"- Best baseline NSE at H=3: **{best_h3:.4f}**\n"
        f"- Best baseline NSE at H=6: **{best_h6:.4f}**\n"
    )
    if best_h6 < 0.3:
        lines.append(
            "Interpretation: at H=6 no baseline reaches a reasonable NSE. The available "
            "predictive signal degrades quickly with the horizon, "
            "consistent with the system's effective lag (short memory). "
            "Work at H=1 (30 min) and, if the TFM requires it, H=3 (15 min) "
            "as the maximum operational horizon.\n"
        )
    else:
        lines.append(
            "Interpretation: there is room for longer horizons. Evaluate H>=3 "
            "as an operational target.\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  shape={df.shape}")

    all_results: Dict[int, Dict[str, object]] = {}
    for h in HORIZONS:
        all_results[h] = run_horizon(df, h)

    # Load TCN v1 metrics for comparison
    tcn_ref_path = ROOT / "outputs" / "data_analysis" / "local_eval_metrics.json"
    with open(tcn_ref_path, "r", encoding="utf-8") as f:
        tcn_ref = json.load(f)

    # Global metadata
    meta = {
        "split": {
            "train_end_idx": IDX_TRAIN_END,
            "val_end_idx": IDX_VAL_END,
            "total_rows": len(df),
            "train_last_ts": str(df.iloc[IDX_TRAIN_END - 1]["timestamp"]),
            "val_last_ts": str(df.iloc[IDX_VAL_END - 1]["timestamp"]),
            "test_last_ts": str(df.iloc[-1]["timestamp"]),
        },
        "seq_length": SEQ_LENGTH,
        "horizons": HORIZONS,
        "features_22": FEATURES_22,
        "features_20": FEATURES_20,
        "physical_features": PHYSICAL_FEATURES,
        "xgb_params": XGB_PARAMS,
        "rf_params": RF_PARAMS,
        "rf_train_subsample": RF_TRAIN_SUBSAMPLE,
        "buckets_mgd": [{"name": b[0], "lo": b[1], "hi": b[2]} for b in BUCKETS],
        "tcn_reference_source": str(tcn_ref_path),
    }

    json_out = {"meta": meta, "results": all_results}
    json_path = OUT_DIR / "S2_baselines.json"
    # Clean inf/NaN for JSON
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
        return o
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean(json_out), f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {json_path}")

    md_body = format_md_table(all_results, tcn_ref)
    md_verdict = build_verdict(all_results, tcn_ref)
    md_path = OUT_DIR / "S2_baselines.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_body + "\n" + md_verdict + "\n")
    print(f"Written: {md_path}")


if __name__ == "__main__":
    main()
