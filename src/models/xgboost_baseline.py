"""XGBoost baseline with explicit target lags for iter17.

Main model of the new system after the diagnostic (see
`outputs/diagnostic/DIAGNOSTIC_REPORT.md`): regressive XGBoost trained
on the 12 target lags (`stormflow_mgd[t-0..t-11]`) plus 10 reduced exogenous
features derived from S5. Horizon h >= 1.

This module is independent from the TCN normalization pipeline:
- It works with real MGD values (no log1p, no z-score).
- It does not use `src/pipeline/normalize.py` or `src/models/loss.py`.
- The split and aligned indices are identical to
  `scripts/diagnostic/s2_baselines.py` so the figures compare directly.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Split and feature constants - cloned from scripts/diagnostic/s2_baselines.py
# to avoid introducing a dependency on scripts/ from src/.
# ---------------------------------------------------------------------------

IDX_TRAIN_END = 771374
IDX_VAL_END = 936669
SEQ_LENGTH = 72
TARGET_COL = "stormflow_mgd"

# Reduced feature subset after S5 (ordered by permutation importance).
# Justification: S5 showed that these 10 features capture all useful
# exogenous signal (PI > 0), and that the other 10 in the original set are
# noise or redundancy. Keep report order for traceability.
FEATURES_10: List[str] = [
    "api_dynamic",
    "rain_sum_360m",
    "rain_sum_120m",
    "rain_sum_15m",
    "temp_daily_f",
    "hour_sin",
    "minutes_since_last_rain",
    "delta_rain_10m",
    "delta_rain_30m",
    "rain_sum_30m",
]

# Hyperparameter starting point from the report (§7.2). early_stopping_rounds=20
# is passed via fit() in XGBoost >= 1.6 using callbacks; here it is stored as a
# separate key and the training function applies it through eval_set.
DEFAULT_XGB_PARAMS: Dict = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
DEFAULT_EARLY_STOPPING_ROUNDS = 20


# ---------------------------------------------------------------------------
# Chronological split and aligned indices (identical to S2)
# ---------------------------------------------------------------------------

def aligned_indices(
    split_start: int,
    split_end: int,
    horizon: int,
    total_len: int,
    seq_length: int = SEQ_LENGTH,
) -> np.ndarray:
    """Valid absolute t indices for a split.

    Rules:
    - a full previous window of `seq_length` steps exists (t >= split_start + seq_length)
    - origin t falls inside [split_start, split_end)
    - y(t+h) exists inside the dataframe (t + h <= total_len - 1)

    Replicates the S2/TCN convention: the target may fall on the boundary with
    the next split as long as it exists in the full dataframe.
    """
    first = split_start + seq_length
    last_in_split = split_end - 1
    last_by_target = total_len - 1 - horizon
    last = min(last_in_split, last_by_target)
    if last < first:
        return np.empty(0, dtype=int)
    return np.arange(first, last + 1)


def get_split_indices(
    df: pd.DataFrame,
    horizon: int,
    seq_length: int = SEQ_LENGTH,
) -> Dict[str, np.ndarray]:
    """Return (idx_train, idx_val, idx_test) aligned to the official split."""
    total = len(df)
    return {
        "train": aligned_indices(0, IDX_TRAIN_END, horizon, total, seq_length),
        "val":   aligned_indices(IDX_TRAIN_END, IDX_VAL_END, horizon, total, seq_length),
        "test":  aligned_indices(IDX_VAL_END, total, horizon, total, seq_length),
    }


# ---------------------------------------------------------------------------
# Feature construction with target lags
# ---------------------------------------------------------------------------

def _lag_matrix(series: np.ndarray, lags: int) -> np.ndarray:
    """Matrix (n, lags) with columns [y(t), y(t-1), ..., y(t-lags+1)].

    The first `lags-1` rows remain NaN and must be filtered by the caller.
    Equivalent to the lag_matrix function from s2_baselines.py.
    """
    n = len(series)
    out = np.full((n, lags), np.nan, dtype=float)
    for k in range(lags):
        out[k:, k] = series[: n - k]
    return out


def build_features_with_lags(
    df: pd.DataFrame,
    idx: np.ndarray,
    horizon: int,
    lags: int = 12,
    features: Optional[List[str]] = None,
    include_lags: bool = True,
    include_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, feature_names) for one index set.

    X contains, in this order:
    - If include_lags: `y(t), y(t-1), ..., y(t-lags+1)` (lags columns)
    - If include_features: columns of `features` evaluated at t

    y is `stormflow_mgd(t + horizon)`.

    Note: the caller guarantees that indices in `idx` are aligned (all
    have enough previous window and a valid target), by construction of
    `aligned_indices`. There should be no rows with NaN because
    SEQ_LENGTH (72) > lags (12) by default, but the function validates
    that for robustness.
    """
    if features is None:
        features = FEATURES_10
    if not include_lags and not include_features:
        raise ValueError("At least one of include_lags or include_features must be True")

    feature_names: List[str] = []
    blocks: List[np.ndarray] = []

    target = df[TARGET_COL].to_numpy()
    if include_lags:
        lag_mat = _lag_matrix(target, lags)  # (n_total, lags)
        x_lags = lag_mat[idx]                # (len(idx), lags)
        if np.isnan(x_lags).any():
            raise ValueError(
                f"NaN lags found in idx: first index={int(idx[0])}, lags={lags}. "
                "Indices must satisfy t >= lags - 1."
            )
        blocks.append(x_lags)
        feature_names.extend([f"lag_{k}" for k in range(lags)])

    if include_features:
        x_feat = df[features].to_numpy()[idx]  # (len(idx), n_features)
        blocks.append(x_feat)
        feature_names.extend(list(features))

    X = np.concatenate(blocks, axis=1)
    y = target[idx + horizon]
    return X.astype(np.float32, copy=False), y.astype(np.float32, copy=False), feature_names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost_h(
    df: pd.DataFrame,
    horizon: int,
    lags: int = 12,
    features: Optional[List[str]] = None,
    include_lags: bool = True,
    include_features: bool = True,
    xgb_params: Optional[Dict] = None,
    early_stopping_rounds: Optional[int] = DEFAULT_EARLY_STOPPING_ROUNDS,
    seq_length: int = SEQ_LENGTH,
    verbose: bool = False,
) -> Dict:
    """Train XGBoost for one horizon and return model + test predictions.

    Returns a dict with:
      - model: trained XGBRegressor
      - y_pred_test, y_true_test: arrays in MGD (without clipping)
      - y_pred_val, y_true_val: same on val
      - timestamps_test: pd.Series aligned with y_*_test (for panel)
      - feature_names: input column names
      - fit_seconds: fit time
      - best_iteration: selected iteration (if early stopping; None otherwise)
      - config: dict with horizon, lags, include_*, n_features_used
    """
    if xgb_params is None:
        xgb_params = DEFAULT_XGB_PARAMS
    splits = get_split_indices(df, horizon, seq_length=seq_length)
    idx_train, idx_val, idx_test = splits["train"], splits["val"], splits["test"]

    X_train, y_train, feat_names = build_features_with_lags(
        df, idx_train, horizon, lags, features, include_lags, include_features
    )
    X_val, y_val, _ = build_features_with_lags(
        df, idx_val, horizon, lags, features, include_lags, include_features
    )
    X_test, y_test, _ = build_features_with_lags(
        df, idx_test, horizon, lags, features, include_lags, include_features
    )

    params = dict(xgb_params)
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds

    model = XGBRegressor(**params)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)] if early_stopping_rounds is not None else None,
        verbose=bool(verbose),
    )
    fit_seconds = time.time() - t0

    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)

    timestamps_test = df.iloc[idx_test + horizon]["timestamp"].reset_index(drop=True)

    best_iter: Optional[int]
    try:
        best_iter = int(model.best_iteration) if early_stopping_rounds is not None else None
    except AttributeError:
        best_iter = None

    return {
        "model": model,
        "y_pred_test": np.asarray(y_pred_test, dtype=float),
        "y_true_test": np.asarray(y_test, dtype=float),
        "y_pred_val": np.asarray(y_pred_val, dtype=float),
        "y_true_val": np.asarray(y_val, dtype=float),
        "timestamps_test": timestamps_test,
        "feature_names": feat_names,
        "fit_seconds": float(fit_seconds),
        "best_iteration": best_iter,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "config": {
            "horizon": int(horizon),
            "lags": int(lags) if include_lags else 0,
            "include_lags": bool(include_lags),
            "include_features": bool(include_features),
            "n_features_input": int(len(feat_names)),
            "features": list(features) if (include_features and features is not None) else
                        (FEATURES_10 if include_features else []),
        },
    }
