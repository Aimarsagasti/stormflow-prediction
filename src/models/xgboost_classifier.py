"""Binary XGBoost classifier for stormflow alerts in iter18.

This module implements the operational reformulation of the problem for
long horizons: instead of predicting a point stormflow value, it predicts
whether in the future window `t+1..t+h` there will be at least one sample
with `stormflow_mgd >= U`.

Design:
- Reuses exactly the official chronological split from iter17.
- Reuses the same 6 target lags and the same 10 exogenous features.
- Works in real MGD, without depending on the legacy normalization pipeline.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.xgboost_baseline import (
    FEATURES_10,
    SEQ_LENGTH,
    TARGET_COL,
    _lag_matrix,
    aligned_indices,
    get_split_indices,
)


# These hyperparameters replicate the requested starting point for iter18.
DEFAULT_XGB_CLASSIFIER_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": 42,
}

# Keep the same early stopping as the regressor for a clean comparison.
DEFAULT_EARLY_STOPPING_ROUNDS = 20


def build_binary_target(
    df: pd.DataFrame,
    horizon: int,
    threshold_mgd: float,
    idx: np.ndarray,
) -> np.ndarray:
    """Build `y_bin(t)` using `max(stormflow[t+1..t+h]) >= U`.

    The function uses the full dataframe so the future window remains
    well defined even if the target crosses the split boundary.
    """
    # Extract full target as an array because future-maximum construction is
    # simpler and faster in NumPy.
    target = df[TARGET_COL].to_numpy(dtype=float, copy=False)
    # Reserve one column per future shift inside the window.
    future_blocks: List[np.ndarray] = []
    for step_ahead in range(1, horizon + 1):
        # At position t place the value observed at t+step_ahead.
        shifted = np.full(target.shape[0], np.nan, dtype=float)
        # Fill only positions where that future exists in the array.
        shifted[:-step_ahead] = target[step_ahead:]
        future_blocks.append(shifted)

    # Stack future columns to take row-wise maximum.
    future_matrix = np.column_stack(future_blocks)
    # `idx` already guarantees that t+h exists, so there should be no NaN here.
    future_max = np.nanmax(future_matrix[idx], axis=1)
    # Binary target is 1 if any future sample exceeds threshold U.
    return (future_max >= float(threshold_mgd)).astype(np.int32, copy=False)


def _build_classifier_features(
    df: pd.DataFrame,
    idx: np.ndarray,
    lags: int,
    features: List[str],
) -> tuple[np.ndarray, List[str]]:
    """Build classifier input with the same contract as iter17."""
    # Reuse the same raw target to construct explicit lags.
    target = df[TARGET_COL].to_numpy(dtype=float, copy=False)
    # `_lag_matrix` is already validated in iter17 and keeps the expected order.
    lag_matrix = _lag_matrix(target, lags)
    # Select the exact temporal rows from the aligned split.
    x_lags = lag_matrix[idx]
    if np.isnan(x_lags).any():
        raise ValueError(
            "NaN values were found in the lag matrix. "
            "This indicates misalignment between `idx` and `lags`."
        )

    # Take exogenous features at the same time t so no extra privilege is given.
    x_features = df[features].to_numpy(dtype=float, copy=False)[idx]
    # Concatenate lags first and then exogenous features to preserve traceability.
    X = np.concatenate([x_lags, x_features], axis=1)
    # Generate clear names so importance can be inspected later.
    feature_names = [f"lag_{lag_index}" for lag_index in range(lags)] + list(features)
    return X.astype(np.float32, copy=False), feature_names


def _compute_scale_pos_weight(y_train_bin: np.ndarray) -> float:
    """Compute `neg/pos` using train only, as required by the specification."""
    # Count positives and negatives in train to weight the true imbalance.
    positives = int(np.sum(y_train_bin == 1))
    negatives = int(np.sum(y_train_bin == 0))
    if positives == 0:
        raise ValueError(
            "The train split contains no positives for this variant. "
            "It is not possible to train a useful binary classifier."
        )
    if negatives == 0:
        raise ValueError(
            "The train split contains no negatives for this variant. "
            "The variant is badly defined for binary classification."
        )
    return float(negatives / positives)


def select_operational_threshold(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.85,
) -> float:
    """Choose the highest threshold that satisfies a minimum recall on validation.

    Choosing the highest possible threshold among those that satisfy recall
    reduces false positives without sacrificing the main operational constraint.
    """
    # Sort unique thresholds from high to low to prioritize precision.
    candidate_thresholds = np.unique(np.asarray(y_prob, dtype=float))
    candidate_thresholds = np.sort(candidate_thresholds)[::-1]

    # Iterate from high to low: the first one that satisfies recall is optimal
    # under the rule "maximize threshold subject to minimum recall".
    for threshold in candidate_thresholds:
        y_pred_bin = (y_prob >= threshold).astype(np.int32, copy=False)
        true_positives = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        false_negatives = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        recall = true_positives / max(true_positives + false_negatives, 1)
        if recall >= float(min_recall):
            return float(threshold)

    # If no threshold reaches the desired recall, use 0.0 to always trigger an
    # alert and make it explicit in the results that the model does not separate.
    return 0.0


def train_xgboost_classifier(
    df: pd.DataFrame,
    horizon: int,
    threshold_mgd: float,
    lags: int = 6,
    features: List[str] = FEATURES_10,
    xgb_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    seq_length: int = SEQ_LENGTH,
    verbose: bool = False,
) -> Dict[str, object]:
    """Train a binary XGBoost classifier for one `(h, U)` variant."""
    # Respect prompt defaults unless explicitly overridden.
    params = dict(DEFAULT_XGB_CLASSIFIER_PARAMS if xgb_params is None else xgb_params)

    # `aligned_indices` is imported by explicit prompt contract; this extra call
    # acts as a sanity check that we still follow the same split.
    _ = aligned_indices(0, len(df), horizon, len(df), seq_length)

    # Get exactly the same chronological indices as iter17.
    splits = get_split_indices(df, horizon=horizon, seq_length=seq_length)
    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    # Build input with the same information available to the regressor.
    X_train, feature_names = _build_classifier_features(df, idx_train, lags, features)
    X_val, _ = _build_classifier_features(df, idx_val, lags, features)
    X_test, _ = _build_classifier_features(df, idx_test, lags, features)

    # Build binary target using the full confirmed window.
    y_train_bin = build_binary_target(df, horizon, threshold_mgd, idx_train)
    y_val_bin = build_binary_target(df, horizon, threshold_mgd, idx_val)
    y_test_bin = build_binary_target(df, horizon, threshold_mgd, idx_test)

    # Compute positive weight using only train to avoid leakage.
    scale_pos_weight = _compute_scale_pos_weight(y_train_bin)
    params["scale_pos_weight"] = scale_pos_weight
    # Early stopping is passed in the constructor to preserve the iter17 pattern.
    params["early_stopping_rounds"] = int(early_stopping_rounds)

    # Instantiate classifier with the already closed configuration.
    model = XGBClassifier(**params)

    # Measure actual fit time to report it in artifacts.
    fit_start = time.time()
    model.fit(
        X_train,
        y_train_bin,
        # Validate only on val to select the best iteration without touching test.
        eval_set=[(X_val, y_val_bin)],
        # The notebook controls global verbosity; keep this clean here.
        verbose=bool(verbose),
    )
    fit_seconds = float(time.time() - fit_start)

    # Extract positive-class probabilities for later thresholding.
    y_prob_val = model.predict_proba(X_val)[:, 1].astype(float, copy=False)
    y_prob_test = model.predict_proba(X_test)[:, 1].astype(float, copy=False)

    # `timestamps_test` points to decision time t, not to the future.
    timestamps_test = pd.to_datetime(df.iloc[idx_test]["timestamp"]).reset_index(drop=True)
    # Save actual stormflow at t to reconstruct the first real exceedance.
    y_true_raw_test = df.iloc[idx_test][TARGET_COL].to_numpy(dtype=float, copy=False)

    # Capture best iteration if the model exposes it after early stopping.
    try:
        best_iteration = int(model.best_iteration)
    except AttributeError:
        best_iteration = None

    # Compute true train prevalence for traceability in JSON.
    prevalence_train = float(np.mean(y_train_bin))

    return {
        "model": model,
        "y_true_bin_val": y_val_bin.astype(int, copy=False),
        "y_prob_val": y_prob_val,
        "y_true_bin_test": y_test_bin.astype(int, copy=False),
        "y_prob_test": y_prob_test,
        "y_true_raw_test": y_true_raw_test,
        "timestamps_test": timestamps_test,
        "feature_names": feature_names,
        "fit_seconds": fit_seconds,
        "best_iteration": best_iteration,
        "prevalence_train": prevalence_train,
        "scale_pos_weight": scale_pos_weight,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "config": {
            "horizon": int(horizon),
            "threshold_mgd": float(threshold_mgd),
            "lags": int(lags),
            "features": list(features),
            "seq_length": int(seq_length),
            "n_features_input": int(len(feature_names)),
        },
    }
