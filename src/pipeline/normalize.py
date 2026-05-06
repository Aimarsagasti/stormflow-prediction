"""Normalization utilities with train-only statistics for MSD pipeline."""

from __future__ import annotations  # Allows modern type annotations in supported versions

from typing import Dict, List, Tuple  # Defines clear return structures for normalization parameters

import numpy as np  # Provides robust numeric operations for scaling
import pandas as pd  # Provides DataFrames to transform each split


def _get_log_columns(
    feature_columns: List[str],
    target_col: str,
    apply_log1p_to_target: bool,
) -> List[str]:
    """Return skewed columns that should receive log1p before scaling."""
    log_columns = []  # Accumulates skewed columns for log transformation
    for column_name in feature_columns:  # Iterates through each candidate input feature
        if column_name == "rain_in" or column_name.startswith("rain_sum_"):  # Selects base rainfall and accumulations according to proposal.md
            log_columns.append(column_name)  # Stores the column to apply log1p before scaling
    if apply_log1p_to_target:  # Also allows compressing the extreme tail of the target when configured that way
        log_columns.append(target_col)  # Adds the target to the list of columns transformed with log1p
    return log_columns  # Returns the final list of transformed columns


def _apply_log_transform(df_split: pd.DataFrame, log_columns: List[str]) -> pd.DataFrame:
    """Apply log1p to selected columns using non-negative clipping."""
    df_out = df_split.copy()  # Works on a copy to avoid mutating original inputs
    for column_name in log_columns:  # Iterates through selected skewed columns
        if column_name in df_out.columns:  # Verifies existence to avoid errors for missing columns
            df_out[column_name] = np.log1p(df_out[column_name].clip(lower=0.0))  # Applies log1p on non-negative values for numeric stability
    return df_out  # Returns DataFrame transformed in log space


def _zscore_scale(
    df_split: pd.DataFrame,
    columns: List[str],
    stats_mean: Dict[str, float],
    stats_std: Dict[str, float],
) -> pd.DataFrame:
    """Scale columns with precomputed z-score statistics."""
    df_out = df_split.copy()  # Creates a copy to avoid touching the original DataFrame
    for column_name in columns:  # Iterates through all columns to normalize
        if column_name in df_out.columns:  # Avoids failure if a column is not available in the split
            df_out[column_name] = (df_out[column_name] - stats_mean[column_name]) / stats_std[column_name]  # Applies z-score with train stats
    return df_out  # Returns normalized DataFrame


def normalize_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "stormflow_mgd",
    apply_log1p_to_target: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Normalize train/val/test with train-only z-score stats and optional log1p."""
    if target_col not in df_train.columns:  # Validates that the target exists in train to compute its statistics
        raise ValueError(f"Target column '{target_col}' not found in train split")  # Gives an explicit message for quick debugging

    log_columns = _get_log_columns(  # Defines skewed columns that are transformed with log1p
        feature_columns=feature_columns,  # Passes model features to detect skewed rainfall/accumulations
        target_col=target_col,  # Passes the target name in case it should also be compressed
        apply_log1p_to_target=apply_log1p_to_target,  # Indicates whether the target enters the log transform
    )

    train_transformed = _apply_log_transform(df_train, log_columns)  # Transforms train before computing statistics according to the proposed pipeline
    val_transformed = _apply_log_transform(df_val, log_columns)  # Applies the same transformation to validation for consistency
    test_transformed = _apply_log_transform(df_test, log_columns)  # Applies the same transformation to test for coherent inference

    all_norm_columns = list(feature_columns) + [target_col]  # Builds the final set of columns to scale with z-score
    stats_mean: Dict[str, float] = {}  # Stores per-column means computed only on train
    stats_std: Dict[str, float] = {}  # Stores per-column standard deviations for scaling and denormalization

    for column_name in all_norm_columns:  # Iterates through features and target to extract base statistics
        if column_name not in train_transformed.columns:  # Validates the expected schema before continuing
            raise ValueError(f"Column '{column_name}' not found in train split")  # Fails clearly if any critical column is missing
        col_mean = float(train_transformed[column_name].mean())  # Computes the train mean for the column
        col_std_raw = float(train_transformed[column_name].std(ddof=0))  # Computes population standard deviation for stability
        col_std = col_std_raw if col_std_raw > 0.0 else 1.0  # Avoids division by zero in constant columns
        stats_mean[column_name] = col_mean  # Stores mean in the parameter dictionary
        stats_std[column_name] = col_std  # Stores the standard deviation used to normalize and reverse

    df_train_norm = _zscore_scale(train_transformed, all_norm_columns, stats_mean, stats_std)  # Scales train using train stats
    df_val_norm = _zscore_scale(val_transformed, all_norm_columns, stats_mean, stats_std)  # Scales val without future information leakage
    df_test_norm = _zscore_scale(test_transformed, all_norm_columns, stats_mean, stats_std)  # Scales test with the same training rules

    norm_params: Dict[str, object] = {  # Packs all information needed to reproduce the transform and inverse
        "feature_columns": list(feature_columns),  # Preserves feature order to reconstruct tensors later
        "target_col": target_col,  # Stores the normalized target name
        "log1p_columns": log_columns,  # List of columns that received log1p transformation
        "apply_log1p_to_target": apply_log1p_to_target,  # Stores explicit flag for target traceability
        "mean": stats_mean,  # Dictionary of per-column means
        "std": stats_std,  # Dictionary of per-column standard deviations
    }

    print(f"[normalize] Columns with log1p: {log_columns}")  # Reports skewed columns transformed before z-score
    print(f"[normalize] Shape train_norm: {df_train_norm.shape}")  # Reports final dimensions of the normalized train split
    print(f"[normalize] Shape val_norm: {df_val_norm.shape}")  # Reports final dimensions of normalized validation
    print(f"[normalize] Shape test_norm: {df_test_norm.shape}")  # Reports final dimensions of normalized test

    return df_train_norm, df_val_norm, df_test_norm, norm_params  # Returns normalized splits and parameters for inference/denormalization


def normalize_target_values(values: np.ndarray, norm_params: Dict[str, object]) -> np.ndarray:
    """Convert raw target values in MGD into the normalized training scale."""
    target_col = str(norm_params["target_col"])  # Retrieves the target name to access its transformation parameters
    values_array = np.asarray(values, dtype=float)  # Converts incoming values to a numpy array for batch transformation
    if target_col in norm_params.get("log1p_columns", []):  # Checks whether the target was compressed with log1p during normalization
        values_array = np.log1p(np.clip(values_array, a_min=0.0, a_max=None))  # Reproduces the same transformation on real non-negative values
    target_mean = float(norm_params["mean"][target_col])  # Extracts the mean used in z-score for the target
    target_std = float(norm_params["std"][target_col])  # Extracts the standard deviation used in z-score for the target
    normalized_values = (values_array - target_mean) / target_std  # Moves values into the same training space as the model
    return normalized_values  # Returns array ready to compare with normalized y_true/y_pred


def denormalize_target(y_norm: np.ndarray, norm_params: Dict[str, object]) -> np.ndarray:
    """Convert normalized target values back to real-world units."""
    target_col = str(norm_params["target_col"])  # Retrieves the target name to look up its scaling parameters
    target_mean = float(norm_params["mean"][target_col])  # Extracts the train mean used in target normalization
    target_std = float(norm_params["std"][target_col])  # Extracts the train standard deviation used in target normalization

    y_norm_array = np.asarray(y_norm, dtype=float)  # Converts input to a numpy array for vectorized operations
    y_real = (y_norm_array * target_std) + target_mean  # Reverses z-score to the transformed space before scaling

    if target_col in norm_params.get("log1p_columns", []):  # Checks whether the target also uses log1p to fully reverse it
        y_real = np.expm1(y_real)  # Reverses log1p and returns real values in original units

    y_real = np.clip(y_real, a_min=0.0, a_max=None)  # Enforces non-negativity for physical stormflow consistency
    return y_real  # Returns target in physical scale for interpretation and metrics
