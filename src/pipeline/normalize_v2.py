"""Z-score normalization + optional log1p on the target. For iter19 (clean TCN).

Independent module from the old pipeline `src/pipeline/normalize.py`, which has
a latent bug documented in `outputs/diagnostic/S1_pipeline_audit.md` (BUG2:
if `stormflow_mgd` appears in feature_columns AND target_col, it is normalized
twice in-place). Here that is avoided by construction: features and target are
kept separate, there is no in-place iteration over lists that may contain
duplicates, and the transformation is applied on numpy copies.

Differences relative to `normalize.py`:
- Features NEVER receive log1p. The old version applied log1p to `rain_*`; here
  we trust the network to handle asymmetry through its nonlinearity. This keeps
  the module simple and predictable.
- The target CAN receive log1p (parameter `log1p_target`). This is the only
  nonlinear treatment in the module.
- It works on numpy arrays in `transform`, not on DataFrames. It returns
  matrices ready to feed into the network.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def fit_scaler(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    log1p_target: bool = True,
) -> Dict:
    """Compute normalization statistics on TRAIN only.

    For each feature in `feature_cols`: mean and population standard deviation
    (ddof=0). For the target: if `log1p_target=True`, applies log1p before
    computing the statistics. Zero standard deviations are replaced by 1.0 to
    avoid division by zero in constant columns.

    Args:
        df_train: DataFrame with all required columns (features + target).
        feature_cols: list of feature column names (order matters).
        target_col: name of the target column.
        log1p_target: if True, log1p(clip(y, 0, None)) before computing mu/sigma.

    Returns:
        Dict with keys: feature_cols, target_col, feature_means, feature_stds,
        target_mean, target_std, log1p_target. Arrays are returned as
        np.float32 so they are directly compatible with torch tensors.
    """
    feature_cols = list(feature_cols)
    if target_col in feature_cols:
        # Safety warning: do not allow the target to appear in features.
        # If ignored, the network sees the target as input (trivial leakage).
        raise ValueError(
            f"target_col='{target_col}' appears in feature_cols. Remove it before calling fit_scaler."
        )

    feat_mat = df_train[feature_cols].to_numpy(dtype=np.float64, copy=False)
    feature_means = feat_mat.mean(axis=0)
    feature_stds = feat_mat.std(axis=0, ddof=0)
    feature_stds = np.where(feature_stds > 0.0, feature_stds, 1.0)

    target_arr = df_train[target_col].to_numpy(dtype=np.float64, copy=True)
    if log1p_target:
        target_arr = np.log1p(np.clip(target_arr, 0.0, None))
    target_mean = float(target_arr.mean())
    target_std_raw = float(target_arr.std(ddof=0))
    target_std = target_std_raw if target_std_raw > 0.0 else 1.0

    return {
        "feature_cols": feature_cols,
        "target_col": str(target_col),
        "feature_means": feature_means.astype(np.float32),
        "feature_stds": feature_stds.astype(np.float32),
        "target_mean": np.float32(target_mean),
        "target_std": np.float32(target_std),
        "log1p_target": bool(log1p_target),
    }


def transform(df: pd.DataFrame, scaler: Dict) -> Dict[str, np.ndarray]:
    """Apply the scaler transformation to a DataFrame.

    Returns a dict with:
        features: np.ndarray (N, F) in float32, normalized with z-score.
        target:   np.ndarray (N,)   in float32, in normalized space
                  (with log1p applied first when appropriate).

    Does not modify the input DataFrame.
    """
    feature_cols: List[str] = scaler["feature_cols"]
    target_col: str = scaler["target_col"]

    feat = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    feat -= scaler["feature_means"]
    feat /= scaler["feature_stds"]

    target = df[target_col].to_numpy(dtype=np.float32, copy=True)
    if scaler["log1p_target"]:
        target = np.log1p(np.clip(target, 0.0, None)).astype(np.float32)
    target = (target - float(scaler["target_mean"])) / float(scaler["target_std"])
    target = target.astype(np.float32, copy=False)

    return {"features": feat, "target": target}


def inverse_transform_target(y_norm, scaler: Dict) -> np.ndarray:
    """Return the target in real MGD from its normalized version.

    First applies the inverse of z-score and then, if appropriate, expm1.
    Enforces non-negativity for physical consistency with stormflow.
    """
    y_norm_arr = np.asarray(y_norm, dtype=np.float64)
    y = y_norm_arr * float(scaler["target_std"]) + float(scaler["target_mean"])
    if scaler["log1p_target"]:
        y = np.expm1(y)
    return np.clip(y, 0.0, None)
