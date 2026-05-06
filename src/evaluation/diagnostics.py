"""Diagnostic utilities for full stormflow model inspection."""

from __future__ import annotations  # Allows using modern annotations without breaking compatibility

from typing import Any, Dict, List, Tuple  # Defines explicit types for diagnostic structures

import numpy as np  # Provides vectorized operations for statistics and metrics
import pandas as pd  # Lets us read normalized split columns consistently
import torch  # Lets us run inference and permutations on model tensors
from torch.utils.data import DataLoader  # Types the DataLoader used in permutation importance
from src.models.tcn import TwoStageTCN  # Imports the two-stage model to use its predict method

from src.pipeline.normalize import denormalize_target  # Reuses the official denormalization from the pipeline


def _to_1d_array(values: np.ndarray) -> np.ndarray:
    """Convert any array-like input into a float 1D numpy array."""
    return np.asarray(values, dtype=float).reshape(-1)  # Forces 1D float vector for stable numeric computations


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE safely for possibly empty arrays."""
    if y_true.size == 0:  # Avoids mean over empty arrays
        return float("nan")  # Marks metric as undefined when there is no data
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))  # Computes root mean squared error


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE safely for possibly empty arrays."""
    if y_true.size == 0:  # Avoids mean over empty arrays
        return float("nan")  # Marks metric as undefined when there is no data
    return float(np.mean(np.abs(y_pred - y_true)))  # Computes mean absolute error


def _safe_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency safely."""
    if y_true.size == 0:  # Avoids computing NSE without samples
        return float("nan")  # Returns NaN when the metric does not apply
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)  # Computes observed variance used as denominator
    if denominator <= 0:  # Controls degenerate case with no variability in y_true
        return float("nan")  # Marks NSE as not interpretable in that case
    numerator = np.sum((y_true - y_pred) ** 2)  # Computes model sum of squared errors
    return float(1.0 - (numerator / denominator))  # Returns final NSE where 1.0 is perfect fit


def _safe_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute Pearson correlation safely for possibly degenerate arrays."""
    if x_values.size == 0 or y_values.size == 0:  # Avoids computing correlation if either vector is missing
        return float("nan")  # Returns NaN to indicate insufficient data
    if x_values.size != y_values.size:  # Validates exact alignment in length between both vectors
        raise ValueError("x_values and y_values must have the same length")  # Raises clear error if misaligned
    if np.isclose(np.std(x_values), 0.0) or np.isclose(np.std(y_values), 0.0):  # Avoids implicit division by zero in correlation
        return float("nan")  # Returns NaN when there is no variability in either vector
    return float(np.corrcoef(x_values, y_values)[0, 1])  # Returns Pearson coefficient for the pair of series


def _distribution_stats(y_norm: np.ndarray) -> Dict[str, float]:
    """Build normalized target distribution summary for one split."""
    if y_norm.size == 0:  # Handles empty split without breaking report structure
        return {  # Returns complete structure with NaN/0 to keep stable contract
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "p999": float("nan"),
            "pct_eq_0": 0.0,
            "pct_lt_0_01": 0.0,
            "pct_lt_0_05": 0.0,
            "pct_lt_0_10": 0.0,
            "n_samples": 0.0,
        }

    return {  # Builds requested summary to diagnose compression in normalized scale
        "min": float(np.min(y_norm)),  # Reports minimum normalized target in the split
        "max": float(np.max(y_norm)),  # Reports maximum normalized target in the split
        "mean": float(np.mean(y_norm)),  # Reports mean normalized target in the split
        "median": float(np.median(y_norm)),  # Reports median for robustness against long tails
        "p95": float(np.quantile(y_norm, 0.95)),  # Reports 95th percentile of normalized target
        "p99": float(np.quantile(y_norm, 0.99)),  # Reports 99th percentile of normalized target
        "p999": float(np.quantile(y_norm, 0.999)),  # Reports 99.9th percentile of normalized target
        "pct_eq_0": float(np.mean(y_norm == 0.0) * 100.0),  # Reports exact percentage of values at zero
        "pct_lt_0_01": float(np.mean(y_norm < 0.01) * 100.0),  # Reports percentage of values below 0.01
        "pct_lt_0_05": float(np.mean(y_norm < 0.05) * 100.0),  # Reports percentage of values below 0.05
        "pct_lt_0_10": float(np.mean(y_norm < 0.10) * 100.0),  # Reports percentage of values below 0.10
        "n_samples": float(y_norm.size),  # Reports total number of samples in the split
    }


def _bias_by_severity(y_real_mgd: np.ndarray, y_pred_mgd: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute bias by severity bucket using requested MGD thresholds."""
    buckets: List[Tuple[str, float | None, float | None]] = [  # Defines severity cuts requested by the user
        ("base", None, 0.5),  # Base bucket for samples below 0.5 MGD
        ("pequeno", 0.5, 2.0),  # Small bucket for 0.5 to 2.0 MGD range
        ("moderado", 2.0, 13.0),  # Moderate bucket for 2.0 to 13.0 MGD range
        ("grande", 13.0, 51.0),  # Large bucket for 13.0 to 51.0 MGD range
        ("extremo", 51.0, None),  # Extreme bucket for values greater than or equal to 51.0 MGD
    ]

    results: Dict[str, Dict[str, float]] = {}  # Prepares bias container per bucket
    for bucket_name, lower_bound, upper_bound in buckets:  # Iterates through each bucket to apply its mask
        if lower_bound is None:  # Handles bucket with upper bound only
            mask = y_real_mgd < upper_bound  # Selects real samples below the upper limit
        elif upper_bound is None:  # Handles bucket with lower bound only
            mask = y_real_mgd >= lower_bound  # Selects real samples above or equal to the lower limit
        else:  # Handles bucket with both lower and upper limits
            mask = (y_real_mgd >= lower_bound) & (y_real_mgd < upper_bound)  # Selects real samples inside the interval

        if not mask.any():  # Controls bucket without samples to avoid invalid means
            results[bucket_name] = {"bias": float("nan"), "n_samples": 0.0}  # Stores empty output while keeping structure
            continue  # Continues with the next bucket without attempting extra calculations

        bucket_bias = float(np.mean(y_pred_mgd[mask] - y_real_mgd[mask]))  # Computes signed bias (pred-real) in current bucket
        results[bucket_name] = {"bias": bucket_bias, "n_samples": float(mask.sum())}  # Stores bias and count for traceability

    return results  # Returns complete structure of bias by severity


def _top_peak_ratios(y_real_mgd: np.ndarray, y_pred_mgd: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    """Compute pred/real ratio for the top-k largest real peaks."""
    if y_real_mgd.size == 0:  # Handles empty evaluations without breaking serialization
        return []  # Returns empty list when there are no samples

    ranked_indices = np.argsort(y_real_mgd)[::-1]  # Sorts indices by descending real magnitude
    selected_indices = ranked_indices[: min(top_k, ranked_indices.size)]  # Takes only the first top_k available indices

    peak_rows: List[Dict[str, Any]] = []  # Initializes list of rows with per-peak detail
    for rank_position, sample_index in enumerate(selected_indices, start=1):  # Iterates selected peaks with human 1-based ranking
        real_value = float(y_real_mgd[sample_index])  # Extracts real magnitude of the current peak
        pred_value = float(y_pred_mgd[sample_index])  # Extracts prediction at the same real-peak index
        ratio_value = float(pred_value / real_value) if real_value > 0.0 else float("nan")  # Computes pred/real ratio while avoiding division by zero
        peak_rows.append(  # Adds serializable row for pointwise audit of the highest real peaks
            {
                "rank": int(rank_position),  # Stores rank position among real peaks
                "sample_index": int(sample_index),  # Stores original index for traceability in the series
                "real_mgd": real_value,  # Stores real peak value in MGD
                "pred_mgd": pred_value,  # Stores predicted peak value in MGD
                "pred_real_ratio": ratio_value,  # Stores prediction/real ratio to inspect compression or overestimation
            }
        )

    return peak_rows  # Returns list with detail of top-10 real peaks


def _extract_xy_from_batch(
    batch: Tuple[torch.Tensor, ...],
    resolved_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract X and y from a training batch with 3 or 4 tensors."""
    if len(batch) == 4:  # Supports loaders with extra event metadata
        x_batch, y_batch, _w_batch, _event_batch = batch  # Ignores weights and metadata because only inference is needed
    elif len(batch) == 3:  # Supports classic loaders that only return X, y, and weights
        x_batch, y_batch, _w_batch = batch  # Ignores weights in this diagnostic because RMSE uses only prediction and target
    else:  # Detects unexpected formats to avoid silent errors
        raise ValueError("Expected dataloader batches with 3 or 4 tensors")  # Raises clear error for pipeline debugging
    x_batch = x_batch.to(resolved_device)  # Moves features to configured inference device
    y_batch = y_batch.to(resolved_device)  # Moves target to the same device to align comparison
    return x_batch, y_batch  # Returns only tensors needed for prediction and RMSE


def _predict_over_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    resolved_device: torch.device,
    permuted_feature_index: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader, optionally permuting one feature column."""
    predictions: List[np.ndarray] = []  # Accumulates predictions batch by batch to concatenate at the end
    is_two_stage = isinstance(model, TwoStageTCN)  # Detects whether the model requires hard switch during prediction
    targets: List[np.ndarray] = []  # Accumulates real targets batch by batch to measure comparable RMSE
    with torch.no_grad():  # Disables gradients because this path is diagnostic only
        for batch in dataloader:  # Iterates through all batches of the received dataloader
            x_batch, y_batch = _extract_xy_from_batch(  # Reuses common routine to accept 3- or 4-tensor signatures
                batch=batch,  # Passes raw batch emitted by the DataLoader
                resolved_device=resolved_device,  # Uses device resolved by the caller
            )
            if permuted_feature_index is not None:  # Only applies permutation when evaluating a specific feature
                if x_batch.ndim != 3:  # Validates expected shape (batch, seq_length, n_features) before permuting
                    raise ValueError("Expected x_batch with shape (batch, seq_length, n_features)")  # Raises clear error if tensor shape does not match
                if not (0 <= permuted_feature_index < x_batch.shape[2]):  # Protects against out-of-range indices in feature columns
                    raise ValueError("permuted_feature_index is out of bounds for input features")  # Raises explicit error for quick debugging
                x_batch = x_batch.clone()  # Clones batch to avoid mutating original tensor shared by the DataLoader
                feature_values = x_batch[:, :, permuted_feature_index].reshape(-1)  # Takes all values of the feature across batch and time
                permutation_indices = torch.randperm(feature_values.numel(), device=resolved_device)  # Builds random permutation on the same device
                feature_values = feature_values[permutation_indices]  # Reorders values to break feature-target association
                x_batch[:, :, permuted_feature_index] = feature_values.view_as(x_batch[:, :, permuted_feature_index])  # Writes permuted feature back while preserving original shape
            if is_two_stage:  # Runs hard-switch prediction when the model is two-stage
                y_pred = model.predict(x_batch, threshold=0.3)  # Uses low threshold to prioritize event recall
            else:  # Preserves direct forward pass when the model is not two-stage
                y_pred = model(x_batch)  # Runs classic forward with original or permuted batch
            if not isinstance(y_pred, torch.Tensor):  # Validates expected output signature to avoid silent incompatibilities
                raise TypeError("Model output must be a torch.Tensor")  # Raises clear error if model output is not a tensor
            predictions.append(y_pred.detach().cpu().numpy().reshape(-1))  # Converts prediction to 1D numpy and accumulates it
            targets.append(y_batch.detach().cpu().numpy().reshape(-1))  # Converts target to 1D numpy and accumulates it
    if not predictions:  # Handles empty loaders to avoid breaking concatenation
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)  # Returns empty arrays if there were no batches
    y_pred_array = np.concatenate(predictions, axis=0)  # Concatenates predictions from all batches into one vector
    y_true_array = np.concatenate(targets, axis=0)  # Concatenates real targets from all batches into one vector
    return y_pred_array, y_true_array  # Returns comparable pairs for RMSE computation


def run_permutation_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    feature_columns: List[str],
    norm_params: Dict[str, object],
) -> List[Dict[str, float]]:
    """Compute permutation importance ranking using RMSE delta per feature."""
    resolved_device = torch.device(device)  # Normalizes received device for consistent use in PyTorch
    model = model.to(resolved_device)  # Moves model to inference device to avoid implicit copies
    model_was_training = model.training  # Stores previous state so it can be restored at the end of the diagnostic
    model.eval()  # Forces evaluation mode to disable dropout and stabilize comparisons
    baseline_pred_norm, baseline_true_norm = _predict_over_loader(  # Gets baseline without permuting any feature
        model=model,  # Reuses the same trained model for the whole comparison
        dataloader=dataloader,  # Iterates over exactly the same sample set for the baseline
        resolved_device=resolved_device,  # Uses same device to keep costs and precision comparable
        permuted_feature_index=None,  # Does not permute columns in the baseline run
    )
    baseline_pred_mgd = denormalize_target(baseline_pred_norm, norm_params)  # Converts baseline predictions to real MGD for hydrologic interpretation
    baseline_true_mgd = denormalize_target(baseline_true_norm, norm_params)  # Converts baseline target to real MGD with the same parameters
    baseline_pred_mgd = np.clip(baseline_pred_mgd, a_min=0.0, a_max=None)  # Enforces physical non-negativity constraint on prediction
    baseline_true_mgd = np.clip(baseline_true_mgd, a_min=0.0, a_max=None)  # Enforces physical non-negativity constraint on real target
    baseline_rmse_mgd = _safe_rmse(y_true=baseline_true_mgd, y_pred=baseline_pred_mgd)  # Computes baseline RMSE in real units
    ranking_rows: List[Dict[str, float]] = []  # Initializes result list per feature for final ranking
    for feature_index, feature_name in enumerate(feature_columns):  # Iterates through all columns to estimate individual impact
        permuted_pred_norm, permuted_true_norm = _predict_over_loader(  # Runs inference by permuting only the current feature
            model=model,  # Reuses same model to keep comparison fair
            dataloader=dataloader,  # Iterates over the same set to isolate the effect of the permutation
            resolved_device=resolved_device,  # Keeps same execution device
            permuted_feature_index=feature_index,  # Selects the specific feature to shuffle across all batches
        )
        permuted_pred_mgd = denormalize_target(permuted_pred_norm, norm_params)  # Converts permuted predictions to real MGD
        permuted_true_mgd = denormalize_target(permuted_true_norm, norm_params)  # Converts permuted target to real MGD for consistency
        permuted_pred_mgd = np.clip(permuted_pred_mgd, a_min=0.0, a_max=None)  # Preserves physical constraint in permuted predictions
        permuted_true_mgd = np.clip(permuted_true_mgd, a_min=0.0, a_max=None)  # Preserves physical constraint in permuted targets
        permuted_rmse_mgd = _safe_rmse(y_true=permuted_true_mgd, y_pred=permuted_pred_mgd)  # Computes RMSE with perturbed feature
        delta_rmse_mgd = float(permuted_rmse_mgd - baseline_rmse_mgd)  # Measures absolute impact of breaking the feature in MGD
        if np.isfinite(baseline_rmse_mgd) and baseline_rmse_mgd > 0.0:  # Avoids division by zero when computing relative change
            relative_increase_pct = float((delta_rmse_mgd / baseline_rmse_mgd) * 100.0)  # Converts impact to percentage relative to baseline
        else:  # Handles degenerate baseline to avoid infinities in the ranking
            relative_increase_pct = float("nan")  # Marks percentage as undefined when baseline is not useful
        ranking_rows.append(  # Adds row with complete importance metric for the current feature
            {
                "feature": str(feature_name),  # Stores feature name for human-readable ranking
                "baseline_rmse_mgd": float(baseline_rmse_mgd),  # Repeats baseline RMSE for traceability in each row
                "permuted_rmse_mgd": float(permuted_rmse_mgd),  # Stores observed RMSE after permuting the feature
                "delta_rmse_mgd": delta_rmse_mgd,  # Stores absolute RMSE increase used as main importance
                "relative_increase_pct": relative_increase_pct,  # Stores relative increase to compare features in standardized form
            }
        )
    ranking_rows.sort(key=lambda row: row["delta_rmse_mgd"], reverse=True)  # Sorts from highest to lowest impact for final ranking
    print("[diag] === Permutation Importance (RMSE delta in MGD) ===")  # Prints header to separate this diagnostic section
    for rank_index, row in enumerate(ranking_rows, start=1):  # Iterates over already sorted ranking to display feature importance
        print(  # Reports rank, feature, and absolute/relative RMSE changes
            f"[diag] #{rank_index:02d} {row['feature']}: "
            f"delta_rmse={row['delta_rmse_mgd']:.6f} MGD | "
            f"permuted_rmse={row['permuted_rmse_mgd']:.6f} | "
            f"rel_increase={row['relative_increase_pct']:.2f}%"
        )
    if model_was_training:  # Restores original state in case the caller continues training after the diagnostic
        model.train()  # Reactivates training mode only if the model was previously in train mode
    return ranking_rows  # Returns serializable ranking to save in JSON or markdown


def run_full_diagnostics(
    y_pred_norm: np.ndarray,
    y_real_norm: np.ndarray,
    norm_params: Dict[str, object],
    df_train_norm: pd.DataFrame,
    df_val_norm: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """Run full diagnostics for normalized target distribution, predictions, residuals, and normalization."""
    y_pred_norm_array = _to_1d_array(y_pred_norm)  # Normalizes prediction input to 1D float vector
    y_real_norm_array = _to_1d_array(y_real_norm)  # Normalizes real-target input to 1D float vector

    if y_pred_norm_array.shape[0] != y_real_norm_array.shape[0]:  # Validates exact alignment between prediction and target
        raise ValueError("y_pred_norm and y_real_norm must have the same length")  # Raises clear error if lengths do not match

    train_target_norm = _to_1d_array(df_train_norm[target_col].to_numpy())  # Extracts normalized train target for summary
    val_target_norm = _to_1d_array(df_val_norm[target_col].to_numpy())  # Extracts normalized val target for summary
    test_target_norm = _to_1d_array(df_test_norm[target_col].to_numpy())  # Extracts normalized test target for summary

    distributions = {  # Groups normalized distribution by split as requested
        "train": _distribution_stats(train_target_norm),  # Computes train statistical summary in normalized scale
        "val": _distribution_stats(val_target_norm),  # Computes val statistical summary in normalized scale
        "test": _distribution_stats(test_target_norm),  # Computes test statistical summary in normalized scale
    }

    y_pred_mgd = denormalize_target(y_pred_norm_array, norm_params)  # Converts normalized predictions to real MGD
    y_real_mgd = denormalize_target(y_real_norm_array, norm_params)  # Converts normalized real target to real MGD
    y_pred_mgd = np.clip(y_pred_mgd, a_min=0.0, a_max=None)  # Applies physical non-negativity constraint to prediction
    y_real_mgd = np.clip(y_real_mgd, a_min=0.0, a_max=None)  # Applies physical non-negativity constraint to real target

    nse_value = _safe_nse(y_real_mgd, y_pred_mgd)  # Computes global NSE in physical units
    rmse_value = _safe_rmse(y_real_mgd, y_pred_mgd)  # Computes global RMSE in MGD
    mae_value = _safe_mae(y_real_mgd, y_pred_mgd)  # Computes global MAE in MGD

    if y_real_mgd.size > 0:  # Checks that data exists to compute global peak diagnostic
        peak_real_mgd = float(np.max(y_real_mgd))  # Gets maximum observed real magnitude
        peak_pred_mgd = float(np.max(y_pred_mgd))  # Gets maximum predicted magnitude by the model
        peak_error_mgd = float(peak_pred_mgd - peak_real_mgd)  # Computes signed global peak error
        peak_error_pct = float((peak_error_mgd / max(peak_real_mgd, 1e-12)) * 100.0)  # Computes relative peak error as percentage
    else:  # Handles empty vector to keep report structure stable
        peak_real_mgd = float("nan")  # Marks real peak as undefined
        peak_pred_mgd = float("nan")  # Marks predicted peak as undefined
        peak_error_mgd = float("nan")  # Marks absolute peak error as undefined
        peak_error_pct = float("nan")  # Marks percentage peak error as undefined

    severity_bias = _bias_by_severity(y_real_mgd=y_real_mgd, y_pred_mgd=y_pred_mgd)  # Computes bias by requested severity buckets
    top_10_ratios = _top_peak_ratios(y_real_mgd=y_real_mgd, y_pred_mgd=y_pred_mgd, top_k=10)  # Summarizes calibration on the 10 highest peaks

    residuals_mgd = y_pred_mgd - y_real_mgd  # Defines signed residual as pred-real to separate over- and underestimation
    residual_mean = float(np.mean(residuals_mgd)) if residuals_mgd.size > 0 else float("nan")  # Computes residual mean for global bias
    residual_median = float(np.median(residuals_mgd)) if residuals_mgd.size > 0 else float("nan")  # Computes robust residual median
    residual_std = float(np.std(residuals_mgd)) if residuals_mgd.size > 0 else float("nan")  # Computes residual spread in MGD
    pct_over_gt_2 = float(np.mean(residuals_mgd > 2.0) * 100.0) if residuals_mgd.size > 0 else 0.0  # Computes percentage with overestimation greater than 2 MGD
    pct_under_gt_2 = float(np.mean(residuals_mgd < -2.0) * 100.0) if residuals_mgd.size > 0 else 0.0  # Computes percentage with underestimation greater than 2 MGD
    residual_corr = _safe_correlation(x_values=y_real_mgd, y_values=residuals_mgd)  # Estimates correlation between real magnitude and residual

    q95_norm = float(np.quantile(y_real_norm_array, 0.95)) if y_real_norm_array.size > 0 else float("nan")  # Computes P95 in normalized space of the evaluated vector
    q99_norm = float(np.quantile(y_real_norm_array, 0.99)) if y_real_norm_array.size > 0 else float("nan")  # Computes P99 in normalized space of the evaluated vector
    q95_real = float(np.quantile(y_real_mgd, 0.95)) if y_real_mgd.size > 0 else float("nan")  # Computes P95 in real MGD of the same vector
    q99_real = float(np.quantile(y_real_mgd, 0.99)) if y_real_mgd.size > 0 else float("nan")  # Computes P99 in real MGD of the same vector

    norm_span = float(q99_norm - q95_norm) if np.isfinite(q99_norm) and np.isfinite(q95_norm) else float("nan")  # Computes normalized P95-P99 band width
    real_span = float(q99_real - q95_real) if np.isfinite(q99_real) and np.isfinite(q95_real) else float("nan")  # Computes real MGD P95-P99 band width
    if np.isfinite(norm_span) and np.isfinite(real_span) and real_span > 0.0:  # Verifies that both spans are valid before dividing
        compression_ratio = float(norm_span / real_span)  # Computes requested ratio to measure relative compression
    else:  # Handles degenerate case with no useful real range
        compression_ratio = float("nan")  # Marks ratio as undefined when there is no valid numeric basis

    flag_train_lt_005 = bool(distributions["train"]["pct_lt_0_05"] > 80.0)  # Marks whether train is highly concentrated below 0.05
    flag_val_lt_005 = bool(distributions["val"]["pct_lt_0_05"] > 80.0)  # Marks whether val is highly concentrated below 0.05
    flag_test_lt_005 = bool(distributions["test"]["pct_lt_0_05"] > 80.0)  # Marks whether test is highly concentrated below 0.05

    normalization_diagnostics = {  # Groups requested normalization section
        "effective_target_range_norm": {  # Reports effective normalized target range by split
            "train": {  # Stores min, max, and range of normalized train
                "min": distributions["train"]["min"],
                "max": distributions["train"]["max"],
                "range": float(distributions["train"]["max"] - distributions["train"]["min"]),
            },
            "val": {  # Stores min, max, and range of normalized val
                "min": distributions["val"]["min"],
                "max": distributions["val"]["max"],
                "range": float(distributions["val"]["max"] - distributions["val"]["min"]),
            },
            "test": {  # Stores min, max, and range of normalized test
                "min": distributions["test"]["min"],
                "max": distributions["test"]["max"],
                "range": float(distributions["test"]["max"] - distributions["test"]["min"]),
            },
        },
        "compression_p95_p99": {  # Reports compression diagnostic across normalized and real scales
            "p95_norm": q95_norm,
            "p99_norm": q99_norm,
            "p95_real_mgd": q95_real,
            "p99_real_mgd": q99_real,
            "norm_span_p95_p99": norm_span,
            "real_span_p95_p99": real_span,
            "norm_real_span_ratio": compression_ratio,
        },
        "high_mass_below_0_05_flags": {  # Reports low-concentration flags by split
            "train_gt_80pct": flag_train_lt_005,
            "val_gt_80pct": flag_val_lt_005,
            "test_gt_80pct": flag_test_lt_005,
            "any_split_gt_80pct": bool(flag_train_lt_005 or flag_val_lt_005 or flag_test_lt_005),
        },
    }

    diagnostics: Dict[str, Any] = {  # Builds final complete dictionary to persist in JSON
        "normalized_target_distribution": distributions,  # Includes normalized target statistics for train/val/test
        "prediction_diagnostics": {  # Includes prediction diagnostics in real MGD
            "global_metrics": {  # Summarizes main global performance metrics
                "nse": nse_value,
                "rmse": rmse_value,
                "mae": mae_value,
            },
            "severity_bias": severity_bias,  # Includes bias by base-small-moderate-large-extreme buckets
            "peak_diagnostics": {  # Includes requested global peak diagnostics
                "peak_real_mgd": peak_real_mgd,
                "peak_pred_mgd": peak_pred_mgd,
                "peak_error_mgd": peak_error_mgd,
                "peak_error_pct": peak_error_pct,
                "max_pred_mgd": peak_pred_mgd,
                "max_real_mgd": peak_real_mgd,
            },
            "top_10_peak_pred_real_ratios": top_10_ratios,  # Includes pred/real ratio of the 10 highest real peaks
        },
        "residual_diagnostics": {  # Includes statistical summary and residual behavior
            "mean": residual_mean,
            "median": residual_median,
            "std": residual_std,
            "pct_overestimation_gt_2_mgd": pct_over_gt_2,
            "pct_underestimation_gt_2_mgd": pct_under_gt_2,
            "corr_real_magnitude_vs_residual": residual_corr,
        },
        "normalization_diagnostics": normalization_diagnostics,  # Includes compression and low-scale concentration checks
    }

    print("[diag] === Normalized Target Distribution ===")  # Prints header for section 1 for quick console reading
    print(  # Prints short summary by split with quantiles and low concentration
        f"[diag] train pct<0.05={distributions['train']['pct_lt_0_05']:.2f}% | "
        f"val pct<0.05={distributions['val']['pct_lt_0_05']:.2f}% | "
        f"test pct<0.05={distributions['test']['pct_lt_0_05']:.2f}%"
    )

    print("[diag] === Predictions (MGD) ===")  # Prints header for section 2 for global metrics
    print(  # Summarizes NSE, RMSE, and MAE for quick comparison between iterations
        f"[diag] NSE={nse_value:.4f} | RMSE={rmse_value:.4f} | MAE={mae_value:.4f}"
    )
    print(  # Summarizes global peak diagnostic to inspect amplitude compression/expansion
        f"[diag] Real peak={peak_real_mgd:.4f} | Pred peak={peak_pred_mgd:.4f} | Error%={peak_error_pct:.2f}%"
    )

    print("[diag] === Residuals ===")  # Prints header for section 3 for bias and spread
    print(  # Summarizes residual central tendency and spread
        f"[diag] mean={residual_mean:.4f} | median={residual_median:.4f} | std={residual_std:.4f}"
    )
    print(  # Summarizes rates of large over/underestimation errors
        f"[diag] over>2MGD={pct_over_gt_2:.2f}% | under>2MGD={pct_under_gt_2:.2f}% | corr(real,res)={residual_corr:.4f}"
    )

    print("[diag] === Normalization ===")  # Prints header for section 4 for compression diagnostic
    print(  # Reports compression ratio between normalized and real span in P95-P99
        f"[diag] Compression ratio P95-P99 (norm/real)={compression_ratio:.6f}"
    )
    print(  # Reports whether there is excessive concentration below 0.05 in any split
        f"[diag] Flag >80% target_norm <0.05 (any split): {normalization_diagnostics['high_mass_below_0_05_flags']['any_split_gt_80pct']}"
    )

    return diagnostics  # Returns the full dictionary for later saving to JSON
