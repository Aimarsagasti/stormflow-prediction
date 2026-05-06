"""Evaluation metrics for stormflow prediction."""

from __future__ import annotations  # Allows modern annotations without version conflicts

from typing import Dict, Optional  # Defines explicit types for metric return and optional mask

import numpy as np  # Provides vectorized numerical operations for metrics

from src.pipeline.normalize import denormalize_target  # Reuses official denormalization function from the pipeline


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE safely for possibly empty arrays."""
    if y_true.size == 0:  # Avoids invalid operations when there are no samples in the subset
        return float("nan")  # Returns NaN to indicate undefined metric
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))  # Computes root mean squared error


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE safely for possibly empty arrays."""
    if y_true.size == 0:  # Avoids mean over empty array
        return float("nan")  # Returns NaN when there is no data to compute MAE
    return float(np.mean(np.abs(y_pred - y_true)))  # Computes mean absolute error


def _safe_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency safely."""
    if y_true.size == 0:  # Avoids computing NSE on empty arrays
        return float("nan")  # Returns NaN when there are no available samples
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)  # Computes total observed variance for NSE denominator
    if denominator <= 0:  # Controls degenerate cases with no variance in the real series
        return float("nan")  # Returns NaN when NSE is not interpretable
    numerator = np.sum((y_true - y_pred) ** 2)  # Computes model sum of squared errors
    return float(1.0 - (numerator / denominator))  # Returns NSE where 1.0 indicates perfect prediction


def _bucket_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute severity bucket metrics using proposal-defined thresholds."""
    buckets = {  # Defines severity cuts according to the project proposal
        "base": (None, 0.5),  # Base bucket for values below 0.5 MGD
        "pequeno": (0.5, 2.0),  # Small bucket for 0.5 to 2 MGD range
        "moderado": (2.0, 12.8),  # Moderate bucket for 2 to 12.8 MGD range
        "grande": (12.8, 51.0),  # Large bucket for 12.8 to 51 MGD range
        "extremo": (51.0, None),  # Extreme bucket for values above 51 MGD
    }
    results: Dict[str, Dict[str, float]] = {}  # Prepares container of metrics by bucket

    for bucket_name, (lower, upper) in buckets.items():  # Iterates buckets to compute metrics by severity
        if lower is None:  # Handles bucket with upper bound only
            mask = y_true < upper  # Selects samples below the upper threshold
        elif upper is None:  # Handles bucket with lower bound only
            mask = y_true > lower  # Selects samples above the lower threshold
        else:  # Handles bucket with lower and upper limits
            mask = (y_true >= lower) & (y_true < upper)  # Selects samples within the defined interval

        bucket_true = y_true[mask]  # Extracts real target of current bucket
        bucket_pred = y_pred[mask]  # Extracts prediction of current bucket

        if bucket_true.size == 0:  # Avoids computing metrics when there are no samples in the bucket
            results[bucket_name] = {  # Fills with NaN and n=0 to keep consistent structure
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "n_samples": 0.0,
            }
            continue  # Moves to the next bucket after recording empty values

        bias_value = float(np.mean(bucket_pred - bucket_true))  # Computes mean bias (pred-real) of the bucket
        results[bucket_name] = {  # Stores metrics by bucket for report and final return
            "rmse": _safe_rmse(bucket_true, bucket_pred),  # Computes RMSE in the current bucket
            "mae": _safe_mae(bucket_true, bucket_pred),  # Computes MAE in the current bucket
            "bias": bias_value,  # Stores mean bias to detect systematic under/overestimation
            "n_samples": float(bucket_true.size),  # Stores number of samples in the bucket
        }

    return results  # Returns complete dictionary of metrics by severity


def evaluate_model(
    y_real_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    norm_params: Dict[str, object],
    is_event: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Evaluate normalized predictions with denormalization and severity breakdown."""
    y_real_norm = np.asarray(y_real_norm, dtype=float).reshape(-1)  # Converts normalized target to consistent 1D vector
    y_pred_norm = np.asarray(y_pred_norm, dtype=float).reshape(-1)  # Converts normalized prediction to consistent 1D vector
    if y_real_norm.shape[0] != y_pred_norm.shape[0]:  # Validates same number of samples for pointwise comparison
        raise ValueError("y_real_norm and y_pred_norm must have the same length")  # Raises clear error if there is misalignment

    y_real = denormalize_target(y_real_norm, norm_params)  # Moves real target from normalized scale to physical MGD units
    y_pred = denormalize_target(y_pred_norm, norm_params)  # Moves prediction from normalized scale to physical MGD units
    y_real = np.clip(y_real, a_min=0.0, a_max=None)  # Reinforces physical non-negative stormflow constraint in final metrics
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)  # Prevents numerically negative outputs from distorting physical metrics

    global_nse = _safe_nse(y_real, y_pred)  # Computes global NSE in real units
    global_rmse = _safe_rmse(y_real, y_pred)  # Computes global RMSE in MGD
    global_mae = _safe_mae(y_real, y_pred)  # Computes global MAE in MGD

    if y_real.size > 0:  # Checks existence of data to compute global peak
        real_peak_value = float(np.max(y_real))  # Gets global observed real peak
        pred_peak_value = float(np.max(y_pred))  # Gets global peak predicted by the model
        peak_error_mgd = pred_peak_value - real_peak_value  # Computes signed peak error (pred - real)
        peak_abs_error_mgd = float(np.abs(peak_error_mgd))  # Computes absolute peak error for robust reporting
        peak_error_pct = float((peak_error_mgd / max(real_peak_value, 1e-12)) * 100.0)  # Computes relative peak error in percentage
    else:  # Handles empty case to keep output consistent
        real_peak_value = float("nan")  # Marks real peak as undefined when there are no samples
        pred_peak_value = float("nan")  # Marks predicted peak as undefined when there are no samples
        peak_error_mgd = float("nan")  # Marks peak error as undefined
        peak_abs_error_mgd = float("nan")  # Marks absolute error as undefined
        peak_error_pct = float("nan")  # Marks percentage error as undefined

    event_nse = float("nan")  # Initializes event-only NSE as NaN by default when no mask is passed
    event_sample_count = 0.0  # Initializes event sample counter for metric traceability
    if is_event is not None:  # Checks whether event-only evaluation was requested
        event_mask = np.asarray(is_event).reshape(-1).astype(bool)  # Converts event mask to 1D bool
        if event_mask.shape[0] != y_real.shape[0]:  # Validates alignment between mask and prediction/real arrays
            raise ValueError("is_event must have the same length as y_real_norm and y_pred_norm")  # Raises clear error on incompatible lengths
        event_sample_count = float(event_mask.sum())  # Counts how many samples in the evaluated vector belong to real events
        event_nse = _safe_nse(y_real[event_mask], y_pred[event_mask])  # Computes NSE only on samples marked as events

    severity_metrics = _bucket_metrics(y_true=y_real, y_pred=y_pred)  # Computes breakdown of metrics by severity buckets

    metrics: Dict[str, object] = {  # Builds final metrics dictionary for structured return
        "global": {  # Groups main global metrics for direct reading
            "nse": global_nse,
            "rmse": global_rmse,
            "mae": global_mae,
            "peak_real_mgd": real_peak_value,
            "peak_pred_mgd": pred_peak_value,
            "peak_error_mgd": peak_error_mgd,
            "peak_abs_error_mgd": peak_abs_error_mgd,
            "peak_error_pct": peak_error_pct,
        },
        "event_only": {  # Groups event-restricted metrics to evaluate operational behavior
            "nse": event_nse,
            "n_samples": event_sample_count,
        },
        "severity": severity_metrics,  # Includes full severity-bucket breakdown
    }

    print("[metrics] === Global Metrics ===")  # Print header for global metrics section
    print(f"[metrics] NSE: {global_nse:.4f}")  # Prints formatted global NSE
    print(f"[metrics] RMSE: {global_rmse:.4f} MGD")  # Prints global RMSE in physical units
    print(f"[metrics] MAE: {global_mae:.4f} MGD")  # Prints global MAE in physical units
    print(f"[metrics] Real peak: {real_peak_value:.4f} MGD")  # Prints real global peak value
    print(f"[metrics] Predicted peak: {pred_peak_value:.4f} MGD")  # Prints predicted global peak value
    print(f"[metrics] Peak error: {peak_error_mgd:.4f} MGD ({peak_error_pct:.2f}%)")  # Prints signed peak error in MGD and percentage
    if is_event is not None:  # Prints event metric only if a mask was provided
        print(f"[metrics] NSE (events): {event_nse:.4f} | samples: {int(event_sample_count)}")  # Reports NSE restricted to event subset and its coverage

    print("[metrics] === Breakdown by Severity ===")  # Print header for severity buckets
    for bucket_name, bucket_values in severity_metrics.items():  # Iterates each bucket to print its metrics
        print(  # Prints per-bucket summary with all requested metrics
            f"[metrics] {bucket_name}: n={int(bucket_values['n_samples'])} | "
            f"RMSE={bucket_values['rmse']:.4f} | "
            f"MAE={bucket_values['mae']:.4f} | "
            f"Bias={bucket_values['bias']:.4f}"
        )

    return metrics  # Returns complete dictionary for logging, reports, and persistence
