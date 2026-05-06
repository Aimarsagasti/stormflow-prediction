"""Sequence and DataLoader utilities for MSD stormflow modeling."""

from __future__ import annotations  # Allows modern type annotations with good compatibility

from typing import Dict, List, Tuple  # Defines clear types for datasets and pipeline returns

import numpy as np  # Provides vectorized operations to build windows efficiently
import pandas as pd  # Provides tabular access to extract input arrays
import torch  # Tensor base used to build datasets and dataloaders in PyTorch
from torch.utils.data import DataLoader, Dataset  # Data loading utilities for training and evaluation


class StormflowSequenceDataset(Dataset):
    """PyTorch dataset that returns (X, y, sample_weight, event_target)."""

    def __init__(
        self,
        x_array: np.ndarray,
        y_array: np.ndarray,
        weight_array: np.ndarray,
        event_array: np.ndarray,
    ) -> None:
        self.x_tensor = torch.tensor(x_array, dtype=torch.float32)  # Converts input windows to float32 tensor for GPU training
        self.y_tensor = torch.tensor(y_array, dtype=torch.float32).unsqueeze(-1)  # Converts target to a column tensor for compatibility with regression models
        self.weight_tensor = torch.tensor(weight_array, dtype=torch.float32).unsqueeze(-1)  # Converts per-sample weight to a column tensor for weighted loss
        self.event_tensor = torch.tensor(event_array.astype(np.float32), dtype=torch.float32).unsqueeze(-1)  # Converts the future event label to a column tensor for multitask learning

    def __len__(self) -> int:
        return int(self.x_tensor.shape[0])  # Returns the total number of sequences available in the dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (  # Returns the full tuple required by multitask training
            self.x_tensor[index],  # Returns the already tensorized input sequence
            self.y_tensor[index],  # Returns the scalar target aligned with the horizon
            self.weight_tensor[index],  # Returns the sample weight for the composite loss
            self.event_tensor[index],  # Returns the boolean event label to predict
        )


def _build_window_arrays(
    df_split: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window arrays and aligned target/event vectors."""
    feature_matrix = df_split[feature_columns].to_numpy(dtype=np.float32)  # Extracts feature matrix in float32 to reduce memory
    target_array = df_split[target_col].to_numpy(dtype=np.float32)  # Extracts target into a contiguous array for fast indexing
    event_array = df_split[aux_col].to_numpy(dtype=bool)  # Extracts auxiliary event indicator for multitask supervision

    total_rows = len(df_split)  # Gets split size to calculate how many windows can be formed
    max_end_index = total_rows - horizon - 1  # Defines the last valid t index that allows access to t+horizon without leaving the array
    if max_end_index < seq_length - 1:  # Detects cases with too little data to build even one window
        empty_x = np.empty((0, seq_length, len(feature_columns)), dtype=np.float32)  # Creates empty tensor with consistent shape so the pipeline does not break
        empty_y = np.empty((0,), dtype=np.float32)  # Creates empty target vector when there are no samples
        empty_event = np.empty((0,), dtype=bool)  # Creates empty event vector to preserve the return contract
        return empty_x, empty_y, empty_event  # Returns empty structures that the caller can handle

    windows_x: List[np.ndarray] = []  # Accumulates input sequences for each valid t index
    windows_y: List[float] = []  # Accumulates scalar target associated with the future horizon
    windows_event: List[bool] = []  # Accumulates event label for the target sample

    for end_index in range(seq_length - 1, max_end_index + 1):  # Iterates through all t indices that allow a full window and full horizon
        start_index = end_index - seq_length + 1  # Computes the start of the causal window [t-seq_length+1, ..., t]
        target_index = end_index + horizon  # Computes the target index at t+horizon according to the user requirement
        windows_x.append(feature_matrix[start_index : end_index + 1])  # Stores feature block of length seq_length
        windows_y.append(float(target_array[target_index]))  # Stores scalar target value at the horizon
        windows_event.append(bool(event_array[target_index]))  # Stores the event label aligned exactly with the same target

    x_array = np.stack(windows_x).astype(np.float32)  # Converts list of windows into 3D array [N, L, F]
    y_array = np.asarray(windows_y, dtype=np.float32)  # Converts targets to 1D array for the dataset
    event_target_array = np.asarray(windows_event, dtype=bool)  # Converts event flags to boolean array aligned with the target

    return x_array, y_array, event_target_array  # Returns base tensors to build datasets and diagnostics


def _compute_quantile_thresholds(y_array: np.ndarray) -> Dict[str, float]:
    """Compute magnitude thresholds from the training target distribution."""
    if y_array.size == 0:  # Avoids computing quantiles when no samples are available
        return {"p95": float("nan"), "p99": float("nan"), "p999": float("nan")}  # Returns undefined thresholds if the split is empty
    return {  # Returns key quantiles for weights and severity diagnostics
        "p95": float(np.quantile(y_array, 0.95)),  # Defines a high threshold in the distribution
        "p99": float(np.quantile(y_array, 0.99)),  # Defines a very high threshold in the distribution
        "p999": float(np.quantile(y_array, 0.999)),  # Defines an extreme threshold in the distribution
    }


def _compute_sample_weights(y_array: np.ndarray, event_array: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Compute sample weights using both event presence and target magnitude."""
    if y_array.size == 0:  # Handles splits without samples to avoid later errors
        return np.empty((0,), dtype=np.float32)  # Returns empty vector when there are no targets

    p95 = thresholds["p95"]  # Retrieves the P95 threshold computed on train for consistency across splits
    p99 = thresholds["p99"]  # Retrieves the P99 threshold computed on train for consistency across splits
    p999 = thresholds["p999"]  # Retrieves the P99.9 threshold computed on train for consistency across splits

    sample_weights = np.ones_like(y_array, dtype=np.float32)  # Initializes base weights for the dominant low-magnitude regime
    sample_weights[event_array] = np.maximum(sample_weights[event_array], 1.75)  # Gives a mild boost to generic events without dominating over true future magnitude
    sample_weights[(y_array >= p95) & (y_array < p99)] = 5.0  # Increases weight for the high tail without overreacting at intermediate severity
    sample_weights[(y_array >= p99) & (y_array < p999)] = 12.0  # Further increases weight for the very high tail where underestimation is costly
    sample_weights[y_array >= p999] = 24.0  # Gives highest priority to critically extreme magnitudes for the operational objective
    return sample_weights  # Returns per-sample weight vector aligned with the target


def _compute_event_pos_weight(event_array: np.ndarray) -> float:
    """Compute a bounded positive-class weight for event BCE."""
    if event_array.size == 0:  # Handles empty datasets without breaking auxiliary classification
        return 1.0  # Returns neutral weight when there are no samples
    positive_count = int(event_array.sum())  # Counts how many target windows belong to an event
    negative_count = int(event_array.size - positive_count)  # Counts how many target windows belong to non-event
    if positive_count == 0 or negative_count == 0:  # Avoids division by zero in degenerate cases
        return 1.0  # Returns neutral weight when only one class exists in the split
    raw_ratio = negative_count / positive_count  # Computes negative/positive ratio to compensate for natural imbalance
    return float(np.clip(raw_ratio, 1.0, 25.0))  # Limits the weight to avoid excessive or unstable gradients


def _build_single_loader(
    df_split: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    thresholds: Dict[str, float],
) -> Tuple[DataLoader, Dict[str, object]]:
    """Create one DataLoader and diagnostics for a split."""
    x_array, y_array, event_array = _build_window_arrays(  # Builds causal windows and targets aligned to the horizon
        df_split=df_split,  # Uses the current split as the source of sequences
        feature_columns=feature_columns,  # Uses the feature order defined by the pipeline
        target_col=target_col,  # Uses the already normalized target column from the split
        aux_col=aux_col,  # Uses the auxiliary event flag for multitask supervision
        seq_length=seq_length,  # Uses the history length defined for the model
        horizon=horizon,  # Uses the target horizon defined for prediction
    )
    sample_weight_array = _compute_sample_weights(y_array=y_array, event_array=event_array, thresholds=thresholds)  # Computes weights with fixed train thresholds and explicit priority on future magnitude
    event_pos_weight = _compute_event_pos_weight(event_array)  # Computes positive-class weight for BCE in this split

    dataset = StormflowSequenceDataset(  # Wraps arrays in a PyTorch dataset with multitask supervision
        x_array=x_array,  # Passes already built input windows
        y_array=y_array,  # Passes scalar targets per window
        weight_array=sample_weight_array,  # Passes sample weights for the composite loss
        event_array=event_array,  # Passes event labels aligned with the future target
    )
    dataloader = DataLoader(  # Creates DataLoader for efficient batch iteration
        dataset,  # Uses the sequence dataset already built for the current split
        batch_size=batch_size,  # Uses batch size configured for training or evaluation
        shuffle=shuffle,  # Shuffles only train and preserves temporal order in validation/test
        drop_last=False,  # Keeps the last incomplete batch so samples are not lost
    )

    diagnostics: Dict[str, object] = {  # Packs useful statistics for pipeline traceability
        "num_windows": int(len(dataset)),  # Stores total number of windows created for the split
        "x_shape": tuple(x_array.shape),  # Stores input shape [N, seq_length, n_features]
        "y_shape": tuple(y_array.shape),  # Stores target shape [N]
        "num_batches": int(len(dataloader)),  # Stores resulting batch count
        "event_windows": int(event_array.sum()),  # Stores number of windows whose target label is an event
        "event_rate": float(event_array.mean()) if event_array.size > 0 else float("nan"),  # Stores the natural event rate at that horizon
        "event_pos_weight": float(event_pos_weight),  # Stores suggested weight for event BCE
        "peak_windows_p95": int((y_array >= thresholds["p95"]).sum()) if y_array.size > 0 else 0,  # Stores how many windows belong to the tail >= P95
        "peak_windows_p99": int((y_array >= thresholds["p99"]).sum()) if y_array.size > 0 else 0,  # Stores how many windows belong to the tail >= P99
        "thresholds": thresholds,  # Stores P95/P99/P99.9 thresholds used in weights and diagnostics
        "shuffle": bool(shuffle),  # Records whether the loader shuffles or preserves temporal order
    }
    setattr(dataloader, "stormflow_diagnostics", diagnostics)  # Attaches diagnostics to the loader so they can be recovered without changing the public signature

    return dataloader, diagnostics  # Returns ready-to-use DataLoader and diagnostic metadata


def create_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    aux_col: str,
    seq_length: int = 72,
    horizon: int = 6,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders preserving the natural validation/test distribution."""
    train_thresholds = _compute_quantile_thresholds(df_train[target_col].to_numpy(dtype=np.float32))  # Computes magnitude thresholds once on train

    train_loader, train_diag = _build_single_loader(  # Creates train loader with natural distribution and light shuffling
        df_split=df_train,  # Uses train split as the sequence source
        feature_columns=feature_columns,  # Passes input columns in the correct order
        target_col=target_col,  # Passes the already normalized target name
        aux_col=aux_col,  # Passes the auxiliary event flag
        seq_length=seq_length,  # Passes model history length
        horizon=horizon,  # Passes the desired prediction horizon
        batch_size=batch_size,  # Passes configured batch size
        shuffle=True,  # Shuffles train to break intra-batch correlation without altering val/test
        thresholds=train_thresholds,  # Uses train quantiles for weights and categories
    )
    val_loader, val_diag = _build_single_loader(  # Creates validation loader with stable order for temporal analysis
        df_split=df_val,  # Uses validation split in chronological order
        feature_columns=feature_columns,  # Passes input columns
        target_col=target_col,  # Passes normalized target name
        aux_col=aux_col,  # Passes auxiliary flag for diagnostics
        seq_length=seq_length,  # Passes model sequence length
        horizon=horizon,  # Passes prediction horizon
        batch_size=batch_size,  # Passes evaluation batch size
        shuffle=False,  # Disables shuffling to preserve temporal alignment
        thresholds=train_thresholds,  # Uses the same train thresholds for consistent loss/metrics
    )
    test_loader, test_diag = _build_single_loader(  # Creates test loader with stable order for temporal analysis
        df_split=df_test,  # Uses test split in chronological order
        feature_columns=feature_columns,  # Passes input columns
        target_col=target_col,  # Passes normalized target name
        aux_col=aux_col,  # Passes auxiliary event flag for later diagnostics
        seq_length=seq_length,  # Passes defined sequence length
        horizon=horizon,  # Passes target horizon
        batch_size=batch_size,  # Passes batch size for inference/evaluation
        shuffle=False,  # Disables shuffling to preserve natural order
        thresholds=train_thresholds,  # Reuses train thresholds to evaluate on the same operational scale
    )

    print(f"[sequences] Train X shape: {train_diag['x_shape']} | y shape: {train_diag['y_shape']} | batches: {train_diag['num_batches']}")  # Reports train sizes to validate windowing and batching
    print(f"[sequences] Val   X shape: {val_diag['x_shape']} | y shape: {val_diag['y_shape']} | batches: {val_diag['num_batches']}")  # Reports validation sizes
    print(f"[sequences] Test  X shape: {test_diag['x_shape']} | y shape: {test_diag['y_shape']} | batches: {test_diag['num_batches']}")  # Reports test sizes

    print(f"[sequences] Event rate(train): {train_diag['event_rate']:.4f} | event BCE pos_weight: {train_diag['event_pos_weight']:.4f}")  # Prints natural event imbalance in train to configure the loss
    print(f"[sequences] Event rate(val): {val_diag['event_rate']:.4f} | event windows: {val_diag['event_windows']} | p95+ windows: {val_diag['peak_windows_p95']}")  # Prints event coverage and high tail in validation
    print(f"[sequences] Event rate(test): {test_diag['event_rate']:.4f} | event windows: {test_diag['event_windows']} | p95+ windows: {test_diag['peak_windows_p95']}")  # Prints event coverage and high tail in test
    print(f"[sequences] Weight thresholds(train): {train_diag['thresholds']}")  # Prints P95/P99/P99.9 used by weights and diagnostics

    return train_loader, val_loader, test_loader  # Returns loaders ready for training and evaluation
