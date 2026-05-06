"""Chronological split utilities for train/validation/test partitions."""

from __future__ import annotations  # Allows modern type annotations with broad compatibility

from typing import Tuple  # Defines explicit return types for the split

import pandas as pd  # Provides DataFrame and temporal ordering functions


def _print_split_diagnostics(split_name: str, split_df: pd.DataFrame) -> None:
    """Print basic diagnostics for a split."""
    if split_df.empty:  # Avoids min/max errors when a split is empty
        print(f"[split] {split_name}: 0 records | range: N/A | % event: N/A")  # Reports empty split for debugging
        return  # Stops the function because there are no more useful statistics

    time_min = split_df["timestamp"].min()  # Gets the temporal start of the split to validate chronological order
    time_max = split_df["timestamp"].max()  # Gets the temporal end of the split to validate temporal coverage
    event_pct = split_df["is_event"].mean() * 100.0 if "is_event" in split_df.columns else float("nan")  # Computes event percentage if the column exists
    event_text = f"{event_pct:.2f}%" if "is_event" in split_df.columns else "N/A"  # Formats event percentage text for printing
    print(f"[split] {split_name}: {len(split_df)} records | range: {time_min} -> {time_max} | % event: {event_text}")  # Prints the summary required by the pipeline


def split_chronological(
    df_feat: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split features DataFrame into train/val/test in strict chronological order."""
    if "timestamp" not in df_feat.columns:  # Validates that the temporal column required by the user exists
        raise ValueError("df_feat must contain a 'timestamp' column for chronological splitting")  # Raises an explicit error to simplify debugging
    if train_ratio <= 0 or val_ratio <= 0:  # Avoids invalid ratios that break the training scheme
        raise ValueError("train_ratio and val_ratio must be positive")  # Clear message to correct configuration
    if train_ratio + val_ratio >= 1.0:  # Guarantees that a segment remains for test
        raise ValueError("train_ratio + val_ratio must be < 1.0")  # Restricts configuration to a valid split

    df_sorted = df_feat.copy()  # Works on a copy to avoid mutating the received DataFrame
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], errors="coerce")  # Ensures datetime type for correct sorting
    df_sorted = df_sorted.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)  # Removes invalid timestamps and sorts in time

    total_rows = len(df_sorted)  # Gets total available records to distribute across splits
    train_end = int(total_rows * train_ratio)  # Computes cut index for train according to requested ratio
    val_end = int(total_rows * (train_ratio + val_ratio))  # Computes cumulative cut index for train+val

    df_train = df_sorted.iloc[:train_end].copy().reset_index(drop=True)  # Extracts initial segment for training without mixing future data
    df_val = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)  # Extracts middle segment for temporal validation
    df_test = df_sorted.iloc[val_end:].copy().reset_index(drop=True)  # Extracts final segment for testing on the most recent data

    print(f"[split] Total records: {total_rows}")  # Reports global size before detailing each split
    _print_split_diagnostics("train", df_train)  # Prints train diagnostics as required
    _print_split_diagnostics("val", df_val)  # Prints validation diagnostics
    _print_split_diagnostics("test", df_test)  # Prints test diagnostics

    return df_train, df_val, df_test  # Returns the three DataFrames for the rest of the pipeline
