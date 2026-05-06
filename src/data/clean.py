"""Data cleaning utilities for MSD time series."""

from __future__ import annotations  # Allows using modern type annotations in all supported versions

import numpy as np  # Provides vectorized operations to build efficient masks
import pandas as pd  # Provides the DataFrame and time-cleaning utilities


def _build_event_mask(timestamps: pd.Series, df_events: pd.DataFrame) -> pd.Series:
    """Build a boolean mask indicating whether each timestamp is within an event interval."""
    valid_events = df_events.dropna(subset=["event_start", "event_end"]).copy()  # Ensures events with valid start and end
    if valid_events.empty:  # Avoids extra work when there are no usable events
        return pd.Series(False, index=timestamps.index)  # Returns an all-false mask for the whole period

    starts = np.sort(valid_events["event_start"].to_numpy(dtype="datetime64[ns]"))  # Sorts starts for binary search
    ends = np.sort(valid_events["event_end"].to_numpy(dtype="datetime64[ns]"))  # Sorts ends for binary search
    ts_values = timestamps.to_numpy(dtype="datetime64[ns]")  # Converts timestamps to a native array to speed up computations

    started_count = np.searchsorted(starts, ts_values, side="right")  # Counts events started at or before each timestamp
    ended_before_count = np.searchsorted(ends, ts_values, side="left")  # Counts events ended strictly before the timestamp
    is_event_mask = started_count > ended_before_count  # Marks active when there is at least one open event

    return pd.Series(is_event_mask, index=timestamps.index)  # Returns the mask aligned with the original DataFrame


def _infer_resolution_minutes(timestamps: pd.Series) -> int:
    """Infer sampling resolution in minutes from timestamp differences."""
    deltas = timestamps.sort_values().diff().dropna()  # Computes differences between consecutive timestamps
    if deltas.empty:  # Handles single-row cases to avoid division by zero
        return 5  # Uses 5 minutes by default according to the project resolution
    resolution = int(round(deltas.dt.total_seconds().median() / 60.0))  # Estimates resolution using the robust median
    return max(resolution, 1)  # Guarantees at least 1 minute to avoid invalid parameters


def clean_timeseries(df_timeseries: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """Clean MSD time series and append is_event flag."""
    df_clean = df_timeseries.copy()  # Works on a copy to avoid mutating the input DataFrame
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")  # Forces datetime type for sorting and interpolation
    initial_rows = len(df_clean)  # Stores the initial size for global diagnostics
    df_clean = df_clean.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)  # Removes invalid timestamps and sorts chronologically
    print(f"[clean] Initial records: {initial_rows}")  # Reports the starting point before cleaning
    print(f"[clean] Records after timestamp validation: {len(df_clean)}")  # Reports how many rows remain after time validation

    negative_stormflow_count = int((df_clean["stormflow_mgd"] < 0).sum())  # Counts negative stormflow identified as an artifact
    df_clean["stormflow_mgd"] = df_clean["stormflow_mgd"].clip(lower=0)  # Corrects negative stormflow to zero due to the physical constraint
    print(f"[clean] Negative stormflow clipped to 0: {negative_stormflow_count}")  # Reports how many rows were affected in step 1

    negative_flow_count = int((df_clean["flow_total_mgd"] < 0).sum())  # Counts negative total flow values due to possible sensor error
    df_clean.loc[df_clean["flow_total_mgd"] < 0, "flow_total_mgd"] = np.nan  # Converts negative values to NaN for controlled repair
    resolution_minutes = _infer_resolution_minutes(df_clean["timestamp"])  # Infers the real time resolution to compute the gap limit
    max_gap_steps = max(int(30 / resolution_minutes), 1)  # Converts 30 minutes into the number of series steps
    nan_before_interp = int(df_clean["flow_total_mgd"].isna().sum())  # Counts gaps before interpolation for diagnostics
    df_clean = df_clean.set_index("timestamp")  # Uses timestamp as index for time-based interpolation with the real clock
    df_clean["flow_total_mgd"] = df_clean["flow_total_mgd"].interpolate(  # Interpolates only short internal gaps
        method="time",  # Time interpolation that considers the real distance between timestamps
        limit=max_gap_steps,  # Limits filling to gaps up to 30 minutes
        limit_area="inside",  # Avoids extrapolating at the start or end of the series
    )
    nan_after_interp = int(df_clean["flow_total_mgd"].isna().sum())  # Counts gaps that remained unrepaired after interpolation
    large_gap_rows_removed = nan_after_interp  # Interprets remaining NaN as a gap larger than the allowed limit
    df_clean = df_clean.dropna(subset=["flow_total_mgd"]).reset_index()  # Removes rows in long gaps and restores timestamp as a column
    print(f"[clean] Negative flow_total converted to NaN: {negative_flow_count}")  # Reports how many readings were marked as missing
    print(f"[clean] NaN in flow_total before interpolation: {nan_before_interp}")  # Reports the size of gaps before filling
    print(f"[clean] NaN in flow_total after interpolation: {nan_after_interp}")  # Reports how many gaps could not be repaired
    print(f"[clean] Rows removed for gaps > 30 min: {large_gap_rows_removed}")  # Reports removals due to long gaps

    storm_above_flow_count = int((df_clean["stormflow_mgd"] > df_clean["flow_total_mgd"]).sum())  # Counts physically inconsistent cases
    df_clean["stormflow_mgd"] = df_clean["stormflow_mgd"].clip(upper=df_clean["flow_total_mgd"])  # Limits stormflow so it does not exceed total flow
    print(f"[clean] Records with stormflow > flow clipped: {storm_above_flow_count}")  # Reports hydrologic consistency corrections

    df_clean["baseflow_mgd"] = df_clean["flow_total_mgd"] - df_clean["stormflow_mgd"]  # Recomputes baseflow from corrected variables
    negative_baseflow_count = int((df_clean["baseflow_mgd"] < 0).sum())  # Counts negative baseflow before the final clip
    df_clean["baseflow_mgd"] = df_clean["baseflow_mgd"].clip(lower=0)  # Reinforces the physical constraint of non-negative baseflow
    print(f"[clean] Baseflow recomputed and clipped to 0 (previous negatives): {negative_baseflow_count}")  # Reports baseflow adjustments

    df_clean["is_event"] = _build_event_mask(df_clean["timestamp"], df_events).astype(bool)  # Marks each row according to event membership
    valid_event_count = int(df_clean["is_event"].sum())  # Counts how many rows fall inside event windows
    print(f"[clean] Records marked as event: {valid_event_count}")  # Reports temporal event coverage in the final series
    print(f"[clean] Final shape: {df_clean.shape}")  # Summarizes final dimensions after all cleaning

    return df_clean  # Returns the cleaned series with auxiliary is_event column for later stages
