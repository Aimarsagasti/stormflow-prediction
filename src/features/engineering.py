"""Feature engineering utilities for MSD stormflow modeling."""

from __future__ import annotations  # Allows modern type annotations with broad compatibility

import numpy as np  # Provides vectorized operations for time-series calculations
import pandas as pd  # Provides tabular structure and rolling/datetime functions


API_K_BASE = 0.90  # Defines the base API persistence under neutral thermal conditions
API_ALPHA = 0.002  # Defines how much K changes per Fahrenheit degree relative to the reference
API_TEMP_REF_F = 50.0  # Uses 50 F as the neutral thermal reference for API decay
API_K_MIN = 0.80  # Prevents K from dropping too much and making hydrologic memory unstable
API_K_MAX = 0.98  # Prevents K from rising too much and accumulating excessive memory for many days


def _infer_resolution_minutes(timestamps: pd.Series) -> int:
    """Infer sampling resolution in minutes from timestamp differences."""
    deltas = timestamps.sort_values().diff().dropna()  # Computes time deltas to estimate the real resolution
    if deltas.empty:  # Avoids errors when there is too little data to estimate differences
        return 5  # Uses 5 minutes by default according to dataset specification
    resolution = int(round(deltas.dt.total_seconds().median() / 60.0))  # Uses median for robustness against gaps
    return max(resolution, 1)  # Guarantees a positive value to avoid invalid divisions


def _add_rain_features(df_feat: pd.DataFrame, resolution_minutes: int) -> pd.DataFrame:
    """Create rainfall rolling and recency features."""
    rolling_sum_minutes = [10, 15, 30, 60, 120, 180, 360]  # Defines requested accumulation windows plus an extra intermediate level for recent moisture
    rolling_max_minutes = [10, 30, 60]  # Defines requested maximum windows for intense rainfall

    for window_minutes in rolling_sum_minutes:  # Iterates through each requested accumulation window
        window_steps = max(int(window_minutes / resolution_minutes), 1)  # Converts minutes to series steps
        column_name = f"rain_sum_{window_minutes}m"  # Builds a consistent feature name
        df_feat[column_name] = df_feat["rain_in"].rolling(window=window_steps, min_periods=1).sum()  # Computes causal rolling sum

    for window_minutes in rolling_max_minutes:  # Iterates through each requested maximum window
        window_steps = max(int(window_minutes / resolution_minutes), 1)  # Converts minutes to steps according to resolution
        column_name = f"rain_max_{window_minutes}m"  # Defines the column name for rolling maximum
        df_feat[column_name] = df_feat["rain_in"].rolling(window=window_steps, min_periods=1).max()  # Computes causal rolling maximum

    rain_positive_mask = df_feat["rain_in"] > 0  # Marks timestamps with rainfall to measure recency
    index_values = np.arange(len(df_feat))  # Creates a numeric index to compute step distances
    last_rain_index = np.where(rain_positive_mask.to_numpy(), index_values, -1)  # Stores the current index only when it rains
    last_rain_index = np.maximum.accumulate(last_rain_index)  # Propagates the last rainfall index forward
    steps_since_rain = index_values - last_rain_index  # Computes how many steps passed since the last rainfall
    has_previous_rain = last_rain_index >= 0  # Identifies rows where rainfall has already occurred before
    cap_minutes = 24 * 60  # Defines a 24-hour cap according to the requirement
    minutes_since_rain = np.where(has_previous_rain, steps_since_rain * resolution_minutes, cap_minutes)  # Converts to minutes and assigns the initial cap
    df_feat["minutes_since_last_rain"] = np.minimum(minutes_since_rain, cap_minutes).astype(float)  # Applies the 24h cap and casts to float

    return df_feat  # Returns DataFrame with the rainfall feature block added


def _prepare_temperature_feature(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Ensure daily temperature is available as a numeric feature without NaN."""
    if "temp_daily_f" not in df_feat.columns:  # Keeps compatibility if the loader has not yet added temperature
        df_feat["temp_daily_f"] = np.nan  # Creates the column so a neutral fallback can be applied later

    df_feat["temp_daily_f"] = pd.to_numeric(  # Forces the temperature column into a stable numeric format
        df_feat["temp_daily_f"],  # Daily temperature column coming from the load merge
        errors="coerce",  # Converts unusual values to NaN to handle them uniformly
    )
    missing_temperature_count = int(df_feat["temp_daily_f"].isna().sum())  # Counts rows without valid temperature before fallback
    if missing_temperature_count > 0:  # Only reports and fills when there are actually missing values
        print(f"[features] Missing daily temperature; fallback of {API_TEMP_REF_F:.1f} F will be used in {missing_temperature_count} rows")  # Explains the neutral fallback applied
        df_feat["temp_daily_f"] = df_feat["temp_daily_f"].fillna(API_TEMP_REF_F)  # Uses 50 F to avoid biasing API decay or introducing NaN

    return df_feat  # Returns the DataFrame with daily temperature ready to use as a direct feature


def _compute_dynamic_api(rain_values: np.ndarray, temp_values: np.ndarray) -> np.ndarray:
    """Compute sequential API with temperature-modulated decay."""
    api_values = np.zeros(shape=rain_values.shape[0], dtype=float)  # Reserves the output array for the sequential API
    previous_api = 0.0  # Initializes the previous API at zero at the start of the series

    for row_index, rain_value in enumerate(rain_values):  # Iterates sequentially because each step depends on the previous one
        temperature_value = temp_values[row_index]  # Takes the daily temperature already aligned with the current timestamp
        if np.isfinite(temperature_value):  # Uses the real temperature when it is available
            k_value = API_K_BASE - (API_ALPHA * (temperature_value - API_TEMP_REF_F))  # Modulates K to speed drying in heat and slow it in cold
        else:  # Keeps stable behavior if any residual NaN still appears
            k_value = API_K_BASE  # Applies the base K as the requested neutral fallback
        k_value = float(np.clip(k_value, API_K_MIN, API_K_MAX))  # Clamps K within the stable range specified by the user
        current_api = float(rain_value) + (k_value * previous_api)  # Applies the recurrence API(t) = rain(t) + K(t) * API(t-1)
        api_values[row_index] = current_api  # Stores the current API value for this timestamp
        previous_api = current_api  # Propagates the state to the next sequence step

    return api_values  # Returns the fully computed dynamic API series


def create_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Create model features, target, and auxiliary columns from cleaned time series."""
    required_columns = ["timestamp", "rain_in", "flow_total_mgd", "stormflow_mgd", "is_event"]  # Defines columns strictly required to build the feature set
    missing_columns = [column for column in required_columns if column not in df_clean.columns]  # Detects missing input columns
    if missing_columns:  # Validates schema to fail with a clear message in Colab
        raise ValueError(f"Missing required columns for feature engineering: {missing_columns}")  # Raises an explicit error to simplify debugging

    df_feat = df_clean.copy()  # Works on a copy to avoid mutating the cleaning output
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], errors="coerce")  # Ensures datetime type to derive temporal signals
    df_feat = df_feat.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)  # Removes invalid timestamps and sorts chronologically
    resolution_minutes = _infer_resolution_minutes(df_feat["timestamp"])  # Estimates resolution to convert windows from minutes to steps

    df_feat = _add_rain_features(df_feat, resolution_minutes)  # Adds rolling sums, rolling max, and rainfall recency
    df_feat = _prepare_temperature_feature(df_feat)  # Ensures daily temperature is available as a direct feature without NaN

    rain_values = pd.to_numeric(  # Converts rainfall to a float vector for recursive API computation
        df_feat["rain_in"],  # 5-minute incremental rainfall series
        errors="coerce",  # Protects against unexpected values although they should not exist
    ).fillna(0.0).to_numpy(dtype=float)  # Sends any eventual missing values to zero to avoid breaking the recurrence
    temp_values = df_feat["temp_daily_f"].to_numpy(dtype=float)  # Extracts daily temperature already aligned to modulate K at each step
    df_feat["api_dynamic"] = _compute_dynamic_api(rain_values=rain_values, temp_values=temp_values)  # Reintroduces dynamic API with temperature-dependent sequential memory

    steps_5m = max(int(5 / resolution_minutes), 1)  # Converts 5 minutes to the number of series steps
    steps_10m = max(int(10 / resolution_minutes), 1)  # Converts 10 minutes to the number of series steps
    steps_15m = max(int(15 / resolution_minutes), 1)  # Converts 15 minutes to the number of series steps
    steps_30m = max(int(30 / resolution_minutes), 1)  # Converts 30 minutes to the number of series steps
    df_feat["delta_flow_5m"] = df_feat["flow_total_mgd"].diff(periods=steps_5m).fillna(0.0)  # Computes flow_total variation over 5 minutes
    df_feat["delta_flow_15m"] = df_feat["flow_total_mgd"].diff(periods=steps_15m).fillna(0.0)  # Computes flow_total variation over 15 minutes
    df_feat["delta_rain_10m"] = df_feat["rain_in"].diff(periods=steps_10m).fillna(0.0)  # Exposes whether recent rainfall is intensifying or weakening at a very short horizon
    df_feat["delta_rain_30m"] = df_feat["rain_in"].diff(periods=steps_30m).fillna(0.0)  # Summarizes rainfall change over a slightly more stable window to distinguish sustained rises

    hour_fraction = (df_feat["timestamp"].dt.hour + (df_feat["timestamp"].dt.minute / 60.0)) / 24.0  # Converts hour of day into a continuous phase
    df_feat["hour_sin"] = np.sin(2.0 * np.pi * hour_fraction)  # Encodes hourly phase with sine to capture periodicity
    df_feat["hour_cos"] = np.cos(2.0 * np.pi * hour_fraction)  # Encodes hourly phase with cosine to close the cycle

    month_fraction = (df_feat["timestamp"].dt.month - 1) / 12.0  # Converts month into a normalized annual phase
    df_feat["month_sin"] = np.sin(2.0 * np.pi * month_fraction)  # Encodes monthly seasonality with sine
    df_feat["month_cos"] = np.cos(2.0 * np.pi * month_fraction)  # Encodes monthly seasonality with cosine

    feature_columns = [  # Lists input columns that will go into the model
        "rain_in",  # Keeps base rainfall as the primary causal feature
        "flow_total_mgd",  # Keeps total flow because of its high correlation with stormflow
        "temp_daily_f",  # Adds direct daily temperature because it modulates hydrologic response and evaporation
        "api_dynamic",  # Reintroduces API with antecedent moisture memory sensitive to temperature
        "rain_sum_10m",  # Rainfall accumulation in a short fast-response window
        "rain_sum_15m",  # Rainfall accumulation over 15 minutes
        "rain_sum_30m",  # Rainfall accumulation over half an hour
        "rain_sum_60m",  # Rainfall accumulation over one hour
        "rain_sum_120m",  # Rainfall accumulation over two hours
        "rain_sum_180m",  # Intermediate rainfall accumulation to distinguish recent saturation without relying only on 120 or 360 min
        "rain_sum_360m",  # Rainfall accumulation over six hours for moisture memory
        "rain_max_10m",  # Maximum recent rainfall over 10 minutes
        "rain_max_30m",  # Maximum recent rainfall over 30 minutes
        "rain_max_60m",  # Maximum recent rainfall over 60 minutes
        "minutes_since_last_rain",  # Rainfall recency with a 24-hour cap
        "delta_flow_5m",  # Short flow_total trend over 5 minutes
        "delta_flow_15m",  # Short flow_total trend over 15 minutes
        "delta_rain_10m",  # Immediate rainfall change to distinguish rapid intensification before the peak
        "delta_rain_30m",  # Slightly more stable rainfall change to separate brief pulses from growing episodes
        "hour_sin",  # Hourly sine cyclic component
        "hour_cos",  # Hourly cosine cyclic component
        "month_sin",  # Monthly sine cyclic component
        "month_cos",  # Monthly cosine cyclic component
    ]

    output_columns = ["timestamp", *feature_columns, "stormflow_mgd", "is_event"]  # Defines output with features, target, and auxiliary column
    df_output = df_feat.loc[:, output_columns].copy()  # Selects only final columns to avoid accidental leakage
    df_output["is_event"] = df_output["is_event"].astype(bool)  # Forces boolean type for the auxiliary column

    feature_count = len(feature_columns)  # Counts effective input features in the final DataFrame
    nan_count = int(df_output.isna().sum().sum())  # Computes global NaN count for quick diagnostics in Colab
    print(f"[features] Number of input features: {feature_count}")  # Reports how many features were generated
    print(f"[features] Final shape: {df_output.shape}")  # Reports dimensions of the resulting table
    print(f"[features] Total NaN count: {nan_count}")  # Reports remaining missing values after feature engineering

    return df_output  # Returns DataFrame ready for the split/sequence pipeline
