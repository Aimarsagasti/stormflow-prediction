"""Load MSD time series and event data from TSF and DAT files."""

from __future__ import annotations  # Allows type annotations as text in Python 3.7+

from pathlib import Path  # Safe handling of local or Google Drive paths
from typing import Any, Dict, List, Tuple  # Types that make the function contract explicit

import pandas as pd  # DataFrames for time series and event tables
import yaml  # YAML reading with configurable paths


DEFAULT_TEMP_F = 50.0  # Defines a neutral temperature as fallback when the file does not exist locally


def _read_config(config_path: str | Path) -> Dict[str, Any]:
    """Read YAML config and return it as a dict."""
    config_path = Path(config_path)  # Normalizes the path to behave the same on Windows or Colab
    raw_text = config_path.read_text(encoding="utf-8")  # Reads the whole file to handle trailing garbage
    try:  # First tries to parse the full YAML
        config = yaml.safe_load(raw_text)  # Uses safe_load to avoid code execution
    except yaml.YAMLError:  # If there is trailing non-YAML text, applies cleanup
        cleaned_text = raw_text.split("```", maxsplit=1)[0]  # Keeps only the YAML block before the fence
        config = yaml.safe_load(cleaned_text)  # Retries parsing with the cleaned content
    return config  # Returns the dictionary to extract paths and names


def _read_tsf_file(file_path: Path, value_name: str) -> pd.DataFrame:
    """Read a TSF file with 3 header lines and return a DataFrame."""
    data = pd.read_csv(  # Reads the tabular file in a single pass for efficiency
        file_path,  # Path to the .tsf file
        sep="\t",  # Tab separator for datetime and value columns
        header=None,  # The data has no useful header
        names=["timestamp", value_name],  # Defines consistent names in the DataFrame
        skiprows=3,  # Skips the 3 header lines according to the format
        parse_dates=[0],  # Converts the datetime column to date type for clean joins
    )
    return data  # Returns the DataFrame with timestamp and variable


def _read_daily_temperature_file(file_path: Path) -> pd.DataFrame:
    """Read the daily temperature TSF file and return one row per day."""
    data = pd.read_csv(  # Reads the daily temperature tabular file following the stated TSF format
        file_path,  # Path to the daily temperature file in Drive
        sep="\t",  # Uses tab because the file is tab-separated
        header=None,  # Does not use a header because the first 3 rows are metadata
        names=["date", "temp_daily_f"],  # Assigns explicit names for merge and feature engineering
        skiprows=3,  # Skips the 3 TSF header lines
    )
    data["date"] = pd.to_datetime(  # Converts the daily date to datetime to align with the main series
        data["date"],  # Date column in M/d/yyyy format
        format="%m/%d/%Y",  # Uses explicit format for stable parsing in Colab and local
        errors="coerce",  # Converts invalid dates to NaT so they can be dropped cleanly
    )
    data["temp_daily_f"] = pd.to_numeric(  # Converts temperature to float in case it arrives as text
        data["temp_daily_f"],  # Daily temperature column in Fahrenheit
        errors="coerce",  # Sends any unexpected value to NaN to filter it later
    )
    data = data.dropna(subset=["date", "temp_daily_f"])  # Keeps only complete rows ready for merge
    data["date"] = data["date"].dt.normalize()  # Moves each date to midnight to use a consistent daily key
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")  # Guarantees a single temperature per day

    print(f"[load] Daily temperature records: {len(data)}")  # Reports how many temperature days remained valid
    if not data.empty:  # Avoids printing invalid ranges if the DataFrame ended up empty
        print(f"[load] Daily temperature range: {data['date'].min()} -> {data['date'].max()}")  # Reports temporal coverage of the daily dataset

    return data  # Returns daily series ready to merge with the 5-minute series


def _merge_daily_temperature(df_timeseries: pd.DataFrame, df_temperature: pd.DataFrame) -> pd.DataFrame:
    """Merge daily temperature into the 5-minute time series using calendar date."""
    df_merged = df_timeseries.copy()  # Works on a copy to avoid mutating the original series outside the function
    df_merged["date"] = pd.to_datetime(  # Derives the daily date from each 5-minute timestamp
        df_merged["timestamp"],  # Base time column of the main dataset
        errors="coerce",  # Protects against invalid timestamps although they are not expected here
    ).dt.normalize()  # Normalizes to midnight to share the same daily key as the temperature dataset

    if df_temperature.empty:  # Handles locally the case where the Drive file does not exist
        df_merged["temp_daily_f"] = DEFAULT_TEMP_F  # Uses 50 F as a neutral value to avoid propagating NaN through the pipeline
        print(f"[load] Daily temperature unavailable; fallback of {DEFAULT_TEMP_F:.1f} F will be used")  # Explains the applied fallback
        return df_merged.drop(columns=["date"])  # Removes auxiliary key before returning the table

    df_merged = df_merged.merge(  # Joins daily temperature to each 5-minute row by calendar date
        df_temperature,  # Daily DataFrame with a single row per date
        on="date",  # Uses the daily date as join key
        how="left",  # Keeps the full main series even if a specific temperature is missing
    )
    missing_temperature_count = int(df_merged["temp_daily_f"].isna().sum())  # Counts rows without temperature after the merge
    if missing_temperature_count > 0:  # Applies fallback only when real gaps exist in temperature
        df_merged["temp_daily_f"] = df_merged["temp_daily_f"].fillna(DEFAULT_TEMP_F)  # Fills with 50 F to keep API neutrality
    print(f"[load] Rows without daily temperature after merge: {missing_temperature_count}")  # Reports effective coverage of the daily merge

    return df_merged.drop(columns=["date"])  # Removes the auxiliary key because it is not a final model feature


def _read_events_file(file_path: Path) -> pd.DataFrame:
    """Read the events .dat file with a trailing empty column."""
    event_columns = [  # Defines the 32 expected names in the event file
        "event_id",  # Numeric event identifier
        "is_valid",  # Validity flag as True/False text
        "event_start",  # Event start date and time
        "event_end",  # Event end date and time
        "failure_description",  # Failure description if present
        "precond_24h_volume",  # 24h preconditioning volume
        "volume_1h",  # 1h accumulated volume
        "volume_2h",  # 2h accumulated volume
        "volume_4h",  # 4h accumulated volume
        "volume_8h",  # 8h accumulated volume
        "volume_24h",  # 24h accumulated volume
        "total_volume",  # Total event volume
        "duration_hours",  # Duration in hours
        "total_rainfall",  # Total event rainfall
        "peak_wwf",  # Peak wet weather flow
        "reactive_wwf",  # Response variable to wwf
        "precond_24h_base_vol",  # Previous 24h base volume
        "precond_24h_base_vol_str",  # String field associated with previous baseflow
        "max_hourly_flow",  # Maximum hourly flow
        "avg_sewage_at_max",  # Average sewage at the maximum
        "event_end_index",  # Event end index
        "max_hourly_storm_intensity",  # Maximum hourly storm intensity
        "precond_24h_vol_str",  # String field associated with 24h volume
        "total_rain",  # Alternative total rainfall
        "total_baseflow_vol",  # Total baseflow volume
        "total_storm_vol",  # Total stormflow volume
        "ignore_has_impact",  # Ignore flag with impact
        "ignore_no_impact",  # Ignore flag without impact
        "runoff_coefficient",  # Runoff coefficient
        "event_start_index",  # Event start index
        "use_prev_24h_baseflow",  # Flag to use previous 24h baseflow
        "event_id_str",  # Event identifier as string
    ]
    data = pd.read_csv(  # Reads the event file in one block to keep speed
        file_path,  # Path to the .dat file
        sep="|",  # Pipe separator according to the event format
        header=None,  # The file has no header
        engine="python",  # Engine tolerant to irregular separators
    )
    if data.shape[1] > 0:  # Verifies that there are columns before trimming
        data = data.iloc[:, :-1]  # Drops the empty column generated by the trailing pipe
    data = data.iloc[:, : len(event_columns)]  # Ensures exactly the 32 expected columns
    data.columns = event_columns  # Assigns column names according to the schema

    data["event_start"] = pd.to_datetime(  # Converts start to datetime for temporal filtering
        data["event_start"],  # Original column with date text
        errors="coerce",  # Converts errors to NaT for safe filtering
    )
    data["event_end"] = pd.to_datetime(  # Converts end to datetime for temporal filtering
        data["event_end"],  # Original column with date text
        errors="coerce",  # Converts errors to NaT for safe filtering
    )

    data["is_valid"] = (  # Converts text flag to a real boolean
        data["is_valid"].astype(str)  # Ensures string type so it can be normalized
        .str.strip()  # Removes surrounding spaces
        .str.lower()  # Normalizes uppercase/lowercase
        .map({"true": True, "false": False})  # Maps text to boolean
    )

    numeric_columns = [  # List of key numeric columns to convert
        "total_volume",  # Total event volume
        "duration_hours",  # Duration in hours
        "total_rainfall",  # Total event rainfall
        "peak_wwf",  # Peak flow
        "total_rain",  # Alternative total rainfall
        "total_storm_vol",  # Total stormflow volume
        "runoff_coefficient",  # Runoff coefficient
    ]
    for column_name in numeric_columns:  # Iterates through columns for numeric conversion
        data[column_name] = pd.to_numeric(  # Converts text to float safely
            data[column_name],  # Column to convert
            errors="coerce",  # Converts invalid values to NaN
        )

    data = data[  # Filters valid events with correct dates
        (data["is_valid"] == True)  # Only events marked as valid
        & data["event_start"].notna()  # Excludes invalid starts
        & data["event_end"].notna()  # Excludes invalid ends
    ]

    return data  # Returns clean typed events


def _load_part(base_path: Path, tsf_files: Dict[str, str], events_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a single part (1parte or 2parte) and return time series and events."""
    rain_path = base_path / tsf_files["rain"]  # Builds the rainfall file path
    flow_path = base_path / tsf_files["flow"]  # Builds the total flow file path
    stormflow_path = base_path / tsf_files["stormflow"]  # Builds the stormflow file path
    events_path = base_path / events_filename  # Builds the event file path

    rain_df = _read_tsf_file(rain_path, "rain_in")  # Loads rainfall with the target name
    flow_df = _read_tsf_file(flow_path, "flow_total_mgd")  # Loads total flow with the target name
    stormflow_df = _read_tsf_file(stormflow_path, "stormflow_mgd")  # Loads stormflow with the target name

    merged_df = rain_df.merge(  # Joins rainfall with total flow by timestamp
        flow_df,  # Second DataFrame for the join
        on="timestamp",  # Exact join by time to align measurements
        how="inner",  # Inner join to keep only common timestamps
    )
    merged_df = merged_df.merge(  # Joins the result with stormflow by timestamp
        stormflow_df,  # Stormflow DataFrame
        on="timestamp",  # Aligns on the same time column
        how="inner",  # Inner join to avoid incomplete rows
    )

    merged_df["baseflow_mgd"] = (  # Computes baseflow by physical definition
        merged_df["flow_total_mgd"] - merged_df["stormflow_mgd"]  # Subtracts to separate baseflow
    )
    merged_df["baseflow_mgd"] = merged_df["baseflow_mgd"].clip(  # Avoids non-physical negative baseflow
        lower=0  # Lower bound at zero
    )

    events_df = _read_events_file(events_path)  # Loads events with the specific format

    return merged_df, events_df  # Returns both tables for later concatenation


def load_msd_data(config_path: str | Path = "configs/default.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MSD time series and event data from both parts defined in the config."""
    config = _read_config(config_path)  # Reads the configuration for paths and names
    data_cfg = config["data"]  # Extracts the data block to avoid long indexes
    base_paths = [Path(path) for path in data_cfg["base_paths"]]  # Normalizes part paths
    tsf_files = data_cfg["tsf_files"]  # Dictionary with TSF file names
    events_filename = data_cfg["events_filename"]  # Event file name
    temperature_daily_path = data_cfg.get("temperature_daily_path")  # Reads the optional daily temperature file path

    all_timeseries: List[pd.DataFrame] = []  # Accumulates time-series DataFrames from each part
    all_events: List[pd.DataFrame] = []  # Accumulates event DataFrames from each part

    for base_path in base_paths:  # Iterates through each part (1parte and 2parte)
        part_timeseries, part_events = _load_part(  # Loads time series and events for the part
            base_path,  # Part path
            tsf_files,  # TSF file names
            events_filename,  # Event file name
        )
        all_timeseries.append(part_timeseries)  # Stores the time series for concatenation
        all_events.append(part_events)  # Stores events for concatenation

        print(  # Size diagnostic per part
            f"Records in {base_path.name}: {len(part_timeseries)}"  # Reports row count
        )
        print(  # Temporal range diagnostic per part
            f"Range in {base_path.name}: {part_timeseries['timestamp'].min()} -> {part_timeseries['timestamp'].max()}"  # Reports minimum and maximum
        )

    df_timeseries = pd.concat(all_timeseries, ignore_index=True)  # Concatenates parts into a single table
    df_events = pd.concat(all_events, ignore_index=True)  # Concatenates events from both parts

    duplicate_count = df_timeseries.duplicated(subset=["timestamp"]).sum()  # Counts duplicates before removal
    df_timeseries = df_timeseries.drop_duplicates(  # Removes overlaps by timestamp
        subset=["timestamp"],  # Uses timestamp as temporal key
        keep="first",  # Keeps the first record for each time
    )
    print(  # Removed duplicate diagnostic
        f"Temporal duplicates removed: {duplicate_count}"  # Reports removed count
    )
    df_timeseries = df_timeseries.sort_values("timestamp").reset_index(drop=True)  # Sorts chronologically for consistency and resets the index

    if temperature_daily_path:  # Only tries to read temperature if the path is declared in config
        temperature_path = Path(temperature_daily_path)  # Normalizes the daily temperature path for portable behavior
        if temperature_path.exists():  # In Colab it should exist; locally it probably will not
            df_temperature = _read_daily_temperature_file(temperature_path)  # Loads the daily temperature table from Drive
            df_timeseries = _merge_daily_temperature(df_timeseries, df_temperature)  # Replicates daily temperature across each 5-minute block
        else:  # Handles the local case where the hardcoded Drive path is not mounted
            print(f"[load] Temperature file not found: {temperature_path}")  # Reports why the external dataset could not be read
            df_timeseries = _merge_daily_temperature(  # Keeps the column available even if the real file does not exist
                df_timeseries,  # Main series already concatenated and sorted
                pd.DataFrame(columns=["date", "temp_daily_f"]),  # Empty DataFrame to trigger the neutral fallback
            )
    else:  # Keeps compatibility if someone removes the YAML path in an experimental session
        df_timeseries = _merge_daily_temperature(  # Guarantees the column exists even without explicit config
            df_timeseries,  # Main series already concatenated and sorted
            pd.DataFrame(columns=["date", "temp_daily_f"]),  # Empty DataFrame to use the neutral fallback
        )

    df_events = df_events.drop_duplicates()  # Removes exact duplicate events
    print(  # Valid event diagnostic
        f"Valid events: {len(df_events)}"  # Reports total valid events
    )

    nan_count = df_timeseries.isna().sum().sum()  # Counts total NaN in the final table
    print(  # Basic final DataFrame statistics diagnostic
        f"Timeseries final shape: {df_timeseries.shape}, NaN count: {nan_count}"  # Reports shape and NaN
    )

    return df_timeseries, df_events  # Returns both DataFrames according to the contract
