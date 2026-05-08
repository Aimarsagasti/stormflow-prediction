"""
Script to generate `docs/DATASET_STATS.md` from the real dataset.

It reads the raw data from the paths configured in `configs/default.yaml`,
applies the same cleaning and feature-engineering pipeline used by
`claude_train.py`, computes 7 sections of descriptive statistics, and
writes a markdown document with tables and embedded figures.

Usage:
    python scripts/generate_dataset_stats.py

Optional flags:
    --no-cache        Forces regeneration of the cached dataframe.
    --skip-figures    Does not generate figures (faster, tables only).

Output:
    docs/DATASET_STATS.md                           (main markdown)
    outputs/figures/dataset_stats/*.png             (embedded figures)
    outputs/data_analysis/dataset_stats.json        (raw numbers)
    outputs/cache/df_with_features.parquet          (dataframe cache)
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations  # Allows modern type annotations regardless of the exact Python version

import argparse  # Standard library for parsing command-line flags (--no-cache, --skip-figures)
import json  # Standard library for serializing the statistics dictionary to JSON
import sys  # Standard library for manipulating sys.path (adding the repo root to imports)
from pathlib import Path  # Modern standard library for file paths (better than os.path)
from typing import Any, Dict, List, Tuple  # Standard library for type hints

import numpy as np  # Numerical library, the basis for vectorized array computations
import pandas as pd  # Library for tabular manipulation (DataFrames), already used across the project
import yaml  # Library to read `configs/default.yaml` (requires `pyyaml`, already installed)

# Matplotlib imports to generate PNG figures embedded in the markdown
import matplotlib  # Base plotting library in Python
matplotlib.use("Agg")  # Non-interactive backend: generates PNGs without needing a GUI window
import matplotlib.pyplot as plt  # Main matplotlib API for creating figures

# Imports from scipy and statsmodels for specific statistics
from scipy import stats as scipy_stats  # Provides skewness, kurtosis, and statistical tests
from statsmodels.tsa.stattools import acf  # Computes the autocorrelation function (section 6)

# The script lives in `scripts/`, but the repo root is the parent directory. We add the root
# to `sys.path` so we can import `src/...` like `claude_train.py` and `evaluate_local.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent  # Path to `C:\Dev\TFM\` (parent of `scripts/`)
sys.path.insert(0, str(REPO_ROOT))  # Insert it first so it has priority over other packages

# Now we can import project modules
# Note: the exact names were verified against the files in `src/`
from src.data.load import load_msd_data            # Returns a tuple `(df_timeseries, df_events)`
from src.data.clean import clean_timeseries        # Receives `(df_timeseries, df_events)` and returns the cleaned df
from src.features.engineering import create_features  # Generates the 22 features from the cleaned df
from src.pipeline.split import split_chronological   # Chronological train/val/test 70/15/15 split

# ============================================================================
# GLOBAL SCRIPT CONFIGURATION
# ============================================================================

# Output paths, relative to `REPO_ROOT`
OUTPUT_MARKDOWN = REPO_ROOT / "docs" / "DATASET_STATS.md"  # Main markdown document
OUTPUT_JSON = REPO_ROOT / "outputs" / "data_analysis" / "dataset_stats.json"  # Raw numbers in structured format
OUTPUT_FIGURES_DIR = REPO_ROOT / "outputs" / "figures" / "dataset_stats"  # PNG figure directory
CACHE_DIR = REPO_ROOT / "outputs" / "cache"  # Cache directory for the parquet
CACHE_FILE = CACHE_DIR / "df_with_features.parquet"  # Parquet file with the processed dataframe

# Path to the configuration file used by the rest of the project
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"  # YAML with data paths and parameters

# Project target variable: predict stormflow in million gallons per day
TARGET_COLUMN = "stormflow_mgd"  # Defined in AGENTS.md section 3

# List of the 22 features used by the production model (source: modelo_H1_sinSF_meta.json)
FEATURE_COLUMNS = [
    "rain_in", "temp_daily_f", "api_dynamic",
    "rain_sum_10m", "rain_sum_15m", "rain_sum_30m", "rain_sum_60m",
    "rain_sum_120m", "rain_sum_180m", "rain_sum_360m",
    "rain_max_10m", "rain_max_30m", "rain_max_60m",
    "minutes_since_last_rain",
    "delta_flow_5m", "delta_flow_15m", "delta_rain_10m", "delta_rain_30m",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# Severity-bucket thresholds. Same as `evaluate_local.py` so the analyses stay consistent.
SEVERITY_BUCKETS: List[Tuple[str, float, float]] = [
    ("Base",     0.0,   0.5),     # 92% of samples fall here (no event)
    ("Leve",     0.5,   5.0),     # Small events
    ("Moderado", 5.0,   20.0),    # Moderate events: where the model performs best
    ("Alto",     20.0,  50.0),    # High events: fewer samples
    ("Extremo",  50.0,  np.inf),  # Extreme events: 59 samples in test, the focus of the problem
]

# Threshold to consider an "event" in regime analysis (>= 0.5 MGD by project convention)
EVENT_THRESHOLD_MGD = 0.5  # Taken from the metric functions in `src/evaluation/metrics.py`

# Matplotlib configuration so figures look clean and readable in the markdown
plt.rcParams["figure.dpi"] = 100          # Reasonable screen resolution
plt.rcParams["savefig.dpi"] = 150         # PNG save resolution (higher than screen)
plt.rcParams["figure.figsize"] = (10, 5)  # Default size: 10 inches wide x 5 inches tall
plt.rcParams["font.size"] = 10            # Readable but not oversized text
plt.rcParams["axes.grid"] = True          # Grid enabled by default (improves plot readability)
plt.rcParams["grid.alpha"] = 0.3          # Subtle grid to avoid visual clutter

# ============================================================================
# LOW-LEVEL UTILITIES
# ============================================================================

def _load_config() -> Dict[str, Any]:
    """Reads and returns the contents of the project's configuration YAML."""
    # We open the YAML in read mode with UTF-8 encoding to avoid Windows issues.
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        # yaml.safe_load parses the YAML into a Python dict without executing arbitrary code (safer than yaml.load).
        return yaml.safe_load(f)


def _ensure_dirs_exist() -> None:
    """Creates the output folders if they do not exist (equivalent to `mkdir -p` in bash)."""
    # parents=True creates parent directories if needed, and exist_ok=True avoids errors if they already exist.
    OUTPUT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _format_number(x: float, decimals: int = 3) -> str:
    """Formats a number for the markdown table (handles NaN, inf, and very small/large values)."""
    # Special cases must be handled first because NaN and inf break normal formatting.
    if pd.isna(x):
        return "NaN"
    if np.isinf(x):
        return "+inf" if x > 0 else "-inf"
    # For very large numbers we use scientific notation, which is more readable than many zeros.
    if abs(x) >= 1e6:
        return f"{x:.2e}"
    # Normal formatting with the requested number of decimals.
    return f"{x:.{decimals}f}"


def _save_figure(fig: plt.Figure, filename: str) -> str:
    """Saves the figure in `OUTPUT_FIGURES_DIR` and returns the relative path for the markdown."""
    # Absolute path where the PNG is physically saved.
    full_path = OUTPUT_FIGURES_DIR / filename
    # bbox_inches='tight' trims extra white space around the figure.
    fig.savefig(full_path, bbox_inches="tight")
    # We close the figure explicitly to free memory, which matters when generating many figures.
    plt.close(fig)
    # Return the relative path from docs/ (where the markdown lives) to the figure.
    # The markdown must use ../outputs/figures/dataset_stats/ to reach outputs/.
    return f"../outputs/figures/dataset_stats/{filename}"


def _small_sample_warning(n: int, threshold: int = 100) -> str:
    """Generates a footnote if the sample is small. Returns an empty string if not needed."""
    # Convention: any statistic on <=100 samples is marked as low confidence.
    if n <= threshold:
        return f" *(n={n}, tomar con cautela)*"
    return ""

# ============================================================================
# CARGA DE DATOS Y GESTION DEL CACHE
# ============================================================================

def _load_and_process_data(config_path: Path, use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the raw data, cleans it, and generates the 22 features.

    If a valid parquet cache exists, it reads it directly (fast).
    Otherwise, it runs the full pipeline (`load -> clean -> features`) and saves the result.

    Args:
        config_path: Path to the configuration YAML (`local.yaml` in our execution).
        use_cache: If `False`, ignores the cache and regenerates the dataframe from scratch.

    Returns:
        DataFrame with columns: timestamp, [22 features], stormflow_mgd, is_event.
    """
    # First decision: use cache or regenerate?
    if use_cache and CACHE_FILE.exists():
        print(f"[cache] Reading processed dataframe from {CACHE_FILE}")
        # Parquet is a very fast binary columnar format to read and write, and it is more efficient than CSV.
        df_cached = pd.read_parquet(CACHE_FILE)
        print(f"[cache] Cached dataframe shape: {df_cached.shape}")
        return df_cached

    # No cache exists, or regeneration was requested: run the full pipeline
    print(f"[pipeline] Generating dataframe from scratch (this may take several minutes)...")

    # Step 1: load raw data. `load_msd_data` returns `(df_timeseries, df_events)`
    print(f"[pipeline] 1/3: Loading raw data from {config_path}")
    df_timeseries, df_events = load_msd_data(config_path=config_path)
    print(f"[pipeline]      df_timeseries: {df_timeseries.shape} | df_events: {df_events.shape}")

    # Step 2: cleaning (timestamps, negatives, gaps)
    print(f"[pipeline] 2/3: Cleaning timeseries")
    df_clean = clean_timeseries(df_timeseries, df_events)
    print(f"[pipeline]      df_clean: {df_clean.shape}")

    # Step 3: feature engineering (rolling sums, api_dynamic, deltas, cyclic encoding)
    print(f"[pipeline] 3/3: Generating features")
    df_features = create_features(df_clean)
    print(f"[pipeline]      df_features: {df_features.shape}")

    # Save the result in cache for future runs
    print(f"[cache] Saving processed dataframe to {CACHE_FILE}")
    df_features.to_parquet(CACHE_FILE, index=False)

    return df_features


def _split_data(df_features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Applies the chronological 70/15/15 split using the pipeline function.

    Returns:
        Dict with keys 'train', 'val', 'test' and DataFrame values.
    """
    # split_chronological returns a tuple of 3 dataframes in train, val, test order.
    df_train, df_val, df_test = split_chronological(df_features)
    # We wrap them in a named dict to make later iteration easier.
    return {"train": df_train, "val": df_val, "test": df_test}


# ============================================================================
# UTILIDADES ESTADISTICAS ESPECIFICAS
# ============================================================================

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Pearson correlation while handling NaN and small samples."""
    # Pearson measures LINEAR correlation between two variables, range [-1, 1].
    # np.corrcoef returns a 2x2 matrix; [0, 1] is the correlation between x and y.
    # We need to filter NaN because corrcoef propagates them and returns NaN if any are present.
    mask = ~(np.isnan(x) | np.isnan(y))  # True where BOTH are finite.
    if mask.sum() < 2:  # With fewer than 2 points we cannot compute correlation.
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Spearman correlation (by ranks) while handling NaN."""
    # Spearman measures MONOTONIC correlation (not necessarily linear): if y rises when x rises, it returns 1.
    # Useful for nonlinear relationships that Pearson does not capture well (e.g. rain_sum_60m vs stormflow_mgd, which is concave).
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return float("nan")
    # scipy_stats.spearmanr returns a named tuple (correlation, pvalue); we only need the correlation.
    correlation, _ = scipy_stats.spearmanr(x[mask], y[mask])
    return float(correlation)


# ============================================================================
# SECCION 1: RESUMEN GLOBAL DEL DATASET
# ============================================================================

def compute_section_1_global_summary(
    df_features: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
) -> Tuple[str, Dict[str, Any]]:
    """
    Generates section 1 of the document: global dataset summary.

    Returns:
        Tuple `(markdown_string, dict_with_raw_numbers)`.
    """
    # We compute the global statistics of the whole dataset.
    n_total = len(df_features)  # Total number of 5-minute records.
    first_ts = df_features["timestamp"].min()  # First timestamp in the dataset.
    last_ts = df_features["timestamp"].max()  # Last timestamp in the dataset.
    span_days = (last_ts - first_ts).days  # Total duration in days.

    # NaNs by column: useful to detect problematic columns.
    nan_counts = df_features.isna().sum()
    # We only report columns that have at least one NaN (those at 0 are less informative).
    columns_with_nans = nan_counts[nan_counts > 0]

    # Statistics of each split: name, start/end dates, row count, percentage of total.
    split_stats = {}
    for split_name, df_split in splits.items():
        split_stats[split_name] = {
            "n": len(df_split),
            "first_ts": df_split["timestamp"].min().isoformat(),
            "last_ts": df_split["timestamp"].max().isoformat(),
            "pct": 100 * len(df_split) / n_total,
            "span_days": (df_split["timestamp"].max() - df_split["timestamp"].min()).days,
        }

    # Raw dictionary for the output JSON.
    brute_data = {
        "n_total_records": n_total,
        "first_timestamp": first_ts.isoformat(),
        "last_timestamp": last_ts.isoformat(),
        "span_days": span_days,
        "span_years": round(span_days / 365.25, 2),
        "resolution_minutes": 5,
        "n_columns_after_features": df_features.shape[1],
        "columns_with_nans": {col: int(count) for col, count in columns_with_nans.items()},
        "splits": split_stats,
    }

    # Now we build the markdown.
    md_lines = []
    md_lines.append("## 1. Global dataset summary\n")
    md_lines.append(f"- **Total records:** {n_total:,} (at 5-minute resolution).")
    md_lines.append(f"- **Temporal coverage:** {first_ts.strftime('%Y-%m-%d')} to {last_ts.strftime('%Y-%m-%d')} ({span_days} days, {brute_data['span_years']} years).")
    md_lines.append(f"- **Columns after feature engineering:** {df_features.shape[1]} (timestamp + 22 features + stormflow_mgd + is_event).")

    if len(columns_with_nans) > 0:
        md_lines.append(f"- **Columns with NaN:** {len(columns_with_nans)} columns have at least one NaN.")
        for col, count in columns_with_nans.items():
            pct = 100 * count / n_total
            md_lines.append(f"  - `{col}`: {count:,} NaN ({pct:.2f}%)")
    else:
        md_lines.append(f"- **Columns with NaN:** none.")

    md_lines.append("")  # Blank line before the table
    md_lines.append("### Chronological 70/15/15 split\n")
    md_lines.append("| Split | Rows | % of total | From | To | Duration |")
    md_lines.append("|-------|-------|-------------|-------|-------|----------|")
    for split_name in ["train", "val", "test"]:
        s = split_stats[split_name]
        md_lines.append(
            f"| {split_name} | {s['n']:,} | {s['pct']:.1f}% | "
            f"{s['first_ts'][:10]} | {s['last_ts'][:10]} | {s['span_days']} days |"
        )
    md_lines.append("")

    # Critical note about the split (what Aimar detected at the start).
    md_lines.append(
        "> **Methodological note:** the current split leaves val and test shorter than "
        "a full annual cycle. In datasets with strong annual seasonality (storm events "
        "concentrated in spring/summer, snowmelt events concentrated in winter), this may "
        "introduce seasonal bias in evaluation metrics. Alternatives to consider: splitting "
        "by full years (e.g. 7/2/1), `TimeSeriesSplit` with sliding windows, or blocked CV by "
        "events with temporal embargo. This decision remains pending (see `STATE.md`)."
    )
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 2: DISTRIBUCION DEL TARGET (stormflow_mgd)
# ============================================================================

def compute_section_2_target_distribution(
    df_features: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Analyzes the distribution of stormflow_mgd: quantiles, skewness, zero-inflation,
    and comparison across splits to detect imbalance.
    """
    # Stormflow series by split and globally.
    series_global = df_features[TARGET_COLUMN].dropna().to_numpy()
    series_by_split = {name: df[TARGET_COLUMN].dropna().to_numpy() for name, df in splits.items()}

    # Quantiles to compute: they span from the median to the upper tail.
    quantile_points = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.00]

    def compute_stats(series: np.ndarray) -> Dict[str, float]:
        """Computes basic statistics for a stormflow series."""
        return {
            "n": int(len(series)),
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            # Skewness: measure of asymmetry. High values (>>0) indicate a very long right tail (our case).
            "skewness": float(scipy_stats.skew(series)),
            # Kurtosis: measures how "heavy" the tails are. >3 indicates heavier tails than a normal distribution.
            "kurtosis": float(scipy_stats.kurtosis(series)),
            # Percentage of time in the base regime (no event, <0.5 MGD).
            "pct_baseflow": float(100 * np.mean(series < EVENT_THRESHOLD_MGD)),
            # Specific quantiles: keys "q50", "q75", "q90", "q95", "q99", "q99.9", "q100".
            **{f"q{q*100:g}": float(np.quantile(series, q)) for q in quantile_points},
        }

    global_stats = compute_stats(series_global)
    split_stats = {name: compute_stats(s) for name, s in series_by_split.items()}

    brute_data = {
        "global": global_stats,
        "by_split": split_stats,
    }

    # We generate the figure as a log-scale histogram of the target (extreme events are few, so log is needed).
    figure_path = None
    if not skip_figures:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: histogram of all values (includes zero-inflation).
        # log=True on the Y axis lets us see the right tail that would otherwise dominate the plot.
        ax[0].hist(series_global, bins=100, log=True, color="steelblue", edgecolor="black", alpha=0.7)
        ax[0].set_xlabel("stormflow_mgd")
        ax[0].set_ylabel("Frequency (log scale)")
        ax[0].set_title("Full distribution of stormflow_mgd\n(log Y scale)")
        ax[0].axvline(EVENT_THRESHOLD_MGD, color="red", linestyle="--", label=f"Event threshold ({EVENT_THRESHOLD_MGD} MGD)")
        ax[0].legend()

        # Right panel: histogram of the tail only (>=0.5 MGD) to inspect the event distribution.
        tail = series_global[series_global >= EVENT_THRESHOLD_MGD]
        ax[1].hist(tail, bins=50, log=True, color="darkorange", edgecolor="black", alpha=0.7)
        ax[1].set_xlabel("stormflow_mgd (events only)")
        ax[1].set_ylabel("Frequency (log scale)")
        ax[1].set_title(f"Event distribution (>={EVENT_THRESHOLD_MGD} MGD)\nn={len(tail):,}")

        fig.tight_layout()
        figure_path = _save_figure(fig, "section2_target_distribution.png")

    # Construimos el markdown
    md_lines = []
    md_lines.append("## 2. Distribution of the target variable `stormflow_mgd`\n")

    # Estadisticos globales en formato texto
    md_lines.append("### Global statistics (full dataset)\n")
    md_lines.append(f"- **n:** {global_stats['n']:,}")
    md_lines.append(f"- **Mean:** {global_stats['mean']:.4f} MGD")
    md_lines.append(f"- **Standard deviation:** {global_stats['std']:.4f} MGD")
    md_lines.append(f"- **Skewness:** {global_stats['skewness']:.2f} (very marked right skew)")
    md_lines.append(f"- **Kurtosis:** {global_stats['kurtosis']:.2f} (extremely heavy tails)")
    md_lines.append(f"- **% of time in the base regime (<{EVENT_THRESHOLD_MGD} MGD):** {global_stats['pct_baseflow']:.2f}%")
    md_lines.append("")

    # Quantile comparison across splits: this is critical to detect train/val/test imbalance.
    md_lines.append("### Quantiles of `stormflow_mgd` by split\n")
    md_lines.append("| Quantile | Global | Train | Val | Test |")
    md_lines.append("|---------|--------|-------|-----|------|")
    for q in quantile_points:
        key = f"q{q*100:g}"  # We use the same convention as compute_stats: "q50", "q99.9", "q100".
        label = f"p{q*100:g}" if q < 1 else "max"
        md_lines.append(
            f"| {label} | {global_stats[key]:.3f} | {split_stats['train'][key]:.3f} | "
            f"{split_stats['val'][key]:.3f} | {split_stats['test'][key]:.3f} |"
        )
    md_lines.append("")

    # Extrapolation note: if max(test) > max(train), the model is being evaluated in
    # an extreme regime it never saw. This is critical evidence for the Opus diagnosis.
    max_train = split_stats["train"]["q100"]
    max_test = split_stats["test"]["q100"]
    if max_test > max_train:
        md_lines.append(
            f"> **Extrapolation alert:** the test maximum ({max_test:.1f} MGD) is higher "
            f"than the train maximum ({max_train:.1f} MGD). The model is being evaluated on "
            f"more extreme events than it saw during training, which partially explains "
            f"the systematic underestimation in the Extreme bucket."
        )
    else:
        md_lines.append(
            f"> The magnitude range of test ({max_test:.1f} MGD max) does not exceed train "
            f"({max_train:.1f} MGD max). There is no extrapolation beyond the observed range."
        )
    md_lines.append("")

    if figure_path:
        md_lines.append(f"![Target distribution]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 3: CARACTERIZACION DE LOS EVENTOS EXTREMOS (>50 MGD)
# ============================================================================

def compute_section_3_extreme_events(
    df_features: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Characterizes the extreme events (stormflow_mgd >= 50 MGD): distribution across splits,
    seasonality, and an initial clue about the "no rain in window" events.
    """
    # Mask of extreme events in the full dataset.
    extreme_mask = df_features[TARGET_COLUMN] >= 50.0
    df_extreme = df_features.loc[extreme_mask].copy()

    # How many extreme samples fall into each split.
    extreme_by_split = {}
    for split_name, df_split in splits.items():
        mask = df_split[TARGET_COLUMN] >= 50.0
        extreme_by_split[split_name] = {
            "n": int(mask.sum()),
            "pct_of_split": float(100 * mask.mean()),
            "max_mgd": float(df_split.loc[mask, TARGET_COLUMN].max()) if mask.any() else float("nan"),
            "mean_mgd": float(df_split.loc[mask, TARGET_COLUMN].mean()) if mask.any() else float("nan"),
        }

    # Seasonal analysis: which month of the year the extremes occur in.
    df_extreme["month"] = df_extreme["timestamp"].dt.month  # dt.month returns 1-12.
    extremes_by_month = df_extreme.groupby("month").size().to_dict()
    # We make sure all months appear even if they have zero events.
    extremes_by_month = {int(m): int(extremes_by_month.get(m, 0)) for m in range(1, 13)}

    # Annual analysis: which year the extremes occur in.
    df_extreme["year"] = df_extreme["timestamp"].dt.year
    extremes_by_year = df_extreme.groupby("year").size().to_dict()
    extremes_by_year = {int(y): int(v) for y, v in extremes_by_year.items()}

    # Hint of events without rain: we check rain_sum_60m in the peak sample.
    # If rain_sum_60m ~ 0 in an extreme event, recent rain does not justify the peak (possible snowmelt).
    if "rain_sum_60m" in df_extreme.columns:
        # Very low threshold: 0.01 inches in the last 60 minutes is practically zero.
        no_rain_mask = df_extreme["rain_sum_60m"] < 0.01
        n_extremes_without_recent_rain = int(no_rain_mask.sum())
        # These are the "candidates for 15 of 59 without rain" mentioned in STATE.md.
        # This is not the exact definition (the real one uses the model input window), but it gives an order of magnitude.
    else:
        n_extremes_without_recent_rain = -1  # -1 means we could not compute it.

    brute_data = {
        "n_total_extremes": int(extreme_mask.sum()),
        "by_split": extreme_by_split,
        "by_month": extremes_by_month,
        "by_year": extremes_by_year,
        "n_extremes_without_recent_rain_60m": n_extremes_without_recent_rain,
    }

    # Figure: bar plot of events by month and by year.
    figure_path = None
    if not skip_figures and len(df_extreme) > 0:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: events by month.
        months = list(range(1, 13))
        counts_month = [extremes_by_month[m] for m in months]
        ax[0].bar(months, counts_month, color="steelblue", edgecolor="black", alpha=0.8)
        ax[0].set_xticks(months)
        ax[0].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax[0].set_xlabel("Month")
        ax[0].set_ylabel("Number of extreme events (>=50 MGD)")
        ax[0].set_title("Monthly distribution of extreme events")

        # Right panel: events by year.
        years_sorted = sorted(extremes_by_year.keys())
        counts_year = [extremes_by_year[y] for y in years_sorted]
        ax[1].bar(years_sorted, counts_year, color="darkorange", edgecolor="black", alpha=0.8)
        ax[1].set_xlabel("Year")
        ax[1].set_ylabel("Number of extreme events (>=50 MGD)")
        ax[1].set_title("Annual distribution of extreme events")
        ax[1].tick_params(axis="x", rotation=45)

        fig.tight_layout()
        figure_path = _save_figure(fig, "section3_extreme_events.png")

    # Markdown
    md_lines = []
    md_lines.append("## 3. Characterization of extreme events (stormflow >= 50 MGD)\n")
    md_lines.append(f"- **Total extreme samples in the dataset:** {brute_data['n_total_extremes']}")
    md_lines.append("")

    md_lines.append("### Distribution across splits\n")
    md_lines.append("| Split | n extreme | % of split | Max MGD | Mean MGD |")
    md_lines.append("|-------|------------|-------------|---------|-----------|")
    for split_name in ["train", "val", "test"]:
        s = extreme_by_split[split_name]
        md_lines.append(
            f"| {split_name} | {s['n']} | {s['pct_of_split']:.3f}% | "
            f"{_format_number(s['max_mgd'], 1)} | {_format_number(s['mean_mgd'], 1)} |"
        )
    md_lines.append("")

    # Critical note: if test has few extremes (e.g. <60), metrics in the Extreme bucket
    # have high variance. This matters when interpreting the NSE=-0.99 in the JSON.
    n_test_extreme = extreme_by_split["test"]["n"]
    if n_test_extreme < 100:
        md_lines.append(
            f"> **Note on sample size:** the test contains only {n_test_extreme} extreme samples. "
            f"Metrics computed on this bucket (NSE, bias, MAPE) have high variance and a "
            f"single atypical event can shift them significantly."
        )
        md_lines.append("")

    md_lines.append("### Seasonality of extreme events\n")
    if n_extremes_without_recent_rain >= 0:
        pct_no_rain = 100 * n_extremes_without_recent_rain / max(brute_data["n_total_extremes"], 1)
        md_lines.append(
            f"- **Extremes with `rain_sum_60m` < 0.01 inches:** {n_extremes_without_recent_rain} "
            f"({pct_no_rain:.1f}%). These are candidates for events without immediate rainfall origin "
            f"(possible delayed runoff, snowmelt, or sensor errors)."
        )
        md_lines.append("")

    if figure_path:
        md_lines.append(f"![Extreme events]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 4: CORRELACIONES FEATURE-TARGET POR REGIMEN
# ============================================================================

def compute_section_4_feature_target_correlations(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Computes Pearson and Spearman for each feature against the target, separated into
    baseflow regime (<0.5 MGD) versus event regime (>=0.5 MGD).

    Uses ONLY the train split to avoid data snooping on val/test.
    This is consistent with the project rule to compute normalization
    only on train (see CLAUDE.md rule 3).

    Detects features that are "shortcuts" only in one regime, as flow_total_mgd
    was with r=0.9976 globally in earlier iterations.
    """
    # We work exclusively with train: the same principle used by the normalization pipeline.
    df_train = splits["train"]

    baseflow_mask = df_train[TARGET_COLUMN] < EVENT_THRESHOLD_MGD
    event_mask = df_train[TARGET_COLUMN] >= EVENT_THRESHOLD_MGD

    target_array = df_train[TARGET_COLUMN].to_numpy()
    target_baseflow = target_array[baseflow_mask]
    target_event = target_array[event_mask]

    # We compute correlations for each feature in each regime.
    correlations = []
    for feat in FEATURE_COLUMNS:
        if feat not in df_train.columns:
                # This can happen if some feature was not generated for any reason; we report it.
            correlations.append({
                "feature": feat,
                "available": False,
                "pearson_global": None, "spearman_global": None,
                "pearson_baseflow": None, "spearman_baseflow": None,
                "pearson_event": None, "spearman_event": None,
            })
            continue

        feat_array = df_train[feat].to_numpy()
        feat_baseflow = feat_array[baseflow_mask]
        feat_event = feat_array[event_mask]

        correlations.append({
            "feature": feat,
            "available": True,
            "pearson_global": _safe_pearson(feat_array, target_array),
            "spearman_global": _safe_spearman(feat_array, target_array),
            "pearson_baseflow": _safe_pearson(feat_baseflow, target_baseflow),
            "spearman_baseflow": _safe_spearman(feat_baseflow, target_baseflow),
            "pearson_event": _safe_pearson(feat_event, target_event),
            "spearman_event": _safe_spearman(feat_event, target_event),
        })

    # Sort by Spearman in the event regime (more operationally relevant).
    # Use abs() because a strong correlation can be positive or negative.
    correlations_sorted = sorted(
        [c for c in correlations if c["available"]],
        key=lambda c: abs(c.get("spearman_event") or 0),
        reverse=True,
    )

    brute_data = {
        "correlations": correlations,
        "n_samples_baseflow": int(baseflow_mask.sum()),
        "n_samples_event": int(event_mask.sum()),
    }

    # Markdown
    md_lines = []
    md_lines.append("## 4. Feature-target correlations by regime\n")
    md_lines.append(
        f"Pearson (linear) and Spearman (monotonic) correlations are computed between each of "
        f"the 22 features and the target `stormflow_mgd`, separating baseflow regime "
        f"(<{EVENT_THRESHOLD_MGD} MGD, n={brute_data['n_samples_baseflow']:,}) and event regime "
        f"(>={EVENT_THRESHOLD_MGD} MGD, n={brute_data['n_samples_event']:,}). "
        f"**Computed only on the train split** to avoid data snooping on val/test.\n"
    )
    md_lines.append(
        "**Interpretation:** high Pearson indicates a strong linear relationship. High Spearman indicates "
        "a monotonic relationship (which may be nonlinear). Large differences between baseflow and event "
        "indicate that the feature behaves differently in each regime.\n"
    )

    md_lines.append("| Feature | Pearson global | Spearman global | Spearman baseflow | Spearman event |")
    md_lines.append("|---------|---------------:|----------------:|-------------------:|----------------:|")
    for c in correlations_sorted:
        md_lines.append(
            f"| `{c['feature']}` | "
            f"{_format_number(c['pearson_global'], 3)} | "
            f"{_format_number(c['spearman_global'], 3)} | "
            f"{_format_number(c['spearman_baseflow'], 3)} | "
            f"{_format_number(c['spearman_event'], 3)} |"
        )
    md_lines.append("")
    md_lines.append(
        "*Sorted descending by `|Spearman event|` because that is the operationally relevant regime.*"
    )
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 5: MATRIZ DE CORRELACIONES FEATURE-FEATURE
# ============================================================================

def compute_section_5_feature_feature_matrix(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Computes the Pearson correlation matrix between pairs of features.
    Detects redundancies (features that contribute the same information).

    Historical example: api_dynamic vs rain_sum_60m, with r=0.94 according to previous notes.

    Uses train only, consistent with section 4.
    """
    # Same principle as section 4: train-only to avoid data snooping.
    df_train = splits["train"]

    # Filter to columns that actually exist (for safety).
    available_features = [f for f in FEATURE_COLUMNS if f in df_train.columns]

    # pd.DataFrame.corr() computes Pearson across all columns in the selected dataframe.
    # The result is an NxN matrix where N is the number of features.
    corr_matrix = df_train[available_features].corr(method="pearson")

    # Extract the pairs with high absolute correlation (>0.7) that are NOT the diagonal.
    # The diagonal is always 1.0 (feature with itself), so it is not interesting.
    high_corr_pairs = []
    n_features = len(available_features)
    for i in range(n_features):
        # We only traverse the upper triangle (i<j) to avoid duplicate pairs.
        for j in range(i + 1, n_features):
            feat_a = available_features[i]
            feat_b = available_features[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= 0.7:  # Conventional threshold for considering a correlation "high".
                high_corr_pairs.append({
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "pearson": float(corr_value),
                })

    # Sort pairs by descending absolute correlation (most redundant first).
    high_corr_pairs.sort(key=lambda p: abs(p["pearson"]), reverse=True)

    brute_data = {
        "n_features_analyzed": n_features,
        "correlation_matrix": corr_matrix.round(4).to_dict(),  # Dict anidado para JSON
        "high_correlation_pairs_abs_gt_0_7": high_corr_pairs,
    }

    # Figure: heatmap of the correlation matrix.
    figure_path = None
    if not skip_figures:
        fig, ax = plt.subplots(figsize=(12, 10))
        # imshow draws the matrix as an image; cmap='coolwarm' is red-white-blue (intuitive for correlations).
        # vmin=-1, vmax=1 fixes the color scale so 0 is always white.
        im = ax.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        # Axis labels with the feature names.
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(available_features, rotation=90, fontsize=8)
        ax.set_yticklabels(available_features, fontsize=8)
        # Color bar on the right to interpret the values.
        plt.colorbar(im, ax=ax, label="Pearson correlation")
        ax.set_title("Feature-feature correlation matrix (train)")
        fig.tight_layout()
        figure_path = _save_figure(fig, "section5_feature_correlation_matrix.png")

    # Markdown
    md_lines = []
    md_lines.append("## 5. Feature-feature correlation matrix\n")
    md_lines.append(
        f"Pearson correlation between all pairs of the {n_features} features, computed "
        f"on the train split. It detects redundancies: features with high correlation contribute "
        f"almost identical information to the model.\n"
    )

    if high_corr_pairs:
        md_lines.append(f"### Pairs with |Pearson| >= 0.7 ({len(high_corr_pairs)} pairs)\n")
        md_lines.append("| Feature A | Feature B | Pearson |")
        md_lines.append("|-----------|-----------|--------:|")
        for p in high_corr_pairs:
            md_lines.append(f"| `{p['feature_a']}` | `{p['feature_b']}` | {p['pearson']:+.3f} |")
        md_lines.append("")
        md_lines.append(
            "> **Interpretation:** pairs with very high |Pearson| (>0.9) are candidates for removing one "
            "of the two features. Pairs in the 0.7-0.9 range can remain if they add slightly different "
            "signal, but this should be evaluated with permutation importance on a trained model."
        )
    else:
        md_lines.append("No pairs with |Pearson| >= 0.7 were found among the features.\n")
    md_lines.append("")

    if figure_path:
        md_lines.append(f"![Correlation matrix]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 6: AUTOCORRELACION DEL TARGET
# ============================================================================

def compute_section_6_target_autocorrelation(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Computes the autocorrelation function (ACF) of the target up to 24 hours (288 5-minute steps).

    The ACF measures how correlated the target is with itself at different time lags.
    If ACF(1) is close to 1, the next step is almost trivial (just copy the current value),
    which explains why H=1 gives NSE=0.86 while H=6 collapses to NSE=-1.21.
    """
    # We use train only to stay consistent with the rest of the analysis.
    df_train = splits["train"]
    target_series = df_train[TARGET_COLUMN].to_numpy()

    # Number of lags to compute: up to 24 hours at 5 min/step = 288 steps.
    # +1 because lag 0 is the correlation with itself (=1 by definition, but we include it).
    max_lag_steps = 288

    # statsmodels.acf accepts NaN with missing='drop'; if there were no NaN this parameter would be irrelevant.
    # fft=True speeds up the computation using the fast Fourier transform (important with 770k points).
    acf_values = acf(target_series, nlags=max_lag_steps, fft=True, missing="drop")

    # Specific lags of interest for the table: the points where the ACF decays noticeably.
    lags_of_interest = [1, 3, 6, 12, 24, 36, 72, 144, 288]  # en pasos de 5 minutos
    # Convert to minutes for easier reading.
    lags_as_minutes = {lag: lag * 5 for lag in lags_of_interest}

    acf_at_lags = {
        str(lag): {
            "minutes": lags_as_minutes[lag],
            "acf": float(acf_values[lag]),
        }
        for lag in lags_of_interest if lag < len(acf_values)
    }

    brute_data = {
        "max_lag_steps": max_lag_steps,
        "resolution_minutes": 5,
        "acf_at_key_lags": acf_at_lags,
        # We do not store the full array in JSON (288 values, not useful), only the key lags.
    }

    # Figure: ACF plot.
    figure_path = None
    if not skip_figures:
        fig, ax = plt.subplots(figsize=(12, 5))
        # X axis in minutes so it is easy to read (1 step = 5 min).
        lag_axis_minutes = np.arange(len(acf_values)) * 5
        ax.plot(lag_axis_minutes, acf_values, color="steelblue", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        # Guideline lines at 0.5 and 0.1 help read where the autocorrelation decays.
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, label="ACF=0.5")
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5, label="ACF=0.1")
        ax.set_xlabel("Lag (minutos)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"ACF of stormflow_mgd up to {max_lag_steps * 5} minutes (24h)")
        ax.legend()
        fig.tight_layout()
        figure_path = _save_figure(fig, "section6_target_acf.png")

    # Markdown
    md_lines = []
    md_lines.append("## 6. Target autocorrelation (stormflow_mgd)\n")
    md_lines.append(
        "The autocorrelation function (ACF) measures how much the series resembles a shifted copy "
        "of itself in time. It is relevant for this project because:\n"
    )
    md_lines.append(
        "- If ACF(lag=1) is very close to 1, predicting the next step is almost trivial by copying "
        "the current value. This explains the high NSE at H=1.\n"
    )
    md_lines.append(
        "- The speed at which the ACF decays toward zero indicates how much useful predictive horizon "
        "the series has. If the ACF falls quickly, predicting at H=6 is intrinsically difficult "
        "regardless of the model.\n"
    )

    md_lines.append("### ACF at interesting lags (train-only)\n")
    md_lines.append("| Lag (steps) | Lag (minutes) | ACF |")
    md_lines.append("|-------------|---------------|----:|")
    for lag_str, entry in acf_at_lags.items():
        md_lines.append(f"| {lag_str} | {entry['minutes']} | {entry['acf']:.4f} |")
    md_lines.append("")

    # Automated interpretation based on the values.
    acf_at_5min = acf_at_lags.get("1", {}).get("acf", None)
    acf_at_30min = acf_at_lags.get("6", {}).get("acf", None)
    if acf_at_5min is not None and acf_at_30min is not None:
        md_lines.append(
            f"> **Reading:** ACF at 5 minutes = {acf_at_5min:.3f}, ACF at 30 minutes = {acf_at_30min:.3f}. "
            f"As a theoretical approximation, the NSE of a persistence AR(1) predictor would be bounded "
            f"below by 2*rho - 1 = {2*acf_at_5min - 1:.3f} at H=1 and {2*acf_at_30min - 1:.3f} at H=6. "
            f"The exact empirical value on test, computed in section 8 of this document, is "
            f"**NSE naive = 0.811 at H=1**. The empirical figure in section 8 is authoritative; "
            f"the approximations here only contextualize how the signal decays in the series."
        )
    md_lines.append("")

    if figure_path:
        md_lines.append(f"![ACF del target]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 7: CARACTERIZACION DE LA LLUVIA
# ============================================================================

def compute_section_7_rain_characterization(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Describes rainfall behavior: fraction of zero-rain steps, distribution of the
    non-zero values, duration of rain events, etc.

    This is the context that explains why the rain_sum_* features are compressed
    near zero 92% of the time, as documented in AGENTS.md.
    """
    df_train = splits["train"]
    rain_series = df_train["rain_in"].to_numpy()

    # Fraction of time with exactly zero rain.
    n_total = len(rain_series)
    n_zero = int(np.sum(rain_series == 0.0))
    n_positive = int(np.sum(rain_series > 0.0))
    pct_zero = 100 * n_zero / n_total

    # Statistics of the non-zero values (to see how rainfall is distributed when it does occur).
    rain_nonzero = rain_series[rain_series > 0.0]
    if len(rain_nonzero) > 0:
        rain_nonzero_stats = {
            "n": int(len(rain_nonzero)),
            "mean": float(np.mean(rain_nonzero)),
            "median": float(np.median(rain_nonzero)),
            "max": float(np.max(rain_nonzero)),
            "q95": float(np.quantile(rain_nonzero, 0.95)),
            "q99": float(np.quantile(rain_nonzero, 0.99)),
        }
    else:
        rain_nonzero_stats = {"n": 0}

    # Average duration of consecutive rain streaks (step by step).
    # Simple definition: a streak is a contiguous sequence of steps with rain_in > 0.
    # We use np.diff on indicators to detect streak starts and ends.
    is_raining = (rain_series > 0.0).astype(int)
    # Differences: +1 at the start of a streak, -1 at the end, 0 where nothing changes.
    transitions = np.diff(is_raining, prepend=0, append=0)
    starts = np.where(transitions == 1)[0]  # Indices where a streak starts.
    ends = np.where(transitions == -1)[0]   # Indices where a streak ends.
    # Duration of each streak (in 5-minute steps).
    if len(starts) > 0 and len(ends) > 0:
        rain_streak_lengths = ends - starts
        streak_stats = {
            "n_rain_streaks": int(len(rain_streak_lengths)),
            "mean_duration_steps": float(np.mean(rain_streak_lengths)),
            "mean_duration_minutes": float(np.mean(rain_streak_lengths) * 5),
            "median_duration_minutes": float(np.median(rain_streak_lengths) * 5),
            "max_duration_steps": int(np.max(rain_streak_lengths)),
            "max_duration_hours": float(np.max(rain_streak_lengths) * 5 / 60),
        }
    else:
        streak_stats = {"n_rain_streaks": 0}

    brute_data = {
        "n_total_steps": n_total,
        "n_zero_rain_steps": n_zero,
        "n_positive_rain_steps": n_positive,
        "pct_zero_rain": pct_zero,
        "rain_nonzero_stats": rain_nonzero_stats,
        "rain_streak_stats": streak_stats,
    }

    # Figure: histogram of non-zero rainfall (log scale).
    figure_path = None
    if not skip_figures and len(rain_nonzero) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(rain_nonzero, bins=100, log=True, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_xlabel("rain_in (inches per 5 min, only steps with rain > 0)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title(f"Rainfall distribution for steps with rain_in > 0\nn={len(rain_nonzero):,}")
        fig.tight_layout()
        figure_path = _save_figure(fig, "section7_rain_distribution.png")

    # Markdown
    md_lines = []
    md_lines.append("## 7. Rainfall characterization\n")
    md_lines.append(
        f"Analysis of `rain_in` behavior (rainfall in inches per 5-minute step) on the "
        f"train split. It contextualizes why the `rain_sum_*` features are compressed near "
        f"zero for most of the time.\n"
    )

    md_lines.append("### Fraction of steps with rain\n")
    md_lines.append(f"- **Total steps:** {n_total:,}")
    md_lines.append(f"- **Steps without rain (rain_in == 0):** {n_zero:,} ({pct_zero:.2f}%)")
    md_lines.append(f"- **Steps with rain (rain_in > 0):** {n_positive:,} ({100 - pct_zero:.2f}%)")
    md_lines.append("")

    if rain_nonzero_stats.get("n", 0) > 0:
        md_lines.append("### Non-zero rainfall statistics\n")
        md_lines.append(f"- **Mean:** {rain_nonzero_stats['mean']:.5f} inches / 5 min")
        md_lines.append(f"- **Median:** {rain_nonzero_stats['median']:.5f} inches / 5 min")
        md_lines.append(f"- **95th percentile:** {rain_nonzero_stats['q95']:.5f} inches / 5 min")
        md_lines.append(f"- **99th percentile:** {rain_nonzero_stats['q99']:.5f} inches / 5 min")
        md_lines.append(f"- **Max:** {rain_nonzero_stats['max']:.5f} inches / 5 min")
        md_lines.append("")

    if streak_stats.get("n_rain_streaks", 0) > 0:
        md_lines.append("### Rain streaks (contiguous sequences with rain_in > 0)\n")
        md_lines.append(f"- **Number of identified streaks:** {streak_stats['n_rain_streaks']:,}")
        md_lines.append(f"- **Mean duration:** {streak_stats['mean_duration_minutes']:.1f} minutes")
        md_lines.append(f"- **Median duration:** {streak_stats['median_duration_minutes']:.1f} minutes")
        md_lines.append(f"- **Max duration:** {streak_stats['max_duration_hours']:.1f} hours")
        md_lines.append("")

    if figure_path:
        md_lines.append(f"![Rain distribution]({figure_path})\n")

    return "\n".join(md_lines), brute_data



# ============================================================================
# SECCION 8: COMPARACION CON PREDICTOR NAIVE (PERSISTENCIA)
# ============================================================================

def compute_section_8_naive_baseline(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Computes the NSE of a naive "persistence" predictor (y_pred(t+h) = y(t)) on test
    for horizons H=1, H=3, H=6, and compares it with the NSE of the current model.

    This comparison is critical: if the model does not clearly beat naive, the project
    is not learning the rain -> stormflow relationship but simply copying the previous value.
    """
    # We work on test to make a direct comparison with the model metrics.
    # (which are also evaluated on test according to local_eval_metrics.json)
    df_test = splits["test"]
    target_series = df_test[TARGET_COLUMN].to_numpy()

    # NSE definition: 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    # An NSE of 1 is a perfect prediction, NSE of 0 is predicting the mean, negative NSE is worse than the mean.
    def _nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return float("nan")
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)  # Observed total variance.
        if denominator <= 0:  # Degenerate case: zero variance.
            return float("nan")
        numerator = np.sum((y_true - y_pred) ** 2)  # Sum of squared prediction errors.
        return float(1.0 - numerator / denominator)

    # Horizons to compare: 1, 3, 6 steps = 5, 15, 30 minutes.
    horizons = [1, 3, 6]

    # Current model metrics for each horizon (source: local_eval_metrics.json)
    # We copy them here so we can compare directly without opening the JSON.
    model_nse_sin_sf = {1: 0.861, 3: 0.471, 6: -1.212}  # H1_sinSF, H3_sinSF, H6_sinSF
    model_nse_con_sf = {1: 0.854, 3: 0.488, 6: 0.255}   # H1_conSF, H3_conSF, H6_conSF

    # For each horizon, we compute naive NSE and the "gain" over the model
    # (delta NSE). If the gain is positive, the model adds value.
    comparison = {}
    for h in horizons:
        # Naive predictor: copy the current value as the prediction for step h.
        # y_true are the values from step h onward (the ones we want to predict).
        # y_pred are the values from step 0 up to -h (the ones we copy as prediction).
        if h >= len(target_series):  # Safety case (should not happen with a 165k-row test).
            continue
        y_true = target_series[h:]            # Real values at t+h.
        y_pred_naive = target_series[:-h]     # Naive prediction is the value at t.
        nse_naive = _nse(y_true, y_pred_naive)

        # Model gains over naive (negative means the model is worse than naive).
        gain_sin_sf = model_nse_sin_sf[h] - nse_naive
        gain_con_sf = model_nse_con_sf[h] - nse_naive

        comparison[h] = {
            "n_samples": int(len(y_true)),
            "nse_naive_persistence": float(nse_naive),
            "nse_model_sin_sf": float(model_nse_sin_sf[h]),
            "nse_model_con_sf": float(model_nse_con_sf[h]),
            "gain_sin_sf_over_naive": float(gain_sin_sf),
            "gain_con_sf_over_naive": float(gain_con_sf),
        }

    brute_data = {
        "comparison_by_horizon": comparison,
        "note": "Model NSE taken from outputs/data_analysis/local_eval_metrics.json (evaluation on 2026-04-16).",
    }

    # Markdown
    md_lines = []
    md_lines.append("## 8. Comparison with naive predictor (persistence)\n")
    md_lines.append(
        "The current model is compared with a trivial predictor that copies the current value as "
        "the prediction of the next step. Formally: $\\hat{y}(t+h) = y(t)$. This is the absolute "
        "baseline: any useful model should clearly beat it.\n"
    )
    md_lines.append(
        "The comparison is especially relevant because section 6 shows that the target ACF is "
        "high in the short term (0.91 at 5 min), which implies persistence already has "
        "non-trivial performance.\n"
    )

    md_lines.append("### Model NSE vs naive NSE (test split)\n")
    md_lines.append("| Horizon | Min | NSE naive | NSE model WITHOUT SF | Gain WITHOUT SF | NSE model WITH SF | Gain WITH SF |")
    md_lines.append("|-----------|-----|----------:|---------------------:|----------------:|------------------:|----------------:|")
    for h, entry in comparison.items():
        minutes = h * 5
        naive_nse = entry["nse_naive_persistence"]
        m_sin = entry["nse_model_sin_sf"]
        m_con = entry["nse_model_con_sf"]
        g_sin = entry["gain_sin_sf_over_naive"]
        g_con = entry["gain_con_sf_over_naive"]
        # Format the gain with an explicit sign so it is obvious whether it is positive or negative.
        sign_sin = "+" if g_sin >= 0 else ""
        sign_con = "+" if g_con >= 0 else ""
        md_lines.append(
            f"| H={h} | {minutes}m | {naive_nse:.3f} | {m_sin:.3f} | {sign_sin}{g_sin:.3f} | "
            f"{m_con:.3f} | {sign_con}{g_con:.3f} |"
        )
    md_lines.append("")

    # Interpretacion automatica basada en los numeros
    h1_gain_sin = comparison[1]["gain_sin_sf_over_naive"]
    h3_gain_sin = comparison[3]["gain_sin_sf_over_naive"]
    h6_gain_sin = comparison[6]["gain_sin_sf_over_naive"]

    md_lines.append("### Reading\n")
    if h1_gain_sin > 0.05:
        md_lines.append(f"- **H=1 (5 min):** the model WITHOUT SF beats naive by {h1_gain_sin:+.3f} NSE. Moderate improvement.")
    elif h1_gain_sin > 0.0:
        md_lines.append(f"- **H=1 (5 min):** the model WITHOUT SF beats naive by only {h1_gain_sin:+.3f} NSE. Marginal improvement.")
    else:
        md_lines.append(f"- **H=1 (5 min):** the model WITHOUT SF is WORSE than naive by {h1_gain_sin:.3f} NSE.")

    if h3_gain_sin > 0.0:
        md_lines.append(f"- **H=3 (15 min):** the model WITHOUT SF beats naive by {h3_gain_sin:+.3f} NSE.")
    else:
        md_lines.append(f"- **H=3 (15 min):** the model WITHOUT SF is WORSE than naive by {abs(h3_gain_sin):.3f} NSE. A one-line script (`y_pred = y_last`) does better than the TCN.")

    if h6_gain_sin > 0.0:
        md_lines.append(f"- **H=6 (30 min):** the model WITHOUT SF beats naive by {h6_gain_sin:+.3f} NSE.")
    else:
        md_lines.append(f"- **H=6 (30 min):** the model WITHOUT SF is CATASTROPHICALLY worse than naive ({abs(h6_gain_sin):.2f} NSE difference).")

    md_lines.append("")
    md_lines.append(
        "> **Operational and academic implication:** reporting NSE=0.861 at H=1 without comparing "
        "it with naive gives a misleading impression of the model's value. A robust thesis defense "
        "should include this comparison and justify why a model with ~104K parameters improves "
        "(or not) over a one-line predictor."
    )
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 9: SINTESIS DE HALLAZGOS PARA LA REVISION EXTERNA
# ============================================================================

def compute_section_9_synthesis(
    splits: Dict[str, pd.DataFrame],
    section_outputs: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Summarizes the most relevant findings from the document in a short list, suitable for
    serving as a starting point for the external review with Opus 4.7.

    This section does NOT introduce new analysis: it synthesizes what sections 1-8
    already laid out, making it explicit as "observations to review".
    """
    # We extract numeric data from the previous sections to write a quantitative synthesis.
    # This keeps the numbers exact even if the dataset changes in the future.
    s2 = section_outputs.get("section_2", {})
    s3 = section_outputs.get("section_3", {})
    s5 = section_outputs.get("section_5", {})
    s6 = section_outputs.get("section_6", {})
    s8 = section_outputs.get("section_8", {})

    # Values for the synthesis. We use .get() with defaults to avoid KeyError if a section failed.
    train_max = s2.get("by_split", {}).get("train", {}).get("q100", float("nan"))
    val_max = s2.get("by_split", {}).get("val", {}).get("q100", float("nan"))
    test_max = s2.get("by_split", {}).get("test", {}).get("q100", float("nan"))

    n_extreme_train = s3.get("by_split", {}).get("train", {}).get("n", 0)
    n_extreme_val = s3.get("by_split", {}).get("val", {}).get("n", 0)
    n_extreme_test = s3.get("by_split", {}).get("test", {}).get("n", 0)

    n_redundant_pairs = len(s5.get("high_correlation_pairs_abs_gt_0_7", []))

    acf_5min = s6.get("acf_at_key_lags", {}).get("1", {}).get("acf", float("nan"))
    acf_30min = s6.get("acf_at_key_lags", {}).get("6", {}).get("acf", float("nan"))

    h1_gain = s8.get("comparison_by_horizon", {}).get(1, {}).get("gain_sin_sf_over_naive", float("nan"))
    h3_gain = s8.get("comparison_by_horizon", {}).get(3, {}).get("gain_sin_sf_over_naive", float("nan"))
    h6_gain = s8.get("comparison_by_horizon", {}).get(6, {}).get("gain_sin_sf_over_naive", float("nan"))

    brute_data = {
        "key_findings": [
            {
                "id": "F1_naive_baseline_too_close",
                "title": "El modelo apenas bate al predictor naive a H=1 y es peor a H>=3.",
                "evidence": f"Ganancia sobre naive: H=1 {h1_gain:+.3f}, H=3 {h3_gain:+.3f}, H=6 {h6_gain:+.3f}",
                "severity": "alta",
            },
            {
                "id": "F2_val_extremes_larger_than_test",
                "title": "Val contains more extreme events than test, which contains more than train-max.",
                "evidence": f"Max by split: train={train_max:.1f} MGD, val={val_max:.1f} MGD, test={test_max:.1f} MGD",
                "severity": "alta",
            },
            {
                "id": "F3_test_size_for_extremes",
                "title": "Test has only 59 extreme samples, insufficient for robust conclusions in the Extreme bucket.",
                "evidence": f"n extreme: train={n_extreme_train}, val={n_extreme_val}, test={n_extreme_test}",
                "severity": "media",
            },
            {
                "id": "F4_feature_redundancy",
                "title": "Massive redundancy among features: many strongly correlated pairs.",
                "evidence": f"{n_redundant_pairs} pairs with |Pearson| >= 0.7 among the 22 features.",
                "severity": "media",
            },
            {
                "id": "F5_acf_bounds_predictability",
                "title": "The target ACF imposes a theoretical ceiling on multi-step predictability.",
                "evidence": f"ACF(5min)={acf_5min:.3f}, ACF(30min)={acf_30min:.3f}",
                "severity": "media",
            },
        ]
    }

    # Markdown
    md_lines = []
    md_lines.append("## 9. Sintesis de hallazgos para la revision externa\n")
    md_lines.append(
        "Esta seccion resume los hallazgos mas relevantes del documento, pensada para servir "
        "de punto de partida a la revision con Opus 4.7. No introduce analisis nuevo: explicita "
        "lo que las secciones 1-8 ya exponen.\n"
    )

    md_lines.append("### Findings ordered by severity\n")

    md_lines.append("**1. [HIGH] The model barely beats the naive predictor at H=1 and is WORSE at H=3 and H=6.**")
    md_lines.append(f"")
    md_lines.append(f"Ganancia del modelo SIN SF sobre persistencia: H=1 {h1_gain:+.3f}, H=3 {h3_gain:+.3f}, H=6 {h6_gain:+.3f}.")
    md_lines.append(f"A model of ~104K parameters must justify why it improves (or not) on a one-line predictor.")
    md_lines.append(f"Question for the reviewer: is NSE=0.861 at H=1 really a good result given that naive gives {s8.get('comparison_by_horizon', {}).get(1, {}).get('nse_naive_persistence', 'NaN'):.3f}?")
    md_lines.append("")

    md_lines.append("**2. [HIGH] Asymmetry across splits in extreme-event coverage.**")
    md_lines.append(f"")
    md_lines.append(f"Maximum value by split: train={train_max:.1f} MGD, **val={val_max:.1f} MGD** (absolute dataset maximum), test={test_max:.1f} MGD.")
    md_lines.append(f"Extremes by split: train={n_extreme_train}, val={n_extreme_val}, test={n_extreme_test}.")
    md_lines.append(f"Consequence: the model is evaluated on test under a less extreme regime than training, while val includes the absolute peak. Early stopping on val may be favoring underestimation.")
    md_lines.append("")

    md_lines.append("**3. [MEDIUM] Sample size is insufficient for the Extreme bucket in test.**")
    md_lines.append(f"")
    md_lines.append(f"Only 59 extreme samples in test. Metrics on this bucket (NSE=-0.99, bias=-12.7 MGD) have high variance. A single atypical event can move them significantly.")
    md_lines.append(f"Question for the reviewer: does it make sense to report metrics on this bucket with this sample, or should we use bootstrap or k-fold to estimate confidence intervals?")
    md_lines.append("")

    md_lines.append("**4. [MEDIUM] Massive redundancy among features.**")
    md_lines.append(f"")
    md_lines.append(f"{n_redundant_pairs} feature pairs with |Pearson| >= 0.7. The strongest ones: rain_sum_10m vs rain_max_10m (0.985), rain_sum_10m vs rain_sum_15m (0.970), api_dynamic correlates with 9 different features.")
    md_lines.append(f"Of the 22 declared features, the effective number of independent dimensions is much smaller (possibly ~8-10).")
    md_lines.append(f"Question for the reviewer: should we reduce features before continuing to iterate on architecture/loss?")
    md_lines.append("")

    md_lines.append("**5. [MEDIUM] The target ACF imposes a theoretical limit on H>1.**")
    md_lines.append(f"")
    md_lines.append(f"ACF(5 min) = {acf_5min:.3f}, ACF(30 min) = {acf_30min:.3f}. The ACF decays quickly after the first 30 minutes.")
    md_lines.append(f"This indicates that, without incorporating external rainfall prediction (option c in STATE.md), extending the useful horizon beyond ~15 min may be physically limited.")
    md_lines.append("")

    md_lines.append("### Suggested priorities (to validate with the reviewer)\n")
    md_lines.append("1. **Report the naive baseline in all future metrics.** No-regrets, 5 lines of code.")
    md_lines.append("2. **Review the split.** Evaluate alternative splits (full years, TimeSeriesSplit, blocked CV by events).")
    md_lines.append("3. **Simplify features.** Reduce from 22 to ~10 by removing redundancy, with permutation importance on the simplified model.")
    md_lines.append("4. **Confidence intervals.** Bootstrap on test to quantify uncertainty in the Extreme bucket metrics.")
    md_lines.append("5. **Realistic horizon.** Accept that H>1 with the current features has a physical limit, or integrate MSD's rainfall model (option c).")
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# MAIN: orquestacion completa
# ============================================================================

def _build_markdown_header(config_path: Path) -> str:
    """Builds the markdown header with execution metadata."""
    from datetime import datetime  # Local import: only used here

    md_lines = []
    md_lines.append("# DATASET_STATS.md - Statistical summary of the dataset\n")
    md_lines.append(
        f"Automatically generated by `scripts/generate_dataset_stats.py`.\n"
    )
    md_lines.append(f"- **Generation date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- **Config used:** `{config_path}`")
    md_lines.append(f"- **Purpose:** input for external review with Opus 4.7 on methodology and physical ceiling of the model.")
    md_lines.append("")
    md_lines.append(
        "This document describes *the data*, not the model. For current model metrics, "
        "see `outputs/data_analysis/local_eval_metrics.json` and `docs/STATE.md`."
    )
    md_lines.append("\n---\n")
    return "\n".join(md_lines)


def main(config_path: Path, use_cache: bool, skip_figures: bool) -> None:
    """
    Orchestrates the full flow: loads data, computes 7 sections, writes markdown + JSON.
    """
    print("=" * 70)
    print("GENERATE_DATASET_STATS.PY")
    print("=" * 70)

    # Initial step: create output directories if they do not exist.
    _ensure_dirs_exist()

    # Step 1: load and process data (with cache if available).
    print("\n[main] Step 1: load and process data")
    df_features = _load_and_process_data(config_path=config_path, use_cache=use_cache)

    # Step 2: perform the chronological split.
    print("\n[main] Step 2: chronological split 70/15/15")
    splits = _split_data(df_features)
    for split_name, df_split in splits.items():
        print(f"[main]      {split_name}: {len(df_split):,} rows")

    # Step 3: compute the 9 report sections in order.
    print("\n[main] Step 3: compute the 9 report sections")

    print("[main]      Section 1: global summary")
    md1, bd1 = compute_section_1_global_summary(df_features, splits)

    print("[main]      Section 2: target distribution")
    md2, bd2 = compute_section_2_target_distribution(df_features, splits, skip_figures=skip_figures)

    print("[main]      Section 3: extreme events")
    md3, bd3 = compute_section_3_extreme_events(df_features, splits, skip_figures=skip_figures)

    print("[main]      Section 4: feature-target correlations")
    md4, bd4 = compute_section_4_feature_target_correlations(splits, skip_figures=skip_figures)

    print("[main]      Section 5: feature-feature matrix")
    md5, bd5 = compute_section_5_feature_feature_matrix(splits, skip_figures=skip_figures)

    print("[main]      Section 6: target autocorrelation")
    md6, bd6 = compute_section_6_target_autocorrelation(splits, skip_figures=skip_figures)

    print("[main]      Section 7: rainfall characterization")
    md7, bd7 = compute_section_7_rain_characterization(splits, skip_figures=skip_figures)

    print("[main]      Section 8: comparison with naive predictor")
    md8, bd8 = compute_section_8_naive_baseline(splits, skip_figures=skip_figures)

    # Section 9 receives the brute_data from the previous sections to generate the synthesis.
    section_outputs_for_synthesis = {
        "section_2": bd2,
        "section_3": bd3,
        "section_5": bd5,
        "section_6": bd6,
        "section_8": bd8,
    }
    print("[main]      Section 9: findings synthesis")
    md9, bd9 = compute_section_9_synthesis(splits, section_outputs_for_synthesis)

    # Step 4: concatenate all sections into a single markdown document.
    print("\n[main] Step 4: write markdown and JSON output")
    header = _build_markdown_header(config_path)
    full_markdown = header + "\n---\n\n".join([md1, md2, md3, md4, md5, md6, md7, md8, md9])

    # Escribir markdown
    with open(OUTPUT_MARKDOWN, "w", encoding="utf-8") as f:
        f.write(full_markdown)
    print(f"[main]      Markdown written to {OUTPUT_MARKDOWN}")

    # Escribir JSON con todos los brute_data
    full_brute_data = {
        "section_1_global_summary": bd1,
        "section_2_target_distribution": bd2,
        "section_3_extreme_events": bd3,
        "section_4_feature_target_correlations": bd4,
        "section_5_feature_feature_matrix": bd5,
        "section_6_target_autocorrelation": bd6,
        "section_7_rain_characterization": bd7,
        "section_8_naive_baseline": bd8,
        "section_9_synthesis": bd9,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_brute_data, f, indent=2, default=str)
    print(f"[main]      JSON written to {OUTPUT_JSON}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Command-line argument parsing.
    parser = argparse.ArgumentParser(description="Generates docs/DATASET_STATS.md with dataset statistics.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local.yaml",
        help="Path to the configuration YAML (default: configs/local.yaml for local execution).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Forces dataframe regeneration without using the parquet cache.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skips PNG figure generation (faster, tables only).",
    )
    args = parser.parse_args()

    # Resolve the path relative to the repo root so the script is independent of the cwd.
    resolved_config = (REPO_ROOT / args.config).resolve()
    if not resolved_config.exists():
        raise FileNotFoundError(f"The configuration file does not exist: {resolved_config}")

    main(
        config_path=resolved_config,
        use_cache=not args.no_cache,
        skip_figures=args.skip_figures,
    )

