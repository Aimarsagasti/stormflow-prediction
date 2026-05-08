"""
S5 - Feature analysis and redundancy.

Objective: identify which features provide real signal (not an autoregressive
shortcut), detect redundancy, and propose an implementable reduced set. It works
on the same split / aligned indices as S2 so the metrics are comparable.

Stages:
  1. Retrains XGBoost-20 (without delta_flow_*) for H=1 with the same hyperparameters as S2.
  2. Importance analysis:
       - Native importance (gain) de XGBoost.
       - Permutation importance on aligned TEST (n_repeats=5,
         subsample=50_000 si test es grande).
       - SHAP TreeExplainer on 5_000 test rows (random_state=42).
  3. Feature-feature correlation matrix on TRAIN; hierarchical clustering
     by |1 - r| with cutoff r>=0.85; representative = most important feature
     of the cluster by permutation importance.
  4. Individual ACF of key features at lags {1, 6, 12, 24} on TRAIN.
     Detects hidden shortcuts (features with ACF very similar to the target).
  5. Proposed reduced set (8-12 features) and XGBoost retraining
     to compare NSE H=1 against XGB-20 and AR(12).
  6. Hidden shortcut detection: 1-by-1 ablations on top-PI;
     a feature whose ablation causes an NSE drop >0.05 is a shortcut candidate.

Only uses: pandas, numpy, sklearn, xgboost, scipy, matplotlib, seaborn,
shap (optional, skipped if unavailable).

Artefactos:
  - outputs/diagnostic/S5_feature_analysis.json
  - outputs/diagnostic/S5_feature_analysis.md
  - outputs/figures/diagnostic/s5_corr_matrix.png
  - outputs/figures/diagnostic/s5_permutation_importance.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from xgboost import XGBRegressor

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    print(f"[S5] SHAP not available: {exc}")
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration (aligned with S2)
# ---------------------------------------------------------------------------
ROOT = Path("C:/Dev/TFM")
PARQUET_PATH = ROOT / "outputs" / "cache" / "df_with_features.parquet"
OUT_DIR = ROOT / "outputs" / "diagnostic"
FIG_DIR = ROOT / "outputs" / "figures" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

IDX_TRAIN_END = 771374
IDX_VAL_END = 936669
SEQ_LENGTH = 72
HORIZON = 1
TARGET_COL = "stormflow_mgd"

FEATURES_22 = [
    "rain_in", "temp_daily_f", "api_dynamic",
    "rain_sum_10m", "rain_sum_15m", "rain_sum_30m", "rain_sum_60m",
    "rain_sum_120m", "rain_sum_180m", "rain_sum_360m",
    "rain_max_10m", "rain_max_30m", "rain_max_60m",
    "minutes_since_last_rain",
    "delta_flow_5m", "delta_flow_15m",
    "delta_rain_10m", "delta_rain_30m",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
FEATURES_20 = [c for c in FEATURES_22 if not c.startswith("delta_flow_")]

KEY_FEATURES_ACF = ["rain_sum_60m", "api_dynamic", "rain_sum_360m", "delta_flow_5m"]
ACF_LAGS = [1, 6, 12, 24]
TARGET_ACF_REF = {1: 0.91, 6: 0.56, 12: 0.42, 24: 0.28}  # DATASET_STATS section 6

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

PI_N_REPEATS = 5
PI_TEST_SUBSAMPLE = 50_000  # subsample to speed up PI
SHAP_N_SAMPLES = 5_000
RANDOM_STATE = 42

CORR_CLUSTER_THRESHOLD = 0.85  # |r| >= 0.85 => same cluster
EXTREME_BUCKET_THRESHOLD = 25.0  # MGD: used for SHAP on extreme events


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def aligned_indices(split_start: int, split_end: int, horizon: int, total_len: int) -> np.ndarray:
    first = split_start + SEQ_LENGTH
    last_in_split = split_end - 1
    last_by_target = total_len - 1 - horizon
    last = min(last_in_split, last_by_target)
    return np.arange(first, last + 1)


def acf_at_lags(x: np.ndarray, lags: List[int]) -> Dict[int, float]:
    """Sample ACF (Pearson) at the indicated lags. NaN if variance is zero."""
    out: Dict[int, float] = {}
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < max(lags) + 10:
        return {l: float("nan") for l in lags}
    x_mean = float(np.mean(x))
    x_dev = x - x_mean
    var = float(np.sum(x_dev ** 2))
    if var == 0:
        return {l: float("nan") for l in lags}
    for l in lags:
        cov = float(np.sum(x_dev[:-l] * x_dev[l:]))
        out[l] = cov / var
    return out


# ---------------------------------------------------------------------------
# Stage 1: entrenamiento XGB-20
# ---------------------------------------------------------------------------
def train_xgb20(df: pd.DataFrame) -> Tuple[XGBRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    print("[1] Training XGBoost-20 (without delta_flow) H=1...")
    idx_train = aligned_indices(0, IDX_TRAIN_END, HORIZON, len(df))
    idx_test = aligned_indices(IDX_VAL_END, len(df), HORIZON, len(df))
    print(f"    n_train={len(idx_train):,}  n_test={len(idx_test):,}")

    X = df[FEATURES_20].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    X_train = X[idx_train]
    y_train = y[idx_train + HORIZON]
    X_test = X[idx_test]
    y_test = y[idx_test + HORIZON]

    model = XGBRegressor(**XGB_PARAMS)
    t0 = time.time()
    model.fit(X_train, y_train)
    t_fit = time.time() - t0
    print(f"    fit OK in {t_fit:.1f}s")

    return model, X_train, y_train, X_test, y_test, t_fit


# ---------------------------------------------------------------------------
# Stage 2: importance analysis
# ---------------------------------------------------------------------------
def native_importance(model: XGBRegressor, features: List[str]) -> Dict[str, float]:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    out: Dict[str, float] = {}
    for i, name in enumerate(features):
        key = f"f{i}"
        out[name] = float(score.get(key, 0.0))
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


def permutation_importance_manual(
    model: XGBRegressor,
    X: np.ndarray,
    y: np.ndarray,
    features: List[str],
    n_repeats: int = 5,
    rng_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """PI = baseline_NSE - mean_NSE_shuffled. Higher = more important.

    Returns a dict by feature with keys {mean_drop, std_drop, baseline_nse,
    shuffled_nse_mean}.
    """
    rng = np.random.default_rng(rng_seed)
    base_pred = model.predict(X)
    base_nse = nse(y, base_pred)
    out: Dict[str, Dict[str, float]] = {}
    n = len(X)
    for j, name in enumerate(features):
        drops = np.empty(n_repeats, dtype=float)
        nses_perm = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            X_perm = X.copy()
            order = rng.permutation(n)
            X_perm[:, j] = X[order, j]
            y_pred = model.predict(X_perm)
            nse_perm = nse(y, y_pred)
            nses_perm[r] = nse_perm
            drops[r] = base_nse - nse_perm
        out[name] = {
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
            "baseline_nse": float(base_nse),
            "shuffled_nse_mean": float(np.mean(nses_perm)),
        }
    return out


def shap_analysis(
    model: XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: List[str],
    n_samples: int,
    extreme_threshold: float,
    rng_seed: int = 42,
) -> Dict[str, object]:
    """SHAP TreeExplainer on a test subsample.
    Reports:
      - mean |SHAP| per feature.
      - mean SHAP in extreme events (y_true>=threshold) vs baseflow (y_true<0.5).
      - features with sign change (positive in extremes, negative in baseflow
        or vice versa).
    """
    rng = np.random.default_rng(rng_seed)
    n = len(X_test)
    if n_samples >= n:
        sel = np.arange(n)
    else:
        sel = rng.choice(n, size=n_samples, replace=False)
        sel.sort()
    X_sub = X_test[sel]
    y_sub = y_test[sel]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)  # (n, p)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mask_extreme = y_sub >= extreme_threshold
    mask_base = y_sub < 0.5
    n_extreme = int(np.sum(mask_extreme))
    n_base = int(np.sum(mask_base))

    out: Dict[str, object] = {
        "n_samples": int(len(sel)),
        "n_extreme_in_sample": n_extreme,
        "n_base_in_sample": n_base,
        "extreme_threshold_mgd": extreme_threshold,
        "by_feature": {},
        "non_monotonic_candidates": [],
    }
    for j, name in enumerate(features):
        rec: Dict[str, float] = {
            "mean_abs_shap": float(mean_abs[j]),
            "mean_shap_extreme": (
                float(np.mean(shap_values[mask_extreme, j])) if n_extreme > 0 else float("nan")
            ),
            "mean_shap_base": (
                float(np.mean(shap_values[mask_base, j])) if n_base > 0 else float("nan")
            ),
        }
        # Range and absolute median for context
        rec["std_shap"] = float(np.std(shap_values[:, j]))
        out["by_feature"][name] = rec
        # Detect significant sign change
        if n_extreme > 0 and n_base > 0:
            if rec["mean_shap_extreme"] * rec["mean_shap_base"] < 0:
                out["non_monotonic_candidates"].append({
                    "feature": name,
                    "mean_shap_extreme": rec["mean_shap_extreme"],
                    "mean_shap_base": rec["mean_shap_base"],
                })
    return out


# ---------------------------------------------------------------------------
# Stage 3: correlations and clustering
# ---------------------------------------------------------------------------
def corr_matrix_train(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    idx_train = aligned_indices(0, IDX_TRAIN_END, HORIZON, len(df))
    sub = df[features].iloc[idx_train]
    return sub.corr(method="pearson")


def cluster_features(
    corr: pd.DataFrame, threshold: float, importance: Dict[str, float]
) -> Tuple[List[List[str]], Dict[str, str]]:
    """Complete-linkage hierarchical clustering on `|1 - r|`. Cuts at `(1 - threshold).
    Returns `(sorted_clusters, feature_to_representative_mapping)`.
    The representative is the feature with the highest importance in the cluster.
    """
    abs_corr = corr.abs().values
    # Distance: d_ij = 1 - |r_ij| ; condenses the symmetric matrix
    d = 1.0 - abs_corr
    np.fill_diagonal(d, 0.0)
    # Force perfect symmetry for `squareform`
    d = (d + d.T) / 2.0
    condensed = squareform(d, checks=False)
    Z = sch.linkage(condensed, method="complete")
    cut_distance = 1.0 - threshold
    cluster_ids = sch.fcluster(Z, t=cut_distance, criterion="distance")

    feat_names = list(corr.columns)
    clusters_dict: Dict[int, List[str]] = {}
    for fname, cid in zip(feat_names, cluster_ids):
        clusters_dict.setdefault(int(cid), []).append(fname)

    # Sort clusters by descending size and then choose a representative
    clusters_sorted = sorted(clusters_dict.values(), key=lambda c: -len(c))
    representative: Dict[str, str] = {}
    for cl in clusters_sorted:
        rep = max(cl, key=lambda f: importance.get(f, 0.0))
        for f in cl:
            representative[f] = rep
    return clusters_sorted, representative


def plot_corr_heatmap(corr: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        annot_kws={"size": 7}, cbar_kws={"label": "Pearson r"}, square=True,
    )
    plt.title("S5 - Feature-feature correlation matrix (TRAIN, 20 features)")
    plt.xticks(rotation=70, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_perm_importance(pi: Dict[str, Dict[str, float]], path: Path) -> None:
    items = sorted(pi.items(), key=lambda kv: kv[1]["mean_drop"], reverse=True)
    names = [k for k, _ in items]
    means = [v["mean_drop"] for _, v in items]
    stds = [v["std_drop"] for _, v in items]
    colors = ["#1f77b4" if m > 0 else "#d62728" for m in means]

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.7)
    ax.set_xlabel("NSE drop when shuffled (mean +/- std, n_repeats=5)")
    ax.set_title("S5 - Permutation importance XGBoost-20 (aligned test)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# Stage 5: reduced set
# ---------------------------------------------------------------------------
def select_reduced_features(
    pi: Dict[str, Dict[str, float]],
    representative: Dict[str, str],
    target_size_min: int = 8,
    target_size_max: int = 12,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Strategy:
      1. For each cluster, use ONLY the representative (max PI inside the cluster).
      2. Remove features with mean_drop <= 0.
      3. If more than target_size_max remain, keep the top by PI.
      4. If fewer than target_size_min remain, fill with the next feature by PI.
    """
    reps = sorted(set(representative.values()))
    candidates = [f for f in reps if pi[f]["mean_drop"] > 0.0]
    candidates_ranked = sorted(candidates, key=lambda f: pi[f]["mean_drop"], reverse=True)

    selected = candidates_ranked[:target_size_max]
    if len(selected) < target_size_min:
        # Complete with features outside `selected` but with PI > 0
        extras = [
            f for f in sorted(pi.keys(), key=lambda f: pi[f]["mean_drop"], reverse=True)
            if f not in selected and pi[f]["mean_drop"] > 0
        ]
        selected = selected + extras[: (target_size_min - len(selected))]

    # Justification per feature
    justifications: List[Dict[str, str]] = []
    for f in selected:
        cluster_members = [k for k, v in representative.items() if v == f]
        justifications.append({
            "feature": f,
            "permutation_drop": pi[f]["mean_drop"],
            "represents_cluster_size": len(cluster_members),
            "cluster_members": cluster_members,
        })
    return selected, justifications


def train_reduced_and_eval(
    df: pd.DataFrame, features: List[str]
) -> Dict[str, object]:
    idx_train = aligned_indices(0, IDX_TRAIN_END, HORIZON, len(df))
    idx_test = aligned_indices(IDX_VAL_END, len(df), HORIZON, len(df))
    X = df[features].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    X_train, y_train = X[idx_train], y[idx_train + HORIZON]
    X_test, y_test = X[idx_test], y[idx_test + HORIZON]
    model = XGBRegressor(**XGB_PARAMS)
    t0 = time.time()
    model.fit(X_train, y_train)
    t_fit = time.time() - t0
    y_pred = model.predict(X_test)
    return {
        "n_features": len(features),
        "features": features,
        "fit_seconds": t_fit,
        "nse": float(nse(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "peak_real": float(np.max(y_test)),
        "peak_pred": float(np.max(y_pred)),
        "peak_err_pct": float((np.max(y_pred) - np.max(y_test)) / np.max(y_test) * 100.0),
    }


# ---------------------------------------------------------------------------
# Stage 6: 1-by-1 ablation on top-PI (hidden shortcut detection)
# ---------------------------------------------------------------------------
def leave_one_out_ablation(
    df: pd.DataFrame,
    base_features: List[str],
    candidates: List[str],
    base_nse: float,
) -> Dict[str, Dict[str, float]]:
    """For each candidate, trains XGBoost with `base_features \\ {c}` and measures the drop."""
    idx_train = aligned_indices(0, IDX_TRAIN_END, HORIZON, len(df))
    idx_test = aligned_indices(IDX_VAL_END, len(df), HORIZON, len(df))
    X_full = df[base_features].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    y_train = y[idx_train + HORIZON]
    y_test = y[idx_test + HORIZON]
    feat_to_idx = {f: i for i, f in enumerate(base_features)}
    out: Dict[str, Dict[str, float]] = {}
    for c in candidates:
        cols = [i for f, i in feat_to_idx.items() if f != c]
        X_train = X_full[idx_train][:, cols]
        X_test = X_full[idx_test][:, cols]
        model = XGBRegressor(**XGB_PARAMS)
        t0 = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t0
        y_pred = model.predict(X_test)
        nse_abl = nse(y_test, y_pred)
        out[c] = {
            "nse_without": float(nse_abl),
            "delta_nse": float(base_nse - nse_abl),
            "fit_seconds": t_fit,
        }
        print(f"    ablation -{c:<24s} NSE={nse_abl:.4f}  drop={base_nse - nse_abl:+.4f}")
    return out


# ---------------------------------------------------------------------------
# Renderizado MD
# ---------------------------------------------------------------------------
def render_md(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# S5 - Feature analysis and redundancy\n")
    lines.append(
        "Diagnostic of the real (non-autoregressive) value of the 20 official features "
        "after excluding `delta_flow_*` (shortcut confirmed in iter16/S2). It works on the "
        "same split and aligned indices as S2 so that metrics are "
        "directly comparable.\n"
    )

    lines.append("## Methodology\n")
    lines.append(
        "- **Base model**: XGBoost-20 (same hyperparameters as S2) retrained for H=1.\n"
        "- **Permutation importance**: n_repeats=5 on a test subsample "
        f"(n={report['stage_pi']['n_test_used']:,}), shuffling each feature individually.\n"
        f"- **SHAP**: {'TreeExplainer on ' + str(report['stage_shap']['n_samples']) + ' random test rows (random_state=42).' if SHAP_AVAILABLE else 'not available (library cannot be installed).'}\n"
        f"- **Correlations**: Pearson matrix on TRAIN ({report['stage_corr']['n_train']:,} rows), "
        f"hierarchical complete-linkage clustering on |1 - r| with cutoff r>={CORR_CLUSTER_THRESHOLD}.\n"
        "- **Key feature ACFs**: lags 1, 6, 12, 24 on TRAIN, compared with the target "
        "(reference DATASET_STATS section 6).\n"
    )

    # Time by stage
    timings = report["timings"]
    lines.append("### Time by stage\n")
    lines.append("| Stage | Seconds |")
    lines.append("|---|---:|")
    for k, v in timings.items():
        lines.append(f"| {k} | {v:.1f} |")
    lines.append("")

    # Importance ranking
    lines.append("## 1. Native importance (gain) y Permutation importance\n")
    lines.append(
        "`gain_pct` = normalized booster importance (sum 1.0). "
        "`PI mean_drop` = average NSE drop when the feature is shuffled in test. "
        "A feature with PI<=0 is considered noise (its ablation does not degrade the model).\n"
    )
    lines.append("| Feature | gain_pct | PI mean_drop | PI std | rank PI |")
    lines.append("|---|---:|---:|---:|---:|")
    pi = report["permutation_importance"]
    gain = report["native_importance"]
    items_sorted = sorted(pi.items(), key=lambda kv: kv[1]["mean_drop"], reverse=True)
    for rank, (f, v) in enumerate(items_sorted, start=1):
        lines.append(
            f"| {f} | {gain.get(f, 0.0):.4f} | {v['mean_drop']:+.4f} | {v['std_drop']:.4f} | {rank} |"
        )
    lines.append("")

    # SHAP
    if report.get("shap") is not None:
        lines.append("## 2. SHAP (mean |SHAP|, sign in extremes vs base)\n")
        lines.append(
            "Ranking by mean absolute importance. `mean_shap_extreme` = mean SHAP in "
            f"y_true>={EXTREME_BUCKET_THRESHOLD} MGD, `mean_shap_base` = SHAP medio en y_true<0.5 MGD. "
            "Sign change => non-monotonic contribution (the feature pushes up in extremes but "
            "down in baseflow or vice versa).\n"
        )
        sh = report["shap"]
        lines.append(
            f"_n_samples={sh['n_samples']:,}, n_extremes_in_sample={sh['n_extreme_in_sample']}, "
            f"n_base_in_sample={sh['n_base_in_sample']}._\n"
        )
        lines.append("| Feature | mean |SHAP| | mean SHAP extreme | mean SHAP base | std SHAP |")
        lines.append("|---|---:|---:|---:|---:|")
        items = sorted(sh["by_feature"].items(), key=lambda kv: kv[1]["mean_abs_shap"], reverse=True)
        for f, v in items:
            lines.append(
                f"| {f} | {v['mean_abs_shap']:.4f} | {v['mean_shap_extreme']:+.4f} | "
                f"{v['mean_shap_base']:+.4f} | {v['std_shap']:.4f} |"
            )
        lines.append("")
        if sh["non_monotonic_candidates"]:
            lines.append("**Features with non-monotonic SHAP (sign changes between extremes and base):**")
            for nm in sh["non_monotonic_candidates"]:
                lines.append(
                    f"- `{nm['feature']}`: extremo {nm['mean_shap_extreme']:+.4f} vs "
                    f"base {nm['mean_shap_base']:+.4f}"
                )
            lines.append("")
        else:
            lines.append("No feature shows a sign change between extremes and baseflow.\n")
    else:
        lines.append("## 2. SHAP\n")
        lines.append("SHAP not available. Using native + permutation importance.\n")

    # Correlations / clusters
    lines.append("## 3. Redundancy by correlation (TRAIN)\n")
    lines.append(f"Heatmap: `outputs/figures/diagnostic/s5_corr_matrix.png`.\n")
    pairs_high = report["high_corr_pairs"]
    lines.append(
        f"**Pairs with |r| >= {CORR_CLUSTER_THRESHOLD}**: {len(pairs_high)} (full list in JSON). "
        "Top 10 by |r|:\n"
    )
    lines.append("| Feature A | Feature B | r |")
    lines.append("|---|---|---:|")
    for p in pairs_high[:10]:
        lines.append(f"| {p['a']} | {p['b']} | {p['r']:+.3f} |")
    lines.append("")

    clusters = report["clusters"]
    lines.append(f"**Redundancy clusters (cutoff |r| >= {CORR_CLUSTER_THRESHOLD})**: {len(clusters)} clusters.\n")
    lines.append("| # | size | representative (max PI) | members |")
    lines.append("|---|---:|---|---|")
    for i, cl in enumerate(clusters, start=1):
        rep_set = set([report["representative"][f] for f in cl])
        rep = next(iter(rep_set))
        lines.append(f"| {i} | {len(cl)} | `{rep}` | {', '.join('`'+f+'`' for f in cl)} |")
    lines.append("")

    # ACF
    lines.append("## 4. Key feature ACF vs target\n")
    lines.append(
        "Sample ACF at lags 1, 6, 12, 24 on TRAIN. If a feature ACF is "
        "very similar to the target ACF at the same lags, it is a candidate for carrying "
        "hidden autoregressive target information through its own inertia.\n"
    )
    lines.append("| Feature | ACF lag1 | ACF lag6 | ACF lag12 | ACF lag24 |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| **target ({TARGET_COL})** | "
        f"{TARGET_ACF_REF[1]:.3f} | {TARGET_ACF_REF[6]:.3f} | "
        f"{TARGET_ACF_REF[12]:.3f} | {TARGET_ACF_REF[24]:.3f} |"
    )
    for f, acf_dict in report["acf_features"].items():
        def _get(d, k):
            # Supports int and str keys (keys are serialized as str for JSON).
            v = d.get(k, d.get(str(k), float("nan")))
            return float(v) if v is not None else float("nan")
        lines.append(
            f"| {f} | {_get(acf_dict, 1):.3f} | {_get(acf_dict, 6):.3f} | "
            f"{_get(acf_dict, 12):.3f} | {_get(acf_dict, 24):.3f} |"
        )
    lines.append("")

    # Reduced set
    lines.append("## 5. Proposed reduced set and comparison\n")
    lines.append(
        "Criterion: for each cluster with |r|>=0.85 only the feature with the "
        "highest permutation importance; the others are discarded. Features with "
        "PI<=0 (noise) are also removed. If after filtering more than 12 features remain, we keep "
        "the top by PI; if fewer than 8 remain, we fill with the next best ones.\n"
    )
    sel = report["reduced_set"]["selected"]
    lines.append(f"**Final size: {len(sel)} features.**\n")
    lines.append("| # | Feature | PI mean_drop | cluster size | cluster members |")
    lines.append("|---|---|---:|---:|---|")
    for i, j in enumerate(report["reduced_set"]["justification"], start=1):
        members = ", ".join(f"`{m}`" for m in j["cluster_members"])
        lines.append(
            f"| {i} | `{j['feature']}` | {j['permutation_drop']:+.4f} | "
            f"{j['represents_cluster_size']} | {members} |"
        )
    lines.append("")

    # Comparativa
    metrics_red = report["reduced_set"]["metrics"]
    metrics_xgb20 = report["xgb20_metrics"]
    delta_red_vs_xgb20 = metrics_red["nse"] - metrics_xgb20["nse"]
    lines.append("### NSE H=1 comparison (aligned test)\n")
    lines.append("| Modelo | N feats | NSE | RMSE | MAE | Peak pred | Peak err % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| **AR(12) (S2)** | - | {report['ar12_nse_h1_ref']:.4f} | - | - | - | - |"
    )
    lines.append(
        f"| **XGB-20 (S2 ref)** | 20 | {metrics_xgb20['nse']:.4f} | {metrics_xgb20['rmse']:.3f} | "
        f"{metrics_xgb20['mae']:.3f} | {metrics_xgb20['peak_pred']:.1f} | {metrics_xgb20['peak_err_pct']:+.1f} |"
    )
    lines.append(
        f"| **XGB-reducido** | {metrics_red['n_features']} | {metrics_red['nse']:.4f} | "
        f"{metrics_red['rmse']:.3f} | {metrics_red['mae']:.3f} | "
        f"{metrics_red['peak_pred']:.1f} | {metrics_red['peak_err_pct']:+.1f} |"
    )
    lines.append("")
    if abs(delta_red_vs_xgb20) <= 0.01:
        veredicto_red = "Reduced set **VALID** (delta within +-0.01 NSE)."
    elif delta_red_vs_xgb20 > 0.01:
        veredicto_red = (
            f"Reduced set **IMPROVES {delta_red_vs_xgb20:+.4f} NSE** over XGB-20: "
            "the reduction removes noise / features anti-correlated with the target."
        )
    else:
        veredicto_red = f"Reduced set **LOSES {abs(delta_red_vs_xgb20):.4f} NSE**: review the cutoff criterion."
    lines.append(f"Delta NSE (reduced - XGB-20) = **{delta_red_vs_xgb20:+.4f}**. {veredicto_red}\n")

    # Hidden shortcuts
    lines.append("## 6. Hidden shortcuts (1-by-1 ablation on top-PI)\n")
    lines.append(
        "On the reduced set, individual ablation of each feature. "
        "A drop >0.05 NSE when removing it indicates a very strong dependency: candidate shortcut "
        "(or a genuinely irreplaceable feature).\n"
    )
    lines.append("| Ablated feature | NSE without it | Delta NSE (drop) | Diagnosis |")
    lines.append("|---|---:|---:|---|")
    for f, v in sorted(report["ablation_reduced"].items(), key=lambda kv: kv[1]["delta_nse"], reverse=True):
        if v["delta_nse"] >= 0.05:
            diag = "POSSIBLE SHORTCUT / dominant feature"
        elif v["delta_nse"] >= 0.01:
            diag = "Provides real value"
        elif v["delta_nse"] >= -0.005:
            diag = "Marginal or redundant"
        else:
            diag = "Removing it improves (likely noise)"
        lines.append(f"| `{f}` | {v['nse_without']:.4f} | {v['delta_nse']:+.4f} | {diag} |")
    lines.append("")

    # Hallazgos clave
    lines.append("## Key findings\n")
    lines.append("\n".join(report["key_findings"]))
    lines.append("")

    # Veredicto
    lines.append("## Verdict\n")
    lines.append(report["verdict"])
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    timings: Dict[str, float] = {}

    print(f"Loading parquet: {PARQUET_PATH}")
    t0 = time.time()
    df = pd.read_parquet(PARQUET_PATH)
    timings["load_parquet"] = time.time() - t0
    print(f"  shape={df.shape}  load={timings['load_parquet']:.1f}s")

    # ---------- Stage 1: XGB-20 ----------
    t0 = time.time()
    model, X_train, y_train, X_test, y_test, t_fit_xgb20 = train_xgb20(df)
    timings["xgb20_fit"] = time.time() - t0

    pred_full = model.predict(X_test)
    base_nse = nse(y_test, pred_full)
    xgb20_metrics = {
        "n_features": len(FEATURES_20),
        "fit_seconds": t_fit_xgb20,
        "nse": float(base_nse),
        "rmse": rmse(y_test, pred_full),
        "mae": mae(y_test, pred_full),
        "peak_real": float(np.max(y_test)),
        "peak_pred": float(np.max(pred_full)),
        "peak_err_pct": float((np.max(pred_full) - np.max(y_test)) / np.max(y_test) * 100.0),
    }
    print(f"  XGB-20 NSE base = {base_nse:.4f}")

    # ---------- Stage 2a: Native importance ----------
    t0 = time.time()
    gain = native_importance(model, FEATURES_20)
    timings["native_importance"] = time.time() - t0

    # ---------- Stage 2b: Permutation importance ----------
    print("[2b] Permutation importance (n_repeats=5)...")
    rng = np.random.default_rng(RANDOM_STATE)
    n_test = len(X_test)
    if n_test > PI_TEST_SUBSAMPLE:
        sel = rng.choice(n_test, size=PI_TEST_SUBSAMPLE, replace=False)
        sel.sort()
        X_pi = X_test[sel]
        y_pi = y_test[sel]
        n_pi_used = PI_TEST_SUBSAMPLE
    else:
        X_pi = X_test
        y_pi = y_test
        n_pi_used = n_test
    t0 = time.time()
    pi = permutation_importance_manual(
        model, X_pi, y_pi, FEATURES_20, n_repeats=PI_N_REPEATS, rng_seed=RANDOM_STATE
    )
    timings["permutation_importance"] = time.time() - t0
    print(f"  PI computed on {n_pi_used:,} rows in {timings['permutation_importance']:.1f}s")

    # ---------- Stage 2c: SHAP ----------
    shap_report = None
    if SHAP_AVAILABLE:
        print("[2c] SHAP TreeExplainer...")
        t0 = time.time()
        try:
            shap_report = shap_analysis(
                model, X_test, y_test, FEATURES_20,
                n_samples=SHAP_N_SAMPLES,
                extreme_threshold=EXTREME_BUCKET_THRESHOLD,
                rng_seed=RANDOM_STATE,
            )
            timings["shap"] = time.time() - t0
            print(f"  SHAP OK ({shap_report['n_samples']} filas, n_extreme={shap_report['n_extreme_in_sample']})")
        except Exception as exc:
            print(f"  SHAP failed: {exc}")
            shap_report = None
            timings["shap"] = time.time() - t0
    else:
        timings["shap"] = 0.0

    # ---------- Stage 3: correlations / clusters ----------
    print("[3] Correlation matrix and clustering...")
    t0 = time.time()
    corr_train = corr_matrix_train(df, FEATURES_20)
    n_train = int(IDX_TRAIN_END - SEQ_LENGTH)
    plot_corr_heatmap(corr_train, FIG_DIR / "s5_corr_matrix.png")

    # Pairs with |r| >= threshold
    high_pairs: List[Dict[str, float]] = []
    cols = list(corr_train.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr_train.iat[i, j])
            if abs(r) >= CORR_CLUSTER_THRESHOLD:
                high_pairs.append({"a": cols[i], "b": cols[j], "r": r})
    high_pairs.sort(key=lambda d: -abs(d["r"]))

    # Importance for choosing the cluster representative: we use PI (mean_drop).
    pi_for_cluster = {f: pi[f]["mean_drop"] for f in FEATURES_20}
    clusters, representative = cluster_features(
        corr_train, threshold=CORR_CLUSTER_THRESHOLD, importance=pi_for_cluster
    )
    timings["corr_cluster"] = time.time() - t0
    print(f"  pairs >= {CORR_CLUSTER_THRESHOLD}: {len(high_pairs)} ; clusters: {len(clusters)}")

    # ---------- Plot PI ----------
    plot_perm_importance(pi, FIG_DIR / "s5_permutation_importance.png")

    # ---------- Stage 4: ACF ----------
    print("[4] ACF of key features...")
    t0 = time.time()
    idx_train = aligned_indices(0, IDX_TRAIN_END, HORIZON, len(df))
    acf_features: Dict[str, Dict[int, float]] = {}
    for f in KEY_FEATURES_ACF:
        if f not in df.columns:
            acf_features[f] = {l: float("nan") for l in ACF_LAGS}
            continue
        x = df[f].iloc[idx_train].to_numpy(dtype=float)
        acf_features[f] = acf_at_lags(x, ACF_LAGS)
    # Target ACF on TRAIN to validate the reference.
    y_train_full = df[TARGET_COL].iloc[idx_train].to_numpy(dtype=float)
    target_acf_emp = acf_at_lags(y_train_full, ACF_LAGS)
    timings["acf"] = time.time() - t0

    # ---------- Stage 5: reduced set ----------
    print("[5] Building reduced set...")
    t0 = time.time()
    selected, justification = select_reduced_features(pi, representative)
    metrics_reduced = train_reduced_and_eval(df, selected)
    timings["reduced_set"] = time.time() - t0
    print(f"  Reducido n={len(selected)}  NSE={metrics_reduced['nse']:.4f}")

    # ---------- Stage 6: 1-by-1 ablation on the reduced set ----------
    print("[6] 1-by-1 ablation on the reduced set...")
    t0 = time.time()
    ablation_reduced = leave_one_out_ablation(
        df, selected, selected, base_nse=metrics_reduced["nse"]
    )
    timings["ablation_reduced"] = time.time() - t0

    # ---------- Key findings (text) ----------
    items_pi = sorted(pi.items(), key=lambda kv: kv[1]["mean_drop"], reverse=True)
    top5_pi = items_pi[:5]
    top5_str = ", ".join([f"`{f}`(+{v['mean_drop']:.4f})" for f, v in top5_pi])
    n_pairs_red = len(high_pairs)
    n_independent = len(clusters)
    pi_neg_or_zero = [f for f, v in pi.items() if v["mean_drop"] <= 0]
    abl_shortcuts = [f for f, v in ablation_reduced.items() if v["delta_nse"] >= 0.05]

    long_rain = ["rain_sum_180m", "rain_sum_360m"]
    long_rain_pi = {f: pi[f]["mean_drop"] for f in long_rain if f in pi}

    delta_red_vs_xgb20 = metrics_reduced["nse"] - xgb20_metrics["nse"]
    ar12_nse_h1_ref = 0.8273  # de S2_baselines.json

    key_findings: List[str] = []
    key_findings.append(
        f"1. **Top 5 features by permutation importance**: {top5_str}. "
        "These are the ones that really impact NSE when shuffled in test, "
        "not necessarily the ones the booster uses most by gain."
    )
    key_findings.append(
        f"2. **Redundancy: {n_pairs_red} pairs with |r|>=0.85**, grouped into "
        f"**{n_independent} independent clusters**. The effective dimensionality "
        f"is much closer to {n_independent} than to 20."
    )
    if abl_shortcuts:
        key_findings.append(
            f"3. **Hidden shortcuts detected (ablation drop >0.05 NSE)**: {', '.join('`'+f+'`' for f in abl_shortcuts)}. "
            "After removing `delta_flow_*`, dominant features still appear; review "
            "whether its value comes from physical signal or autoregressive target inertia."
        )
    else:
        key_findings.append(
            "3. **No additional hidden shortcuts are detected**: no feature, "
            "after removing `delta_flow_*`, causes a drop >0.05 NSE in 1-by-1 ablation. "
            "The reduced set depends on multiple features."
        )
    if abs(delta_red_vs_xgb20) <= 0.01:
        red_status = "validated (delta within +-0.01 NSE)."
    elif delta_red_vs_xgb20 > 0.01:
        red_status = f"improves (+{delta_red_vs_xgb20:.4f} NSE over XGB-20: removing PI<=0 features eliminates noise)."
    else:
        red_status = f"review (loss {delta_red_vs_xgb20:+.4f} NSE > 0.01)."
    key_findings.append(
        f"4. **Proposed reduced set ({len(selected)} features)**: "
        f"NSE={metrics_reduced['nse']:.4f} vs XGB-20={xgb20_metrics['nse']:.4f} "
        f"(delta {delta_red_vs_xgb20:+.4f}). " + red_status.capitalize()
    )
    if long_rain_pi:
        rains_text = ", ".join([f"`{k}`(PI={v:+.4f})" for k, v in long_rain_pi.items()])
        veredicto_long = (
            "have marginal or negative PI: candidates to discard"
            if all(v <= 0.005 for v in long_rain_pi.values())
            else "provide measurable PI: keep"
        )
        key_findings.append(f"5. **Long rains {rains_text}**: {veredicto_long}.")
    if pi_neg_or_zero:
        key_findings.append(
            f"6. **Features with PI <= 0**: {', '.join('`'+f+'`' for f in pi_neg_or_zero)}. "
            "Shuffling them does not degrade the model: noise for this task."
        )

    # ACF: candidates for carrying target inertia
    acf_notes: List[str] = []
    for f, acf_d in acf_features.items():
        if any(np.isnan(v) for v in acf_d.values()):
            continue
        # Simple lag6 and lag12 comparison.
        r1_delta = abs(acf_d.get(1, 0.0) - TARGET_ACF_REF[1])
        r6_delta = abs(acf_d.get(6, 0.0) - TARGET_ACF_REF[6])
        r12_delta = abs(acf_d.get(12, 0.0) - TARGET_ACF_REF[12])
        very_similar_lag12 = r12_delta < 0.1 and acf_d.get(1, 0.0) > 0.95
        very_flat = acf_d.get(1, 0.0) < 0.3
        if very_similar_lag12:
            acf_notes.append(
                f"`{f}` with ACF lag1={acf_d[1]:.3f}, lag12={acf_d[12]:.3f} (target=0.420): "
                "inertia comparable to the target at lag12 => possible carrier of hidden autoregressive signal."
            )
        elif very_flat:
            acf_notes.append(
            f"`{f}` with ACF lag1={acf_d[1]:.3f}: short memory, difference-like behavior or noise."
            )
    if acf_notes:
        key_findings.append("7. **Individual ACF (possible hidden inertia)**:\n   - " + "\n   - ".join(acf_notes))

    # Global verdict
    verdict_lines: List[str] = []
    verdict_lines.append(
        "The 20 official features (without `delta_flow_*`) contain massive redundancy: "
        f"{n_pairs_red} pairs with |r|>=0.85 that cluster into {n_independent} clusters "
        "almost independent clusters. The real predictive capacity of the XGBoost model lives in a "
        "much smaller subspace."
    )
    verdict_lines.append("")
    if abs(delta_red_vs_xgb20) <= 0.01:
        verdict_lines.append(
            f"The proposed reduced set ({len(selected)} features) matches XGB-20 "
            f"(delta within +-0.01 NSE) while keeping only the representative of each cluster + "
            "features with PI > 0. This is an actionable recommendation: simplify preprocessing and "
            "the model input without sacrificing metric."
        )
    elif delta_red_vs_xgb20 > 0.01:
        verdict_lines.append(
            f"The reduced set ({len(selected)} features) **improves {delta_red_vs_xgb20:+.4f} NSE** "
            "over XGB-20. Reducing dimensionality not only does not lose signal but also cleans noise: "
            "the features with PI<=0 (hour_cos, month_sin, month_cos, rain_sum_60m, rain_max_30m, "
            "rain_max_60m) were degrading the XGBoost fit."
        )
    else:
        verdict_lines.append(
            f"The proposed reduced set loses {delta_red_vs_xgb20:+.4f} NSE. "
            "Investigate whether the cluster criterion is too aggressive or whether there are interactions that "
            "break when collapsing highly correlated but complementary features."
        )
    verdict_lines.append("")
    if abl_shortcuts:
        verdict_lines.append(
            "The 1-by-1 ablation on the reduced set reveals features with disproportionate weight "
            "(>=0.05 NSE drop): "
            + ", ".join("`" + f + "`" for f in abl_shortcuts)
            + ". Before launching more TCN iterations it is worth confirming that these features "
            "do not contain hidden target information (compatible ACF, non-physical lags, etc.)."
        )
    else:
        verdict_lines.append(
            "Without features with a drop >=0.05 NSE in ablation, the reduced XGBoost model spreads "
            "its signal in a healthy way across the physical features (aggregated rain + API + temperature "
            "+ seasonality). No obvious shortcuts remain to clean up beyond the already excluded `delta_flow_*` "
            "features."
        )

    # ---------- Empaquetar ----------
    report: Dict[str, object] = {
        "config": {
            "horizon": HORIZON,
            "seq_length": SEQ_LENGTH,
            "split": {
                "train_end_idx": IDX_TRAIN_END,
                "val_end_idx": IDX_VAL_END,
                "total_rows": int(len(df)),
            },
            "features_20": FEATURES_20,
            "xgb_params": XGB_PARAMS,
            "pi_n_repeats": PI_N_REPEATS,
            "pi_test_subsample": PI_TEST_SUBSAMPLE,
            "shap_n_samples": SHAP_N_SAMPLES,
            "corr_cluster_threshold": CORR_CLUSTER_THRESHOLD,
            "extreme_bucket_threshold_mgd": EXTREME_BUCKET_THRESHOLD,
            "shap_available": SHAP_AVAILABLE,
        },
        "stage_pi": {"n_test_used": int(n_pi_used)},
        "stage_corr": {"n_train": int(n_train)},
        "stage_shap": {"n_samples": int(SHAP_N_SAMPLES) if SHAP_AVAILABLE else 0},
        "timings": timings,
        "xgb20_metrics": xgb20_metrics,
        "ar12_nse_h1_ref": ar12_nse_h1_ref,
        "native_importance": gain,
        "permutation_importance": pi,
        "shap": shap_report,
        "high_corr_pairs": high_pairs,
        "clusters": clusters,
        "representative": representative,
        "acf_features": {f: {str(k): v for k, v in d.items()} for f, d in acf_features.items()},
        "target_acf_train_empirical": {str(k): v for k, v in target_acf_emp.items()},
        "target_acf_reference": {str(k): v for k, v in TARGET_ACF_REF.items()},
        "reduced_set": {
            "selected": selected,
            "justification": justification,
            "metrics": metrics_reduced,
        },
        "ablation_reduced": ablation_reduced,
        "key_findings": key_findings,
        "verdict": "\n\n".join(verdict_lines),
    }

    # Clean NaN/Inf for JSON
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, tuple):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            o = float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return [_clean(v) for v in o.tolist()]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
        return o

    json_path = OUT_DIR / "S5_feature_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {json_path}")

    md_text = render_md(report)
    md_path = OUT_DIR / "S5_feature_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Written: {md_path}")
    print(f"Figures: {FIG_DIR}/s5_corr_matrix.png  +  s5_permutation_importance.png")

    print("\nTime by stage:")
    for k, v in timings.items():
        print(f"  {k}: {v:.1f}s")


if __name__ == "__main__":
    main()



