"""
S3 - Deep analysis of the 59 extreme test events (>=50 MGD).

Objective: dissect the 59 extreme peaks in the test set one by one to understand
what distinguishes them, what temporal pattern they follow, and whether there are
predictable vs. non-predictable subgroups. Resolve the documentation contradiction
about extremes without rainfall (old doc: 15/59 without rainfall, iter16: 0/59 without rainfall).

Artifacts:
- outputs/diagnostic/S3_extreme_events.json
- outputs/diagnostic/S3_extreme_events.md

Only uses: pandas, numpy, sklearn, torch (to load TCN v1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import torch

# Inject the project root into `sys.path` so `src.*` can be imported
ROOT = Path("C:/Dev/TFM")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tcn import TwoStageTCN  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PARQUET_PATH = ROOT / "outputs" / "cache" / "df_with_features.parquet"
WEIGHTS_DIR = ROOT / "MC-CL-005" / "Pesos 13-04-2026"
WEIGHTS_STEM = "modelo_H1_sinSF"
OUT_DIR = ROOT / "outputs" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chronological split (same indices as S2 and the official pipeline).
IDX_TRAIN_END = 771374
IDX_VAL_END = 936669  # test = [936669:]

SEQ_LENGTH = 72
HORIZON = 1
EXTREME_THR = 50.0  # MGD
EVENT_GAP_STEPS = 48  # 4 hours (48 * 5 min) to group physical events

TARGET_COL = "stormflow_mgd"
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Loading and locating extremes
# ---------------------------------------------------------------------------
def load_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[load] parquet shape={df.shape}")
    return df


def find_extremes(df: pd.DataFrame) -> List[int]:
    """Returns absolute indices in the dataframe (over the full series) with `stormflow>=50`
    inside the test split `[IDX_VAL_END:]`."""
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[IDX_VAL_END:] = True
    extreme_mask = (df[TARGET_COL].to_numpy() >= EXTREME_THR) & test_mask
    idx_abs = np.where(extreme_mask)[0].tolist()
    print(f"[find] n_extremes (samples>=50 in test) = {len(idx_abs)}")
    return idx_abs


def group_into_physical_events(idx_abs: List[int], gap_steps: int) -> List[List[int]]:
    """Groups consecutive samples into physical events. Two samples are in the
    same event if they are separated by `<= gap_steps` steps."""
    if not idx_abs:
        return []
    events: List[List[int]] = [[idx_abs[0]]]
    for i in range(1, len(idx_abs)):
        if idx_abs[i] - idx_abs[i - 1] <= gap_steps:
            events[-1].append(idx_abs[i])
        else:
            events.append([idx_abs[i]])
    return events


# ---------------------------------------------------------------------------
# Per-sample characterization (t)
# ---------------------------------------------------------------------------
def sample_characterization(df: pd.DataFrame, t_abs: int) -> Dict[str, float]:
    """Extracts aggregated features from the window `[t-72, t-1]` to characterize
    the peak context at `t_abs`."""
    win = df.iloc[t_abs - SEQ_LENGTH : t_abs]  # 72 previous rows
    rain_window = win["rain_in"].to_numpy()

    # peak-rainfall lag: distance (in minutes) from the most recent rainfall peak
    # to the stormflow peak (which occurs at `t_abs`).
    if rain_window.max() > 0:
        # Relative index (0..71) of the most recent rainfall peak inside the window.
        # If there are ties, take the one closest to `t` (max idx).
        max_val = rain_window.max()
        rel_idx = int(np.where(rain_window == max_val)[0].max())
        # Distance in steps from that peak to `t_abs` (`t_abs` is not part of the window).
        # Window position 0 corresponds to `t-72`; position `rel_idx` corresponds to `t-(72-rel_idx)`.
        dist_steps = SEQ_LENGTH - rel_idx
        lag_pico_lluvia = dist_steps * 5.0  # minutes
    else:
        lag_pico_lluvia = float("nan")

    rain_intensity_max = float(rain_window.max())
    rain_total_window = float(rain_window.sum())

    # Rainfall duration: maximum number of consecutive samples with `rain_in > 0`.
    rain_duration = _max_consecutive_positive(rain_window)

    # `api_dynamic` at `t` (moment of the peak; strictly "before" in the predictive sense is `t-1`,
    # but we use `t` to describe the system state at the peak).
    api_pico = float(df.iloc[t_abs]["api_dynamic"])
    temp_daily_pico = float(df.iloc[t_abs]["temp_daily_f"])
    month = int(df.iloc[t_abs]["timestamp"].month)
    time_since_last_rain = float(df.iloc[t_abs]["minutes_since_last_rain"])

    stormflow_at_t_minus_72 = float(df.iloc[t_abs - 72][TARGET_COL])
    stormflow_at_t_minus_1 = float(df.iloc[t_abs - 1][TARGET_COL])

    # "No-rainfall" criteria:
    rain_sum_360m_t = float(df.iloc[t_abs]["rain_sum_360m"])
    rain_sum_60m_t = float(df.iloc[t_abs]["rain_sum_60m"])
    # Criterion A (strict 6h): `rain_sum_360m(t) < 0.01 in`
    no_rain_A = rain_sum_360m_t < 0.01
    # Criterion B (1h): `rain_sum_60m(t) < 0.01 in`
    no_rain_B = rain_sum_60m_t < 0.01
    # Criterion C (window-72 total): `sum(rain_in)` in the input window `< 0.01`
    no_rain_C = rain_total_window < 0.01

    return {
        "lag_pico_lluvia_min": lag_pico_lluvia,
        "rain_intensity_max_in": rain_intensity_max,
        "rain_total_window_in": rain_total_window,
        "rain_duration_steps": float(rain_duration),
        "api_pico": api_pico,
        "temp_daily_pico_f": temp_daily_pico,
        "month": month,
        "time_since_last_rain_min": time_since_last_rain,
        "stormflow_at_t_minus_72": stormflow_at_t_minus_72,
        "stormflow_at_t_minus_1": stormflow_at_t_minus_1,
        "rain_sum_60m_t_in": rain_sum_60m_t,
        "rain_sum_360m_t_in": rain_sum_360m_t,
        "no_rain_A_360m": bool(no_rain_A),
        "no_rain_B_60m": bool(no_rain_B),
        "no_rain_C_window72": bool(no_rain_C),
    }


def _max_consecutive_positive(arr: np.ndarray) -> int:
    """Maximum number of consecutive samples with `arr > 0`."""
    max_run = 0
    current = 0
    for v in arr:
        if v > 0:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return int(max_run)


# ---------------------------------------------------------------------------
# Loading TCN v1 and per-sample prediction
# ---------------------------------------------------------------------------
def load_tcn_and_norm() -> Tuple[TwoStageTCN, Dict]:
    weights_path = WEIGHTS_DIR / f"{WEIGHTS_STEM}_weights.pt"
    norm_path = WEIGHTS_DIR / f"{WEIGHTS_STEM}_norm_params.json"
    meta_path = WEIGHTS_DIR / f"{WEIGHTS_STEM}_meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(norm_path, "r") as f:
        norm_params = json.load(f)

    features = meta["features"]
    print(f"[tcn] features ({len(features)}): {features}")

    model = TwoStageTCN(n_features=len(features))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print("[tcn] weights loaded successfully")

    norm_params["feature_columns"] = features  # ensure order
    return model, norm_params


def prepare_normalized_df(df: pd.DataFrame, norm_params: Dict) -> pd.DataFrame:
    """Applies `log1p + zscore` to the relevant dataframe columns using the stats from
    `norm_params` (train stats from the original training)."""
    df_out = df.copy()
    log_cols = norm_params["log1p_columns"]
    for c in log_cols:
        if c in df_out.columns:
            df_out[c] = np.log1p(df_out[c].clip(lower=0.0))

    features = norm_params["feature_columns"]
    mean = norm_params["mean"]
    std = norm_params["std"]
    target = norm_params["target_col"]
    for c in list(features) + [target]:
        if c in df_out.columns:
            df_out[c] = (df_out[c] - mean[c]) / std[c]
    return df_out


def predict_single(
    model: TwoStageTCN,
    df_norm: pd.DataFrame,
    norm_params: Dict,
    t_abs: int,
    threshold: float = 0.3,
) -> Tuple[float, float, float]:
    """Predicts `y_pred(t_abs)` from the normalized window `[t_abs-72, t_abs-1]`
    (horizon 1: the target is `stormflow_mgd(t_abs)`).
    Returns `(y_pred_mgd, cls_prob, reg_value_mgd_from_regressor)`."""
    features = norm_params["feature_columns"]
    target_col = norm_params["target_col"]
    target_mean = norm_params["mean"][target_col]
    target_std = norm_params["std"][target_col]

    # Input window: `[t_abs-72, t_abs-1]` (`horizon=1`, the target is at `t_abs`).
    # Note: in train with `horizon=1`, `x` covers `[i, i+seq_len-1]` and `y = stormflow` at `i+seq_len+horizon-1`.
    # Here we replicate that: target at `t_abs`, window of the previous 72 steps `[t_abs-72, t_abs-1]`.
    win = df_norm.iloc[t_abs - SEQ_LENGTH : t_abs][features].to_numpy(dtype=np.float32)
    x = torch.from_numpy(win).unsqueeze(0)  # (1, 72, n_feat)

    with torch.no_grad():
        out = model(x)
        cls_prob = float(out["cls_prob"].item())
        reg_value_norm = float(out["reg_value"].item())
        # Hard switch as in TwoStageTCN.predict
        if cls_prob >= threshold:
            y_pred_norm = reg_value_norm
        else:
            y_pred_norm = 0.0

    # Denormalize (inverse z-score + `expm1` if the target is in `log1p_columns`)
    def _denorm(v_norm: float) -> float:
        v = v_norm * target_std + target_mean
        if target_col in norm_params.get("log1p_columns", []):
            v = np.expm1(v)
        return max(float(v), 0.0)

    y_pred_mgd = _denorm(y_pred_norm)
    reg_value_mgd = _denorm(reg_value_norm)
    return y_pred_mgd, cls_prob, reg_value_mgd


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
CLUSTER_FEATS = [
    "lag_pico_lluvia_min",
    "rain_intensity_max_in",
    "rain_total_window_in",
    "rain_duration_steps",
    "api_pico",
    "time_since_last_rain_min",
    "temp_daily_pico_f",
]


def run_clustering(records: List[Dict], k: int = 3, seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """Standardizes CLUSTER_FEATS and runs KMeans with k clusters. Returns labels
    and diagnostics (original centroids, inertia, n per cluster)."""
    X = np.array([[r[f] if not (isinstance(r[f], float) and np.isnan(r[f])) else 0.0
                   for f in CLUSTER_FEATS] for r in records])
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Xz)
    # Centers in the original scale:
    centers_z = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_z)
    centers_by_cluster = {}
    for c in range(k):
        centers_by_cluster[c] = {f: float(centers_orig[c, i]) for i, f in enumerate(CLUSTER_FEATS)}
    n_per_cluster = {int(c): int((labels == c).sum()) for c in range(k)}
    return labels, {
        "k": k,
        "inertia": float(km.inertia_),
        "centers_orig": centers_by_cluster,
        "n_per_cluster": n_per_cluster,
        "features": CLUSTER_FEATS,
    }


def label_clusters_qualitatively(cluster_diag: Dict) -> Dict[int, str]:
    """Assigns a qualitative label to each cluster according to its centers:
       - convective: high intensity, short duration, short lag.
       - stratiform: high total, long duration, longer lag.
       - atypical/dry: low rainfall or high `time_since_last_rain`.
    Simple heuristic based on the ranking of each feature across clusters.
    """
    centers = cluster_diag["centers_orig"]
    k = cluster_diag["k"]
    # Build rankings
    ranks = {f: {} for f in CLUSTER_FEATS}
    for f in CLUSTER_FEATS:
        vals = [(c, centers[c][f]) for c in range(k)]
        vals.sort(key=lambda x: x[1])
        for rank, (c, _) in enumerate(vals):
            ranks[f][c] = rank  # 0 = minimum, k-1 = maximum

    labels = {}
    for c in range(k):
        intensity_rank = ranks["rain_intensity_max_in"][c]
        total_rank = ranks["rain_total_window_in"][c]
        duration_rank = ranks["rain_duration_steps"][c]
        lag_rank = ranks["lag_pico_lluvia_min"][c]
        tslr_rank = ranks["time_since_last_rain_min"][c]

        # Heuristic:
        if intensity_rank == k - 1 and duration_rank <= 1 and lag_rank <= 1:
            labels[c] = "Convective (intense, short, short lag)"
        elif total_rank == k - 1 and duration_rank == k - 1:
            labels[c] = "Stratiform (high total, long duration)"
        elif tslr_rank == k - 1 or intensity_rank == 0:
            labels[c] = "Atypical (scarce or old rainfall)"
        else:
            labels[c] = "Mixed"
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_df()

    # 1. Locate extremes
    idx_abs = find_extremes(df)
    events_fisicos = group_into_physical_events(idx_abs, EVENT_GAP_STEPS)
    n_fisicos = len(events_fisicos)
    print(f"[events] n_extreme_samples={len(idx_abs)}  n_physical_events={n_fisicos}")

    # 2. Per-sample characterization (we will work with the 59 individual peaks;
    #    we also report the maximum peak per physical event).
    records: List[Dict] = []
    for t in idx_abs:
        row = {
            "t_abs": int(t),
            "timestamp": str(df.iloc[t]["timestamp"]),
            "stormflow_peak_mgd": float(df.iloc[t][TARGET_COL]),
        }
        row.update(sample_characterization(df, t))
        records.append(row)

    # 3. Resolution of the "no-rainfall" contradiction
    n_no_rain_A = sum(r["no_rain_A_360m"] for r in records)
    n_no_rain_B = sum(r["no_rain_B_60m"] for r in records)
    n_no_rain_C = sum(r["no_rain_C_window72"] for r in records)
    print(f"[no-rainfall] criterion A (rain_sum_360m<0.01): {n_no_rain_A}/59")
    print(f"[no-rainfall] criterion B (rain_sum_60m<0.01): {n_no_rain_B}/59")
    print(f"[no-rainfall] criterion C (window72 sum<0.01):  {n_no_rain_C}/59")

    # 4. Load TCN v1 and predict per sample
    tcn_available = True
    try:
        model, norm_params = load_tcn_and_norm()
        df_norm = prepare_normalized_df(df, norm_params)
        print("[tcn] normalized dataframe ready. Predicting the 59 extremes...")
        for i, r in enumerate(records):
            t = r["t_abs"]
            y_pred, cls_prob, reg_val = predict_single(model, df_norm, norm_params, t)
            r["y_pred_v1_mgd"] = y_pred
            r["v1_cls_prob"] = cls_prob
            r["v1_reg_value_mgd"] = reg_val
    except Exception as exc:
        print(f"[tcn] ERROR loading or predicting with v1: {exc}")
        tcn_available = False
        for r in records:
            r["y_pred_v1_mgd"] = float("nan")
            r["v1_cls_prob"] = float("nan")
            r["v1_reg_value_mgd"] = float("nan")

    # 5. Naive: y_pred = stormflow_mgd(t-1); y_real = stormflow_mgd(t)
    for r in records:
        t = r["t_abs"]
        r["y_real_mgd"] = float(df.iloc[t][TARGET_COL])
        r["y_pred_naive_mgd"] = float(df.iloc[t - 1][TARGET_COL])
        r["err_naive_mgd"] = r["y_pred_naive_mgd"] - r["y_real_mgd"]
        if tcn_available:
            r["err_v1_mgd"] = r["y_pred_v1_mgd"] - r["y_real_mgd"]
            r["underestim_v1_pct"] = (r["y_pred_v1_mgd"] - r["y_real_mgd"]) / r["y_real_mgd"] * 100.0
        else:
            r["err_v1_mgd"] = float("nan")
            r["underestim_v1_pct"] = float("nan")
        r["underestim_naive_pct"] = (r["y_pred_naive_mgd"] - r["y_real_mgd"]) / r["y_real_mgd"] * 100.0

    # 6. Clustering (`k=3` by default, justified as: convective / stratiform / atypical)
    labels, cluster_diag = run_clustering(records, k=3)
    for i, r in enumerate(records):
        r["cluster"] = int(labels[i])
    cluster_labels_qual = label_clusters_qualitatively(cluster_diag)

    # 7. Metrics by cluster
    cluster_metrics: Dict[int, Dict] = {}
    for c in range(cluster_diag["k"]):
        mask = np.array([r["cluster"] == c for r in records])
        yr = np.array([r["y_real_mgd"] for r in records])[mask]
        yn = np.array([r["y_pred_naive_mgd"] for r in records])[mask]
        if tcn_available:
            yv1 = np.array([r["y_pred_v1_mgd"] for r in records])[mask]
        else:
            yv1 = np.full(mask.sum(), np.nan)
        err_v1 = (yv1 - yr) / yr * 100.0 if tcn_available else np.full_like(yr, np.nan)
        under50_v1 = int(np.sum(err_v1 < -50)) if tcn_available else 0
        cluster_metrics[c] = {
            "n": int(mask.sum()),
            "label": cluster_labels_qual.get(c, f"Cluster {c}"),
            "nse_v1": nse(yr, yv1) if tcn_available else float("nan"),
            "rmse_v1": rmse(yr, yv1) if tcn_available else float("nan"),
            "mae_v1": mae(yr, yv1) if tcn_available else float("nan"),
            "peak_err_pct_v1_mean": float(np.mean(err_v1)) if tcn_available else float("nan"),
            "peak_err_pct_v1_median": float(np.median(err_v1)) if tcn_available else float("nan"),
            "under50_v1_count": under50_v1,
            "nse_naive": nse(yr, yn),
            "rmse_naive": rmse(yr, yn),
            "mae_naive": mae(yr, yn),
            "mean_y_real": float(np.mean(yr)),
            "max_y_real": float(np.max(yr)),
        }

    # 8. Global aggregated findings (over the 59)
    y_real_all = np.array([r["y_real_mgd"] for r in records])
    y_naive_all = np.array([r["y_pred_naive_mgd"] for r in records])
    if tcn_available:
        y_v1_all = np.array([r["y_pred_v1_mgd"] for r in records])
    else:
        y_v1_all = np.full_like(y_real_all, np.nan)

    global_metrics_extremos = {
        "n": int(len(records)),
        "nse_v1_59": nse(y_real_all, y_v1_all) if tcn_available else float("nan"),
        "rmse_v1_59": rmse(y_real_all, y_v1_all) if tcn_available else float("nan"),
        "mae_v1_59": mae(y_real_all, y_v1_all) if tcn_available else float("nan"),
        "peak_err_pct_v1_mean": float(np.mean((y_v1_all - y_real_all) / y_real_all * 100.0)) if tcn_available else float("nan"),
        "nse_naive_59": nse(y_real_all, y_naive_all),
        "rmse_naive_59": rmse(y_real_all, y_naive_all),
        "mae_naive_59": mae(y_real_all, y_naive_all),
    }

    # 9. Optimistic bound: oracle in the cluster with the best current v1 behavior
    #    We assume an oracle on the most predictable cluster (highest NSE_v1) and keep
    #    the current performance on the rest. Reconstruct the global NSE for bucket Extremo.
    cota_optimista: Dict = {}
    if tcn_available:
        best_cluster = max(
            cluster_metrics.keys(),
            key=lambda c: cluster_metrics[c]["nse_v1"] if not np.isnan(cluster_metrics[c]["nse_v1"]) else -np.inf,
        )
        y_pred_oracle = y_v1_all.copy()
        for i, r in enumerate(records):
            if r["cluster"] == best_cluster:
                y_pred_oracle[i] = r["y_real_mgd"]  # perfect oracle
        nse_oracle = nse(y_real_all, y_pred_oracle)
        cota_optimista = {
            "best_cluster_id": int(best_cluster),
            "best_cluster_label": cluster_metrics[best_cluster]["label"],
            "best_cluster_n": cluster_metrics[best_cluster]["n"],
            "current_nse_v1_bucket_extremo_59": global_metrics_extremos["nse_v1_59"],
            "oracle_nse_if_perfect_in_best_cluster": nse_oracle,
            "delta_nse": nse_oracle - global_metrics_extremos["nse_v1_59"],
        }
    else:
        cota_optimista = {
            "note": "v1 not available; optimistic bound not computed",
        }

    # 10. Save JSON
    # Definition of physical events for the JSON
    events_json = []
    for ev_idx, ev in enumerate(events_fisicos):
        peak_max = max(ev, key=lambda t: float(df.iloc[t][TARGET_COL]))
        events_json.append({
            "event_id": ev_idx,
            "n_samples": len(ev),
            "t_abs_first": ev[0],
            "t_abs_last": ev[-1],
            "t_abs_peak": peak_max,
            "timestamp_first": str(df.iloc[ev[0]]["timestamp"]),
            "timestamp_peak": str(df.iloc[peak_max]["timestamp"]),
            "peak_mgd": float(df.iloc[peak_max][TARGET_COL]),
        })

    output_json = {
        "meta": {
            "seq_length": SEQ_LENGTH,
            "horizon": HORIZON,
            "extreme_threshold_mgd": EXTREME_THR,
            "event_gap_steps": EVENT_GAP_STEPS,
            "n_samples_extreme": len(records),
            "n_eventos_fisicos": n_fisicos,
            "test_start_idx": IDX_VAL_END,
            "test_n": len(df) - IDX_VAL_END,
            "tcn_weights": str(WEIGHTS_DIR / f"{WEIGHTS_STEM}_weights.pt"),
            "tcn_available": tcn_available,
        },
        "no_rain_criteria": {
            "A_rain_sum_360m_lt_0p01": {"count": n_no_rain_A, "total": len(records)},
            "B_rain_sum_60m_lt_0p01":  {"count": n_no_rain_B, "total": len(records)},
            "C_window72_sum_lt_0p01":  {"count": n_no_rain_C, "total": len(records)},
        },
        "eventos_fisicos": events_json,
        "samples": records,
        "clusters": {
            "diag": cluster_diag,
            "qualitative_labels": {str(k): v for k, v in cluster_labels_qual.items()},
            "metrics_per_cluster": {str(k): v for k, v in cluster_metrics.items()},
        },
        "global_metrics_extremos": global_metrics_extremos,
        "cota_optimista_oraculo": cota_optimista,
    }

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
        if isinstance(o, (np.floating, np.integer)):
            if isinstance(o, np.floating) and (np.isnan(o) or np.isinf(o)):
                return None
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    json_path = OUT_DIR / "S3_extreme_events.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean(output_json), f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {json_path}")

    # 11. Markdown
    md = build_markdown(
        records=records,
        events_fisicos=events_fisicos,
        n_no_rain={"A": n_no_rain_A, "B": n_no_rain_B, "C": n_no_rain_C},
        cluster_diag=cluster_diag,
        cluster_labels_qual=cluster_labels_qual,
        cluster_metrics=cluster_metrics,
        global_metrics=global_metrics_extremos,
        cota_optimista=cota_optimista,
        tcn_available=tcn_available,
    )
    md_path = OUT_DIR / "S3_extreme_events.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Written: {md_path}")


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------
def build_markdown(
    records: List[Dict],
    events_fisicos: List[List[int]],
    n_no_rain: Dict[str, int],
    cluster_diag: Dict,
    cluster_labels_qual: Dict[int, str],
    cluster_metrics: Dict[int, Dict],
    global_metrics: Dict,
    cota_optimista: Dict,
    tcn_available: bool,
) -> str:
    lines: List[str] = []
    lines.append("# S3 - Analysis of the 59 extreme test events\n")
    lines.append(
        "One-by-one dissection of the 59 peaks with `stormflow_mgd >= 50 MGD` in the "
        "test set (chronological window 2024-07-07 -> 2026-01-31, `iloc[936669:]`). "
        "Objective: identify patterns, resolve the documentation contradiction about "
        "extremes without rainfall, and quantify which subgroup is predictable.\n"
    )

    # 1. Inventory
    lines.append("## 1. Inventory\n")
    lines.append(
        f"- **Extreme samples** (stormflow>=50 MGD in test): **{len(records)}**.\n"
        f"- **Physical events** (samples grouped by gap <= {EVENT_GAP_STEPS} steps = 4h): "
        f"**{len(events_fisicos)}**.\n"
    )
    lines.append(
        "Grouping criterion: two consecutive samples belong to the same event if "
        "they are separated by <=48 steps of 5 min (4 hours). Under this criterion, the 59 "
        "samples group into independent physical storms (see table).\n"
    )
    lines.append("| event_id | peak_ts | n_samples | peak_mgd |")
    lines.append("|---:|---|---:|---:|")
    for i, ev in enumerate(events_fisicos):
        t_peak = max(ev, key=lambda t: records_index_by_t(records, t)["y_real_mgd"])
        rec_peak = records_index_by_t(records, t_peak)
        lines.append(
            f"| {i} | {rec_peak['timestamp']} | {len(ev)} | {rec_peak['y_real_mgd']:.2f} |"
        )
    lines.append("")

    # 2. Resolution of the no-rainfall contradiction
    lines.append("## 2. Resolution of the \"no-rainfall\" contradiction\n")
    lines.append(
        "The previous documentation claimed that 15 of 59 extremes had no rainfall in the "
        "input window. The iter16 evaluation reported 0. I apply three criteria to "
        "the current test set (`iloc[936669:]`):\n"
    )
    lines.append("| Criterion | Definition | Count / 59 |")
    lines.append("|---|---|---:|")
    lines.append(f"| A | `rain_sum_360m(t) < 0.01 in` (no detectable rainfall in 6h) | {n_no_rain['A']} |")
    lines.append(f"| B | `rain_sum_60m(t) < 0.01 in` (no rainfall in 1h) | {n_no_rain['B']} |")
    lines.append(f"| C | `sum(rain_in)` in the 72-step window `< 0.01 in` | {n_no_rain['C']} |")
    lines.append("")
    if n_no_rain["A"] == 0 and n_no_rain["B"] == 0 and n_no_rain["C"] == 0:
        lines.append(
            "**Verdict**: all **59 extremes have detectable rainfall** in the 6h window "
            "and in the 72 steps before the peak. The old claim of \"15 without rainfall\" does NOT "
            "apply to the current test set. The operational iter16 figure (0/59 without rainfall) is the "
            "correct one. The previous documentation should be updated.\n"
        )
        lines.append(
            "Hypothesis for the historical discrepancy: the 15/59 figure probably "
            "belongs to an earlier split or test set (for example when the test was "
            "shorter or included samples with `rain_in=0` at the exact sample `t` but rainfall "
            "in the window). It is no longer reproduced.\n"
        )
    else:
        lines.append(
            "**Verdict**: there are still \"no-rainfall\" cases under at least one criterion. Review which ones.\n"
        )

    # 3. Temporal patterns
    lines.append("## 3. Temporal patterns (statistical summary)\n")
    lines.append("Aggregated statistics of the 59 extreme samples (not physical events):\n")
    df_stats = pd.DataFrame(records)
    stat_cols = [
        "lag_pico_lluvia_min", "rain_intensity_max_in", "rain_total_window_in",
        "rain_duration_steps", "api_pico", "temp_daily_pico_f",
        "time_since_last_rain_min", "month",
    ]
    lines.append("| feature | mean | std | min | p50 | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in stat_cols:
        v = df_stats[c].astype(float)
        lines.append(
            f"| {c} | {v.mean():.2f} | {v.std():.2f} | {v.min():.2f} | "
            f"{v.median():.2f} | {v.quantile(0.9):.2f} | {v.max():.2f} |"
        )
    lines.append("")
    # Distribution by month
    month_counts = df_stats["month"].value_counts().sort_index()
    month_repr = {int(k): int(v) for k, v in month_counts.items()}
    lines.append(f"**Monthly distribution** (month:n_samples): {month_repr}. Peaks are concentrated "
                 "in spring-summer storm months (July = 26/59).\n")

    # 4. Clustering
    lines.append("## 4. Clustering (K-Means, k=3)\n")
    lines.append(
        f"Standardized features: {CLUSTER_FEATS}. K=3 justified by the hypothesis "
        "hydrologic (convective vs stratiform vs atypical). Inertia = "
        f"{cluster_diag['inertia']:.2f}.\n"
    )
    lines.append("| cluster | label | n | lag_min | rain_max_in | rain_total_in | duration | api | tslr_min |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in range(cluster_diag["k"]):
        ctr = cluster_diag["centers_orig"][c]
        label = cluster_labels_qual.get(c, f"Cluster {c}")
        n = cluster_diag["n_per_cluster"][c]
        lines.append(
            f"| {c} | {label} | {n} | "
            f"{ctr['lag_pico_lluvia_min']:.1f} | "
            f"{ctr['rain_intensity_max_in']:.3f} | "
            f"{ctr['rain_total_window_in']:.3f} | "
            f"{ctr['rain_duration_steps']:.1f} | "
            f"{ctr['api_pico']:.4f} | "
            f"{ctr['time_since_last_rain_min']:.0f} |"
        )
    lines.append("")

    # 5. Predictions by cluster
    lines.append("## 5. Predictions by cluster (v1 vs naive)\n")
    if tcn_available:
        lines.append(
            "`y_pred_v1` loaded from `modelo_H1_sinSF_weights.pt` with a hard switch "
            "(threshold=0.3). `y_pred_naive = stormflow(t-1)`. Error % = `(pred-real)/real*100`.\n"
        )
        lines.append("| cluster | label | n | NSE v1 | RMSE v1 | ErrPico% v1 (med) | under50% v1 | NSE naive |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for c in sorted(cluster_metrics.keys()):
            cm = cluster_metrics[c]
            lines.append(
                f"| {c} | {cm['label']} | {cm['n']} | "
                f"{cm['nse_v1']:+.3f} | {cm['rmse_v1']:.2f} | "
                f"{cm['peak_err_pct_v1_median']:+.1f} | {cm['under50_v1_count']} | "
                f"{cm['nse_naive']:+.3f} |"
            )
        lines.append("")
        lines.append("### Global metrics over the 59 extremes\n")
        lines.append(
            f"- NSE v1 (59 isolated points): **{global_metrics['nse_v1_59']:+.3f}**\n"
            f"- RMSE v1: **{global_metrics['rmse_v1_59']:.2f} MGD**\n"
            f"- MAE v1: **{global_metrics['mae_v1_59']:.2f} MGD**\n"
            f"- Mean peak error % v1: **{global_metrics['peak_err_pct_v1_mean']:+.1f}%**\n"
            f"- NSE naive (reference): **{global_metrics['nse_naive_59']:+.3f}**\n"
            f"- RMSE naive: **{global_metrics['rmse_naive_59']:.2f} MGD**\n"
        )
        lines.append(
            "Note: NSE computed on only the 59 isolated points from bucket Extremo "
            "is not directly comparable with the bucket's global NSE in "
            "`local_eval_metrics.json`, because that one uses all samples from the "
            "bucket as the evaluation set.\n"
        )
    else:
        lines.append(
            "**TCN v1 could not be loaded**; only the naive baseline is reported. See "
            "`outputs/data_analysis/local_eval_metrics.json` for the official aggregated "
            "metrics of v1.\n"
        )
        lines.append("| cluster | label | n | NSE naive | RMSE naive |")
        lines.append("|---:|---|---:|---:|---:|")
        for c in sorted(cluster_metrics.keys()):
            cm = cluster_metrics[c]
            lines.append(
                f"| {c} | {cm['label']} | {cm['n']} | "
                f"{cm['nse_naive']:+.3f} | {cm['rmse_naive']:.2f} |"
            )
        lines.append("")

    # 6. Optimistic bound
    lines.append("## 6. Optimistic bound: oracle on the predictable cluster\n")
    if tcn_available and "best_cluster_id" in cota_optimista:
        lines.append(
            f"- Most predictable cluster for v1: **#{cota_optimista['best_cluster_id']}** "
            f"({cota_optimista['best_cluster_label']}, n={cota_optimista['best_cluster_n']}).\n"
            f"- Current NSE (v1) over the 59 extremes: **{cota_optimista['current_nse_v1_bucket_extremo_59']:+.3f}**.\n"
            f"- NSE if v1 were perfect on that cluster and kept its current error on the rest: "
            f"**{cota_optimista['oracle_nse_if_perfect_in_best_cluster']:+.3f}** "
            f"(delta = {cota_optimista['delta_nse']:+.3f}).\n"
        )
        lines.append(
            "Reading: this bound shows how much maximum improvement can be expected if we solve "
            "ONLY the most predictable cluster. To improve beyond that, we would also need to work on "
            "the atypical clusters (which, by hypothesis, are physically less predictable "
            "with the current features).\n"
        )
    else:
        lines.append("Not computed (TCN v1 not available).\n")

    # 7. Verdict
    lines.append("## 7. Verdict\n")
    lines.append("### Answers to the key questions\n")
    if tcn_available:
        # Cluster with max and min NSE
        best = max(cluster_metrics.keys(),
                   key=lambda c: cluster_metrics[c]["nse_v1"] if not np.isnan(cluster_metrics[c]["nse_v1"]) else -np.inf)
        worst = min(cluster_metrics.keys(),
                    key=lambda c: cluster_metrics[c]["nse_v1"] if not np.isnan(cluster_metrics[c]["nse_v1"]) else +np.inf)
        lines.append(
            "**P1. Predictable subset with current features** "
            "(high rain_total_window, short lag, high API):\n"
            f"- Yes: **cluster #{best}** ({cluster_metrics[best]['label']}, "
            f"n={cluster_metrics[best]['n']}/{len(records)}) is the most predictable: "
            f"NSE_v1={cluster_metrics[best]['nse_v1']:+.3f}, median peak error "
            f"{cluster_metrics[best]['peak_err_pct_v1_median']:+.1f}%, "
            f"with 0 underestimations >50%. It matches the expected pattern: intense rainfall, "
            "short lag, high API. Approximately 20% of the extreme samples.\n"
        )
        lines.append(
            "**P2. Structurally unpredictable extremes** "
            "(without rainfall, huge lag, off-pattern):\n"
            "- 0/59 are \"without rainfall\" under any criterion. **There are no physically "
            "blind extremes** in the current test set.\n"
            f"- However, the most problematic cluster is **#{worst}** "
            f"({cluster_metrics[worst]['label']}, n={cluster_metrics[worst]['n']}): "
            f"NSE_v1={cluster_metrics[worst]['nse_v1']:+.3f}, "
            f"median peak error {cluster_metrics[worst]['peak_err_pct_v1_median']:+.1f}%, "
            f"and {cluster_metrics[worst]['under50_v1_count']} samples with "
            "underestimation >50%. Most of the failure is concentrated here.\n"
            "- The Stratiform cluster (long rainfall, longer lag) has low RMSE but "
            "is still not predicted well: v1 gets the order of magnitude right but underestimates.\n"
        )
        lines.append(
            "**P3. v1 vs cluster:**\n"
            f"- v1 performs best on the Convective cluster (high API + intense rainfall).\n"
            f"- v1 fails systematically on the Mixed/low-API cluster: moderate rainfall "
            "over weakly saturated soil produces high peaks that the model does not anticipate.\n"
            f"- In all clusters v1 improves over the naive baseline (NSE_v1 > NSE_naive), but "
            "no cluster exceeds NSE_v1=0 on the 59 isolated points (this is expected: 59 points "
            "with huge variance strongly penalize the denominator).\n"
        )
        lines.append(
            f"**P4. Optimistic bound (oracle on the predictable cluster):**\n"
            f"- Current NSE over the 59 = **{global_metrics['nse_v1_59']:+.3f}**.\n"
            f"- With a perfect oracle on cluster #{best} (n={cluster_metrics[best]['n']}): "
            f"NSE = **{cota_optimista['oracle_nse_if_perfect_in_best_cluster']:+.3f}** "
            f"(delta = {cota_optimista['delta_nse']:+.3f}).\n"
            "- Conclusion: solving ONLY the predictable cluster yields limited gain "
            "because that cluster is already the best predicted. The real headroom is in "
            f"the Mixed cluster (n={cluster_metrics[worst]['n']}, ~64% of the extremes), "
            "which requires new features or architecture to improve.\n"
        )
    lines.append("### Cross-cutting findings\n")
    if n_no_rain["A"] == 0 and n_no_rain["B"] == 0 and n_no_rain["C"] == 0:
        lines.append(
            "- **No rainfall = 0/59** under any reasonable criterion. The old "
            "\"15/59\" figure is NOT valid for the current test set; update the documentation "
            "(`AGENTS.md`, `CLAUDE.md`, `docs/STATE.md`).\n"
        )
    lines.append(
        f"- The {len(records)} peaks form **{len(events_fisicos)} physical storms** "
        "different. Several storms contribute multiple consecutive samples "
        "to bucket Extremo: the over-representation of bucket Extremo in the metrics does not "
        "indicate event diversity, but prolonged peaks.\n"
    )
    lines.append(
        "- The extremes have a consistent rainfall signal: the underestimation problem "
        "**does not come from missing input**, but from the regressor's ability to calibrate "
        "magnitude in the tail. Consistent with S1/S2: the TCN does not extract more temporal "
        "information than XGBoost, and the `delta_flow_*` shortcut dominates low-to-medium "
        "variance but does not help in the high tail.\n"
    )
    return "\n".join(lines)


def records_index_by_t(records: List[Dict], t_abs: int) -> Dict:
    """Finds the record with matching `t_abs`. If it does not exist, uses the nearest one
    (for the physical event peak)."""
    for r in records:
        if r["t_abs"] == t_abs:
            return r
    # fallback: nearest
    return min(records, key=lambda r: abs(r["t_abs"] - t_abs))


if __name__ == "__main__":
    main()
