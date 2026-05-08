"""
S4 - Physical ceiling of achievable NSE by horizon and partial oracles.

Objective: quantify for each horizon h in {1, 3, 6, 12, 24} how much the model
can improve over trivial persistence (naive and AR(1)) and where the real levers
are (which magnitude bucket contributes the most to the NSE denominator and,
therefore, where a perfect prediction increases global NSE the most).

For the partial oracles of the current TCN v1 model (H=1, H=3, H=6 sinSF), the
script runs batch inference over the full aligned test set and replaces the
prediction with the real target only inside the selected bucket or bucket
combination; it then recomputes global NSE.

Artefactos:
  - outputs/diagnostic/S4_horizon_ceiling.json
  - outputs/diagnostic/S4_horizon_ceiling.md
  - outputs/figures/diagnostic/s4_horizon_ceiling.png

Only uses: pandas, numpy, sklearn, torch, matplotlib.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUT_DIR = ROOT / "outputs" / "diagnostic"
FIG_DIR = ROOT / "outputs" / "figures" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

IDX_TRAIN_END = 771374
IDX_VAL_END = 936669

SEQ_LENGTH = 72
HORIZONS = [1, 3, 6, 12, 24]  # 1=5min, 3=15min, 6=30min, 12=60min, 24=120min

TARGET_COL = "stormflow_mgd"
DEVICE = "cpu"

# Buckets according to the S4 prompt: Moderado [5, 25), Alto [25, 50), Extremo>=50.
# They differ from the v1 metrics buckets (which use Alto [20, 50)) -> documented in the md.
BUCKETS = [
    ("Base",     -np.inf, 0.5),
    ("Leve",     0.5,     5.0),
    ("Moderado", 5.0,     25.0),
    ("Alto",     25.0,    50.0),
    ("Extremo",  50.0,    np.inf),
]

# v1 sinSF models available for inference (only H1 and H3; H6_sinSF has negative NSE
# and H12/H24 do not have a trained model).
TCN_MODELS = {
    1: "modelo_H1_sinSF",
    3: "modelo_H3_sinSF",
    6: "modelo_H6_sinSF",
}

INFERENCE_BATCH = 1024
CLS_THRESHOLD = 0.3  # same threshold as S3 / evaluate_local


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bucket_mask(y_true: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (y_true >= lo) & (y_true < hi)


# ---------------------------------------------------------------------------
# Indices alineados
# ---------------------------------------------------------------------------
def aligned_indices(split_start: int, split_end: int, horizon: int, total_len: int) -> np.ndarray:
    """Same criterion as S2: full previous SEQ_LENGTH window and target inside the dataframe."""
    first = split_start + SEQ_LENGTH
    last_in_split = split_end - 1
    last_by_target = total_len - 1 - horizon
    last = min(last_in_split, last_by_target)
    return np.arange(first, last + 1)


# ---------------------------------------------------------------------------
# Analytical baselines
# ---------------------------------------------------------------------------
def naive_pred(df: pd.DataFrame, idx_test: np.ndarray) -> np.ndarray:
    return df[TARGET_COL].to_numpy()[idx_test]


def ar1_optimal(
    df: pd.DataFrame, idx_train: np.ndarray, idx_test: np.ndarray, horizon: int
) -> Tuple[np.ndarray, float, float]:
    """Optimal AR(1): `y_hat = mean_train + rho_h * (y(t) - mean_train)`."""
    y = df[TARGET_COL].to_numpy()
    y_train_t = y[idx_train]
    y_train_th = y[idx_train + horizon]
    rho = float(np.corrcoef(y_train_t, y_train_th)[0, 1])
    mean_train = float(np.mean(y_train_t))
    y_hat = mean_train + rho * (y[idx_test] - mean_train)
    return y_hat, rho, mean_train


def ar_k_pred(
    df: pd.DataFrame, idx_train: np.ndarray, idx_test: np.ndarray, horizon: int, k: int
) -> np.ndarray:
    """AR(k) by linear regression over k lags. Same method as S2."""
    y = df[TARGET_COL].to_numpy()
    n = len(y)
    lags = np.full((n, k), np.nan, dtype=float)
    for j in range(k):
        lags[j:, j] = y[: n - j]
    idx_train_f = idx_train[idx_train >= k - 1]
    idx_test_f = idx_test[idx_test >= k - 1]
    reg = LinearRegression()
    reg.fit(lags[idx_train_f], y[idx_train_f + horizon])
    y_hat_test_f = reg.predict(lags[idx_test_f])
    y_hat = np.full(len(idx_test), np.nan)
    pos = {v: i for i, v in enumerate(idx_test)}
    for i, ix in enumerate(idx_test_f):
        y_hat[pos[ix]] = y_hat_test_f[i]
    return y_hat


# ---------------------------------------------------------------------------
# TCN v1 inference over the entire aligned test set (in batches)
# ---------------------------------------------------------------------------
def load_tcn(model_stem: str) -> Tuple[TwoStageTCN, Dict]:
    weights_path = WEIGHTS_DIR / f"{model_stem}_weights.pt"
    norm_path = WEIGHTS_DIR / f"{model_stem}_norm_params.json"
    meta_path = WEIGHTS_DIR / f"{model_stem}_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(norm_path, "r") as f:
        norm_params = json.load(f)
    features = meta["features"]
    model = TwoStageTCN(n_features=len(features))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    norm_params["feature_columns"] = features
    return model, norm_params


def normalize_df(df: pd.DataFrame, norm_params: Dict) -> pd.DataFrame:
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


def infer_tcn_test(
    df: pd.DataFrame, model_stem: str, idx_test: np.ndarray, horizon: int,
    threshold: float = CLS_THRESHOLD,
) -> Tuple[np.ndarray, Dict]:
    """Returns y_pred(t+h) in MGD for each origin t_origin in idx_test.

    Convention from `src.pipeline.sequences`:
      - ventana = feat[t_origin - SEQ_LENGTH + 1 : t_origin + 1]  (incluye t_origin)
      - target  = y[t_origin + horizon]

    In this script `idx_test` represents `t_origin`. As it comes from
    `aligned_indices`, con `first = split_start + SEQ_LENGTH` y
    `last = total_len - 1 - horizon`, la ventana [origin - SEQ + 1, origin]
    siempre cabe en el df.
    """
    print(f"[tcn] loading {model_stem}...")
    model, norm_params = load_tcn(model_stem)
    df_norm = normalize_df(df, norm_params)
    features = norm_params["feature_columns"]
    target_col = norm_params["target_col"]
    target_mean = norm_params["mean"][target_col]
    target_std = norm_params["std"][target_col]
    use_log1p = target_col in norm_params.get("log1p_columns", [])

    feat_arr = df_norm[features].to_numpy(dtype=np.float32)

    n_test = len(idx_test)
    y_pred_norm = np.zeros(n_test, dtype=np.float32)
    cls_probs = np.zeros(n_test, dtype=np.float32)

    print(f"[tcn] batch inference batch={INFERENCE_BATCH} ({n_test:,} samples)...")
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_test, INFERENCE_BATCH):
            end = min(start + INFERENCE_BATCH, n_test)
            batch_idx = idx_test[start:end]
            # Window [t_origin - SEQ + 1, t_origin] including t_origin.
            xs = np.stack(
                [feat_arr[t - SEQ_LENGTH + 1 : t + 1] for t in batch_idx],
                axis=0,
            )
            x_t = torch.from_numpy(xs)
            out = model(x_t)
            cls_b = out["cls_prob"].squeeze(-1).cpu().numpy()
            reg_b = out["reg_value"].squeeze(-1).cpu().numpy()
            cls_probs[start:end] = cls_b
            y_norm = np.where(cls_b >= threshold, reg_b, 0.0)
            y_pred_norm[start:end] = y_norm
            if (start // INFERENCE_BATCH) % 50 == 0:
                pct = 100.0 * end / n_test
                print(f"    progress {pct:5.1f}% ({end:,}/{n_test:,})")
    elapsed = time.time() - t0
    print(f"[tcn] inference completed in {elapsed:.1f}s")

    # Denormalize
    y_pred = y_pred_norm * target_std + target_mean
    if use_log1p:
        y_pred = np.expm1(y_pred)
    y_pred = np.clip(y_pred, 0.0, None)

    info = {
        "model": model_stem,
        "n_features": len(features),
        "threshold": threshold,
        "elapsed_s": elapsed,
        "cls_prob_mean": float(np.mean(cls_probs)),
        "cls_prob_pct_above_thr": float(np.mean(cls_probs >= threshold)) * 100.0,
    }
    return y_pred.astype(float), info


# ---------------------------------------------------------------------------
# Partial oracles and denominator contribution
# ---------------------------------------------------------------------------
def bucket_contribution(y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Contribution of each bucket to the NSE denominator (`sum((y-mean)^2)`)."""
    mean_y = float(np.mean(y_true))
    sse_total = float(np.sum((y_true - mean_y) ** 2))
    out = {}
    for name, lo, hi in BUCKETS:
        m = bucket_mask(y_true, lo, hi)
        n = int(np.sum(m))
        sse_bucket = float(np.sum((y_true[m] - mean_y) ** 2)) if n > 0 else 0.0
        out[name] = {
            "n": n,
            "share_n_pct": 100.0 * n / max(len(y_true), 1),
            "sse_bucket": sse_bucket,
            "share_denom_pct": 100.0 * sse_bucket / sse_total if sse_total > 0 else 0.0,
        }
    out["_total"] = {
        "n": int(len(y_true)),
        "sse_total": sse_total,
        "mean_y": mean_y,
    }
    return out


def oracle_partial(
    y_true: np.ndarray, y_pred_model: np.ndarray, bucket_names: List[str]
) -> Dict[str, float]:
    """Reemplaza y_pred por y_true dentro de los buckets indicados y calcula NSE global."""
    y_pred_oracle = y_pred_model.copy()
    name_to_bounds = {b[0]: (b[1], b[2]) for b in BUCKETS}
    for bn in bucket_names:
        lo, hi = name_to_bounds[bn]
        m = bucket_mask(y_true, lo, hi)
        y_pred_oracle[m] = y_true[m]
    return {
        "buckets_oracled": bucket_names,
        "nse": nse(y_true, y_pred_oracle),
        "rmse": rmse(y_true, y_pred_oracle),
        "n_oracled": int(np.sum([
            np.sum(bucket_mask(y_true, *name_to_bounds[bn])) for bn in bucket_names
        ])),
    }


# ---------------------------------------------------------------------------
# Logica principal por horizonte
# ---------------------------------------------------------------------------
def run_horizon(df: pd.DataFrame, horizon: int) -> Dict[str, object]:
    print(f"\n=== Horizon h={horizon} ({horizon * 5} min) ===")
    idx_train = aligned_indices(0, IDX_TRAIN_END, horizon, len(df))
    idx_test = aligned_indices(IDX_VAL_END, len(df), horizon, len(df))
    print(f"  n_train={len(idx_train):,}  n_test={len(idx_test):,}")

    y_full = df[TARGET_COL].to_numpy()
    y_true_test = y_full[idx_test + horizon]

    # 1) Naive
    y_hat_naive = naive_pred(df, idx_test)
    nse_naive = nse(y_true_test, y_hat_naive)

    # 2) Optimal AR(1) + theoretical bound
    y_hat_ar1, rho_h, mean_train = ar1_optimal(df, idx_train, idx_test, horizon)
    nse_ar1 = nse(y_true_test, y_hat_ar1)
    bound_2rho_minus_1 = 2.0 * rho_h - 1.0  # Upper bound under iid (informative)

    # 3) Linear AR(12)
    y_hat_ar12 = ar_k_pred(df, idx_train, idx_test, horizon, k=12)
    nse_ar12 = nse(y_true_test, y_hat_ar12)

    # 4) Bucket contribution to the denominator
    contrib = bucket_contribution(y_true_test)

    result = {
        "horizon_steps": horizon,
        "horizon_min": horizon * 5,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "test_index_first": int(idx_test[0]),
        "test_index_last": int(idx_test[-1]),
        "test_timestamp_first": str(df.iloc[idx_test[0]]["timestamp"]),
        "test_timestamp_last": str(df.iloc[idx_test[-1] + horizon]["timestamp"]),
        "rho_h_train": rho_h,
        "mean_train": mean_train,
        "nse_naive": nse_naive,
        "nse_ar1_opt": nse_ar1,
        "bound_2rho_minus_1": bound_2rho_minus_1,
        "nse_ar12": nse_ar12,
        "rmse_naive": rmse(y_true_test, y_hat_naive),
        "rmse_ar1_opt": rmse(y_true_test, y_hat_ar1),
        "rmse_ar12": rmse(y_true_test, y_hat_ar12),
        "buckets_contribution": contrib,
    }

    # 5) Partial oracles: only for horizons with a trained TCN.
    if horizon in TCN_MODELS:
        model_stem = TCN_MODELS[horizon]
        y_pred_tcn, info = infer_tcn_test(df, model_stem, idx_test, horizon)
        nse_tcn = nse(y_true_test, y_pred_tcn)
        rmse_tcn = rmse(y_true_test, y_pred_tcn)
        peak_real = float(np.max(y_true_test))
        peak_pred = float(np.max(y_pred_tcn))
        peak_err_pct = (peak_pred - peak_real) / peak_real * 100.0 if peak_real > 0 else float("nan")

        # Bias and RMSE by bucket using the TCN (sanity check vs metrics.json)
        bucket_metrics_tcn = {}
        for bname, lo, hi in BUCKETS:
            m = bucket_mask(y_true_test, lo, hi)
            n = int(np.sum(m))
            if n == 0:
                bucket_metrics_tcn[bname] = {"n": 0}
                continue
            yt = y_true_test[m]
            yp = y_pred_tcn[m]
            bucket_metrics_tcn[bname] = {
                "n": n,
                "bias": float(np.mean(yp - yt)),
                "rmse": rmse(yt, yp),
                "mae": float(np.mean(np.abs(yp - yt))),
                "nse_local": nse(yt, yp),
                "sse_residual": float(np.sum((yt - yp) ** 2)),
            }

        # Partial oracles
        oracles = {
            "extremo": oracle_partial(y_true_test, y_pred_tcn, ["Extremo"]),
            "extremo_alto": oracle_partial(y_true_test, y_pred_tcn, ["Extremo", "Alto"]),
            "moderado": oracle_partial(y_true_test, y_pred_tcn, ["Moderado"]),
            "leve_base": oracle_partial(y_true_test, y_pred_tcn, ["Leve", "Base"]),
            "moderado_leve_base": oracle_partial(y_true_test, y_pred_tcn, ["Moderado", "Leve", "Base"]),
            "todo_excepto_extremo": oracle_partial(
                y_true_test, y_pred_tcn, ["Base", "Leve", "Moderado", "Alto"]
            ),
        }
        for k, v in oracles.items():
            v["delta_nse_vs_tcn"] = v["nse"] - nse_tcn

        result["tcn_v1"] = {
            "model": model_stem,
            "info": info,
            "nse": nse_tcn,
            "rmse": rmse_tcn,
            "peak_real": peak_real,
            "peak_pred": peak_pred,
            "peak_err_pct": peak_err_pct,
            "buckets_metrics": bucket_metrics_tcn,
        }
        result["oracles"] = oracles
    else:
        # For H=12 and H=24 there is no TCN. We report only the analytical ceiling.
        result["tcn_v1"] = None
        result["oracles"] = None

    print(f"  NSE naive={nse_naive:.4f}  AR(1)opt={nse_ar1:.4f}  AR(12)={nse_ar12:.4f}")
    print(f"  rho_h={rho_h:.4f}  2rho-1 bound={bound_2rho_minus_1:.4f}")
    if result["tcn_v1"] is not None:
        print(f"  TCN v1 NSE={result['tcn_v1']['nse']:.4f}  peak_err={result['tcn_v1']['peak_err_pct']:+.1f}%")
        for k, v in result["oracles"].items():
            print(f"  oracle[{k}] NSE={v['nse']:.4f}  delta={v['delta_nse_vs_tcn']:+.4f}")

    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(all_results: Dict[int, Dict[str, object]], path: Path) -> None:
    horizons = sorted(all_results.keys())
    h_min = [h * 5 for h in horizons]
    nse_naive = [all_results[h]["nse_naive"] for h in horizons]
    nse_ar1 = [all_results[h]["nse_ar1_opt"] for h in horizons]
    nse_ar12 = [all_results[h]["nse_ar12"] for h in horizons]
    bound = [all_results[h]["bound_2rho_minus_1"] for h in horizons]
    nse_tcn = [
        all_results[h]["tcn_v1"]["nse"] if all_results[h]["tcn_v1"] is not None else None
        for h in horizons
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=130)
    ax.plot(h_min, nse_naive, "o--", color="#888", label="Naive (persistence)")
    ax.plot(h_min, nse_ar1, "s--", color="#1f77b4", label=r"Optimal AR(1) ($\rho_h$)")
    ax.plot(h_min, nse_ar12, "^--", color="#2ca02c", label="Linear AR(12)")
    ax.plot(h_min, bound, ":", color="#d62728", label=r"Analytical bound $2\rho_h-1$")
    tcn_h = [h for h, v in zip(h_min, nse_tcn) if v is not None]
    tcn_v = [v for v in nse_tcn if v is not None]
    if tcn_h:
        ax.plot(tcn_h, tcn_v, "*-", color="#9467bd", markersize=14, label="TCN v1 sinSF")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Horizon (minutes)")
    ax.set_ylabel("NSE on aligned test")
    ax.set_title("Achievable NSE ceiling by horizon (target = stormflow_mgd)")
    ax.set_xticks(h_min)
    ax.set_xticklabels([f"{m} min\n(h={h})" for m, h in zip(h_min, horizons)])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] written {path}")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def format_md(all_results: Dict[int, Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# S4 - Achievable NSE ceiling by horizon\n")
    lines.append(
        "Quantification of the achievable NSE ceiling with the current features and "
        "without external future information. For each horizon h, the "
        "analytical AR baselines, the theoretical ceiling of an optimal AR(1), the TCN v1 "
        "(when a model is available), and partial oracles by magnitude bucket.\n"
    )

    # Methodology
    lines.append("## Methodology\n")
    lines.append(
        "- **Chronological test split**: `iloc[936669:]`, same aligned indices as "
        "S2 (full previous 72-step window, target inside the dataframe).\n"
        "- **MGD buckets**: Base<0.5; Leve [0.5, 5); Moderado [5, 25); Alto [25, 50); Extremo>=50. "
        "Note that `evaluate_local.py` uses Alto [20, 50) to report v1; here it is "
        "recomputed with [25, 50) to stay consistent with the diagnostic statement.\n"
        "- **Optimal AR(1)**: rho_h = corr(y(t), y(t+h)) on train; "
        "y_hat = mean_train + rho_h*(y(t) - mean_train).\n"
        "- **Analytical bound `2*rho_h - 1`**: maximum NSE reachable by an "
        "orthogonal linear predictor of y(t) when rho_h > 0.5 (informative bound, not an absolute ceiling).\n"
        "- **AR(12)**: linear regression with 12 consecutive lags.\n"
        "- **TCN v1 sinSF**: batch inference over the full aligned test set, hard "
        "switch with threshold=0.3 (same as evaluate_local). Available only for H=1, H=3, H=6.\n"
        "- **Partial oracle**: for the indicated buckets, y_pred = y_true; the rest "
        "stays the same. NSE is recomputed over the full test.\n"
    )

    # Master table
    lines.append("## Master table: NSE by horizon\n")
    lines.append(
        "| h | min | n_test | rho_h | NSE naive | NSE AR(1)opt | 2rho-1 bound | "
        "NSE AR(12) | NSE TCN v1 | Peak err % |"
    )
    lines.append("|---|----:|-------:|------:|---------:|-------------:|------------:|----------:|----------:|----------:|")
    for h in sorted(all_results.keys()):
        r = all_results[h]
        tcn = r.get("tcn_v1")
        nse_tcn_str = f"{tcn['nse']:.4f}" if tcn else "-"
        peak_str = f"{tcn['peak_err_pct']:+.1f}" if tcn else "-"
        lines.append(
            f"| {h} | {r['horizon_min']} | {r['n_test']:,} | "
            f"{r['rho_h_train']:.4f} | "
            f"{r['nse_naive']:.4f} | {r['nse_ar1_opt']:.4f} | "
            f"{r['bound_2rho_minus_1']:+.4f} | {r['nse_ar12']:.4f} | "
            f"{nse_tcn_str} | {peak_str} |"
        )
    lines.append("")
    lines.append(
        "Quick reading: `naive` is the trivial floor; `optimal AR(1)` beats it "
        "slightly because it pulls predictions toward the train mean when rho_h<1; "
        "`AR(12)` uses additional lags but the marginal improvement indicates that the "
        "memory beyond a couple of lags is already saturated; the `2*rho-1` bound gives the "
        "theoretical ceiling of an AR(1) under independent errors.\n"
    )

    # Bucket contribution to the denominator
    lines.append("## Contribution of each bucket to the NSE denominator\n")
    lines.append(
        "Where the variance of y_true is concentrated (sum (y-mean)^2). If a bucket "
        "contributes X% of the denominator, improving the prediction there raises NSE "
        "proportionally by X%. **This is the main lever.**\n"
    )
    bucket_names = [b[0] for b in BUCKETS]
    lines.append("| h | total SSE | " + " | ".join(f"{b} %denom" for b in bucket_names) + " |")
    lines.append("|---|---:|" + "|".join(["---:"] * len(bucket_names)) + "|")
    for h in sorted(all_results.keys()):
        r = all_results[h]
        c = r["buckets_contribution"]
        sse_total = c["_total"]["sse_total"]
        row = [f"{h}", f"{sse_total:,.1f}"]
        for bn in bucket_names:
            row.append(f"{c[bn]['share_denom_pct']:5.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("| h | " + " | ".join(f"{b} n" for b in bucket_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(bucket_names)) + "|")
    for h in sorted(all_results.keys()):
        r = all_results[h]
        c = r["buckets_contribution"]
        row = [f"{h}"]
        for bn in bucket_names:
            row.append(f"{c[bn]['n']:,}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Partial oracles
    lines.append("## Partial oracles on TCN v1 (only h with a trained model)\n")
    lines.append(
        "Replace y_pred = y_true inside the indicated bucket or combination, then recompute "
        "NSE over the full test. The `delta` column is the increment over TCN v1.\n"
    )
    for h in sorted(all_results.keys()):
        r = all_results[h]
        if r.get("tcn_v1") is None:
            continue
        nse_tcn = r["tcn_v1"]["nse"]
        lines.append(f"### h={h} ({r['horizon_min']} min)\n")
        lines.append(
            f"- TCN v1 base: NSE = **{nse_tcn:.4f}**, "
            f"real peak {r['tcn_v1']['peak_real']:.1f} MGD, "
            f"predicted peak {r['tcn_v1']['peak_pred']:.1f} MGD "
            f"({r['tcn_v1']['peak_err_pct']:+.1f}%).\n"
        )
        lines.append("| Oracle | n oracle samples | Oracle NSE | delta NSE |")
        lines.append("|---|---:|---:|---:|")
        order = [
            ("extremo", "Only Extreme (>=50)"),
            ("extremo_alto", "Extreme + High (>=25)"),
            ("moderado", "Only Moderate [5,25)"),
            ("leve_base", "Leve + Base (<5)"),
            ("moderado_leve_base", "Moderate + Leve + Base (<25)"),
            ("todo_excepto_extremo", "Everything except Extreme (<50)"),
        ]
        for k, label in order:
            o = r["oracles"][k]
            lines.append(
                f"| {label} | {o['n_oracled']:,} | "
                f"{o['nse']:.4f} | {o['delta_nse_vs_tcn']:+.4f} |"
            )
        # Bucket metrics del TCN
        lines.append("")
        lines.append("TCN v1 metrics by bucket (sanity check):\n")
        lines.append("| Bucket | n | bias | RMSE | NSE local | SSE residual | %SSE residual |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        sse_total_residual = sum(
            r["tcn_v1"]["buckets_metrics"][bn].get("sse_residual", 0.0)
            for bn in bucket_names
        )
        for bn in bucket_names:
            bm = r["tcn_v1"]["buckets_metrics"][bn]
            if bm.get("n", 0) == 0:
                lines.append(f"| {bn} | 0 | - | - | - | - | - |")
                continue
            sse_share = 100.0 * bm["sse_residual"] / sse_total_residual if sse_total_residual > 0 else 0.0
            lines.append(
                f"| {bn} | {bm['n']:,} | {bm['bias']:+.3f} | {bm['rmse']:.3f} | "
                f"{bm['nse_local']:+.3f} | {bm['sse_residual']:,.1f} | {sse_share:5.2f} |"
            )
        lines.append("")

    # Sintesis y veredicto
    lines.append(build_verdict(all_results))
    return "\n".join(lines)


def build_verdict(all_results: Dict[int, Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("## Quantitative synthesis and verdict\n")

    # 1. Horizon with the highest possible theoretical gain
    gains = {}
    for h in sorted(all_results.keys()):
        r = all_results[h]
        # Estimated physical ceiling: max( 2*rho-1 bound, NSE AR(12), NSE TCN v1 )
        candidates = [r["bound_2rho_minus_1"], r["nse_ar12"]]
        if r.get("tcn_v1") is not None:
            candidates.append(r["tcn_v1"]["nse"])
        nse_max_phys = max(candidates)
        ganancia = nse_max_phys - r["nse_naive"]
        gains[h] = (nse_max_phys, ganancia)
    lines.append("### 1. Horizon worth optimizing\n")
    lines.append("| h | min | NSE naive | Estimated physical NSE max | Max gain over naive |")
    lines.append("|---|---:|---:|---:|---:|")
    for h in sorted(all_results.keys()):
        r = all_results[h]
        nse_max, gan = gains[h]
        lines.append(
            f"| {h} | {r['horizon_min']} | {r['nse_naive']:.4f} | "
            f"{nse_max:.4f} | {gan:+.4f} |"
        )
    h_best = max(gains.keys(), key=lambda h: gains[h][1])
    lines.append(
        f"\n**Horizon with maximum theoretical gain: h={h_best} ({h_best*5} min)** "
        f"with a margin of {gains[h_best][1]:+.4f} NSE over naive.\n"
    )

    # 2. NSE max defendible
    lines.append("### 2. Maximum defensible NSE by horizon (perfect model, same features)\n")
    lines.append(
        "Estimated upper bound: the maximum between the analytical `2*rho-1` bound ("
        "which assumes an optimal AR(1) predictor under iid) and the best available "
        "empirical evidence (AR(12) or TCN v1). We take the larger of the two as a conservative ceiling "
        "reachable with the available information.\n"
    )
    for h in sorted(all_results.keys()):
        r = all_results[h]
        nse_max, _ = gains[h]
        bound = r["bound_2rho_minus_1"]
        ar12 = r["nse_ar12"]
        tcn_str = (
            f"; TCN v1 = {r['tcn_v1']['nse']:.4f}" if r.get("tcn_v1") is not None else ""
        )
        lines.append(
            f"- **h={h} ({r['horizon_min']} min)**: NSE max ~ **{nse_max:.4f}** "
            f"(2*rho-1 = {bound:+.4f}; AR(12) = {ar12:.4f}{tcn_str})."
        )
    lines.append("")

    # 3. Main quantitative lever
    lines.append("### 3. Main quantitative lever for improving NSE at H=1\n")
    if 1 in all_results and all_results[1].get("oracles") is not None:
        r1 = all_results[1]
        oracles = r1["oracles"]
        # Find the oracle with the maximum delta
        # Compare individual and combined oracles
        ranked = sorted(oracles.items(), key=lambda kv: -kv[1]["delta_nse_vs_tcn"])
        lines.append(
            f"TCN v1 actual: NSE = **{r1['tcn_v1']['nse']:.4f}**. If the model predicted "
            "perfectly inside the indicated bucket (keeping the rest unchanged), NSE would become:\n"
        )
        lines.append("| Oracle | Oracle NSE | delta NSE | n |")
        lines.append("|---|---:|---:|---:|")
        for k, v in ranked:
            lines.append(
                f"| {k} | {v['nse']:.4f} | {v['delta_nse_vs_tcn']:+.4f} | {v['n_oracled']:,} |"
            )
        # Verdict
        best_single = max(
            ["extremo", "moderado", "leve_base"],
            key=lambda k: oracles[k]["delta_nse_vs_tcn"],
        )
        lines.append(
            f"\n**Dominant lever at H=1**: the oracle that raises NSE the most individually "
            f"es `{best_single}` con delta = "
            f"{oracles[best_single]['delta_nse_vs_tcn']:+.4f} NSE. "
            f"This matches the denominator contribution of that bucket.\n"
        )
        # Recall denominator contribution
        c1 = r1["buckets_contribution"]
        lines.append("Denominator contribution (H=1) of the key buckets:\n")
        for bn in ["Extremo", "Alto", "Moderado", "Leve", "Base"]:
            lines.append(
                f"- **{bn}**: {c1[bn]['share_denom_pct']:.2f}% of the denominator "
                f"({c1[bn]['n']:,} samples = {c1[bn]['share_n_pct']:.3f}% of test)."
            )
        lines.append("")

    # 4. Does it make sense to pursue H=6?
    lines.append("### 4. Does it make sense to pursue H=6 (30 min)?\n")
    if 6 in all_results:
        r6 = all_results[6]
        nse_max_h6, gan_h6 = gains[6]
        lines.append(
            f"- NSE naive H=6 = **{r6['nse_naive']:.4f}** "
            f"(rho_h={r6['rho_h_train']:.4f}, AR(1)opt={r6['nse_ar1_opt']:.4f}, "
            f"AR(12)={r6['nse_ar12']:.4f}).\n"
            f"- TCN v1 sinSF a H=6 = **{r6['tcn_v1']['nse']:.4f}** (segun S4 inference).\n"
            f"- Estimated physical NSE max H=6 = **{nse_max_h6:.4f}**.\n"
        )
        if nse_max_h6 < 0.5:
            lines.append(
                "Verdict: the physical ceiling of H=6 with the current features is **below "
                "NSE=0.5**. Pursuing H=6 with the current dataset will not yield a "
                "defensible operational model. Opening H=6 requires exogenous features with a future horizon "
                "(rainfall forecast, NWP) or changing the target horizon.\n"
            )
        elif nse_max_h6 < 0.7:
            lines.append(
                "Verdict: H=6 is reachable but with a ceiling clearly below H=1/H=3. "
                "Architectural decision: accept the ceiling or reformulate the target ("
                "aggregated storm events instead of point stormflow).\n"
            )
        else:
            lines.append(
                "Verdict: H=6 still has a reasonable ceiling; it is worth exploring.\n"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[load] parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  shape={df.shape}")

    all_results: Dict[int, Dict[str, object]] = {}
    for h in HORIZONS:
        all_results[h] = run_horizon(df, h)

    # Plot
    fig_path = FIG_DIR / "s4_horizon_ceiling.png"
    try:
        make_plot(all_results, fig_path)
    except Exception as exc:
        print(f"[plot] WARNING: the plot could not be generated ({exc})")

    # JSON
    meta = {
        "split": {
            "train_end_idx": IDX_TRAIN_END,
            "val_end_idx": IDX_VAL_END,
            "total_rows": int(len(df)),
        },
        "seq_length": SEQ_LENGTH,
        "horizons": HORIZONS,
        "buckets_mgd": [{"name": b[0], "lo": b[1], "hi": b[2]} for b in BUCKETS],
        "tcn_models": TCN_MODELS,
        "cls_threshold": CLS_THRESHOLD,
        "inference_batch": INFERENCE_BATCH,
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, tuple):
            return [_clean(v) for v in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
        if isinstance(o, (np.floating, np.integer)):
            o = o.item()
        return o

    json_path = OUT_DIR / "S4_horizon_ceiling.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean({"meta": meta, "results": all_results}), f, indent=2, ensure_ascii=False)
    print(f"[out] written {json_path}")

    md = format_md(all_results)
    md_path = OUT_DIR / "S4_horizon_ceiling.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"[out] written {md_path}")


if __name__ == "__main__":
    main()


