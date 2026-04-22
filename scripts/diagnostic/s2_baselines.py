"""
S2 - Baselines rigurosos para contextualizar el TwoStageTCN v1.

Objetivo: entrenar y evaluar una bateria de baselines clasicos y de ML tabular
(persistencia, AR(k), regresion fisica, XGBoost, RandomForest) sobre el mismo
split cronologico 70/15/15 y los mismos indices de test que usaria la TCN con
seq_length=72 y horizon h en {1, 3, 6}.

Escribe dos artefactos:
  - outputs/diagnostic/S2_baselines.json
  - outputs/diagnostic/S2_baselines.md

Solo se usan: pandas, numpy, sklearn, xgboost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
ROOT = Path("C:/Dev/TFM")
PARQUET_PATH = ROOT / "outputs" / "cache" / "df_with_features.parquet"
OUT_DIR = ROOT / "outputs" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Split cronologico 70/15/15 (indices ya definidos por la pipeline oficial).
IDX_TRAIN_END = 771374
IDX_VAL_END = 936669

SEQ_LENGTH = 72  # mismo contexto temporal que la TCN
HORIZONS = [1, 3, 6]

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

PHYSICAL_FEATURES = ["rain_sum_60m", "api_dynamic"]

BUCKETS = [
    ("Base",     -np.inf, 0.5),
    ("Leve",     0.5,     5.0),
    ("Moderado", 5.0,     25.0),
    ("Alto",     25.0,    50.0),
    ("Extremo",  50.0,    np.inf),
]

# Hiperparametros fijos (sin tuning).
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
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42,
)
RF_TRAIN_SUBSAMPLE = 300_000  # limite para que quepa en RAM / tiempo razonable


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


def peak_err_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    peak_true = float(np.max(y_true))
    peak_pred = float(np.max(y_pred))
    if peak_true == 0:
        return float("nan")
    return (peak_pred - peak_true) / peak_true * 100.0


def bucket_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in BUCKETS:
        mask = (y_true >= lo) & (y_true < hi)
        n = int(np.sum(mask))
        if n == 0:
            out[name] = {"n": 0, "bias": float("nan"), "nse": float("nan"),
                         "rmse": float("nan"), "mae": float("nan")}
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        out[name] = {
            "n": n,
            "bias": float(np.mean(yp - yt)),
            "nse": float(nse(yt, yp)),
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
        }
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        "nse": float(nse(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "peak_real": float(np.max(y_true)),
        "peak_pred": float(np.max(y_pred)),
        "peak_err_pct": peak_err_pct(y_true, y_pred),
        "n": int(len(y_true)),
        "buckets": bucket_stats(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Construccion de datasets
# ---------------------------------------------------------------------------
def build_target(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """y_target(t) = stormflow_mgd(t + horizon). NaN en las ultimas horizon filas."""
    return df[TARGET_COL].shift(-horizon).to_numpy()


def aligned_indices(
    split_start: int, split_end: int, horizon: int, total_len: int
) -> np.ndarray:
    """Indices absolutos t tales que:
      - existe la ventana previa de SEQ_LENGTH pasos completa (t >= split_start + SEQ_LENGTH)
      - el origen t cae dentro del split [split_start, split_end)
      - existe el target y(t+h) dentro del dataframe completo (t+h <= total_len - 1)
    Replica la convencion de la TCN: el target puede caer fuera del split
    (en la frontera entre splits) mientras siga existiendo en el df.
    Conteo esperado test (split [936669, 1101964), N=1101964):
      H=1 -> 165223 ; H=3 -> 165221 ; H=6 -> 165218 (igual a local_eval_metrics)."""
    first = split_start + SEQ_LENGTH
    last_in_split = split_end - 1
    last_by_target = total_len - 1 - horizon
    last = min(last_in_split, last_by_target)
    return np.arange(first, last + 1)


def lag_matrix(series: np.ndarray, lags: int) -> np.ndarray:
    """Devuelve una matriz (n, lags) con columnas [y(t), y(t-1), ..., y(t-lags+1)].
    Las primeras lags-1 filas quedan con NaN."""
    n = len(series)
    out = np.full((n, lags), np.nan, dtype=float)
    for k in range(lags):
        out[k:, k] = series[: n - k]
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_naive(df: pd.DataFrame, idx_test: np.ndarray, horizon: int) -> np.ndarray:
    """y_hat(t+h) = y(t)."""
    return df[TARGET_COL].to_numpy()[idx_test]


def baseline_ar1_analytic(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """AR(1) analitico: rho por correlacion Pearson en train."""
    y_full = df[TARGET_COL].to_numpy()
    y_train_t = y_full[idx_train]
    y_train_th = y_full[idx_train + horizon]
    rho = float(np.corrcoef(y_train_t, y_train_th)[0, 1])
    mean_train = float(np.mean(y_train_t))

    y_test_t = y_full[idx_test]
    y_hat_const = rho * y_test_t + (1.0 - rho) * mean_train
    y_hat_noconst = rho * y_test_t
    return y_hat_const, y_hat_noconst, rho, mean_train


def baseline_ar_k(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
    k: int,
) -> np.ndarray:
    """AR(k) via regresion lineal."""
    y_full = df[TARGET_COL].to_numpy()
    # Construir matriz de lags sobre toda la serie, luego seleccionar filas validas.
    lags = lag_matrix(y_full, k)
    # Filtrar train/test a filas donde no haya NaN en lags (k-1 primeras de todo el dataset).
    idx_train_f = idx_train[idx_train >= k - 1]
    idx_test_f = idx_test[idx_test >= k - 1]
    X_train = lags[idx_train_f]
    y_train = y_full[idx_train_f + horizon]
    X_test = lags[idx_test_f]
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_hat_test_f = reg.predict(X_test)
    # Reindexar a la longitud completa de idx_test (deberia coincidir porque idx_test
    # arranca en split_start + SEQ_LENGTH + horizon - 1 >> k).
    y_hat = np.full(len(idx_test), np.nan)
    # map idx_test -> posicion
    pos = {v: i for i, v in enumerate(idx_test)}
    for i, ix in enumerate(idx_test_f):
        y_hat[pos[ix]] = y_hat_test_f[i]
    return y_hat


def baseline_physical(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
) -> np.ndarray:
    X = df[PHYSICAL_FEATURES].to_numpy()
    y = df[TARGET_COL].to_numpy()
    reg = LinearRegression()
    reg.fit(X[idx_train], y[idx_train + horizon])
    return reg.predict(X[idx_test])


def baseline_xgb(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
    features: List[str],
    params: Dict,
) -> Tuple[np.ndarray, float]:
    X = df[features].to_numpy()
    y = df[TARGET_COL].to_numpy()
    model = XGBRegressor(**params)
    t0 = time.time()
    model.fit(X[idx_train], y[idx_train + horizon])
    t_fit = time.time() - t0
    y_hat = model.predict(X[idx_test])
    return y_hat, t_fit


def baseline_rf(
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    horizon: int,
    features: List[str],
    params: Dict,
    subsample: int,
) -> Tuple[np.ndarray, float, int]:
    X = df[features].to_numpy()
    y = df[TARGET_COL].to_numpy()
    # Subsample cronologico estratificado: tomamos subsample filas espaciadas
    # uniformemente en el train para preservar cobertura temporal.
    if len(idx_train) > subsample:
        sel = np.linspace(0, len(idx_train) - 1, subsample).astype(int)
        idx_train_sub = idx_train[sel]
    else:
        idx_train_sub = idx_train
    model = RandomForestRegressor(**params)
    t0 = time.time()
    model.fit(X[idx_train_sub], y[idx_train_sub + horizon])
    t_fit = time.time() - t0
    y_hat = model.predict(X[idx_test])
    return y_hat, t_fit, len(idx_train_sub)


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def run_horizon(df: pd.DataFrame, horizon: int) -> Dict[str, object]:
    """Evalua todos los baselines para un horizonte concreto."""
    print(f"\n=== Horizonte h={horizon} ===")
    idx_train = aligned_indices(0, IDX_TRAIN_END, horizon, len(df))
    idx_val = aligned_indices(IDX_TRAIN_END, IDX_VAL_END, horizon, len(df))
    idx_test = aligned_indices(IDX_VAL_END, len(df), horizon, len(df))
    print(f"n_train={len(idx_train):,}  n_val={len(idx_val):,}  n_test={len(idx_test):,}")

    y_full = df[TARGET_COL].to_numpy()
    y_true_test = y_full[idx_test + horizon]

    results: Dict[str, object] = {
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "test_index_first": int(idx_test[0]),
        "test_index_last": int(idx_test[-1]),
        "test_timestamp_first": str(df.iloc[idx_test[0]]["timestamp"]),
        "test_timestamp_last": str(df.iloc[idx_test[-1]]["timestamp"]),
        "baselines": {},
    }

    # (a) Naive
    print("  [a] naive...")
    y_hat = baseline_naive(df, idx_test, horizon)
    results["baselines"]["naive"] = compute_metrics(y_true_test, y_hat)

    # (b) AR(1) analitico con y sin termino constante
    print("  [b] AR(1) analitico...")
    yh_c, yh_nc, rho, mean_tr = baseline_ar1_analytic(df, idx_train, idx_test, horizon)
    m = compute_metrics(y_true_test, yh_c)
    m["rho"] = rho
    m["mean_train"] = mean_tr
    results["baselines"]["ar1_analytic"] = m
    results["baselines"]["ar1_noconst"] = compute_metrics(y_true_test, yh_nc)

    # (c) AR(5)
    print("  [c] AR(5)...")
    y_hat = baseline_ar_k(df, idx_train, idx_test, horizon, k=5)
    results["baselines"]["ar5"] = compute_metrics(y_true_test, y_hat)

    # (d) AR(12)
    print("  [d] AR(12)...")
    y_hat = baseline_ar_k(df, idx_train, idx_test, horizon, k=12)
    results["baselines"]["ar12"] = compute_metrics(y_true_test, y_hat)

    # (e) Predictor fisico
    print("  [e] Fisico (rain_sum_60m + api_dynamic)...")
    y_hat = baseline_physical(df, idx_train, idx_test, horizon)
    results["baselines"]["physical_linear"] = compute_metrics(y_true_test, y_hat)

    # (f) XGBoost 20 features (sin delta_flow)
    print("  [f] XGBoost 20 features (sin delta_flow)...")
    y_hat, t_fit = baseline_xgb(df, idx_train, idx_test, horizon, FEATURES_20, XGB_PARAMS)
    m = compute_metrics(y_true_test, y_hat)
    m["fit_seconds"] = t_fit
    m["n_features"] = len(FEATURES_20)
    results["baselines"]["xgb_20"] = m
    print(f"     NSE={m['nse']:.4f}  t_fit={t_fit:.1f}s")

    # (g) Random Forest 20 features
    print(f"  [g] RandomForest 20 features (subsample={RF_TRAIN_SUBSAMPLE})...")
    y_hat, t_fit, n_used = baseline_rf(
        df, idx_train, idx_test, horizon, FEATURES_20, RF_PARAMS, RF_TRAIN_SUBSAMPLE
    )
    m = compute_metrics(y_true_test, y_hat)
    m["fit_seconds"] = t_fit
    m["n_features"] = len(FEATURES_20)
    m["n_train_subsample"] = n_used
    results["baselines"]["rf_20"] = m
    print(f"     NSE={m['nse']:.4f}  t_fit={t_fit:.1f}s")

    # (h) XGBoost 22 features (con delta_flow) - CRITICO
    print("  [h] XGBoost 22 features (con delta_flow)...")
    y_hat, t_fit = baseline_xgb(df, idx_train, idx_test, horizon, FEATURES_22, XGB_PARAMS)
    m = compute_metrics(y_true_test, y_hat)
    m["fit_seconds"] = t_fit
    m["n_features"] = len(FEATURES_22)
    results["baselines"]["xgb_22"] = m
    print(f"     NSE={m['nse']:.4f}  t_fit={t_fit:.1f}s")

    return results


def format_md_table(all_results: Dict[int, Dict[str, object]], tcn_ref: Dict) -> str:
    """Tabla comparativa por horizonte."""
    lines: List[str] = []
    lines.append("# S2 - Baselines rigurosos\n")
    lines.append(
        "Bateria de baselines clasicos / ML tabular entrenados sobre el mismo split "
        "cronologico 70/15/15 y los mismos indices de test que la TCN "
        "(`seq_length=72`, `horizon h`). Objetivo: contextualizar la ganancia real "
        "del TwoStageTCN v1 y diagnosticar el peso del atajo `delta_flow`.\n"
    )
    lines.append("## Metodologia\n")
    lines.append(
        "- **Split**: train `iloc[:771374]` (hasta 2022-12-11), val `[771374:936669]`, "
        "test `[936669:]` (hasta 2026-01-31).\n"
        "- **Ventana de evaluacion**: indices con ventana previa de 72 pasos completa "
        "y `y(t+h)` disponible (mismos indices que consumiria la TCN).\n"
        "- **Target**: `stormflow_mgd(t+h)`.\n"
        "- **Sin tuning**: hiperparametros fijos definidos en `s2_baselines.py`.\n"
    )
    lines.append("## Dimensiones por horizonte\n")
    lines.append("| h | n_train | n_val | n_test | primer ts test | ultimo ts test |")
    lines.append("|---|--------:|------:|-------:|----------------|----------------|")
    for h in HORIZONS:
        r = all_results[h]
        lines.append(
            f"| {h} | {r['n_train']:,} | {r['n_val']:,} | {r['n_test']:,} | "
            f"{r['test_timestamp_first']} | {r['test_timestamp_last']} |"
        )
    lines.append("")

    # Tabla principal por horizonte
    for h in HORIZONS:
        lines.append(f"## Horizonte h={h}\n")
        lines.append("| Baseline | NSE | RMSE | MAE | ErrPico % | Peak pred | N features | Notas |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        # TCN v1 (referencia)
        key = f"H{h}_sinSF"
        if key in tcn_ref:
            g = tcn_ref[key]["global"]
            lines.append(
                f"| **TCN v1 sinSF** (ref) | {g['nse']:.4f} | {g['rmse']:.3f} | "
                f"{g['mae']:.3f} | {g['peak_err_pct']:+.1f} | {g['peak_pred']:.1f} | 22 | "
                f"n_test={g['n_total']:,} |"
            )
        key2 = f"H{h}_conSF"
        if key2 in tcn_ref:
            g = tcn_ref[key2]["global"]
            lines.append(
                f"| TCN v1 conSF (ref) | {g['nse']:.4f} | {g['rmse']:.3f} | "
                f"{g['mae']:.3f} | {g['peak_err_pct']:+.1f} | {g['peak_pred']:.1f} | 22+sf | - |"
            )
        # Baselines S2
        order = [
            ("naive",           "Naive persistencia y(t)"),
            ("ar1_analytic",    "AR(1) analitico (rho + const)"),
            ("ar1_noconst",     "AR(1) analitico (sin const)"),
            ("ar5",             "AR(5) lineal"),
            ("ar12",            "AR(12) lineal"),
            ("physical_linear", "Lineal fisico (rain_60m+API)"),
            ("xgb_20",          "XGBoost 20 feats (sin delta_flow)"),
            ("rf_20",           "RandomForest 20 feats"),
            ("xgb_22",          "XGBoost 22 feats (con delta_flow)"),
        ]
        bl = all_results[h]["baselines"]
        for k, label in order:
            m = bl[k]
            nf = m.get("n_features", "-")
            extra = ""
            if k == "ar1_analytic":
                extra = f"rho={m['rho']:.4f}"
            elif k == "rf_20":
                extra = f"sub={m.get('n_train_subsample', '?'):,}"
            elif k in ("xgb_20", "xgb_22", "rf_20"):
                extra = f"t={m.get('fit_seconds', 0):.0f}s"
            lines.append(
                f"| {label} | {m['nse']:.4f} | {m['rmse']:.3f} | {m['mae']:.3f} | "
                f"{m['peak_err_pct']:+.1f} | {m['peak_pred']:.1f} | {nf} | {extra} |"
            )
        lines.append("")

    # Bias por bucket (solo H=1)
    lines.append("## Bias por bucket (H=1)\n")
    lines.append(
        "`bias = mean(y_pred - y_true)` dentro de cada rango de `y_true` (MGD). "
        "Negativo = subestima, positivo = sobrestima.\n"
    )
    buckets_names = [b[0] for b in BUCKETS]
    lines.append("| Baseline | " + " | ".join(buckets_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(buckets_names)) + "|")
    if "H1_sinSF" in tcn_ref:
        r = tcn_ref["H1_sinSF"]["ranges"]
        row = ["**TCN v1 sinSF**"]
        for bn in buckets_names:
            row.append(f"{r[bn]['bias']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    bl = all_results[1]["baselines"]
    for k, label in [
        ("naive", "Naive"),
        ("ar1_analytic", "AR(1) analitico"),
        ("ar5", "AR(5)"),
        ("ar12", "AR(12)"),
        ("physical_linear", "Fisico lineal"),
        ("xgb_20", "XGB-20"),
        ("rf_20", "RF-20"),
        ("xgb_22", "XGB-22"),
    ]:
        b = bl[k]["buckets"]
        row = [label]
        for bn in buckets_names:
            bv = b[bn].get("bias", float("nan"))
            row.append(f"{bv:+.3f}" if bv == bv else "n/a")  # nan check
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # NSE por bucket (H=1)
    lines.append("## NSE por bucket (H=1)\n")
    lines.append(
        "NSE local dentro del bucket. NSE<0 indica que el modelo es peor que "
        "predecir la media del bucket.\n"
    )
    lines.append("| Baseline | " + " | ".join(buckets_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(buckets_names)) + "|")
    if "H1_sinSF" in tcn_ref:
        r = tcn_ref["H1_sinSF"]["ranges"]
        row = ["**TCN v1 sinSF**"]
        for bn in buckets_names:
            row.append(f"{r[bn]['nse']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    for k, label in [
        ("naive", "Naive"),
        ("ar1_analytic", "AR(1)"),
        ("ar5", "AR(5)"),
        ("ar12", "AR(12)"),
        ("physical_linear", "Fisico"),
        ("xgb_20", "XGB-20"),
        ("rf_20", "RF-20"),
        ("xgb_22", "XGB-22"),
    ]:
        b = bl[k]["buckets"]
        row = [label]
        for bn in buckets_names:
            bv = b[bn].get("nse", float("nan"))
            row.append(f"{bv:+.3f}" if bv == bv else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


def build_verdict(all_results: Dict[int, Dict[str, object]], tcn_ref: Dict) -> str:
    r1 = all_results[1]["baselines"]
    r3 = all_results[3]["baselines"]
    r6 = all_results[6]["baselines"]

    nse_tcn_h1 = tcn_ref["H1_sinSF"]["global"]["nse"]
    nse_xgb20_h1 = r1["xgb_20"]["nse"]
    nse_xgb22_h1 = r1["xgb_22"]["nse"]
    nse_naive_h1 = r1["naive"]["nse"]
    nse_ar12_h1 = r1["ar12"]["nse"]
    nse_phys_h1 = r1["physical_linear"]["nse"]

    delta_shortcut = nse_xgb22_h1 - nse_xgb20_h1
    delta_xgb_vs_naive = nse_xgb20_h1 - nse_naive_h1
    delta_xgb_vs_ar12 = nse_xgb20_h1 - nse_ar12_h1
    delta_tcn_vs_xgb20 = nse_tcn_h1 - nse_xgb20_h1
    delta_tcn_vs_xgb22 = nse_tcn_h1 - nse_xgb22_h1

    lines: List[str] = []
    lines.append("## Veredicto\n")
    lines.append(
        "Respuestas directas a las preguntas clave del diagnostico. "
        "Todos los numeros se refieren al test alineado (ventana de 72 pasos), "
        "sin tuning ni early stopping, y al target `stormflow_mgd(t+h)`.\n"
    )

    # P1
    lines.append(
        f"### 1. XGBoost-20 vs TCN v1 (H=1)\n"
        f"- NSE XGB-20 = **{nse_xgb20_h1:.4f}**\n"
        f"- NSE TCN v1 sinSF = **{nse_tcn_h1:.4f}**\n"
        f"- Diferencia TCN - XGB-20 = **{delta_tcn_vs_xgb20:+.4f}** NSE.\n"
    )
    if delta_tcn_vs_xgb20 >= 0.02:
        lines.append(
            "Interpretacion: la TCN **si aporta** algo de valor arquitectonico sobre "
            "XGBoost con las mismas 20 features (sin el atajo delta_flow), aunque "
            "la ganancia es modesta. Habra que ver si compensa la complejidad.\n"
        )
    elif abs(delta_tcn_vs_xgb20) < 0.02:
        lines.append(
            "Interpretacion: XGBoost-20 **empata** con el TCN v1. La arquitectura "
            "temporal (convoluciones causales, backbone compartido, two-stage) "
            "no esta aportando valor medible frente a un GBM tabular con las "
            "mismas features sin el atajo.\n"
        )
    else:
        lines.append(
            "Interpretacion: XGBoost-20 **supera** al TCN v1. La TCN no esta "
            "extrayendo informacion util de la dinamica temporal mas alla de lo "
            "que captura XGBoost con las features agregadas (rain_sum_*, api_dynamic, "
            "delta_rain_*). Toda la ganancia aparente del TCN venia del atajo.\n"
        )

    # P2
    lines.append(
        f"### 2. Peso real del atajo `delta_flow`\n"
        f"- NSE XGB-22 (con delta_flow) = **{nse_xgb22_h1:.4f}**\n"
        f"- NSE XGB-20 (sin delta_flow) = **{nse_xgb20_h1:.4f}**\n"
        f"- Contribucion atajo = **{delta_shortcut:+.4f}** NSE.\n"
        f"- NSE XGB-22 - TCN v1 = **{delta_tcn_vs_xgb22:+.4f}** (negativo = XGB-22 gana).\n"
    )
    if delta_shortcut >= 0.03:
        lines.append(
            "Interpretacion: el atajo `delta_flow_*` aporta una ganancia claramente "
            "mensurable tambien en XGBoost. Confirma el hallazgo de iter16: el "
            "salto del TCN sobre naive venia en gran parte de esas dos features.\n"
        )
    else:
        lines.append(
            "Interpretacion: `delta_flow_*` aporta poco en XGBoost (<0.03 NSE). "
            "El atajo puede ser mas util al TCN por como lo combina internamente.\n"
        )

    # P3
    lines.append(
        f"### 3. XGBoost-20 vs baselines triviales\n"
        f"- XGB-20 - naive = **{delta_xgb_vs_naive:+.4f}** NSE\n"
        f"- XGB-20 - AR(12) = **{delta_xgb_vs_ar12:+.4f}** NSE\n"
    )
    if delta_xgb_vs_naive < 0.02:
        lines.append(
            "Interpretacion: XGBoost-20 **apenas mejora** sobre persistencia. Las "
            "features de lluvia estan siendo ignoradas (o no aportan senal a H=1).\n"
        )
    else:
        lines.append(
            "Interpretacion: XGBoost-20 **si usa** las features de lluvia/API. "
            "La senal exogena tiene valor predictivo.\n"
        )

    # P4
    lines.append(
        f"### 4. Predictor fisico (2 features) como sanity check (H=1)\n"
        f"- NSE lineal(rain_sum_60m + api_dynamic) = **{nse_phys_h1:.4f}**\n"
    )
    if nse_phys_h1 >= 0.3:
        lines.append(
            "Interpretacion: con solo dos features fisicas se alcanza un NSE >=0.3. "
            "Existe senal lluvia->stormflow aprendible, y un modelo simple ya la captura "
            "parcialmente.\n"
        )
    elif nse_phys_h1 > 0:
        lines.append(
            "Interpretacion: NSE positivo pero modesto. La relacion lluvia->stormflow "
            "es no lineal y necesita mas features / modelo mas expresivo para "
            "extraerla bien.\n"
        )
    else:
        lines.append(
            "Interpretacion: NSE negativo o cero. La combinacion lineal de "
            "rain_sum_60m + api_dynamic no basta; el problema tiene no-linealidades "
            "fuertes o el horizonte confunde las escalas.\n"
        )

    # P5: que horizonte tiene sentido
    def best_nse(res):
        return max((v["nse"] for v in res.values() if isinstance(v, dict) and "nse" in v))
    best_h1 = best_nse(r1)
    best_h3 = best_nse(r3)
    best_h6 = best_nse(r6)
    lines.append(
        f"### 5. Horizonte operativo\n"
        f"- Mejor NSE baseline a H=1: **{best_h1:.4f}**\n"
        f"- Mejor NSE baseline a H=3: **{best_h3:.4f}**\n"
        f"- Mejor NSE baseline a H=6: **{best_h6:.4f}**\n"
    )
    if best_h6 < 0.3:
        lines.append(
            "Interpretacion: a H=6 ningun baseline alcanza NSE razonable. La senal "
            "predictiva disponible se degrada rapido con el horizonte, "
            "consistente con el lag efectivo del sistema (memoria corta). "
            "Trabajar en H=1 (30 min) y, si el TFM lo exige, H=3 (15 min) "
            "como maximo operativo.\n"
        )
    else:
        lines.append(
            "Interpretacion: hay margen para horizontes mas largos. Evaluar H>=3 "
            "como objetivo operativo.\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Cargando parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  shape={df.shape}")

    all_results: Dict[int, Dict[str, object]] = {}
    for h in HORIZONS:
        all_results[h] = run_horizon(df, h)

    # Cargar metricas del TCN v1 para comparacion
    tcn_ref_path = ROOT / "outputs" / "data_analysis" / "local_eval_metrics.json"
    with open(tcn_ref_path, "r", encoding="utf-8") as f:
        tcn_ref = json.load(f)

    # Metadatos globales
    meta = {
        "split": {
            "train_end_idx": IDX_TRAIN_END,
            "val_end_idx": IDX_VAL_END,
            "total_rows": len(df),
            "train_last_ts": str(df.iloc[IDX_TRAIN_END - 1]["timestamp"]),
            "val_last_ts": str(df.iloc[IDX_VAL_END - 1]["timestamp"]),
            "test_last_ts": str(df.iloc[-1]["timestamp"]),
        },
        "seq_length": SEQ_LENGTH,
        "horizons": HORIZONS,
        "features_22": FEATURES_22,
        "features_20": FEATURES_20,
        "physical_features": PHYSICAL_FEATURES,
        "xgb_params": XGB_PARAMS,
        "rf_params": RF_PARAMS,
        "rf_train_subsample": RF_TRAIN_SUBSAMPLE,
        "buckets_mgd": [{"name": b[0], "lo": b[1], "hi": b[2]} for b in BUCKETS],
        "tcn_reference_source": str(tcn_ref_path),
    }

    json_out = {"meta": meta, "results": all_results}
    json_path = OUT_DIR / "S2_baselines.json"
    # Sanear inf/NaN para JSON
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
        return o
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean(json_out), f, indent=2, ensure_ascii=False)
    print(f"\nEscrito: {json_path}")

    md_body = format_md_table(all_results, tcn_ref)
    md_verdict = build_verdict(all_results, tcn_ref)
    md_path = OUT_DIR / "S2_baselines.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_body + "\n" + md_verdict + "\n")
    print(f"Escrito: {md_path}")


if __name__ == "__main__":
    main()
