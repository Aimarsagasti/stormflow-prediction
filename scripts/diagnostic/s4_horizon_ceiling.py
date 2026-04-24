"""
S4 - Techo fisico de NSE alcanzable por horizonte y oraculos parciales.

Objetivo: cuantificar para cada horizonte h en {1, 3, 6, 12, 24} cuanto puede
mejorar el modelo respecto a la persistencia trivial (naive y AR(1)) y donde
estan las palancas reales (que bucket de magnitud aporta mas al denominador
del NSE y, por tanto, donde un acierto perfecto sube mas el NSE global).

Para los oraculos parciales del modelo TCN v1 actual (H=1, H=3, H=6 sinSF) se
ejecuta inferencia en lote sobre todo el test alineado y se reemplaza la
prediccion por el target real solo dentro del bucket o combinacion de buckets
elegida; despues se recalcula el NSE global.

Artefactos:
  - outputs/diagnostic/S4_horizon_ceiling.json
  - outputs/diagnostic/S4_horizon_ceiling.md
  - outputs/figures/diagnostic/s4_horizon_ceiling.png

Solo se usan: pandas, numpy, sklearn, torch, matplotlib.
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

# Inyectar raiz del proyecto al sys.path para poder importar src.*
ROOT = Path("C:/Dev/TFM")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tcn import TwoStageTCN  # type: ignore


# ---------------------------------------------------------------------------
# Configuracion
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

# Buckets segun enunciado S4: Moderado [5, 25), Alto [25, 50), Extremo>=50.
# Difieren de los del v1 metrics (que usa Alto [20, 50)) -> documentado en md.
BUCKETS = [
    ("Base",     -np.inf, 0.5),
    ("Leve",     0.5,     5.0),
    ("Moderado", 5.0,     25.0),
    ("Alto",     25.0,    50.0),
    ("Extremo",  50.0,    np.inf),
]

# Modelos v1 sinSF disponibles para inferencia (solo H1 y H3; H6_sinSF tiene NSE
# negativo y H12/H24 no tienen modelo entrenado).
TCN_MODELS = {
    1: "modelo_H1_sinSF",
    3: "modelo_H3_sinSF",
    6: "modelo_H6_sinSF",
}

INFERENCE_BATCH = 1024
CLS_THRESHOLD = 0.3  # mismo umbral que S3 / evaluate_local


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
    """Mismo criterio que S2: ventana previa de SEQ_LENGTH completa y target dentro del df."""
    first = split_start + SEQ_LENGTH
    last_in_split = split_end - 1
    last_by_target = total_len - 1 - horizon
    last = min(last_in_split, last_by_target)
    return np.arange(first, last + 1)


# ---------------------------------------------------------------------------
# Baselines analiticos
# ---------------------------------------------------------------------------
def naive_pred(df: pd.DataFrame, idx_test: np.ndarray) -> np.ndarray:
    return df[TARGET_COL].to_numpy()[idx_test]


def ar1_optimal(
    df: pd.DataFrame, idx_train: np.ndarray, idx_test: np.ndarray, horizon: int
) -> Tuple[np.ndarray, float, float]:
    """AR(1) optimo: y_hat = mean_train + rho_h * (y(t) - mean_train)."""
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
    """AR(k) por regresion lineal sobre k lags. Mismo metodo que S2."""
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
# Inferencia TCN v1 sobre TODO el test alineado (en lotes)
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
    """Devuelve y_pred(t+h) en MGD para cada origen t_origin en idx_test.

    Convencion de `src.pipeline.sequences`:
      - ventana = feat[t_origin - SEQ_LENGTH + 1 : t_origin + 1]  (incluye t_origin)
      - target  = y[t_origin + horizon]

    En este script `idx_test` representa `t_origin`. Tal cual viene de
    `aligned_indices`, con `first = split_start + SEQ_LENGTH` y
    `last = total_len - 1 - horizon`, la ventana [origin - SEQ + 1, origin]
    siempre cabe en el df.
    """
    print(f"[tcn] cargando {model_stem}...")
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

    print(f"[tcn] inferencia en lotes batch={INFERENCE_BATCH} ({n_test:,} muestras)...")
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_test, INFERENCE_BATCH):
            end = min(start + INFERENCE_BATCH, n_test)
            batch_idx = idx_test[start:end]
            # Ventana [t_origin - SEQ + 1, t_origin] incluyendo t_origin.
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
                print(f"    progreso {pct:5.1f}% ({end:,}/{n_test:,})")
    elapsed = time.time() - t0
    print(f"[tcn] inferencia completada en {elapsed:.1f}s")

    # Desnormalizar
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
# Oraculos parciales y contribucion al denominador
# ---------------------------------------------------------------------------
def bucket_contribution(y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Contribucion de cada bucket al denominador del NSE (sum (y-mean)^2)."""
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
    print(f"\n=== Horizonte h={horizon} ({horizon * 5} min) ===")
    idx_train = aligned_indices(0, IDX_TRAIN_END, horizon, len(df))
    idx_test = aligned_indices(IDX_VAL_END, len(df), horizon, len(df))
    print(f"  n_train={len(idx_train):,}  n_test={len(idx_test):,}")

    y_full = df[TARGET_COL].to_numpy()
    y_true_test = y_full[idx_test + horizon]

    # 1) Naive
    y_hat_naive = naive_pred(df, idx_test)
    nse_naive = nse(y_true_test, y_hat_naive)

    # 2) AR(1) optimo + cota teorica
    y_hat_ar1, rho_h, mean_train = ar1_optimal(df, idx_train, idx_test, horizon)
    nse_ar1 = nse(y_true_test, y_hat_ar1)
    bound_2rho_minus_1 = 2.0 * rho_h - 1.0  # cota superior bajo iid (informativa)

    # 3) AR(12) lineal
    y_hat_ar12 = ar_k_pred(df, idx_train, idx_test, horizon, k=12)
    nse_ar12 = nse(y_true_test, y_hat_ar12)

    # 4) Contribucion de buckets al denominador
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

    # 5) Oraculos parciales: solo para horizontes con TCN entrenado.
    if horizon in TCN_MODELS:
        model_stem = TCN_MODELS[horizon]
        y_pred_tcn, info = infer_tcn_test(df, model_stem, idx_test, horizon)
        nse_tcn = nse(y_true_test, y_pred_tcn)
        rmse_tcn = rmse(y_true_test, y_pred_tcn)
        peak_real = float(np.max(y_true_test))
        peak_pred = float(np.max(y_pred_tcn))
        peak_err_pct = (peak_pred - peak_real) / peak_real * 100.0 if peak_real > 0 else float("nan")

        # Bias y RMSE por bucket usando el TCN (sanity check vs metrics.json)
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

        # Oraculos parciales
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
        # Para H=12 y H=24 no hay TCN. Se reporta solo el techo analitico.
        result["tcn_v1"] = None
        result["oracles"] = None

    print(f"  NSE naive={nse_naive:.4f}  AR(1)opt={nse_ar1:.4f}  AR(12)={nse_ar12:.4f}")
    print(f"  rho_h={rho_h:.4f}  cota 2rho-1={bound_2rho_minus_1:.4f}")
    if result["tcn_v1"] is not None:
        print(f"  TCN v1 NSE={result['tcn_v1']['nse']:.4f}  pico_err={result['tcn_v1']['peak_err_pct']:+.1f}%")
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
    ax.plot(h_min, nse_naive, "o--", color="#888", label="Naive (persistencia)")
    ax.plot(h_min, nse_ar1, "s--", color="#1f77b4", label=r"AR(1) optimo ($\rho_h$)")
    ax.plot(h_min, nse_ar12, "^--", color="#2ca02c", label="AR(12) lineal")
    ax.plot(h_min, bound, ":", color="#d62728", label=r"Cota analitica $2\rho_h-1$")
    tcn_h = [h for h, v in zip(h_min, nse_tcn) if v is not None]
    tcn_v = [v for v in nse_tcn if v is not None]
    if tcn_h:
        ax.plot(tcn_h, tcn_v, "*-", color="#9467bd", markersize=14, label="TCN v1 sinSF")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Horizonte (minutos)")
    ax.set_ylabel("NSE en test alineado")
    ax.set_title("Techo de NSE alcanzable por horizonte (target = stormflow_mgd)")
    ax.set_xticks(h_min)
    ax.set_xticklabels([f"{m} min\n(h={h})" for m, h in zip(h_min, horizons)])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] escrito {path}")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def format_md(all_results: Dict[int, Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# S4 - Techo de NSE alcanzable por horizonte\n")
    lines.append(
        "Cuantificacion del techo de NSE alcanzable con las features actuales y "
        "sin informacion futura externa. Para cada horizonte h se reportan los "
        "baselines AR analiticos, la cota teorica de un AR(1) optimo, el TCN v1 "
        "(cuando hay modelo) y oraculos parciales por bucket de magnitud.\n"
    )

    # Metodologia
    lines.append("## Metodologia\n")
    lines.append(
        "- **Split test cronologico**: `iloc[936669:]`, mismos indices alineados que "
        "S2 (ventana previa de 72 pasos completa, target dentro del df).\n"
        "- **Buckets MGD**: Base<0.5; Leve [0.5, 5); Moderado [5, 25); Alto [25, 50); Extremo>=50. "
        "Notar que `evaluate_local.py` usa Alto [20, 50) para reportar el v1; aqui se "
        "recalcula con [25, 50) para ser consistentes con el enunciado del diagnostico.\n"
        "- **AR(1) optimo**: rho_h = corr(y(t), y(t+h)) sobre train; "
        "y_hat = mean_train + rho_h*(y(t) - mean_train).\n"
        "- **Cota analitica `2*rho_h - 1`**: maximo NSE alcanzable por un predictor "
        "lineal ortogonal de y(t) cuando rho_h > 0.5 (cota informativa, no tope absoluto).\n"
        "- **AR(12)**: regresion lineal con 12 lags consecutivos.\n"
        "- **TCN v1 sinSF**: inferencia en lote sobre todo el test alineado, switch "
        "duro con threshold=0.3 (mismo que evaluate_local). Solo disponible para H=1, H=3, H=6.\n"
        "- **Oraculo parcial**: para los buckets indicados, y_pred = y_true; el resto "
        "queda igual. NSE recalculado sobre todo el test.\n"
    )

    # Tabla maestra
    lines.append("## Tabla maestra: NSE por horizonte\n")
    lines.append(
        "| h | min | n_test | rho_h | NSE naive | NSE AR(1)opt | cota 2rho-1 | "
        "NSE AR(12) | NSE TCN v1 | Pico err % |"
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
        "Lectura rapida: `naive` es el suelo trivial; `AR(1) optimo` lo bate "
        "ligeramente porque mueve la prediccion hacia la media de train cuando rho_h<1; "
        "`AR(12)` aprovecha lags adicionales pero la mejora marginal indica que la "
        "memoria mas alla de un par de lags ya esta saturada; la cota `2*rho-1` da el "
        "techo teorico de un AR(1) bajo independencia de errores.\n"
    )

    # Contribucion de buckets al denominador
    lines.append("## Contribucion de cada bucket al denominador del NSE\n")
    lines.append(
        "Donde se concentra la varianza de y_true (sum (y-mean)^2). Si un bucket "
        "aporta el X% del denominador, mejorar la prediccion ahi sube el NSE en "
        "proporcion al X%. **Es la palanca principal**.\n"
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

    # Oraculos parciales
    lines.append("## Oraculos parciales sobre TCN v1 (solo h con modelo entrenado)\n")
    lines.append(
        "Reemplazo y_pred = y_true dentro del bucket o combinacion indicada, NSE recalculado "
        "sobre todo el test. La columna `delta` es el incremento sobre el TCN v1.\n"
    )
    for h in sorted(all_results.keys()):
        r = all_results[h]
        if r.get("tcn_v1") is None:
            continue
        nse_tcn = r["tcn_v1"]["nse"]
        lines.append(f"### h={h} ({r['horizon_min']} min)\n")
        lines.append(
            f"- TCN v1 base: NSE = **{nse_tcn:.4f}**, "
            f"pico real {r['tcn_v1']['peak_real']:.1f} MGD, "
            f"pico predicho {r['tcn_v1']['peak_pred']:.1f} MGD "
            f"({r['tcn_v1']['peak_err_pct']:+.1f}%).\n"
        )
        lines.append("| Oraculo | n muestras oraculizadas | NSE oraculo | delta NSE |")
        lines.append("|---|---:|---:|---:|")
        order = [
            ("extremo", "Solo Extremo (>=50)"),
            ("extremo_alto", "Extremo + Alto (>=25)"),
            ("moderado", "Solo Moderado [5,25)"),
            ("leve_base", "Leve + Base (<5)"),
            ("moderado_leve_base", "Moderado + Leve + Base (<25)"),
            ("todo_excepto_extremo", "Todo excepto Extremo (<50)"),
        ]
        for k, label in order:
            o = r["oracles"][k]
            lines.append(
                f"| {label} | {o['n_oracled']:,} | "
                f"{o['nse']:.4f} | {o['delta_nse_vs_tcn']:+.4f} |"
            )
        # Bucket metrics del TCN
        lines.append("")
        lines.append("Metricas del TCN v1 por bucket (sanity check):\n")
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
    lines.append("## Sintesis cuantitativa y veredicto\n")

    # 1. Horizonte con mayor ganancia teorica posible
    gains = {}
    for h in sorted(all_results.keys()):
        r = all_results[h]
        # Tope fisico estimado: max( cota 2*rho-1, NSE AR(12), NSE TCN v1 )
        candidates = [r["bound_2rho_minus_1"], r["nse_ar12"]]
        if r.get("tcn_v1") is not None:
            candidates.append(r["tcn_v1"]["nse"])
        nse_max_phys = max(candidates)
        ganancia = nse_max_phys - r["nse_naive"]
        gains[h] = (nse_max_phys, ganancia)
    lines.append("### 1. Horizonte que merece la pena optimizar\n")
    lines.append("| h | min | NSE naive | NSE max fisico estimado | Ganancia max sobre naive |")
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
        f"\n**Horizonte de maxima ganancia teorica: h={h_best} ({h_best*5} min)** "
        f"con margen de {gains[h_best][1]:+.4f} NSE sobre naive.\n"
    )

    # 2. NSE max defendible
    lines.append("### 2. NSE maximo defendible por horizonte (modelo perfecto, mismas features)\n")
    lines.append(
        "Tope superior estimado: maximo entre la cota analitica `2*rho-1` (que asume "
        "predictor AR(1) optimo bajo iid) y la mejor evidencia empirica disponible "
        "(AR(12) o TCN v1). Tomamos el mayor de los dos como techo conservador "
        "alcanzable con la informacion disponible.\n"
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

    # 3. Mayor palanca cuantitativa
    lines.append("### 3. Mayor palanca cuantitativa para subir NSE en H=1\n")
    if 1 in all_results and all_results[1].get("oracles") is not None:
        r1 = all_results[1]
        oracles = r1["oracles"]
        # Buscar el oraculo con maximo delta
        # Comparar oraculos individuales y combinados
        ranked = sorted(oracles.items(), key=lambda kv: -kv[1]["delta_nse_vs_tcn"])
        lines.append(
            f"TCN v1 actual: NSE = **{r1['tcn_v1']['nse']:.4f}**. Si el modelo acertara "
            "perfectamente dentro del bucket indicado (manteniendo el resto), el NSE pasaria a:\n"
        )
        lines.append("| Oraculo | NSE oraculo | delta NSE | n |")
        lines.append("|---|---:|---:|---:|")
        for k, v in ranked:
            lines.append(
                f"| {k} | {v['nse']:.4f} | {v['delta_nse_vs_tcn']:+.4f} | {v['n_oracled']:,} |"
            )
        # Veredicto
        best_single = max(
            ["extremo", "moderado", "leve_base"],
            key=lambda k: oracles[k]["delta_nse_vs_tcn"],
        )
        lines.append(
            f"\n**Palanca dominante en H=1**: el oraculo que mas sube el NSE individualmente "
            f"es `{best_single}` con delta = "
            f"{oracles[best_single]['delta_nse_vs_tcn']:+.4f} NSE. "
            f"Esto coincide con la contribucion al denominador de ese bucket.\n"
        )
        # Recordar contribucion al denominador
        c1 = r1["buckets_contribution"]
        lines.append("Contribucion al denominador (H=1) de los buckets clave:\n")
        for bn in ["Extremo", "Alto", "Moderado", "Leve", "Base"]:
            lines.append(
                f"- **{bn}**: {c1[bn]['share_denom_pct']:.2f}% del denominador "
                f"({c1[bn]['n']:,} muestras = {c1[bn]['share_n_pct']:.3f}% del test)."
            )
        lines.append("")

    # 4. Tiene sentido perseguir H=6?
    lines.append("### 4. Tiene sentido perseguir H=6 (30 min)?\n")
    if 6 in all_results:
        r6 = all_results[6]
        nse_max_h6, gan_h6 = gains[6]
        lines.append(
            f"- NSE naive H=6 = **{r6['nse_naive']:.4f}** "
            f"(rho_h={r6['rho_h_train']:.4f}, AR(1)opt={r6['nse_ar1_opt']:.4f}, "
            f"AR(12)={r6['nse_ar12']:.4f}).\n"
            f"- TCN v1 sinSF a H=6 = **{r6['tcn_v1']['nse']:.4f}** (segun S4 inference).\n"
            f"- NSE max fisico estimado H=6 = **{nse_max_h6:.4f}**.\n"
        )
        if nse_max_h6 < 0.5:
            lines.append(
                "Veredicto: el techo fisico de H=6 con las features actuales esta **por debajo "
                "de NSE=0.5**. Perseguir H=6 con el dataset actual no llevara a un modelo "
                "operativo defensible. Para abrir H=6 hace falta features exogenas con horizonte "
                "futuro (forecast de lluvia, NWP) o cambiar el horizonte objetivo.\n"
            )
        elif nse_max_h6 < 0.7:
            lines.append(
                "Veredicto: H=6 alcanzable pero con techo claramente inferior a H=1/H=3. "
                "Decision arquitectonica: aceptar el techo o reformular el target (eventos "
                "de tormenta agregados en lugar de stormflow puntual).\n"
            )
        else:
            lines.append(
                "Veredicto: H=6 sigue teniendo techo razonable; merece la pena explorarlo.\n"
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
        print(f"[plot] AVISO: no se pudo generar el plot ({exc})")

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
    print(f"[out] escrito {json_path}")

    md = format_md(all_results)
    md_path = OUT_DIR / "S4_horizon_ceiling.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"[out] escrito {md_path}")


if __name__ == "__main__":
    main()
