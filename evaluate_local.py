"""
evaluate_local.py
=================
Script de evaluación local para los modelos entrenados.
Ejecutar desde VS Code en C:\\Dev\\TFM\\ con Python sin GPU.

Genera:
  - Métricas globales y por rangos de magnitud para los 6 modelos v1
  - Métricas del modelo extra v2 (loss function gradual)
  - Curvas de predicibilidad (NSE vs horizonte, con/sin SF)
  - Series temporales: zoom a los 3 mayores picos por modelo
  - Scatter predicho vs real por modelo
  - Análisis de eventos extremos: con lluvia vs sin lluvia
  - Plots de comparación directa v1 vs v2 (hidrogramas, scatter, rangos)
  - Todos los plots se guardan en outputs/figures/local_eval/

Uso:
  cd C:\\Dev\\TFM
  python evaluate_local.py
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import sys
import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Sin pantalla, guarda los plots directamente a disco
import matplotlib.pyplot as plt
import torch

# Añadir la raíz del proyecto al path para que los imports de src/ funcionen
PROJECT_ROOT = Path(__file__).parent   # C:\Dev\TFM\
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_msd_data
from src.data.clean import clean_timeseries
from src.features.engineering import create_features
from src.pipeline.split import split_chronological
from src.pipeline.normalize import normalize_splits, denormalize_target
from src.pipeline.sequences import create_dataloaders
from src.models.tcn import TwoStageTCN
from src.training.trainer import predict

# ===========================================================================
# 1. CONFIGURACIÓN
# ===========================================================================

# Rutas locales Windows (sobreescriben las rutas de Colab del YAML)
LOCAL_DATA_PATHS = [
    "C:/Dev/TFM/MC-CL-005/1parte/",
    "C:/Dev/TFM/MC-CL-005/2parte/",
]
LOCAL_TEMP_PATH = "C:/Dev/TFM/MC-CL-005/daily_temperatures_2006_2026.tsf"

# Carpeta con los checkpoints (v1 + v2)
WEIGHTS_DIR = Path("C:/Dev/TFM/MC-CL-005/Pesos 13-04-2026")

# Carpeta de salida para plots y métricas
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "local_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_OUTPUT = PROJECT_ROOT / "outputs" / "data_analysis" / "local_eval_metrics.json"

# Rutas del YAML y del config
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# Horizontes y variantes a evaluar
HORIZONS = [1, 3, 6]
VARIANTS = ["sinSF", "conSF"]

# Modelos extra fuera del bucle regular (p.ej. variantes experimentales v2 que solo
# se han entrenado para ciertos horizontes). Cada entrada es un dict con:
#   - key: identificador único del modelo para resultados y plots
#   - weights_stem: nombre base del checkpoint sin extensión (sin "_weights.pt")
#   - horizon: horizonte del modelo (H=1, 3, 6)
#   - variant: "sinSF" o "conSF", determina qué lista de features usar
#   - compare_against: key del modelo v1 contra el que se compara en plots
EXTRA_MODELS = [
    {
        "key":             "H1_sinSF_v2",
        "weights_stem":    "modelo_H1_sinSF_v2",
        "horizon":         1,
        "variant":         "sinSF",
        "compare_against": "H1_sinSF",
    },
]

# Features de cada variante
FEATURES_SIN_SF = [
    "rain_in",
    "temp_daily_f", "api_dynamic",
    "rain_sum_10m", "rain_sum_15m", "rain_sum_30m",
    "rain_sum_60m", "rain_sum_120m", "rain_sum_180m", "rain_sum_360m",
    "rain_max_10m", "rain_max_30m", "rain_max_60m",
    "minutes_since_last_rain",
    "delta_flow_5m", "delta_flow_15m",
    "delta_rain_10m", "delta_rain_30m",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
FEATURES_CON_SF = FEATURES_SIN_SF + ["stormflow_mgd"]

TARGET_COL = "stormflow_mgd"
AUX_COL    = "is_event"
SEQ_LENGTH = 72
BATCH_SIZE = 256    # Más grande que en Colab porque no hay límite de GPU RAM

DEVICE = torch.device("cpu")   # Sin GPU en local

# Umbrales para análisis por rangos de magnitud (MGD)
RANGE_LABELS   = ["Base", "Leve", "Moderado", "Alto", "Extremo"]
RANGE_LOWER    = [None,  0.5,   5.0,   20.0,  50.0]
RANGE_UPPER    = [0.5,   5.0,  20.0,   50.0,  None]

# Paleta fija para los 6 modelos v1 (consistencia entre figuras)
MODEL_PLOT_STYLE = {
    "H1_sinSF": {"color": "#2196F3", "linestyle": "-",  "label": "H1 sinSF"},
    "H1_conSF": {"color": "#2196F3", "linestyle": "--", "label": "H1 conSF"},
    "H3_sinSF": {"color": "#FF9800", "linestyle": "-",  "label": "H3 sinSF"},
    "H3_conSF": {"color": "#FF9800", "linestyle": "--", "label": "H3 conSF"},
    "H6_sinSF": {"color": "#F44336", "linestyle": "-",  "label": "H6 sinSF"},
    "H6_conSF": {"color": "#F44336", "linestyle": "--", "label": "H6 conSF"},
}

# ===========================================================================
# 2. FUNCIONES AUXILIARES
# ===========================================================================

def _nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    if y_true.size == 0:
        return float("nan")
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """MAPE solo sobre muestras con y_true > eps para evitar división por cero."""
    mask = y_true > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def range_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas (NSE, RMSE, MAE, MAPE, N) por cada rango de magnitud."""
    results = {}
    for label, lo, hi in zip(RANGE_LABELS, RANGE_LOWER, RANGE_UPPER):
        if lo is None:
            mask = y_true < hi
        elif hi is None:
            mask = y_true >= lo
        else:
            mask = (y_true >= lo) & (y_true < hi)
        yt = y_true[mask]
        yp = y_pred[mask]
        results[label] = {
            "n":    int(mask.sum()),
            "nse":  _nse(yt, yp),
            "rmse": _rmse(yt, yp),
            "mae":  _mae(yt, yp),
            "mape": _mape(yt, yp),
            "bias": float(np.mean(yp - yt)) if yt.size > 0 else float("nan"),
        }
    return results


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas globales + error de pico + bias en base y extremos."""
    peak_idx  = int(np.argmax(y_true))
    peak_real = float(y_true[peak_idx])
    peak_pred = float(y_pred[peak_idx])
    peak_err  = (peak_pred - peak_real) / max(peak_real, 1e-9) * 100

    base_mask = y_true < 0.5
    ext_mask  = y_true > 50.0
    bias_base = float(np.mean(y_pred[base_mask] - y_true[base_mask])) if base_mask.sum() > 0 else float("nan")
    bias_ext  = float(np.mean(y_pred[ext_mask]  - y_true[ext_mask]))  if ext_mask.sum()  > 0 else float("nan")

    return {
        "nse":       _nse(y_true, y_pred),
        "rmse":      _rmse(y_true, y_pred),
        "mae":       _mae(y_true, y_pred),
        "peak_real": peak_real,
        "peak_pred": peak_pred,
        "peak_err_pct": peak_err,
        "bias_base": bias_base,
        "bias_ext":  bias_ext,
        "n_total":   int(y_true.size),
    }


def print_metrics_table(results: dict) -> None:
    """Imprime en consola una tabla resumen de todos los modelos."""
    print("\n" + "=" * 90)
    print(f"{'Modelo':<20} {'NSE':>7} {'RMSE':>7} {'MAE':>6} {'ErrPico':>9} {'BiasBase':>9} {'BiasExt':>9}")
    print("=" * 90)
    for model_key, data in sorted(results.items()):
        g = data["global"]
        print(
            f"{model_key:<20} {g['nse']:>7.3f} {g['rmse']:>7.2f} {g['mae']:>6.2f} "
            f"{g['peak_err_pct']:>+8.1f}% {g['bias_base']:>+9.2f} {g['bias_ext']:>+9.2f}"
        )
    print("=" * 90)


# ===========================================================================
# 3. CARGA Y PREPARACIÓN DE DATOS (una sola vez para todos los modelos)
# ===========================================================================

def load_and_prepare_data():
    """
    Carga los datos brutos, limpia, genera features y hace el split.
    Sobreescribe las rutas del YAML para que apunten a C:/Dev/TFM/.
    Devuelve df_train, df_val, df_test.
    """
    import yaml

    # Leer config y sobreescribir rutas
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = f.read().split("```", maxsplit=1)[0]   # Limpia posible fence YAML al final
    config = yaml.safe_load(raw)
    config["data"]["base_paths"] = LOCAL_DATA_PATHS
    config["data"]["temperature_daily_path"] = LOCAL_TEMP_PATH

    # Escribir config temporal con rutas corregidas
    tmp_config = PROJECT_ROOT / "configs" / "_local_eval_tmp.yaml"
    with open(tmp_config, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    print("\n[eval] Cargando datos...")
    df_ts, df_events = load_msd_data(tmp_config)
    tmp_config.unlink()   # Borra config temporal

    print("[eval] Limpiando datos...")
    df_clean = clean_timeseries(df_ts, df_events)
    del df_ts, df_events
    gc.collect()

    print("[eval] Generando features...")
    df_feat = create_features(df_clean)
    del df_clean
    gc.collect()

    # Convertir a float32 para ahorrar RAM
    cols_f64 = df_feat.select_dtypes(include=["float64"]).columns
    df_feat[cols_f64] = df_feat[cols_f64].astype("float32")

    print("[eval] Dividiendo en train/val/test...")
    df_train, df_val, df_test = split_chronological(df_feat)
    del df_feat
    gc.collect()

    print(f"[eval] Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    return df_train, df_val, df_test


# ===========================================================================
# 4. EVALUACIÓN DE UN MODELO
# ===========================================================================

def evaluate_one_model(
    horizon: int,
    variant: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    weights_stem: Optional[str] = None,
    result_key:   Optional[str] = None,
) -> dict:
    """
    Carga el checkpoint de un modelo, predice sobre test y devuelve
    métricas globales, por rangos, y los arrays y_true/y_pred en MGD.

    Parámetros:
      - horizon: horizonte de predicción (1, 3 o 6)
      - variant: "sinSF" o "conSF" (determina qué lista de features usar)
      - weights_stem: nombre base del checkpoint. Si None, se usa el patrón
        estándar "modelo_H{h}_{v}". Permite evaluar modelos con nombres
        no estándar (e.g. v2 experimentales) sin duplicar código.
      - result_key: identificador del modelo en los diccionarios de salida.
        Si None, se usa "H{h}_{v}" (el patrón estándar).
    """
    # Si no se especifica stem, construir el estándar a partir de horizon/variant
    default_key = f"H{horizon}_{variant}"
    model_key   = result_key   if result_key   is not None else default_key
    stem        = weights_stem if weights_stem is not None else f"modelo_{default_key}"

    weights_path = WEIGHTS_DIR / f"{stem}_weights.pt"
    norm_path    = WEIGHTS_DIR / f"{stem}_norm_params.json"

    print(f"\n[eval] === {model_key} (stem={stem}) ===")

    # Cargar norm_params desde JSON
    if not norm_path.exists():
        print(f"[eval] AVISO: no se encuentra {norm_path}. Saltando.")
        return {}
    with open(norm_path, "r") as f:
        norm_params = json.load(f)

    # Determinar qué features usar
    features = FEATURES_CON_SF if variant == "conSF" else FEATURES_SIN_SF

    # Normalizar splits
    # Para conSF: pasar lista sin stormflow a normalize_splits para evitar doble normalización
    features_for_norm = [f for f in features if f != TARGET_COL]
    df_tn, df_vn, df_tsn, norm_params_computed = normalize_splits(
        df_train, df_val, df_test, features_for_norm, TARGET_COL
    )
    # Usar norm_params del JSON (stats de train del entrenamiento original) en lugar de los
    # recalculados, para que la desnormalización sea consistente con los pesos guardados.
    norm_params_to_use = norm_params

    # Crear DataLoaders, recortamos train/val a 200 filas para no gastar RAM
    # (solo necesitamos el test_loader, pero create_dataloaders exige los 3 splits)
    df_tn_small = df_tn.iloc[:200].copy()
    df_vn_small = df_vn.iloc[:200].copy()
    del df_tn, df_vn
    gc.collect()

    _, _, test_loader = create_dataloaders(
        df_tn_small, df_vn_small, df_tsn,
        features, TARGET_COL, AUX_COL,
        seq_length=SEQ_LENGTH, horizon=horizon, batch_size=BATCH_SIZE,
    )
    del df_tn_small, df_vn_small, df_tsn
    gc.collect()

    # Cargar modelo
    if not weights_path.exists():
        print(f"[eval] AVISO: no se encuentra {weights_path}. Saltando.")
        return {}

    model = TwoStageTCN(n_features=len(features))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # Predecir
    y_pred_norm, y_real_norm = predict(model, test_loader, DEVICE)
    del model
    gc.collect()

    # Desnormalizar
    y_real_mgd = denormalize_target(y_real_norm, norm_params_to_use)
    y_pred_mgd = denormalize_target(y_pred_norm, norm_params_to_use)
    y_pred_mgd = np.clip(y_pred_mgd, 0, None)

    # Métricas
    g_metrics = global_metrics(y_real_mgd, y_pred_mgd)
    r_metrics = range_metrics(y_real_mgd, y_pred_mgd)

    print(f"  NSE={g_metrics['nse']:.3f} | RMSE={g_metrics['rmse']:.2f} MGD | "
          f"ErrPico={g_metrics['peak_err_pct']:+.1f}% | BiasBase={g_metrics['bias_base']:+.2f}")

    return {
        "global":   g_metrics,
        "ranges":   r_metrics,
        "y_real":   y_real_mgd,   # Guardamos para plots
        "y_pred":   y_pred_mgd,
    }


# ===========================================================================
# 5. PLOTS
# ===========================================================================

def plot_scatter(y_real: np.ndarray, y_pred: np.ndarray, model_key: str) -> None:
    """Scatter predicho vs real con línea 1:1."""
    fig, ax = plt.subplots(figsize=(7, 7))
    n = len(y_real)
    idx = np.random.choice(n, min(n, 40_000), replace=False)
    ax.scatter(y_real[idx], y_pred[idx], s=2, alpha=0.12, color="steelblue", rasterized=True)
    max_val = max(y_real.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 perfecto")
    ax.set_xlabel("Stormflow Real (MGD)")
    ax.set_ylabel("Stormflow Predicho (MGD)")
    ax.set_title(f"{model_key} - Predicho vs Real (test)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"scatter_{model_key}.png", dpi=120)
    plt.close()


def plot_top_peaks(y_real: np.ndarray, y_pred: np.ndarray, model_key: str, top_k: int = 3) -> None:
    """Serie temporal con zoom a los top_k mayores picos reales."""
    peak_indices = np.argsort(y_real)[-top_k:][::-1]
    for rank, peak_idx in enumerate(peak_indices, 1):
        window = 500
        start = max(0, peak_idx - window)
        end   = min(len(y_real), peak_idx + window)
        x     = np.arange(start, end)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, y_real[start:end], color="black",      lw=1.2, label="Real")
        ax.plot(x, y_pred[start:end], color="darkorange",  lw=1.2, ls="--", label="Predicho")
        ax.fill_between(x, y_real[start:end], y_pred[start:end], alpha=0.12, color="red")

        p_real = y_real[peak_idx]
        p_pred = y_pred[peak_idx]
        err    = (p_pred - p_real) / max(p_real, 1e-9) * 100

        ax.set_title(
            f"{model_key} - Pico #{rank}: Real {p_real:.1f} MGD -> Pred {p_pred:.1f} MGD ({err:+.0f}%)",
            fontweight="bold"
        )
        ax.set_xlabel("Índice temporal (cada unidad = 5 min)")
        ax.set_ylabel("Stormflow (MGD)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"peaks_{model_key}_rank{rank}.png", dpi=120)
        plt.close()


def plot_range_barplot(results: dict) -> None:
    """
    Para cada rango de magnitud, muestra NSE de los 6 modelos v1 en un bar plot.
    Útil para comparar qué variante funciona mejor en cada régimen.
    """
    models_sorted = [f"H{h}_{v}" for h in HORIZONS for v in VARIANTS]
    x = np.arange(len(models_sorted))
    width = 0.15

    fig, axes = plt.subplots(1, len(RANGE_LABELS), figsize=(20, 5), sharey=False)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    for ax, rng_label, color in zip(axes, RANGE_LABELS, colors):
        nse_values = []
        for mk in models_sorted:
            nse_val = results.get(mk, {}).get("ranges", {}).get(rng_label, {}).get("nse", float("nan"))
            nse_values.append(nse_val)
        bars = ax.bar(x, nse_values, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models_sorted, rotation=45, ha="right", fontsize=8)
        ax.set_title(rng_label, fontweight="bold")
        ax.set_ylabel("NSE")
        # Anotar valor encima/debajo de cada barra
        for bar, val in zip(bars, nse_values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.12,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7
                )

    fig.suptitle("NSE por rango de magnitud y modelo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nse_por_rango.png", dpi=130)
    plt.close()


def plot_predictability_curve(results: dict) -> None:
    """
    Curva de predicibilidad: NSE vs horizonte para variante SIN SF y CON SF.
    El gráfico más importante del TFM.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    horizon_labels = [f"H={h}\n({h*5} min)" for h in HORIZONS]

    for variant, color, marker in [("sinSF", "#2196F3", "o"), ("conSF", "#FF9800", "s")]:
        nse_vals = [results.get(f"H{h}_{variant}", {}).get("global", {}).get("nse", float("nan")) for h in HORIZONS]
        label = "Sin stormflow (rain-only)" if variant == "sinSF" else "Con stormflow (autoregresivo)"
        ax.plot(horizon_labels, nse_vals, marker=marker, color=color, lw=2, ms=8, label=label)
        for x_pos, val in enumerate(nse_vals):
            if not np.isnan(val):
                ax.annotate(f"{val:.3f}", (x_pos, val), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9, color=color)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5, label="NSE=0 (media como predictor)")
    ax.set_xlabel("Horizonte de predicción")
    ax.set_ylabel("NSE (Nash-Sutcliffe Efficiency)")
    ax.set_title("Curva de predicibilidad - NSE vs horizonte", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictability_curve.png", dpi=150)
    plt.close()
    print(f"[eval] Curva de predicibilidad guardada.")


def plot_extreme_events_analysis(y_real: np.ndarray, y_pred: np.ndarray, model_key: str) -> None:
    """
    Análisis de eventos extremos (>50 MGD): scatter y tabla mostrando
    error individual por evento. Permite detectar los 15 eventos sin lluvia.
    """
    ext_mask = y_real > 50.0
    if ext_mask.sum() == 0:
        return

    yt_ext = y_real[ext_mask]
    yp_ext = y_pred[ext_mask]
    err_pct = (yp_ext - yt_ext) / yt_ext * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter de extremos
    ax = axes[0]
    ax.scatter(yt_ext, yp_ext, s=40, alpha=0.7, color="crimson", edgecolors="black", lw=0.5)
    max_val = max(yt_ext.max(), yp_ext.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 perfecto")
    ax.set_xlabel("Real (MGD)")
    ax.set_ylabel("Predicho (MGD)")
    ax.set_title(f"{model_key} - Eventos extremos (>50 MGD)", fontweight="bold")
    ax.legend(fontsize=9)

    # Histograma de errores porcentuales
    ax = axes[1]
    ax.hist(err_pct, bins=20, color="crimson", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, ls="--", label="Error=0 (perfecto)")
    ax.axvline(np.median(err_pct), color="orange", lw=1.5, ls="--",
               label=f"Mediana: {np.median(err_pct):.1f}%")
    ax.set_xlabel("Error relativo (%)")
    ax.set_ylabel("N eventos")
    ax.set_title(f"Distribución del error en extremos\n(n={int(ext_mask.sum())})", fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle(f"Análisis eventos extremos - {model_key}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"extremos_{model_key}.png", dpi=130)
    plt.close()


# ===========================================================================
# 6. ANÁLISIS: EVENTOS EXTREMOS CON LLUVIA VS SIN LLUVIA
# ===========================================================================

def analyze_extreme_events_rain_split(df_test: pd.DataFrame, results: dict) -> None:
    """
    Identifica los eventos extremos (>50 MGD) y los clasifica en:
    - CON señal de lluvia en la ventana de entrada
    - SIN señal de lluvia (escorrentía retardada / deshielo)

    Calcula métricas separadas para cada grupo usando el mejor modelo (H1_sinSF).
    """
    model_key = "H1_sinSF"
    if model_key not in results or len(results[model_key]) == 0:
        print(f"[eval] Análisis de extremos: no hay resultados para {model_key}.")
        return

    y_real = results[model_key]["y_real"]
    y_pred = results[model_key]["y_pred"]

    ext_mask = y_real > 50.0
    n_ext = int(ext_mask.sum())
    print(f"\n[eval] Eventos extremos (>50 MGD) en test: {n_ext}")

    if n_ext == 0:
        return

    # Para detectar si hay señal de lluvia, necesitamos alinear el índice de extremos
    # con df_test. El test_loader usa seq_length=72, horizon=1, así que los primeros
    # seq_length+horizon-1 = 72 registros de df_test no tienen ventana.
    # El índice i del array y_real corresponde al registro (72 + i) de df_test (para H=1).
    OFFSET = SEQ_LENGTH + 1 - 1   # = 72 para H=1

    ext_indices = np.where(ext_mask)[0]  # Índices en el array y_real

    rain_cols = ["rain_in", "rain_sum_60m", "rain_sum_180m"]
    rain_cols_available = [c for c in rain_cols if c in df_test.columns]

    events_with_rain    = []
    events_without_rain = []

    for idx in ext_indices:
        df_idx = OFFSET + idx   # Índice correspondiente en df_test
        if df_idx >= len(df_test):
            continue

        # Ventana de historia: los 72 timesteps anteriores al target
        win_start = max(0, df_idx - SEQ_LENGTH)
        win_end   = df_idx
        window_df = df_test.iloc[win_start:win_end]

        # Hay señal de lluvia si alguna rain_sum_60m > 0 en la ventana
        has_rain = False
        if "rain_sum_60m" in window_df.columns:
            has_rain = bool((window_df["rain_sum_60m"] > 0).any())
        elif "rain_in" in window_df.columns:
            has_rain = bool((window_df["rain_in"] > 0).any())

        entry = {
            "idx":       int(idx),
            "real_mgd":  float(y_real[idx]),
            "pred_mgd":  float(y_pred[idx]),
            "err_pct":   float((y_pred[idx] - y_real[idx]) / y_real[idx] * 100),
            "has_rain":  has_rain,
        }
        if has_rain:
            events_with_rain.append(entry)
        else:
            events_without_rain.append(entry)

    print(f"  Eventos extremos CON lluvia en ventana:  {len(events_with_rain)}")
    print(f"  Eventos extremos SIN lluvia en ventana:  {len(events_without_rain)}")

    # Métricas separadas
    def _group_metrics(events: list, label: str) -> None:
        if not events:
            return
        yt = np.array([e["real_mgd"] for e in events])
        yp = np.array([e["pred_mgd"] for e in events])
        errs = np.array([e["err_pct"] for e in events])
        print(f"\n  [{label}] n={len(events)} | "
              f"NSE={_nse(yt, yp):.3f} | RMSE={_rmse(yt, yp):.1f} MGD | "
              f"Error mediano={np.median(errs):+.1f}%")

    _group_metrics(events_with_rain,    "Extremos CON lluvia")
    _group_metrics(events_without_rain, "Extremos SIN lluvia")

    # Guardar lista de eventos sin lluvia para el TFM
    output_path = PROJECT_ROOT / "outputs" / "data_analysis" / "extreme_events_no_rain.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": model_key,
            "total_extreme": n_ext,
            "with_rain":    events_with_rain,
            "without_rain": events_without_rain,
        }, f, indent=2)
    print(f"\n[eval] Análisis de extremos guardado en {output_path}")

    # Plot comparativo
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, events, label, color in [
        (axes[0], events_with_rain,    "Con señal de lluvia",     "#2196F3"),
        (axes[1], events_without_rain, "Sin señal de lluvia",  "#FF5252"),
    ]:
        if not events:
            ax.text(0.5, 0.5, "Sin muestras", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        yt = np.array([e["real_mgd"] for e in events])
        yp = np.array([e["pred_mgd"] for e in events])
        ax.scatter(yt, yp, s=60, color=color, edgecolors="black", lw=0.5, alpha=0.8)
        max_val = max(yt.max(), yp.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 perfecto")
        ax.set_xlabel("Real (MGD)")
        ax.set_ylabel("Predicho (MGD)")
        ax.set_title(f"Extremos {label}\n(n={len(events)}, NSE={_nse(yt, yp):.3f})", fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle(f"Eventos extremos >50 MGD - {model_key}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "extreme_rain_vs_norain.png", dpi=130)
    plt.close()
    print(f"[eval] Plot de extremos con/sin lluvia guardado.")


def _parse_horizon_from_model_key(model_key: str) -> int:
    """Extrae horizonte H desde keys tipo H1_sinSF."""
    try:
        # Separar prefijo Hx y convertir x a entero para calcular offset
        return int(model_key.split("_")[0].replace("H", ""))
    except (ValueError, IndexError):
        # Valor por defecto seguro para no interrumpir evaluacion completa
        return 1


def _extract_datetime_index(df_test: pd.DataFrame) -> pd.DatetimeIndex:
    """Obtiene indice temporal desde indice o columnas comunes de fecha."""
    if isinstance(df_test.index, pd.DatetimeIndex):
        # Si ya viene como DatetimeIndex, no hacemos transformaciones extra
        return df_test.index

    # Fallback robusto para distintos nombres de columna temporal
    for col in ["timestamp", "datetime", "date_time", "fecha_hora", "date"]:
        if col in df_test.columns:
            # Convertimos a datetime para alinear por tiempo entre modelos con distinto H
            return pd.to_datetime(df_test[col], errors="coerce")

    # Si no existe tiempo explicito, creamos uno sintetico para no romper graficos
    return pd.date_range(start="2000-01-01", periods=len(df_test), freq="5min")


def _build_prediction_series(
    df_test: pd.DataFrame,
    all_results: dict,
    use_pred: bool = True,
) -> Dict[str, pd.Series]:
    """
    Alinea arrays de cada modelo a timestamps de df_test usando:
    idx_df = seq_length + horizon - 1 + i
    """
    ts_index = _extract_datetime_index(df_test)
    series_by_model: Dict[str, pd.Series] = {}

    for model_key, model_data in all_results.items():
        if not model_data:
            # Saltamos modelos vacios para tolerar ejecuciones parciales
            continue

        horizon = _parse_horizon_from_model_key(model_key)
        offset = SEQ_LENGTH + horizon - 1
        raw_array = model_data["y_pred"] if use_pred else model_data["y_real"]
        valid_len = max(0, min(len(raw_array), len(df_test) - offset))
        if valid_len == 0:
            continue

        # Asociamos cada prediccion al timestamp correcto del test set
        aligned_times = ts_index[offset: offset + valid_len]
        aligned_vals = np.asarray(raw_array[:valid_len], dtype=float)
        series_by_model[model_key] = pd.Series(aligned_vals, index=aligned_times)

    return series_by_model


def _extract_events_from_is_event(df_test: pd.DataFrame) -> List[dict]:
    """Agrupa bloques consecutivos con is_event=True y resume cada evento."""
    if "is_event" not in df_test.columns or TARGET_COL not in df_test.columns:
        return []

    is_event = df_test["is_event"].fillna(False).astype(bool).to_numpy()
    timestamps = _extract_datetime_index(df_test)
    events: List[dict] = []
    i = 0

    while i < len(is_event):
        if not is_event[i]:
            i += 1
            continue

        # Inicio y fin del evento consecutivo
        start = i
        while i + 1 < len(is_event) and is_event[i + 1]:
            i += 1
        end = i

        # Pico real dentro del bloque para clasificar magnitud del evento
        seg = df_test.iloc[start: end + 1]
        peak_local = int(np.argmax(seg[TARGET_COL].to_numpy(dtype=float)))
        peak_idx = start + peak_local
        peak_val = float(df_test.iloc[peak_idx][TARGET_COL])
        rain_360_peak = float(df_test.iloc[peak_idx]["rain_sum_360m"]) if "rain_sum_360m" in df_test.columns else float("nan")

        events.append(
            {
                "start_idx": start,
                "end_idx": end,
                "peak_idx": peak_idx,
                "peak_val": peak_val,
                "rain_360_peak": rain_360_peak,
                "peak_time": timestamps[peak_idx],
            }
        )
        i += 1

    return events


def _choose_representative_events(events: List[dict], n_select: int = 2) -> List[dict]:
    """Elige eventos mas cercanos a la mediana de pico (representativos)."""
    if len(events) <= n_select:
        return events
    peaks = np.array([e["peak_val"] for e in events], dtype=float)
    median_peak = float(np.median(peaks))
    order = np.argsort(np.abs(peaks - median_peak))
    return [events[int(idx)] for idx in order[:n_select]]


def _season_from_month(month: int) -> str:
    """Mapea mes a estacion climatica simple."""
    if month in [12, 1, 2]:
        return "invierno"
    if month in [3, 4, 5]:
        return "primavera"
    if month in [6, 7, 8]:
        return "verano"
    return "otono"


def plot_hydrographs_by_category(df_test: pd.DataFrame, all_results: dict) -> None:
    """
    Plotea hidrogramas de eventos representativos por categoria:
    moderado, alto, extremo con lluvia.
    """
    # Obtenemos eventos continuos desde is_event para respetar definicion solicitada
    events = _extract_events_from_is_event(df_test)
    if not events:
        print("[eval] plot_hydrographs_by_category: no hay eventos disponibles.")
        return

    pred_series = _build_prediction_series(df_test, all_results, use_pred=True)
    ts_index = _extract_datetime_index(df_test)

    # Definimos categorias con reglas explicitas del requerimiento
    categories = {
        "moderado": [e for e in events if 5.0 <= e["peak_val"] < 20.0],
        "alto": [e for e in events if 20.0 <= e["peak_val"] < 50.0],
        "extremo_lluvia": [e for e in events if (e["peak_val"] > 50.0 and e["rain_360_peak"] > 0.1)],
    }

    for category, cat_events in categories.items():
        selected = _choose_representative_events(cat_events, n_select=2)
        for n_plot, event in enumerate(selected, start=1):
            # Extendemos +/-2 horas alrededor del evento (24 pasos de 5 min)
            pad_steps = 24
            start_idx = max(0, event["start_idx"] - pad_steps)
            end_idx = min(len(df_test) - 1, event["end_idx"] + pad_steps)
            win_df = df_test.iloc[start_idx: end_idx + 1]
            win_ts = ts_index[start_idx: end_idx + 1]

            # Eje temporal en horas desde inicio del evento real (no de la ventana)
            x_hours = ((np.arange(start_idx, end_idx + 1) - event["start_idx"]) * 5.0) / 60.0

            fig, ax_left = plt.subplots(figsize=(12, 5))
            ax_right = ax_left.twinx()

            # Curva real en negro grueso para priorizar referencia observada
            ax_left.plot(x_hours, win_df[TARGET_COL].to_numpy(dtype=float),
                         color="black", lw=2.4, label="Real")

            # Trazamos los 6 modelos alineando por timestamp (cada uno con su offset propio)
            for model_key in ["H1_sinSF", "H1_conSF", "H3_sinSF", "H3_conSF", "H6_sinSF", "H6_conSF"]:
                if model_key not in pred_series:
                    continue
                style = MODEL_PLOT_STYLE[model_key]
                aligned_pred = pred_series[model_key].reindex(win_ts).to_numpy(dtype=float)
                ax_left.plot(
                    x_hours,
                    aligned_pred,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    lw=1.7,
                    label=style["label"],
                )

            # Barras de lluvia en eje secundario invertido (arriba=0)
            rain_vals = win_df["rain_in"].to_numpy(dtype=float) if "rain_in" in win_df.columns else np.zeros(len(win_df))
            ax_right.bar(x_hours, rain_vals, width=(5.0 / 60.0) * 0.9, color="#9E9E9E", alpha=0.6, label="Rain in (in/5min)")
            rain_top = max(float(np.nanmax(rain_vals)), 0.1)
            ax_right.set_ylim(rain_top * 1.2, 0.0)

            peak_time_txt = pd.to_datetime(event["peak_time"]).strftime("%Y-%m-%d %H:%M")
            title = f"Evento {category} - {peak_time_txt} - Pico real: {event['peak_val']:.1f} MGD"
            ax_left.set_title(title, fontweight="bold")
            ax_left.set_xlabel("Horas desde inicio del evento (h)")
            ax_left.set_ylabel("Stormflow (MGD)")
            ax_right.set_ylabel("Lluvia incremental (in/5 min, invertido)")

            # Unificamos leyenda de ambos ejes y la ubicamos fuera del area de dibujo
            h1, l1 = ax_left.get_legend_handles_labels()
            h2, l2 = ax_right.get_legend_handles_labels()
            ax_left.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"hydrograph_{category}_{n_plot}.png", dpi=150, bbox_inches="tight")
            plt.close()

    print("[eval] plot_hydrographs_by_category: plots generados.")


def plot_temporal_ranges(df_test: pd.DataFrame, all_results: dict) -> None:
    """Plotea una ventana continua de 14 dias con maxima alternancia evento/calma."""
    if "is_event" not in df_test.columns or TARGET_COL not in df_test.columns:
        print("[eval] plot_temporal_ranges: faltan columnas is_event o stormflow_mgd.")
        return

    window = 4032  # 14 dias a 5 minutos
    if len(df_test) < window:
        window = min(len(df_test), 2016)  # fallback a 7 dias si el test fuera corto
    if window <= 10:
        return

    is_event = df_test["is_event"].fillna(False).astype(bool).to_numpy()
    transitions = (is_event[1:] != is_event[:-1]).astype(int)

    # Contamos transiciones en cada ventana con convolucion para hacerlo eficiente
    kernel = np.ones(window - 1, dtype=int)
    transition_counts = np.convolve(transitions, kernel, mode="valid")
    best_start = int(np.argmax(transition_counts))
    best_end = best_start + window

    # Si la mejor ventana no tiene ambos regimenes, buscamos alternativa que si cumpla
    for start in np.argsort(transition_counts)[::-1][:100]:
        end = int(start + window)
        win = is_event[int(start):end]
        # Contamos inicios de evento dentro de la ventana para exigir variedad real
        event_starts = int(np.sum((~win[:-1]) & win[1:])) + int(win[0])
        if win.any() and (~win).any() and event_starts >= 2:
            best_start = int(start)
            best_end = end
            break

    ts_index = _extract_datetime_index(df_test)
    win_df = df_test.iloc[best_start:best_end]
    win_ts = ts_index[best_start:best_end]
    pred_series = _build_prediction_series(df_test, all_results, use_pred=True)

    fig, ax_left = plt.subplots(figsize=(15, 5))
    ax_right = ax_left.twinx()

    # Serie real
    ax_left.plot(win_ts, win_df[TARGET_COL].to_numpy(dtype=float), color="black", lw=2.0, label="Real")

    # Solo H1 para evitar saturacion visual, como se solicito
    for model_key, color in [("H1_sinSF", "#2196F3"), ("H1_conSF", "#FF9800")]:
        if model_key in pred_series:
            y_aligned = pred_series[model_key].reindex(win_ts).to_numpy(dtype=float)
            ax_left.plot(win_ts, y_aligned, color=color, lw=1.5, label=model_key)

    rain_vals = win_df["rain_in"].to_numpy(dtype=float) if "rain_in" in win_df.columns else np.zeros(len(win_df))
    bar_width_days = (5.0 / (24.0 * 60.0)) * 0.9
    ax_right.bar(win_ts, rain_vals, width=bar_width_days, color="#B0B0B0", alpha=0.45, label="Rain in (in/5min)")
    rain_top = max(float(np.nanmax(rain_vals)), 0.1)
    ax_right.set_ylim(rain_top * 1.2, 0.0)

    start_txt = pd.to_datetime(win_ts.iloc[0]).strftime("%Y-%m-%d")
    end_txt = pd.to_datetime(win_ts.iloc[-1]).strftime("%Y-%m-%d")
    ax_left.set_title(f"Rango temporal continuo (~14 dias) con transiciones evento/calma: {start_txt} a {end_txt}", fontweight="bold")
    ax_left.set_xlabel("Fecha")
    ax_left.set_ylabel("Stormflow (MGD)")
    ax_right.set_ylabel("Lluvia incremental (in/5 min, invertido)")

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "temporal_range_1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[eval] plot_temporal_ranges: plot generado.")


def plot_scatter_all_models(all_results: dict) -> None:
    """Genera scatter 2x3 con los 6 modelos v1 y color por rango hidrologico."""
    order = [
        "H1_sinSF", "H3_sinSF", "H6_sinSF",
        "H1_conSF", "H3_conSF", "H6_conSF",
    ]
    range_styles = [
        ("Base", (None, 0.5), "#9E9E9E"),
        ("Leve", (0.5, 5.0), "#4CAF50"),
        ("Moderado", (5.0, 20.0), "#2196F3"),
        ("Alto", (20.0, 50.0), "#FF9800"),
        ("Extremo", (50.0, None), "#F44336"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, model_key in zip(axes.ravel(), order):
        if model_key not in all_results:
            ax.set_title(f"{model_key} (sin datos)")
            ax.axis("off")
            continue

        y_true = np.asarray(all_results[model_key]["y_real"], dtype=float)
        y_pred = np.asarray(all_results[model_key]["y_pred"], dtype=float)
        if y_true.size == 0:
            ax.set_title(f"{model_key} (vacio)")
            ax.axis("off")
            continue

        # Subsample uniforme para limitar peso del archivo sin perder estructura global
        n = y_true.size
        if n > 30_000:
            idx = np.random.choice(n, 30_000, replace=False)
            ys = y_true[idx]
            ps = y_pred[idx]
        else:
            ys = y_true
            ps = y_pred

        for label, (lo, hi), color in range_styles:
            if lo is None:
                mask = ys < hi
            elif hi is None:
                mask = ys >= lo
            else:
                mask = (ys >= lo) & (ys < hi)
            if mask.sum() == 0:
                continue
            ax.scatter(ys[mask], ps[mask], s=4, alpha=0.35, color=color, label=label, rasterized=True)

        lim_max = float(max(np.nanmax(ys), np.nanmax(ps))) * 1.05
        ax.plot([0, lim_max], [0, lim_max], color="black", lw=1.0)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_title(model_key, fontweight="bold")
        ax.set_xlabel("Stormflow real (MGD)")
        ax.set_ylabel("Stormflow predicho (MGD)")

        nse_val = float(all_results[model_key]["global"]["nse"])
        ax.text(
            0.03, 0.95, f"NSE={nse_val:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Scatter Predicho vs Real - 6 modelos v1 (color por rango de magnitud)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_all_models.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[eval] plot_scatter_all_models: plot generado.")


def plot_sf_comparison(df_test: pd.DataFrame, all_results: dict) -> None:
    """Compara directamente sinSF vs conSF en los mismos eventos moderados-altos."""
    events = _extract_events_from_is_event(df_test)
    if not events:
        print("[eval] plot_sf_comparison: no hay eventos para comparar.")
        return

    # Nos enfocamos en eventos moderados-altos solicitados (10-40 MGD)
    candidate_events = [e for e in events if 10.0 <= e["peak_val"] <= 40.0]
    if not candidate_events:
        print("[eval] plot_sf_comparison: no hay eventos entre 10 y 40 MGD.")
        return

    pred_series = _build_prediction_series(df_test, all_results, use_pred=True)
    needed_models = ["H1_sinSF", "H3_sinSF", "H1_conSF", "H3_conSF"]
    ts_index = _extract_datetime_index(df_test)

    valid_events: List[dict] = []
    for event in candidate_events:
        peak_ts = pd.to_datetime(event["peak_time"])
        # Exigimos disponibilidad de prediccion en timestamp del pico para comparar justo
        if all(m in pred_series and not np.isnan(float(pred_series[m].get(peak_ts, np.nan))) for m in needed_models):
            valid_events.append(event)

    if not valid_events:
        print("[eval] plot_sf_comparison: no hay eventos con prediccion en todas las variantes.")
        return

    selected = _choose_representative_events(valid_events, n_select=3)

    for n_plot, event in enumerate(selected, start=1):
        pad_steps = 24
        start_idx = max(0, event["start_idx"] - pad_steps)
        end_idx = min(len(df_test) - 1, event["end_idx"] + pad_steps)
        win_df = df_test.iloc[start_idx: end_idx + 1]
        win_ts = ts_index[start_idx: end_idx + 1]
        x_hours = ((np.arange(start_idx, end_idx + 1) - event["start_idx"]) * 5.0) / 60.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Panel izquierdo: modelos sin SF
        axes[0].plot(x_hours, win_df[TARGET_COL].to_numpy(dtype=float), color="black", lw=2.1, label="Real")
        for model_key in ["H1_sinSF", "H3_sinSF"]:
            if model_key in pred_series:
                style = MODEL_PLOT_STYLE[model_key]
                y_aligned = pred_series[model_key].reindex(win_ts).to_numpy(dtype=float)
                axes[0].plot(x_hours, y_aligned, color=style["color"], linestyle=style["linestyle"], lw=1.7, label=style["label"])
        axes[0].set_title("Sin stormflow", fontweight="bold")
        axes[0].set_xlabel("Horas desde inicio del evento (h)")
        axes[0].set_ylabel("Stormflow (MGD)")
        axes[0].legend(fontsize=9, loc="upper left")

        # Panel derecho: modelos con SF
        axes[1].plot(x_hours, win_df[TARGET_COL].to_numpy(dtype=float), color="black", lw=2.1, label="Real")
        for model_key in ["H1_conSF", "H3_conSF"]:
            if model_key in pred_series:
                style = MODEL_PLOT_STYLE[model_key]
                y_aligned = pred_series[model_key].reindex(win_ts).to_numpy(dtype=float)
                axes[1].plot(x_hours, y_aligned, color=style["color"], linestyle=style["linestyle"], lw=1.7, label=style["label"])
        axes[1].set_title("Con stormflow", fontweight="bold")
        axes[1].set_xlabel("Horas desde inicio del evento (h)")
        axes[1].legend(fontsize=9, loc="upper left")

        peak_time_txt = pd.to_datetime(event["peak_time"]).strftime("%Y-%m-%d %H:%M")
        fig.suptitle(f"Comparacion sin/con stormflow - {peak_time_txt} - Pico: {event['peak_val']:.1f} MGD", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"comparison_sf_vs_nosf_{n_plot}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("[eval] plot_sf_comparison: plots generados.")


def analyze_no_rain_events(df_test: pd.DataFrame, all_results: dict) -> None:
    """Detalla eventos extremos sin lluvia y evalua hipotesis de deshielo."""
    events = _extract_events_from_is_event(df_test)
    if not events:
        print("[eval] analyze_no_rain_events: no hay eventos detectados.")
        return

    # Seleccionamos extremos sin lluvia segun condicion solicitada
    no_rain_extreme = [e for e in events if (e["peak_val"] > 50.0 and e["rain_360_peak"] < 0.1)]
    if not no_rain_extreme:
        print("[eval] analyze_no_rain_events: no hay extremos sin lluvia.")
        return

    pred_series = _build_prediction_series(df_test, all_results, use_pred=True)
    h1_series: Optional[pd.Series] = pred_series.get("H1_sinSF")
    ts_index = _extract_datetime_index(df_test)

    rows = []
    for event in no_rain_extreme:
        peak_idx = event["peak_idx"]
        peak_ts = pd.to_datetime(ts_index[peak_idx])
        month = int(peak_ts.month)
        season = _season_from_month(month)
        temp_val = float(df_test.iloc[peak_idx]["temp_daily_f"]) if "temp_daily_f" in df_test.columns else float("nan")
        pred_h1 = float(h1_series.get(peak_ts, np.nan)) if h1_series is not None else float("nan")

        # Regla simple de hipotesis: posible deshielo en meses frios y temp sobre congelacion
        hypothesis = "posible deshielo" if (month in [12, 1, 2, 3] and temp_val > 32.0) else "sin evidencia clara"

        rows.append(
            {
                "peak_datetime": peak_ts.strftime("%Y-%m-%d %H:%M"),
                "month": month,
                "season": season,
                "temp_daily_f": temp_val,
                "peak_real_mgd": float(event["peak_val"]),
                "pred_h1_sinSF_mgd": pred_h1,
                "hypothesis": hypothesis,
            }
        )

    # Resumen por estacion para validar predominio invernal/primavera temprana
    season_counts = pd.Series([r["season"] for r in rows]).value_counts().to_dict()
    thaw_like_months = sum(1 for r in rows if r["month"] in [12, 1, 2, 3])
    thaw_ratio = float(thaw_like_months / max(len(rows), 1))

    output_payload = {
        "model_reference": "H1_sinSF",
        "n_extreme_no_rain_events": len(rows),
        "season_counts": season_counts,
        "winter_early_spring_ratio": thaw_ratio,
        "events": rows,
    }

    output_path = PROJECT_ROOT / "outputs" / "data_analysis" / "extreme_no_rain_detailed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n[eval] Tabla de extremos sin lluvia (H1_sinSF):")
    print("-" * 96)
    print(f"{'Fecha':<17} {'Pico(MGD)':>10} {'Temp(F)':>9} {'Mes':>5} {'Hipotesis':>20}")
    print("-" * 96)
    for row in rows:
        print(
            f"{row['peak_datetime']:<17} "
            f"{row['peak_real_mgd']:>10.1f} "
            f"{row['temp_daily_f']:>9.1f} "
            f"{row['month']:>5d} "
            f"{row['hypothesis']:>20}"
        )
    print("-" * 96)
    print(f"[eval] analyze_no_rain_events: detalle guardado en {output_path}")


# ===========================================================================
# 7. PLOTS DE COMPARACIÓN DIRECTA v1 vs v2
# ===========================================================================

def plot_v1_vs_v2_comparison(all_results: dict) -> None:
    """
    Para cada modelo extra con compare_against definido, genera plots de
    comparación directa contra su v1:
      - Hidrogramas de los top-3 picos reales con ambas predicciones superpuestas
      - Scatter 1:1 con ambos modelos en subplots lado a lado
      - Barras comparativas de NSE y Bias por rango de magnitud
      - Tabla de métricas clave en consola
    """
    for extra in EXTRA_MODELS:
        v2_key = extra["key"]
        v1_key = extra["compare_against"]

        # Verificar que ambos modelos se evaluaron con éxito
        if v1_key not in all_results or v2_key not in all_results:
            print(f"[eval] Comparación {v1_key} vs {v2_key}: falta alguno de los dos. Saltando.")
            continue
        if not all_results[v1_key] or not all_results[v2_key]:
            print(f"[eval] Comparación {v1_key} vs {v2_key}: resultados vacíos. Saltando.")
            continue

        y_real = all_results[v1_key]["y_real"]   # y_real es el mismo en v1 y v2
        y_pred_v1 = all_results[v1_key]["y_pred"]
        y_pred_v2 = all_results[v2_key]["y_pred"]

        # ----- PLOT 1: hidrogramas top-3 picos con v1 y v2 superpuestos -----
        peak_indices = np.argsort(y_real)[-3:][::-1]
        for rank, peak_idx in enumerate(peak_indices, 1):
            window = 500
            start = max(0, peak_idx - window)
            end   = min(len(y_real), peak_idx + window)
            x = np.arange(start, end)

            fig, ax = plt.subplots(figsize=(14, 4.5))
            ax.plot(x, y_real[start:end], color="black", lw=1.5, label="Real")
            ax.plot(x, y_pred_v1[start:end], color="#2196F3", lw=1.3, ls="--", label=f"{v1_key} (v1)")
            ax.plot(x, y_pred_v2[start:end], color="#F44336", lw=1.3, ls="--", label=f"{v2_key} (v2)")

            p_real = y_real[peak_idx]
            p_v1   = y_pred_v1[peak_idx]
            p_v2   = y_pred_v2[peak_idx]
            err_v1 = (p_v1 - p_real) / max(p_real, 1e-9) * 100
            err_v2 = (p_v2 - p_real) / max(p_real, 1e-9) * 100

            ax.set_title(
                f"v1 vs v2 - Pico #{rank}: Real {p_real:.1f} MGD | "
                f"v1: {p_v1:.1f} ({err_v1:+.0f}%) | v2: {p_v2:.1f} ({err_v2:+.0f}%)",
                fontweight="bold"
            )
            ax.set_xlabel("Índice temporal (cada unidad = 5 min)")
            ax.set_ylabel("Stormflow (MGD)")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"v1vs_v2_peaks_{v2_key}_rank{rank}.png", dpi=130)
            plt.close()

        # ----- PLOT 2: scatter 1:1 lado a lado -----
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
        n = len(y_real)
        idx = np.random.choice(n, min(n, 40_000), replace=False)
        max_val = max(y_real.max(), y_pred_v1.max(), y_pred_v2.max())

        nse_v1_global = all_results[v1_key]["global"]["nse"]
        nse_v2_global = all_results[v2_key]["global"]["nse"]

        for ax, y_pred, label, color, nse in [
            (axes[0], y_pred_v1, f"{v1_key} (v1)", "#2196F3", nse_v1_global),
            (axes[1], y_pred_v2, f"{v2_key} (v2)", "#F44336", nse_v2_global),
        ]:
            ax.scatter(y_real[idx], y_pred[idx], s=3, alpha=0.12, color=color, rasterized=True)
            ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 perfecto")
            ax.set_xlabel("Stormflow Real (MGD)")
            ax.set_ylabel("Stormflow Predicho (MGD)")
            ax.set_title(f"{label} | NSE={nse:.3f}", fontweight="bold")
            ax.legend(fontsize=9)
            ax.set_aspect("equal")

        fig.suptitle(f"Comparación scatter v1 vs v2 - {v2_key}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"v1vs_v2_scatter_{v2_key}.png", dpi=130)
        plt.close()

        # ----- PLOT 3: barras comparativas por rango de magnitud -----
        range_labels = RANGE_LABELS
        nse_v1 = [all_results[v1_key]["ranges"][r]["nse"] for r in range_labels]
        nse_v2 = [all_results[v2_key]["ranges"][r]["nse"] for r in range_labels]
        bias_v1 = [all_results[v1_key]["ranges"][r]["bias"] for r in range_labels]
        bias_v2 = [all_results[v2_key]["ranges"][r]["bias"] for r in range_labels]

        x = np.arange(len(range_labels))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NSE por rango
        ax = axes[0]
        ax.bar(x - width/2, nse_v1, width, label=f"{v1_key} (v1)", color="#2196F3", alpha=0.85)
        ax.bar(x + width/2, nse_v2, width, label=f"{v2_key} (v2)", color="#F44336", alpha=0.85)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(range_labels)
        ax.set_ylabel("NSE")
        ax.set_title("NSE por rango de magnitud", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Bias por rango
        ax = axes[1]
        ax.bar(x - width/2, bias_v1, width, label=f"{v1_key} (v1)", color="#2196F3", alpha=0.85)
        ax.bar(x + width/2, bias_v2, width, label=f"{v2_key} (v2)", color="#F44336", alpha=0.85)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(range_labels)
        ax.set_ylabel("Bias (MGD)")
        ax.set_title("Bias por rango de magnitud", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Comparación por rangos - {v2_key}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"v1vs_v2_ranges_{v2_key}.png", dpi=130)
        plt.close()

        # ----- Tabla en consola -----
        g1 = all_results[v1_key]["global"]
        g2 = all_results[v2_key]["global"]
        print(f"\n  Comparación {v1_key} vs {v2_key}:")
        print(f"  {'Métrica':<20} {'v1':>10} {'v2':>10} {'Delta':>10}")
        print(f"  {'-'*52}")
        for metric in ["nse", "rmse", "mae", "peak_err_pct", "bias_base", "bias_ext"]:
            v1_val = g1[metric]
            v2_val = g2[metric]
            delta  = v2_val - v1_val
            print(f"  {metric:<20} {v1_val:>10.3f} {v2_val:>10.3f} {delta:>+10.3f}")

    print("[eval] plot_v1_vs_v2_comparison: plots generados.")


# ===========================================================================
# 8. BARRIDA DE THRESHOLD DEL CLASIFICADOR
# ===========================================================================
# OBJETIVO: Encontrar el threshold óptimo del clasificador sin reentrenar.
# En vez de usar predict() que aplica threshold=0.3 fijo, hacemos el forward
# pass manualmente para extraer cls_prob y reg_value por separado, y luego
# probamos diferentes thresholds recalculando métricas cada vez.
# ===========================================================================

def threshold_sweep(
    horizon: int,
    variant: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict:
    """
    Hace una barrida de thresholds para un modelo específico.
    Devuelve dict con resultados por threshold + arrays de cls_prob y reg_value.
    """
    model_key = f"H{horizon}_{variant}"
    weights_path = WEIGHTS_DIR / f"modelo_{model_key}_weights.pt"
    norm_path    = WEIGHTS_DIR / f"modelo_{model_key}_norm_params.json"

    print(f"\n[sweep] === Barrida de threshold: {model_key} ===")

    # --- Cargar norm_params del JSON ---
    with open(norm_path, "r") as f:
        norm_params = json.load(f)

    # --- Preparar datos (misma lógica que evaluate_one_model) ---
    features = FEATURES_CON_SF if variant == "conSF" else FEATURES_SIN_SF
    features_for_norm = [f for f in features if f != TARGET_COL]

    df_tn, df_vn, df_tsn, _ = normalize_splits(
        df_train, df_val, df_test, features_for_norm, TARGET_COL
    )

    # Recortar train/val para ahorrar RAM (solo necesitamos test)
    df_tn_small = df_tn.iloc[:200].copy()
    df_vn_small = df_vn.iloc[:200].copy()
    del df_tn, df_vn
    gc.collect()

    _, _, test_loader = create_dataloaders(
        df_tn_small, df_vn_small, df_tsn,
        features, TARGET_COL, AUX_COL,
        seq_length=SEQ_LENGTH, horizon=horizon, batch_size=BATCH_SIZE,
    )
    del df_tn_small, df_vn_small, df_tsn
    gc.collect()

    # --- Cargar modelo ---
    model = TwoStageTCN(n_features=len(features))
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # --- Forward pass manual: extraer cls_prob y reg_value por separado ---
    all_cls_probs = []      # probabilidades del clasificador (0 a 1)
    all_reg_values = []     # magnitudes del regresor (escala normalizada)
    all_y_real = []         # targets reales (escala normalizada)

    with torch.no_grad():
        for batch in test_loader:
            # batch tiene 4 tensores: (X, y, weights, event)
            x_batch = batch[0].to(DEVICE)
            y_batch = batch[1]              # se queda en CPU

            # Forward pass: el modelo devuelve dict con 'cls_prob' y 'reg_value'
            output = model(x_batch)
            cls_prob  = output['cls_prob'].cpu().numpy().squeeze()   # (batch_size,)
            reg_value = output['reg_value'].cpu().numpy().squeeze()  # (batch_size,)

            all_cls_probs.append(cls_prob)
            all_reg_values.append(reg_value)
            all_y_real.append(y_batch.numpy().squeeze())

    del model
    gc.collect()

    # Concatenar todos los batches
    cls_probs     = np.concatenate(all_cls_probs)         # (N,)
    reg_values_norm = np.concatenate(all_reg_values)      # (N,)
    y_real_norm   = np.concatenate(all_y_real)             # (N,)

    # --- Desnormalizar a escala real (MGD) ---
    target_col = norm_params['target_col']
    target_mean = norm_params['mean'][target_col]
    target_std  = norm_params['std'][target_col]

    y_real_mgd = np.expm1(y_real_norm * target_std + target_mean)
    y_real_mgd = np.clip(y_real_mgd, 0, None)

    reg_values_mgd = np.expm1(reg_values_norm * target_std + target_mean)
    reg_values_mgd = np.clip(reg_values_mgd, 0, None)

    # --- Barrida de thresholds ---
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

    # Etiquetas reales de evento (mismo umbral que la loss: 0.5 MGD)
    real_events = (y_real_mgd > 0.5).astype(int)

    print(f"\n  Total muestras test: {len(cls_probs):,}")
    print(f"  Eventos reales (>0.5 MGD): {real_events.sum():,} / {len(real_events):,} "
          f"({real_events.mean()*100:.1f}%)")
    print(f"  Eventos reales > 10 MGD: {(y_real_mgd > 10).sum():,}")
    print(f"  Eventos reales > 50 MGD: {(y_real_mgd > 50).sum():,}")

    print(f"\n  {'Thresh':>7} | {'NSE':>7} | {'RMSE':>6} | {'MAE':>6} | "
          f"{'ErrPico':>9} | {'BiasBase':>9} | {'FP':>7} | {'FN>10':>5} | {'FN>50':>5}")
    print("  " + "-" * 95)

    sweep_results = {}

    for thresh in thresholds:
        # Aplicar switch duro con este threshold
        pred_mgd = np.where(cls_probs >= thresh, reg_values_mgd, 0.0)

        # Métricas globales
        nse  = _nse(y_real_mgd, pred_mgd)
        rmse = _rmse(y_real_mgd, pred_mgd)
        mae  = _mae(y_real_mgd, pred_mgd)

        # Error de pico
        peak_idx  = int(np.argmax(y_real_mgd))
        peak_real = float(y_real_mgd[peak_idx])
        peak_pred = float(pred_mgd[peak_idx])
        peak_err  = (peak_pred - peak_real) / max(peak_real, 1e-9) * 100

        # Bias en base
        base_mask = y_real_mgd < 0.5
        bias_base = float(np.mean(pred_mgd[base_mask] - y_real_mgd[base_mask]))

        # Falsos positivos y falsos negativos
        pred_events = (cls_probs >= thresh).astype(int)
        fp = int(((pred_events == 1) & (real_events == 0)).sum())

        fn_mask = (real_events == 1) & (pred_events == 0)
        fn_real = y_real_mgd[fn_mask]
        fn_gt10 = int((fn_real > 10).sum())
        fn_gt50 = int((fn_real > 50).sum())

        # Métricas por rango
        r_metrics = range_metrics(y_real_mgd, pred_mgd)

        sweep_results[thresh] = {
            "nse": nse, "rmse": rmse, "mae": mae,
            "peak_err_pct": peak_err, "bias_base": bias_base,
            "fp": fp, "fn_gt10": fn_gt10, "fn_gt50": fn_gt50,
            "peak_real": peak_real, "peak_pred": peak_pred,
            "ranges": r_metrics,
        }

        # Marcar el threshold actual (0.3) y el recomendado
        marker = " << actual" if abs(thresh - 0.3) < 0.01 else ""
        print(f"  {thresh:>7.2f} | {nse:>7.3f} | {rmse:>6.2f} | {mae:>6.2f} | "
              f"{peak_err:>+8.1f}% | {bias_base:>+9.3f} | {fp:>7,} | {fn_gt10:>5} | {fn_gt50:>5}"
              f"{marker}")

    # --- Guardar resultados ---
    output_path = PROJECT_ROOT / "outputs" / "data_analysis" / f"threshold_sweep_{model_key}.json"
    # Convertir claves float a string para JSON
    json_results = {str(k): v for k, v in sweep_results.items()}
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\n[sweep] Resultados guardados en {output_path}")

    # --- Plot: NSE y BiasBase vs threshold ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    thresh_list = sorted(sweep_results.keys())
    nse_list    = [sweep_results[t]["nse"] for t in thresh_list]
    bias_list   = [sweep_results[t]["bias_base"] for t in thresh_list]
    fn50_list   = [sweep_results[t]["fn_gt50"] for t in thresh_list]

    # Panel 1: NSE vs threshold
    ax = axes[0]
    ax.plot(thresh_list, nse_list, "o-", color="#2196F3", lw=2, ms=7)
    ax.axvline(0.3, color="red", ls="--", lw=1, alpha=0.7, label="Actual (0.3)")
    ax.set_xlabel("Threshold del clasificador")
    ax.set_ylabel("NSE")
    ax.set_title("NSE vs Threshold", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Bias baseflow vs threshold
    ax = axes[1]
    ax.plot(thresh_list, bias_list, "o-", color="#FF9800", lw=2, ms=7)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.axvline(0.3, color="red", ls="--", lw=1, alpha=0.7, label="Actual (0.3)")
    ax.set_xlabel("Threshold del clasificador")
    ax.set_ylabel("Bias baseflow (MGD)")
    ax.set_title("Bias en baseflow vs Threshold", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: FN>50 MGD vs threshold
    ax = axes[2]
    ax.plot(thresh_list, fn50_list, "o-", color="#F44336", lw=2, ms=7)
    ax.axvline(0.3, color="red", ls="--", lw=1, alpha=0.7, label="Actual (0.3)")
    ax.set_xlabel("Threshold del clasificador")
    ax.set_ylabel("Eventos >50 MGD perdidos")
    ax.set_title("Eventos extremos perdidos vs Threshold", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Barrida de threshold - {model_key}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"threshold_sweep_{model_key}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[sweep] Plot guardado en {OUTPUT_DIR / f'threshold_sweep_{model_key}.png'}")

    return sweep_results


# ===========================================================================
# 9. BUCLE PRINCIPAL
# ===========================================================================

def main():
    print("\n" + "=" * 70)
    print("  EVALUACIÓN LOCAL - MODELOS STORMFLOW (v1 + v2)")
    print("=" * 70)

    # Cargar datos una sola vez
    df_train, df_val, df_test = load_and_prepare_data()

    all_results = {}

    # Bucle principal: modelos v1 estándar (6 combinaciones)
    for horizon in HORIZONS:
        for variant in VARIANTS:
            model_key = f"H{horizon}_{variant}"
            result = evaluate_one_model(horizon, variant, df_train, df_val, df_test)
            if result:
                all_results[model_key] = result

    # Modelos extra (p.ej. v2 experimentales) evaluados aparte
    for extra in EXTRA_MODELS:
        result = evaluate_one_model(
            horizon=extra["horizon"],
            variant=extra["variant"],
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            weights_stem=extra["weights_stem"],
            result_key=extra["key"],
        )
        if result:
            all_results[extra["key"]] = result

    # Imprimir tabla resumen (incluye v1 y extras)
    print_metrics_table(all_results)

    # Guardar métricas en JSON (sin los arrays numpy que no son serializables)
    metrics_to_save = {}
    for mk, data in all_results.items():
        metrics_to_save[mk] = {
            "global": data["global"],
            "ranges": data["ranges"],
        }
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT, "w") as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    print(f"\n[eval] Métricas guardadas en {METRICS_OUTPUT}")

    # Generar plots por modelo
    print("\n[eval] Generando plots...")
    for mk, data in all_results.items():
        plot_scatter(data["y_real"], data["y_pred"], mk)
        plot_top_peaks(data["y_real"], data["y_pred"], mk)
        plot_extreme_events_analysis(data["y_real"], data["y_pred"], mk)

    # Plots comparativos globales
    plot_predictability_curve(all_results)
    plot_range_barplot(all_results)

    # Análisis especial: extremos con/sin lluvia
    analyze_extreme_events_rain_split(df_test, all_results)

    # Comparación directa v1 vs v2 (o cualquier modelo extra con compare_against)
    plot_v1_vs_v2_comparison(all_results)

    # Plots nuevos (scatter global, hidrogramas, rangos temporales, etc.)
    plot_hydrographs_by_category(df_test, all_results)
    plot_temporal_ranges(df_test, all_results)
    plot_scatter_all_models(all_results)
    plot_sf_comparison(df_test, all_results)
    analyze_no_rain_events(df_test, all_results)

    print(f"\n[eval] Todos los plots guardados en {OUTPUT_DIR}")

    # Barrida de threshold para los modelos H1
    print("\n" + "=" * 70)
    print("  BARRIDA DE THRESHOLD DEL CLASIFICADOR")
    print("=" * 70)
    threshold_sweep(1, "sinSF", df_train, df_val, df_test)
    threshold_sweep(1, "conSF", df_train, df_val, df_test)

    print("[eval] Evaluación completada.")


if __name__ == "__main__":
    main()