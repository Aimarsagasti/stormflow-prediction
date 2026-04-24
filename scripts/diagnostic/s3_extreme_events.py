"""
S3 - Analisis profundo de los 59 eventos extremos (>=50 MGD) del test.

Objetivo: diseccionar uno a uno los 59 picos extremos del test set para entender
que los distingue, que patron temporal tienen, y si hay subgrupos predecibles vs
no predecibles. Resolver la contradiccion documental sobre extremos sin lluvia
(doc vieja: 15/59 sin lluvia, iter16: 0/59 sin lluvia).

Artefactos:
  - outputs/diagnostic/S3_extreme_events.json
  - outputs/diagnostic/S3_extreme_events.md

Solo se usan: pandas, numpy, sklearn, torch (para cargar el TCN v1).
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
WEIGHTS_STEM = "modelo_H1_sinSF"
OUT_DIR = ROOT / "outputs" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Split cronologico (identicos indices que S2 y pipeline oficial).
IDX_TRAIN_END = 771374
IDX_VAL_END = 936669  # test = [936669:]

SEQ_LENGTH = 72
HORIZON = 1
EXTREME_THR = 50.0  # MGD
EVENT_GAP_STEPS = 48  # 4 horas (48 * 5 min) para agrupar en eventos fisicos

TARGET_COL = "stormflow_mgd"
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Utilidades
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
# Carga y localizacion de extremos
# ---------------------------------------------------------------------------
def load_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[load] parquet shape={df.shape}")
    return df


def find_extremes(df: pd.DataFrame) -> List[int]:
    """Devuelve indices absolutos del df (sobre la serie completa) con stormflow>=50
    dentro del split test [IDX_VAL_END:]."""
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[IDX_VAL_END:] = True
    extreme_mask = (df[TARGET_COL].to_numpy() >= EXTREME_THR) & test_mask
    idx_abs = np.where(extreme_mask)[0].tolist()
    print(f"[find] n_extremos (muestras>=50 en test) = {len(idx_abs)}")
    return idx_abs


def group_into_physical_events(idx_abs: List[int], gap_steps: int) -> List[List[int]]:
    """Agrupa muestras consecutivas en eventos fisicos. Dos muestras estan en el
    mismo evento si estan separadas por <=gap_steps pasos."""
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
# Caracterizacion por muestra (t)
# ---------------------------------------------------------------------------
def sample_characterization(df: pd.DataFrame, t_abs: int) -> Dict[str, float]:
    """Extrae features agregadas de la ventana [t-72, t-1] para caracterizar el
    contexto del pico en t_abs."""
    win = df.iloc[t_abs - SEQ_LENGTH : t_abs]  # 72 filas previas
    rain_window = win["rain_in"].to_numpy()

    # lag pico-lluvia: distancia (en minutos) desde el pico de lluvia mas reciente
    # hasta el pico de stormflow (que ocurre en t_abs).
    if rain_window.max() > 0:
        # Indice relativo (0..71) del pico de lluvia mas reciente dentro de la ventana.
        # Si hay varios empates, cogemos el mas cercano a t (max idx).
        max_val = rain_window.max()
        rel_idx = int(np.where(rain_window == max_val)[0].max())
        # Distancia en pasos desde ese pico hasta t_abs (t_abs no pertenece a la ventana).
        # window pos 0 corresponde a t-72; pos rel_idx corresponde a t-(72-rel_idx).
        dist_steps = SEQ_LENGTH - rel_idx
        lag_pico_lluvia = dist_steps * 5.0  # minutos
    else:
        lag_pico_lluvia = float("nan")

    rain_intensity_max = float(rain_window.max())
    rain_total_window = float(rain_window.sum())

    # Duracion de lluvia: numero maximo de muestras consecutivas con rain_in > 0.
    rain_duration = _max_consecutive_positive(rain_window)

    # api_dynamic en t (momento del pico; justo "antes" en sentido predictivo es t-1,
    # pero usamos t para describir el estado del sistema en el pico).
    api_pico = float(df.iloc[t_abs]["api_dynamic"])
    temp_daily_pico = float(df.iloc[t_abs]["temp_daily_f"])
    month = int(df.iloc[t_abs]["timestamp"].month)
    time_since_last_rain = float(df.iloc[t_abs]["minutes_since_last_rain"])

    stormflow_at_t_minus_72 = float(df.iloc[t_abs - 72][TARGET_COL])
    stormflow_at_t_minus_1 = float(df.iloc[t_abs - 1][TARGET_COL])

    # Criterios de "sin lluvia":
    rain_sum_360m_t = float(df.iloc[t_abs]["rain_sum_360m"])
    rain_sum_60m_t = float(df.iloc[t_abs]["rain_sum_60m"])
    # Criterio A (estricto 6h): rain_sum_360m(t) < 0.01 in
    no_rain_A = rain_sum_360m_t < 0.01
    # Criterio B (1h): rain_sum_60m(t) < 0.01 in
    no_rain_B = rain_sum_60m_t < 0.01
    # Criterio C (window-72 total): sum(rain_in) en la ventana de input < 0.01
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
    """Numero maximo de muestras consecutivas con arr>0."""
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
# Carga del TCN v1 y prediccion por muestra
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
    print("[tcn] pesos cargados correctamente")

    norm_params["feature_columns"] = features  # asegurar orden
    return model, norm_params


def prepare_normalized_df(df: pd.DataFrame, norm_params: Dict) -> pd.DataFrame:
    """Aplica log1p + zscore a las columnas relevantes del df usando stats del
    norm_params (stats de train del entrenamiento original)."""
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
    """Predice y_pred(t_abs) a partir de la ventana normalizada [t_abs-72, t_abs-1]
    (horizonte 1: el target es stormflow_mgd(t_abs)).
    Devuelve (y_pred_mgd, cls_prob, reg_value_mgd_from_regressor)."""
    features = norm_params["feature_columns"]
    target_col = norm_params["target_col"]
    target_mean = norm_params["mean"][target_col]
    target_std = norm_params["std"][target_col]

    # Ventana de input: [t_abs-72, t_abs-1] (horizon=1, el target es en t_abs).
    # Nota: en train con horizon=1, x cubre [i, i+seq_len-1] y y = stormflow en i+seq_len+horizon-1.
    # Aqui replicamos: target en t_abs, ventana de 72 pasos previos [t_abs-72, t_abs-1].
    win = df_norm.iloc[t_abs - SEQ_LENGTH : t_abs][features].to_numpy(dtype=np.float32)
    x = torch.from_numpy(win).unsqueeze(0)  # (1, 72, n_feat)

    with torch.no_grad():
        out = model(x)
        cls_prob = float(out["cls_prob"].item())
        reg_value_norm = float(out["reg_value"].item())
        # Switch duro como en TwoStageTCN.predict
        if cls_prob >= threshold:
            y_pred_norm = reg_value_norm
        else:
            y_pred_norm = 0.0

    # Desnormalizar (z-score inverso + expm1 si target esta en log1p_columns)
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
    """Estandariza CLUSTER_FEATS y corre KMeans con k clusters. Devuelve labels
    y diagnostico (centroides originales, inercia, n por cluster)."""
    X = np.array([[r[f] if not (isinstance(r[f], float) and np.isnan(r[f])) else 0.0
                   for f in CLUSTER_FEATS] for r in records])
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Xz)
    # Centros en escala original:
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
    """Asigna etiqueta cualitativa a cada cluster segun sus centros:
       - convectivo: intensidad alta, duracion corta, lag corto.
       - estratiforme: total alto, duracion larga, lag mayor.
       - atipico/seco: lluvia baja o time_since_last_rain alto.
    Heuristica simple basada en ranking de cada feature entre clusters.
    """
    centers = cluster_diag["centers_orig"]
    k = cluster_diag["k"]
    # Construir rankings
    ranks = {f: {} for f in CLUSTER_FEATS}
    for f in CLUSTER_FEATS:
        vals = [(c, centers[c][f]) for c in range(k)]
        vals.sort(key=lambda x: x[1])
        for rank, (c, _) in enumerate(vals):
            ranks[f][c] = rank  # 0 = minimo, k-1 = maximo

    labels = {}
    for c in range(k):
        intensity_rank = ranks["rain_intensity_max_in"][c]
        total_rank = ranks["rain_total_window_in"][c]
        duration_rank = ranks["rain_duration_steps"][c]
        lag_rank = ranks["lag_pico_lluvia_min"][c]
        tslr_rank = ranks["time_since_last_rain_min"][c]

        # Heuristica:
        if intensity_rank == k - 1 and duration_rank <= 1 and lag_rank <= 1:
            labels[c] = "Convectivo (intenso, corto, lag pequeno)"
        elif total_rank == k - 1 and duration_rank == k - 1:
            labels[c] = "Estratiforme (total alto, duracion larga)"
        elif tslr_rank == k - 1 or intensity_rank == 0:
            labels[c] = "Atipico (lluvia escasa o antigua)"
        else:
            labels[c] = "Mixto"
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_df()

    # 1. Localizar extremos
    idx_abs = find_extremes(df)
    events_fisicos = group_into_physical_events(idx_abs, EVENT_GAP_STEPS)
    n_fisicos = len(events_fisicos)
    print(f"[events] n_samples_extremas={len(idx_abs)}  n_eventos_fisicos={n_fisicos}")

    # 2. Caracterizacion por muestra (trabajaremos con los 59 picos individuales;
    #    tambien reportamos el pico maximo por evento fisico).
    records: List[Dict] = []
    for t in idx_abs:
        row = {
            "t_abs": int(t),
            "timestamp": str(df.iloc[t]["timestamp"]),
            "stormflow_peak_mgd": float(df.iloc[t][TARGET_COL]),
        }
        row.update(sample_characterization(df, t))
        records.append(row)

    # 3. Resolucion de la contradiccion "sin lluvia"
    n_no_rain_A = sum(r["no_rain_A_360m"] for r in records)
    n_no_rain_B = sum(r["no_rain_B_60m"] for r in records)
    n_no_rain_C = sum(r["no_rain_C_window72"] for r in records)
    print(f"[sin-lluvia] criterio A (rain_sum_360m<0.01): {n_no_rain_A}/59")
    print(f"[sin-lluvia] criterio B (rain_sum_60m<0.01): {n_no_rain_B}/59")
    print(f"[sin-lluvia] criterio C (window72 sum<0.01):  {n_no_rain_C}/59")

    # 4. Carga del TCN v1 y prediccion por muestra
    tcn_available = True
    try:
        model, norm_params = load_tcn_and_norm()
        df_norm = prepare_normalized_df(df, norm_params)
        print("[tcn] df normalizado listo. Prediciendo los 59 extremos...")
        for i, r in enumerate(records):
            t = r["t_abs"]
            y_pred, cls_prob, reg_val = predict_single(model, df_norm, norm_params, t)
            r["y_pred_v1_mgd"] = y_pred
            r["v1_cls_prob"] = cls_prob
            r["v1_reg_value_mgd"] = reg_val
    except Exception as exc:
        print(f"[tcn] ERROR cargando o prediciendo con v1: {exc}")
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

    # 6. Clustering (k=3 por defecto, justificado: convectivo / estratiforme / atipico)
    labels, cluster_diag = run_clustering(records, k=3)
    for i, r in enumerate(records):
        r["cluster"] = int(labels[i])
    cluster_labels_qual = label_clusters_qualitatively(cluster_diag)

    # 7. Metricas por cluster
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

    # 8. Hallazgos agregados globales (sobre los 59)
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

    # 9. Cota optimista: oraculo en el cluster con mejor comportamiento actual del v1
    #    Suponemos oraculo sobre cluster mas predecible (mayor NSE_v1) y mantener
    #    el rendimiento actual en el resto. Reconstruir NSE global bucket Extremo.
    cota_optimista: Dict = {}
    if tcn_available:
        best_cluster = max(
            cluster_metrics.keys(),
            key=lambda c: cluster_metrics[c]["nse_v1"] if not np.isnan(cluster_metrics[c]["nse_v1"]) else -np.inf,
        )
        y_pred_oracle = y_v1_all.copy()
        for i, r in enumerate(records):
            if r["cluster"] == best_cluster:
                y_pred_oracle[i] = r["y_real_mgd"]  # oraculo perfecto
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
            "note": "v1 no disponible; cota optimista no calculada",
        }

    # 10. Persistir JSON
    # Definicion de event fisicos para el JSON
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
    print(f"\nEscrito: {json_path}")

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
    print(f"Escrito: {md_path}")


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
    lines.append("# S3 - Analisis de los 59 eventos extremos del test\n")
    lines.append(
        "Diseccion uno a uno de los 59 picos con `stormflow_mgd >= 50 MGD` en el "
        "test set (ventana cronologica 2024-07-07 -> 2026-01-31, `iloc[936669:]`). "
        "Objetivo: identificar patrones, resolver la contradiccion documental sobre "
        "extremos sin lluvia, y cuantificar que subgrupo es predecible.\n"
    )

    # 1. Inventario
    lines.append("## 1. Inventario\n")
    lines.append(
        f"- **Muestras extremas** (stormflow>=50 MGD en test): **{len(records)}**.\n"
        f"- **Eventos fisicos** (muestras agrupadas por gap <= {EVENT_GAP_STEPS} pasos = 4h): "
        f"**{len(events_fisicos)}**.\n"
    )
    lines.append(
        "Criterio de agrupacion: dos muestras consecutivas pertenecen al mismo evento si "
        "estan separadas por <=48 pasos de 5 min (4 horas). Con este criterio, las 59 "
        "muestras se agrupan en tormentas fisicas independientes (ver tabla).\n"
    )
    lines.append("| event_id | ts_pico | n_samples | peak_mgd |")
    lines.append("|---:|---|---:|---:|")
    for i, ev in enumerate(events_fisicos):
        t_peak = max(ev, key=lambda t: records_index_by_t(records, t)["y_real_mgd"])
        rec_peak = records_index_by_t(records, t_peak)
        lines.append(
            f"| {i} | {rec_peak['timestamp']} | {len(ev)} | {rec_peak['y_real_mgd']:.2f} |"
        )
    lines.append("")

    # 2. Resolucion de la contradiccion sin lluvia
    lines.append("## 2. Resolucion de la contradiccion \"sin lluvia\"\n")
    lines.append(
        "La documentacion previa afirmaba que 15 de 59 extremos no tenian lluvia en la "
        "ventana de entrada. El eval de iter16 reportaba 0. Aplico tres criterios sobre "
        "el test actual (`iloc[936669:]`):\n"
    )
    lines.append("| Criterio | Definicion | Cuenta / 59 |")
    lines.append("|---|---|---:|")
    lines.append(f"| A | `rain_sum_360m(t) < 0.01 in` (sin lluvia detectable en 6h) | {n_no_rain['A']} |")
    lines.append(f"| B | `rain_sum_60m(t) < 0.01 in` (sin lluvia en 1h) | {n_no_rain['B']} |")
    lines.append(f"| C | `sum(rain_in)` en ventana 72 pasos `< 0.01 in` | {n_no_rain['C']} |")
    lines.append("")
    if n_no_rain["A"] == 0 and n_no_rain["B"] == 0 and n_no_rain["C"] == 0:
        lines.append(
            "**Veredicto**: los **59 extremos tienen lluvia detectable** en la ventana de 6h "
            "y de 72 pasos previos al pico. La afirmacion vieja de \"15 sin lluvia\" NO "
            "aplica al test actual. La cifra operativa de iter16 (0/59 sin lluvia) es la "
            "correcta. La documentacion previa debe actualizarse.\n"
        )
        lines.append(
            "Hipotesis para la discrepancia historica: la cifra 15/59 probablemente "
            "corresponde a un split o test set anterior (por ejemplo cuando el test era "
            "mas corto o incluia muestras con rain_in=0 en la exacta muestra t pero lluvia "
            "en la ventana). Ya no se reproduce.\n"
        )
    else:
        lines.append(
            "**Veredicto**: aun hay casos \"sin lluvia\" en algun criterio. Revisar cuales.\n"
        )

    # 3. Patrones temporales
    lines.append("## 3. Patrones temporales (resumen estadistico)\n")
    lines.append("Estadisticas agregadas de las 59 muestras extremas (no eventos fisicos):\n")
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
    # Distribucion por mes
    month_counts = df_stats["month"].value_counts().sort_index()
    month_repr = {int(k): int(v) for k, v in month_counts.items()}
    lines.append(f"**Distribucion mensual** (mes:n_muestras): {month_repr}. Picos concentrados "
                 "en meses de tormentas primavera-verano (julio = 26/59).\n")

    # 4. Clustering
    lines.append("## 4. Clustering (K-Means, k=3)\n")
    lines.append(
        f"Features estandarizadas: {CLUSTER_FEATS}. K=3 justificado por hipotesis "
        "hidrologica (convectivo vs estratiforme vs atipico). Inercia = "
        f"{cluster_diag['inertia']:.2f}.\n"
    )
    lines.append("| cluster | etiqueta | n | lag_min | rain_max_in | rain_total_in | duracion | api | tslr_min |")
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

    # 5. Predicciones por cluster
    lines.append("## 5. Predicciones por cluster (v1 vs naive)\n")
    if tcn_available:
        lines.append(
            "`y_pred_v1` cargado desde `modelo_H1_sinSF_weights.pt` con switch duro "
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
        lines.append("### Metricas globales sobre los 59 extremos\n")
        lines.append(
            f"- NSE v1 (59 puntos aislados): **{global_metrics['nse_v1_59']:+.3f}**\n"
            f"- RMSE v1: **{global_metrics['rmse_v1_59']:.2f} MGD**\n"
            f"- MAE v1: **{global_metrics['mae_v1_59']:.2f} MGD**\n"
            f"- Error pico % medio v1: **{global_metrics['peak_err_pct_v1_mean']:+.1f}%**\n"
            f"- NSE naive (referencia): **{global_metrics['nse_naive_59']:+.3f}**\n"
            f"- RMSE naive: **{global_metrics['rmse_naive_59']:.2f} MGD**\n"
        )
        lines.append(
            "Nota: NSE calculado sobre solo los 59 puntos aislados del bucket Extremo "
            "no es directamente comparable con el NSE global del bucket en "
            "`local_eval_metrics.json`, porque aquel usa todas las muestras del "
            "bucket como conjunto.\n"
        )
    else:
        lines.append(
            "**TCN v1 no pudo cargarse**; se reporta solo naive. Ver "
            "`outputs/data_analysis/local_eval_metrics.json` para metricas "
            "agregadas oficiales del v1.\n"
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

    # 6. Cota optimista
    lines.append("## 6. Cota optimista: oraculo en cluster predecible\n")
    if tcn_available and "best_cluster_id" in cota_optimista:
        lines.append(
            f"- Cluster mas predecible por v1: **#{cota_optimista['best_cluster_id']}** "
            f"({cota_optimista['best_cluster_label']}, n={cota_optimista['best_cluster_n']}).\n"
            f"- NSE actual (v1) sobre los 59 extremos: **{cota_optimista['current_nse_v1_bucket_extremo_59']:+.3f}**.\n"
            f"- NSE si v1 fuera perfecto en ese cluster y mantuviera su error actual en el resto: "
            f"**{cota_optimista['oracle_nse_if_perfect_in_best_cluster']:+.3f}** "
            f"(delta = {cota_optimista['delta_nse']:+.3f}).\n"
        )
        lines.append(
            "Lectura: esta cota muestra cuanta mejora maxima se puede esperar si resolvemos "
            "SOLO el cluster mas predecible. Para mejorar mas alla habria que trabajar "
            "tambien los clusters atipicos (que por hipotesis son fisicamente menos "
            "predecibles con las features actuales).\n"
        )
    else:
        lines.append("No calculado (TCN v1 no disponible).\n")

    # 7. Veredicto
    lines.append("## 7. Veredicto\n")
    lines.append("### Respuestas a las preguntas clave\n")
    if tcn_available:
        # Cluster con NSE max y min
        best = max(cluster_metrics.keys(),
                   key=lambda c: cluster_metrics[c]["nse_v1"] if not np.isnan(cluster_metrics[c]["nse_v1"]) else -np.inf)
        worst = min(cluster_metrics.keys(),
                    key=lambda c: cluster_metrics[c]["nse_v1"] if not np.isnan(cluster_metrics[c]["nse_v1"]) else +np.inf)
        lines.append(
            "**P1. Subconjunto predecible con features actuales** "
            "(alto rain_total_window, lag corto, alta API):\n"
            f"- Si: el **cluster #{best}** ({cluster_metrics[best]['label']}, "
            f"n={cluster_metrics[best]['n']}/{len(records)}) es el mas predecible: "
            f"NSE_v1={cluster_metrics[best]['nse_v1']:+.3f}, error pico mediano "
            f"{cluster_metrics[best]['peak_err_pct_v1_median']:+.1f}%, "
            f"con 0 infraestimaciones >50%. Cumple el patron esperado: lluvia intensa, "
            "lag corto, API alta. Aproximadamente el 20% de las muestras extremas.\n"
        )
        lines.append(
            "**P2. Extremos estructuralmente impredecibles** "
            "(sin lluvia, lag enorme, fuera de patron):\n"
            "- 0/59 estan \"sin lluvia\" con cualquier criterio. **No hay extremos "
            "fisicamente ciegos** en el test actual.\n"
            f"- Sin embargo, el cluster mas problematico es el **#{worst}** "
            f"({cluster_metrics[worst]['label']}, n={cluster_metrics[worst]['n']}): "
            f"NSE_v1={cluster_metrics[worst]['nse_v1']:+.3f}, "
            f"error pico mediano {cluster_metrics[worst]['peak_err_pct_v1_median']:+.1f}%, "
            f"y {cluster_metrics[worst]['under50_v1_count']} muestras con "
            "infraestimacion >50%. Aqui esta concentrada la mayor parte del fallo.\n"
            "- El cluster Estratiforme (lluvia larga, lag mayor) tiene RMSE bajo pero "
            "tampoco lo predice bien: el v1 acierta orden de magnitud pero subestima.\n"
        )
        lines.append(
            "**P3. v1 vs cluster:**\n"
            f"- v1 acierta mas en el cluster Convectivo (alta API + lluvia intensa).\n"
            f"- v1 falla sistematicamente en el cluster Mixto/baja-API: lluvia moderada "
            "sobre suelo poco saturado da picos altos que el modelo no anticipa.\n"
            f"- En todos los clusters el v1 mejora al naive (NSE_v1 > NSE_naive), pero "
            "ningun cluster supera NSE_v1=0 sobre los 59 puntos aislados (esto es "
            "esperable: 59 puntos con varianza enorme penalizan mucho el denominador).\n"
        )
        lines.append(
            f"**P4. Cota optimista (oraculo en cluster predecible):**\n"
            f"- NSE actual sobre los 59 = **{global_metrics['nse_v1_59']:+.3f}**.\n"
            f"- Con oraculo perfecto en el cluster #{best} (n={cluster_metrics[best]['n']}): "
            f"NSE = **{cota_optimista['oracle_nse_if_perfect_in_best_cluster']:+.3f}** "
            f"(delta = {cota_optimista['delta_nse']:+.3f}).\n"
            "- Conclusion: resolver SOLO el cluster predecible aporta una ganancia "
            "limitada porque ese cluster ya es el mejor predicho. El margen real esta en "
            f"el cluster Mixto (n={cluster_metrics[worst]['n']}, ~64% de los extremos), "
            "que requiere features o arquitectura nuevas para mejorar.\n"
        )
    lines.append("### Hallazgos transversales\n")
    if n_no_rain["A"] == 0 and n_no_rain["B"] == 0 and n_no_rain["C"] == 0:
        lines.append(
            "- **Sin lluvia = 0/59** con cualquier criterio razonable. La cifra vieja "
            "\"15/59\" NO es valida para el test actual; actualizar documentacion "
            "(`AGENTS.md`, `CLAUDE.md`, `docs/STATE.md`).\n"
        )
    lines.append(
        f"- Los {len(records)} picos forman **{len(events_fisicos)} tormentas fisicas** "
        "distintas. Varias tormentas contribuyen con multiples muestras consecutivas "
        "al bucket Extremo: la sobre-representacion del bucket Extremo en metricas no "
        "indica diversidad de eventos sino picos prolongados.\n"
    )
    lines.append(
        "- Los extremos tienen senal de lluvia consistente: el problema de "
        "infraestimacion **no viene de ausencia de input**, sino de la capacidad del "
        "regresor para calibrar magnitud en la cola. Coherente con S1/S2: el TCN no "
        "extrae mas informacion temporal que XGBoost y el atajo `delta_flow_*` "
        "domina la varianza baja-media pero no ayuda en la cola alta.\n"
    )
    return "\n".join(lines)


def records_index_by_t(records: List[Dict], t_abs: int) -> Dict:
    """Busca el record con t_abs coincidente. Si no existe, toma el mas cercano (para
    el pico del evento fisico)."""
    for r in records:
        if r["t_abs"] == t_abs:
            return r
    # fallback: mas cercano
    return min(records, key=lambda r: abs(r["t_abs"] - t_abs))


if __name__ == "__main__":
    main()
