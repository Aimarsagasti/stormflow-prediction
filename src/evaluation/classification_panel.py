"""Panel de evaluacion para clasificadores binarios de alerta en iter18.

El panel esta pensado para responder la pregunta operativa del MSD:
"si el modelo emite una alerta, con que recall, precision y anticipacion
real lo hace?".

La salida es JSON-serializable para poder persistir resultados completos,
graficarlos despues y compararlos entre variantes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Definicion fija del gap operativo entre eventos positivos independientes.
DEFAULT_EVENT_GAP_MINUTES = 240.0


def _to_numpy_int(y_true_bin: Union[np.ndarray, Sequence[int]]) -> np.ndarray:
    """Normaliza el target binario a `int32` para metrica y serializacion."""
    return np.asarray(y_true_bin, dtype=np.int32).reshape(-1)


def _to_numpy_float(y_prob: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Normaliza las probabilidades a `float64` para calculos estables."""
    return np.asarray(y_prob, dtype=float).reshape(-1)


def _safe_metric(metric_fn, *args: Any) -> float:
    """Ejecuta una metrica y devuelve NaN si la definicion no aplica."""
    try:
        return float(metric_fn(*args))
    except ValueError:
        return float("nan")


def _class_distribution(y_true_bin: np.ndarray) -> Dict[str, float]:
    """Resume el balance de clases para interpretar el resto del panel."""
    n_total = int(y_true_bin.size)
    n_positives = int(np.sum(y_true_bin == 1))
    n_negatives = int(np.sum(y_true_bin == 0))
    prevalence = float(n_positives / n_total) if n_total > 0 else float("nan")
    return {
        "n_total": n_total,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "prevalence": prevalence,
    }


def _metrics_at_threshold(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calcula metricas de clasificacion y matriz de confusion a un umbral."""
    # Binarizamos usando `>=` para mantener consistencia con el notebook.
    y_pred_bin = (y_prob >= float(threshold)).astype(np.int32, copy=False)
    # Contamos las cuatro celdas de la confusion matrix manualmente para que
    # el JSON no dependa de otra API externa.
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    return {
        "threshold": float(threshold),
        # `zero_division=0` evita warnings cuando no hay positivos predichos.
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "accuracy": _safe_metric(accuracy_score, y_true_bin, y_pred_bin),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _extract_positive_events(
    y_true_bin: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex],
    gap_minutes: float = DEFAULT_EVENT_GAP_MINUTES,
) -> list[Dict[str, int]]:
    """Segmenta intervalos positivos usando continuidad + gap temporal minimo."""
    positive_idx = np.where(y_true_bin == 1)[0]
    if positive_idx.size == 0:
        return []

    # Si no hay timestamps, solo podemos separar por continuidad de indices.
    if timestamps is None:
        events: list[Dict[str, int]] = []
        start = int(positive_idx[0])
        previous = int(positive_idx[0])
        for current in positive_idx[1:]:
            if int(current) != previous + 1:
                events.append({"start": start, "end": previous})
                start = int(current)
            previous = int(current)
        events.append({"start": start, "end": previous})
        return events

    # Convertimos a nanosegundos para comparar gaps sin perder precision.
    ts_ns = timestamps.to_numpy().astype("datetime64[ns]").astype("int64")
    gap_ns = int(gap_minutes * 60.0 * 1_000_000_000)
    events = []
    start = int(positive_idx[0])
    previous = int(positive_idx[0])
    for current_raw in positive_idx[1:]:
        current = int(current_raw)
        # Cortamos evento si hay salto de indices o si el tiempo supera el gap.
        if current != previous + 1 or (ts_ns[current] - ts_ns[previous]) > gap_ns:
            events.append({"start": start, "end": previous})
            start = current
        previous = current
    events.append({"start": start, "end": previous})
    return events


def _resolve_raw_block(
    y_true_raw: Optional[Union[np.ndarray, Sequence[float], Dict[str, Any]]],
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Extrae serie cruda y umbral MGD desde un bloque flexible.

    Para conservar la firma publica pedida, `y_true_raw` puede ser:
    - `None`
    - array/sequence con la serie cruda
    - dict con claves `values` y `threshold_mgd`
    """
    if y_true_raw is None:
        return None, None
    if isinstance(y_true_raw, dict):
        raw_values = np.asarray(y_true_raw.get("values"), dtype=float).reshape(-1)
        threshold_mgd = y_true_raw.get("threshold_mgd")
        return raw_values, float(threshold_mgd) if threshold_mgd is not None else None
    return np.asarray(y_true_raw, dtype=float).reshape(-1), None


def _lead_time_block(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex],
    y_true_raw: Optional[Union[np.ndarray, Sequence[float], Dict[str, Any]]],
    threshold_operational: Optional[float],
    gap_minutes: float = DEFAULT_EVENT_GAP_MINUTES,
) -> Dict[str, Any]:
    """Calcula lead time por evento positivo si hay datos suficientes."""
    raw_values, threshold_mgd = _resolve_raw_block(y_true_raw)
    if (
        timestamps is None
        or raw_values is None
        or threshold_mgd is None
        or threshold_operational is None
    ):
        return {
            "available": False,
            "threshold_operational": float(threshold_operational)
            if threshold_operational is not None
            else None,
            "threshold_mgd": float(threshold_mgd) if threshold_mgd is not None else None,
            "n_positive_events": 0,
            "n_events_with_alert": 0,
            "mean_minutes": float("nan"),
            "median_minutes": float("nan"),
            "p10_minutes": float("nan"),
            "p90_minutes": float("nan"),
            "per_event": [],
        }

    # Extraemos eventos positivos de la etiqueta binaria ya alineada en t.
    events = _extract_positive_events(y_true_bin, timestamps, gap_minutes=gap_minutes)
    ts_ns = timestamps.to_numpy().astype("datetime64[ns]").astype("int64")
    per_event: list[Dict[str, Any]] = []

    for event_id, event in enumerate(events):
        start = int(event["start"])
        end = int(event["end"])
        event_slice = slice(start, end + 1)

        # Buscamos el primer instante en que el modelo supera el umbral operativo.
        alert_rel_idx = np.where(y_prob[event_slice] >= float(threshold_operational))[0]
        # Buscamos el primer instante en que el stormflow real actual supera U.
        exceed_rel_idx = np.where(raw_values[event_slice] >= float(threshold_mgd))[0]

        if exceed_rel_idx.size == 0:
            # Si esto ocurre, el evento positivo no contiene el rebasamiento real
            # esperado y conviene dejarlo documentado para diagnostico.
            per_event.append(
                {
                    "event_id": event_id,
                    "start_index": start,
                    "end_index": end,
                    "detected": False,
                    "has_real_exceedance": False,
                    "lead_time_minutes": None,
                }
            )
            continue

        exceed_idx = start + int(exceed_rel_idx[0])
        if alert_rel_idx.size == 0:
            per_event.append(
                {
                    "event_id": event_id,
                    "start_index": start,
                    "end_index": end,
                    "detected": False,
                    "has_real_exceedance": True,
                    "first_exceedance_ts": str(timestamps[exceed_idx]),
                    "lead_time_minutes": None,
                }
            )
            continue

        alert_idx = start + int(alert_rel_idx[0])
        # Lead time positivo significa alerta antes del rebasamiento.
        lead_time_minutes = float((ts_ns[exceed_idx] - ts_ns[alert_idx]) / (60.0 * 1_000_000_000))
        per_event.append(
            {
                "event_id": event_id,
                "start_index": start,
                "end_index": end,
                "detected": True,
                "has_real_exceedance": True,
                "first_alert_ts": str(timestamps[alert_idx]),
                "first_exceedance_ts": str(timestamps[exceed_idx]),
                "lead_time_minutes": lead_time_minutes,
            }
        )

    # Resumimos solo eventos que efectivamente generaron alerta y excedencia real.
    detected_times = [
        float(row["lead_time_minutes"])
        for row in per_event
        if row.get("lead_time_minutes") is not None
    ]
    if detected_times:
        lead_array = np.asarray(detected_times, dtype=float)
        return {
            "available": True,
            "threshold_operational": float(threshold_operational),
            "threshold_mgd": float(threshold_mgd),
            "n_positive_events": int(len(events)),
            "n_events_with_alert": int(len(detected_times)),
            "mean_minutes": float(np.mean(lead_array)),
            "median_minutes": float(np.median(lead_array)),
            "p10_minutes": float(np.percentile(lead_array, 10)),
            "p90_minutes": float(np.percentile(lead_array, 90)),
            "per_event": per_event,
        }

    return {
        "available": True,
        "threshold_operational": float(threshold_operational),
        "threshold_mgd": float(threshold_mgd),
        "n_positive_events": int(len(events)),
        "n_events_with_alert": 0,
        "mean_minutes": float("nan"),
        "median_minutes": float("nan"),
        "p10_minutes": float("nan"),
        "p90_minutes": float("nan"),
        "per_event": per_event,
    }


def _sample_pr_curve(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 20,
) -> list[Dict[str, float]]:
    """Muestrea la curva precision-recall para graficarla sin inflar el JSON."""
    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_prob)
    if thresholds.size == 0:
        return []

    # `precision_recall_curve` devuelve una observacion extra sin umbral; por
    # eso alineamos precision/recall con `thresholds` descartando el ultimo punto.
    precision = precision[:-1]
    recall = recall[:-1]
    sample_count = min(int(n_points), int(thresholds.size))
    sample_idx = np.linspace(0, thresholds.size - 1, sample_count, dtype=int)
    sample_idx = np.unique(sample_idx)

    return [
        {
            "threshold": float(thresholds[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
        }
        for index in sample_idx
    ]


def _calibration_deciles(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[Dict[str, float]]:
    """Construye una tabla de calibracion por cuantiles de probabilidad."""
    frame = pd.DataFrame({"y_true": y_true_bin, "y_prob": y_prob})
    try:
        # `qcut` reparte por masa de probabilidad y refleja mejor calibracion
        # cuando el desbalance es severo.
        bins = pd.qcut(frame["y_prob"], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.Series(["all"] * len(frame))

    grouped = frame.groupby(bins, observed=False)
    rows: list[Dict[str, float]] = []
    for bin_id, (_, group) in enumerate(grouped):
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(len(group)),
                "pred_mean": float(group["y_prob"].mean()),
                "observed_freq": float(group["y_true"].mean()),
                "pred_min": float(group["y_prob"].min()),
                "pred_max": float(group["y_prob"].max()),
            }
        )
    return rows


def evaluate_classification_panel(
    y_true_bin: Union[np.ndarray, Sequence[int]],
    y_prob: Union[np.ndarray, Sequence[float]],
    timestamps: Optional[Union[pd.DatetimeIndex, pd.Series, np.ndarray, Sequence]] = None,
    y_true_raw: Optional[Union[np.ndarray, Sequence[float], Dict[str, Any]]] = None,
    threshold_default: float = 0.5,
    threshold_operational: Optional[float] = None,
) -> Dict[str, Any]:
    """Evalua un clasificador binario y devuelve un panel JSON-serializable."""
    y_true_bin_arr = _to_numpy_int(y_true_bin)
    y_prob_arr = _to_numpy_float(y_prob)
    if y_true_bin_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("`y_true_bin` y `y_prob` deben tener la misma longitud.")

    # Normalizamos timestamps a `DatetimeIndex` para operar siempre igual.
    ts_index: Optional[pd.DatetimeIndex]
    if timestamps is None:
        ts_index = None
    else:
        ts_index = pd.to_datetime(pd.Index(timestamps))
        if len(ts_index) != len(y_true_bin_arr):
            raise ValueError("`timestamps` debe tener la misma longitud que `y_true_bin`.")

    # Si no se especifica umbral operativo, reutilizamos el default para que
    # el panel siga siendo consistente y no deje campos vacios.
    op_threshold = float(threshold_default if threshold_operational is None else threshold_operational)

    distribution = _class_distribution(y_true_bin_arr)
    default_metrics = _metrics_at_threshold(y_true_bin_arr, y_prob_arr, float(threshold_default))
    operational_metrics = _metrics_at_threshold(y_true_bin_arr, y_prob_arr, op_threshold)

    threshold_free = {
        "roc_auc": _safe_metric(roc_auc_score, y_true_bin_arr, y_prob_arr),
        "auc_pr": _safe_metric(average_precision_score, y_true_bin_arr, y_prob_arr),
        "brier_score": _safe_metric(brier_score_loss, y_true_bin_arr, y_prob_arr),
    }

    lead_time = _lead_time_block(
        y_true_bin=y_true_bin_arr,
        y_prob=y_prob_arr,
        timestamps=ts_index,
        y_true_raw=y_true_raw,
        threshold_operational=op_threshold,
        gap_minutes=DEFAULT_EVENT_GAP_MINUTES,
    )

    return {
        "class_distribution": distribution,
        "default_threshold_metrics": default_metrics,
        "operational_threshold_metrics": operational_metrics,
        "threshold_free_metrics": threshold_free,
        "lead_time": lead_time,
        "pr_curve": _sample_pr_curve(y_true_bin_arr, y_prob_arr, n_points=20),
        "calibration": _calibration_deciles(y_true_bin_arr, y_prob_arr, n_bins=10),
    }
