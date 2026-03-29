"""Evaluation metrics for stormflow prediction."""

from __future__ import annotations  # Permite anotaciones modernas sin conflictos de version

from typing import Dict, Optional  # Define tipos explicitos para retorno de metricas y mascara opcional

import numpy as np  # Provee operaciones numericas vectorizadas para metricas

from src.pipeline.normalize import denormalize_target  # Reutiliza funcion oficial de desnormalizacion del pipeline


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE safely for possibly empty arrays."""
    if y_true.size == 0:  # Evita operaciones invalidas cuando no hay muestras en el subconjunto
        return float("nan")  # Retorna NaN para indicar metrica no definida
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))  # Calcula raiz del error cuadratico medio


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE safely for possibly empty arrays."""
    if y_true.size == 0:  # Evita media sobre arreglo vacio
        return float("nan")  # Retorna NaN cuando no hay datos para calcular MAE
    return float(np.mean(np.abs(y_pred - y_true)))  # Calcula error absoluto medio


def _safe_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency safely."""
    if y_true.size == 0:  # Evita calcular NSE en arreglos vacios
        return float("nan")  # Retorna NaN cuando no hay muestras disponibles
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)  # Calcula varianza total observada para denominador NSE
    if denominator <= 0:  # Controla casos degenerados sin varianza en la serie real
        return float("nan")  # Retorna NaN cuando NSE no es interpretable
    numerator = np.sum((y_true - y_pred) ** 2)  # Calcula suma de errores cuadrados del modelo
    return float(1.0 - (numerator / denominator))  # Retorna NSE donde 1.0 indica prediccion perfecta


def _bucket_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute severity bucket metrics using proposal-defined thresholds."""
    buckets = {  # Define cortes de severidad segun la propuesta del proyecto
        "base": (None, 0.5),  # Bucket base para valores inferiores a 0.5 MGD
        "pequeno": (0.5, 2.0),  # Bucket pequeno para rango 0.5 a 2 MGD
        "moderado": (2.0, 12.8),  # Bucket moderado para rango 2 a 12.8 MGD
        "grande": (12.8, 51.0),  # Bucket grande para rango 12.8 a 51 MGD
        "extremo": (51.0, None),  # Bucket extremo para valores mayores a 51 MGD
    }
    results: Dict[str, Dict[str, float]] = {}  # Prepara contenedor de metricas por bucket

    for bucket_name, (lower, upper) in buckets.items():  # Recorre buckets para calcular metricas por severidad
        if lower is None:  # Maneja bucket con solo limite superior
            mask = y_true < upper  # Selecciona muestras por debajo del umbral superior
        elif upper is None:  # Maneja bucket con solo limite inferior
            mask = y_true > lower  # Selecciona muestras por encima del umbral inferior
        else:  # Maneja bucket con limite inferior y superior
            mask = (y_true >= lower) & (y_true < upper)  # Selecciona muestras dentro del intervalo definido

        bucket_true = y_true[mask]  # Extrae target real del bucket actual
        bucket_pred = y_pred[mask]  # Extrae prediccion del bucket actual

        if bucket_true.size == 0:  # Evita calcular metricas cuando no hay muestras en bucket
            results[bucket_name] = {  # Llena con NaN y n=0 para mantener estructura consistente
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "n_samples": 0.0,
            }
            continue  # Pasa al siguiente bucket tras registrar valores vacios

        bias_value = float(np.mean(bucket_pred - bucket_true))  # Calcula sesgo medio (pred-real) del bucket
        results[bucket_name] = {  # Guarda metricas por bucket para reporte y retorno final
            "rmse": _safe_rmse(bucket_true, bucket_pred),  # Calcula RMSE en el bucket actual
            "mae": _safe_mae(bucket_true, bucket_pred),  # Calcula MAE en el bucket actual
            "bias": bias_value,  # Guarda sesgo medio para detectar infra/sobreestimacion sistematica
            "n_samples": float(bucket_true.size),  # Guarda cantidad de muestras del bucket
        }

    return results  # Devuelve diccionario completo de metricas por severidad


def evaluate_model(
    y_real_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    norm_params: Dict[str, object],
    is_event: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Evaluate normalized predictions with denormalization and severity breakdown."""
    y_real_norm = np.asarray(y_real_norm, dtype=float).reshape(-1)  # Convierte target normalizado a vector 1D consistente
    y_pred_norm = np.asarray(y_pred_norm, dtype=float).reshape(-1)  # Convierte prediccion normalizada a vector 1D consistente
    if y_real_norm.shape[0] != y_pred_norm.shape[0]:  # Valida mismo numero de muestras para comparar punto a punto
        raise ValueError("y_real_norm and y_pred_norm must have the same length")  # Lanza error claro si hay desalineacion

    y_real = denormalize_target(y_real_norm, norm_params)  # Lleva target real desde escala normalizada a unidades fisicas MGD
    y_pred = denormalize_target(y_pred_norm, norm_params)  # Lleva prediccion desde escala normalizada a unidades fisicas MGD
    y_real = np.clip(y_real, a_min=0.0, a_max=None)  # Refuerza restriccion fisica de stormflow no negativo en las metricas finales
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)  # Evita que una salida numericamente negativa distorsione las metricas fisicas

    global_nse = _safe_nse(y_real, y_pred)  # Calcula NSE global en unidades reales
    global_rmse = _safe_rmse(y_real, y_pred)  # Calcula RMSE global en MGD
    global_mae = _safe_mae(y_real, y_pred)  # Calcula MAE global en MGD

    if y_real.size > 0:  # Verifica existencia de datos para calcular pico global
        real_peak_value = float(np.max(y_real))  # Obtiene pico real global observado
        pred_peak_value = float(np.max(y_pred))  # Obtiene pico global predicho por el modelo
        peak_error_mgd = pred_peak_value - real_peak_value  # Calcula error signed de pico (pred - real)
        peak_abs_error_mgd = float(np.abs(peak_error_mgd))  # Calcula error absoluto de pico para reporte robusto
        peak_error_pct = float((peak_error_mgd / max(real_peak_value, 1e-12)) * 100.0)  # Calcula error relativo de pico en porcentaje
    else:  # Maneja caso vacio para mantener salida consistente
        real_peak_value = float("nan")  # Marca pico real como no definido cuando no hay muestras
        pred_peak_value = float("nan")  # Marca pico predicho como no definido cuando no hay muestras
        peak_error_mgd = float("nan")  # Marca error de pico como no definido
        peak_abs_error_mgd = float("nan")  # Marca error absoluto como no definido
        peak_error_pct = float("nan")  # Marca error porcentual como no definido

    event_nse = float("nan")  # Inicializa NSE en eventos como NaN por defecto cuando no se pasa mascara
    event_sample_count = 0.0  # Inicializa contador de muestras de evento para trazabilidad de la metrica
    if is_event is not None:  # Verifica si se solicito evaluacion restringida a eventos
        event_mask = np.asarray(is_event).reshape(-1).astype(bool)  # Convierte mascara de evento a bool 1D
        if event_mask.shape[0] != y_real.shape[0]:  # Valida alineacion entre mascara y arrays de prediccion/real
            raise ValueError("is_event must have the same length as y_real_norm and y_pred_norm")  # Lanza error claro ante longitudes incompatibles
        event_sample_count = float(event_mask.sum())  # Cuenta cuantas muestras del vector evaluado pertenecen a eventos reales
        event_nse = _safe_nse(y_real[event_mask], y_pred[event_mask])  # Calcula NSE solo en muestras marcadas como evento

    severity_metrics = _bucket_metrics(y_true=y_real, y_pred=y_pred)  # Calcula desglose de metricas por buckets de severidad

    metrics: Dict[str, object] = {  # Construye diccionario final de metricas para retorno estructurado
        "global": {  # Agrupa metricas globales principales para lectura directa
            "nse": global_nse,
            "rmse": global_rmse,
            "mae": global_mae,
            "peak_real_mgd": real_peak_value,
            "peak_pred_mgd": pred_peak_value,
            "peak_error_mgd": peak_error_mgd,
            "peak_abs_error_mgd": peak_abs_error_mgd,
            "peak_error_pct": peak_error_pct,
        },
        "event_only": {  # Agrupa metricas restringidas a eventos para evaluar desempeno operativo
            "nse": event_nse,
            "n_samples": event_sample_count,
        },
        "severity": severity_metrics,  # Incluye desglose completo por buckets de severidad
    }

    print("[metrics] === Metricas Globales ===")  # Encabezado de impresion para lectura de metrica global
    print(f"[metrics] NSE: {global_nse:.4f}")  # Imprime NSE global formateado
    print(f"[metrics] RMSE: {global_rmse:.4f} MGD")  # Imprime RMSE global en unidades fisicas
    print(f"[metrics] MAE: {global_mae:.4f} MGD")  # Imprime MAE global en unidades fisicas
    print(f"[metrics] Pico real: {real_peak_value:.4f} MGD")  # Imprime valor real del pico global
    print(f"[metrics] Pico predicho: {pred_peak_value:.4f} MGD")  # Imprime valor predicho del pico global
    print(f"[metrics] Error pico: {peak_error_mgd:.4f} MGD ({peak_error_pct:.2f}%)")  # Imprime error signed de pico en MGD y porcentaje
    if is_event is not None:  # Imprime metrica de eventos solo si se proporciono mascara
        print(f"[metrics] NSE (eventos): {event_nse:.4f} | muestras: {int(event_sample_count)}")  # Reporta NSE restringido al subconjunto de eventos y su cobertura

    print("[metrics] === Desglose por Severidad ===")  # Encabezado de impresion para buckets de severidad
    for bucket_name, bucket_values in severity_metrics.items():  # Recorre cada bucket para imprimir sus metricas
        print(  # Imprime resumen por bucket con todas las metricas solicitadas
            f"[metrics] {bucket_name}: n={int(bucket_values['n_samples'])} | "
            f"RMSE={bucket_values['rmse']:.4f} | "
            f"MAE={bucket_values['mae']:.4f} | "
            f"Bias={bucket_values['bias']:.4f}"
        )

    return metrics  # Devuelve diccionario completo para logging, reportes y persistencia
