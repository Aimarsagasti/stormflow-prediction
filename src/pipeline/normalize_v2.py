"""Normalizacion z-score + log1p opcional sobre el target. Para iter19 (TCN limpia).

Modulo independiente del pipeline antiguo `src/pipeline/normalize.py`, que tiene
un bug latente documentado en `outputs/diagnostic/S1_pipeline_audit.md` (BUG2:
si `stormflow_mgd` aparece en feature_columns Y target_col se normaliza dos
veces in-place). Aqui se evita por construccion: las features y el target se
mantienen separados, no se itera in-place sobre listas que puedan contener
duplicados, y la transformacion se aplica sobre copias numpy.

Diferencias respecto a `normalize.py`:
- Las features NUNCA reciben log1p. El antiguo aplicaba log1p a `rain_*`; aqui
  se confia en que la red maneja la asimetria con su no linealidad. Mantener
  el modulo simple y predecible.
- El target SI puede recibir log1p (parametro `log1p_target`). Esto es el
  unico tratamiento no lineal del modulo.
- Trabaja sobre arrays numpy en `transform`, no sobre DataFrames. Devuelve
  matrices listas para alimentar a la red.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def fit_scaler(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    log1p_target: bool = True,
) -> Dict:
    """Calcula estadisticas de normalizacion sobre TRAIN unicamente.

    Para cada feature en `feature_cols`: media y desviacion estandar poblacional
    (ddof=0). Para el target: si `log1p_target=True`, aplica log1p antes de
    calcular las estadisticas. Las desviaciones nulas se sustituyen por 1.0 para
    evitar division por cero en columnas constantes.

    Args:
        df_train: DataFrame con todas las columnas necesarias (features + target).
        feature_cols: lista de nombres de columnas de features (orden importa).
        target_col: nombre de la columna del target.
        log1p_target: si True, log1p(clip(y, 0, None)) antes de calcular mu/sigma.

    Returns:
        Dict con claves: feature_cols, target_col, feature_means, feature_stds,
        target_mean, target_std, log1p_target. Los arrays se devuelven como
        np.float32 para que sean compatibles directamente con tensores torch.
    """
    feature_cols = list(feature_cols)
    if target_col in feature_cols:
        # Aviso de seguridad: no permitir que el target aparezca en features.
        # Si esto se ignora, la red ve el target como input (leakage trivial).
        raise ValueError(
            f"target_col='{target_col}' aparece en feature_cols. Sacalo antes de llamar a fit_scaler."
        )

    feat_mat = df_train[feature_cols].to_numpy(dtype=np.float64, copy=False)
    feature_means = feat_mat.mean(axis=0)
    feature_stds = feat_mat.std(axis=0, ddof=0)
    feature_stds = np.where(feature_stds > 0.0, feature_stds, 1.0)

    target_arr = df_train[target_col].to_numpy(dtype=np.float64, copy=True)
    if log1p_target:
        target_arr = np.log1p(np.clip(target_arr, 0.0, None))
    target_mean = float(target_arr.mean())
    target_std_raw = float(target_arr.std(ddof=0))
    target_std = target_std_raw if target_std_raw > 0.0 else 1.0

    return {
        "feature_cols": feature_cols,
        "target_col": str(target_col),
        "feature_means": feature_means.astype(np.float32),
        "feature_stds": feature_stds.astype(np.float32),
        "target_mean": np.float32(target_mean),
        "target_std": np.float32(target_std),
        "log1p_target": bool(log1p_target),
    }


def transform(df: pd.DataFrame, scaler: Dict) -> Dict[str, np.ndarray]:
    """Aplica la transformacion del scaler a un DataFrame.

    Devuelve dict con:
        features: np.ndarray (N, F) en float32, normalizado con z-score.
        target:   np.ndarray (N,)   en float32, en espacio normalizado
                  (con log1p aplicado al inicio si procede).

    No modifica el DataFrame de entrada.
    """
    feature_cols: List[str] = scaler["feature_cols"]
    target_col: str = scaler["target_col"]

    feat = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    feat -= scaler["feature_means"]
    feat /= scaler["feature_stds"]

    target = df[target_col].to_numpy(dtype=np.float32, copy=True)
    if scaler["log1p_target"]:
        target = np.log1p(np.clip(target, 0.0, None)).astype(np.float32)
    target = (target - float(scaler["target_mean"])) / float(scaler["target_std"])
    target = target.astype(np.float32, copy=False)

    return {"features": feat, "target": target}


def inverse_transform_target(y_norm, scaler: Dict) -> np.ndarray:
    """Devuelve el target en MGD reales a partir de su version normalizada.

    Aplica primero la inversa del z-score y luego, si procede, expm1.
    Fuerza no-negatividad por consistencia fisica con el stormflow.
    """
    y_norm_arr = np.asarray(y_norm, dtype=np.float64)
    y = y_norm_arr * float(scaler["target_std"]) + float(scaler["target_mean"])
    if scaler["log1p_target"]:
        y = np.expm1(y)
    return np.clip(y, 0.0, None)
