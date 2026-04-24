"""XGBoost baseline con lags explicitos del target para iter17.

Modelo principal del sistema nuevo tras el diagnostico (ver
`outputs/diagnostic/DIAGNOSTIC_REPORT.md`): XGBoost regresivo entrenado
sobre los 12 lags del target (`stormflow_mgd[t-0..t-11]`) mas 10 features
exogenas reducidas derivadas de S5. Horizonte h >= 1.

El modulo es independiente del pipeline de normalizacion del TCN:
- Trabaja con valores en MGD reales (sin log1p, sin z-score).
- No usa `src/pipeline/normalize.py` ni `src/models/loss.py`.
- El split y los indices alineados son identicos a
  `scripts/diagnostic/s2_baselines.py` para que las cifras comparen directo.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Constantes del split y features - clonadas de scripts/diagnostic/s2_baselines.py
# para evitar introducir una dependencia de scripts/ desde src/.
# ---------------------------------------------------------------------------

IDX_TRAIN_END = 771374
IDX_VAL_END = 936669
SEQ_LENGTH = 72
TARGET_COL = "stormflow_mgd"

# Subconjunto reducido de features tras S5 (orden por permutation importance).
# Justificacion: S5 demostro que estas 10 features capturan toda la senal
# exogena util (PI > 0), y que las otras 10 del set original son ruido o
# redundancia. Mantener el orden del reporte para trazabilidad.
FEATURES_10: List[str] = [
    "api_dynamic",
    "rain_sum_360m",
    "rain_sum_120m",
    "rain_sum_15m",
    "temp_daily_f",
    "hour_sin",
    "minutes_since_last_rain",
    "delta_rain_10m",
    "delta_rain_30m",
    "rain_sum_30m",
]

# Punto de partida de hiperparametros del reporte (§7.2). early_stopping_rounds=20
# se pasa via fit() en XGBoost >= 1.6 usando callbacks; aqui se guarda como clave
# separada y la funcion de entrenamiento la aplica a traves de eval_set.
DEFAULT_XGB_PARAMS: Dict = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
DEFAULT_EARLY_STOPPING_ROUNDS = 20


# ---------------------------------------------------------------------------
# Split cronologico e indices alineados (identicos a S2)
# ---------------------------------------------------------------------------

def aligned_indices(
    split_start: int,
    split_end: int,
    horizon: int,
    total_len: int,
    seq_length: int = SEQ_LENGTH,
) -> np.ndarray:
    """Indices absolutos t validos para un split.

    Reglas:
    - existe ventana previa completa de `seq_length` pasos (t >= split_start + seq_length)
    - el origen t cae dentro de [split_start, split_end)
    - existe y(t+h) dentro del dataframe (t + h <= total_len - 1)

    Replica la convencion de S2/TCN: el target puede caer en la frontera con
    el siguiente split mientras exista en el dataframe completo.
    """
    first = split_start + seq_length
    last_in_split = split_end - 1
    last_by_target = total_len - 1 - horizon
    last = min(last_in_split, last_by_target)
    if last < first:
        return np.empty(0, dtype=int)
    return np.arange(first, last + 1)


def get_split_indices(
    df: pd.DataFrame,
    horizon: int,
    seq_length: int = SEQ_LENGTH,
) -> Dict[str, np.ndarray]:
    """Devuelve (idx_train, idx_val, idx_test) alineados al split oficial."""
    total = len(df)
    return {
        "train": aligned_indices(0, IDX_TRAIN_END, horizon, total, seq_length),
        "val":   aligned_indices(IDX_TRAIN_END, IDX_VAL_END, horizon, total, seq_length),
        "test":  aligned_indices(IDX_VAL_END, total, horizon, total, seq_length),
    }


# ---------------------------------------------------------------------------
# Construccion de features con lags del target
# ---------------------------------------------------------------------------

def _lag_matrix(series: np.ndarray, lags: int) -> np.ndarray:
    """Matriz (n, lags) con columnas [y(t), y(t-1), ..., y(t-lags+1)].

    Las primeras `lags-1` filas quedan NaN y deben filtrarse por el caller.
    Equivalente a la funcion lag_matrix de s2_baselines.py.
    """
    n = len(series)
    out = np.full((n, lags), np.nan, dtype=float)
    for k in range(lags):
        out[k:, k] = series[: n - k]
    return out


def build_features_with_lags(
    df: pd.DataFrame,
    idx: np.ndarray,
    horizon: int,
    lags: int = 12,
    features: Optional[List[str]] = None,
    include_lags: bool = True,
    include_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Construye (X, y, feature_names) para un conjunto de indices.

    X contiene, en este orden:
    - Si include_lags: `y(t), y(t-1), ..., y(t-lags+1)` (lags columnas)
    - Si include_features: columnas de `features` evaluadas en t

    y es `stormflow_mgd(t + horizon)`.

    Nota: el caller garantiza que los indices en `idx` estan alineados (todos
    tienen ventana previa suficiente y target valido), por construccion de
    `aligned_indices`. No hay filas con NaN ya que SEQ_LENGTH (72) > lags (12)
    por defecto, pero la funcion lo valida por robustez.
    """
    if features is None:
        features = FEATURES_10
    if not include_lags and not include_features:
        raise ValueError("Al menos uno de include_lags o include_features debe ser True")

    feature_names: List[str] = []
    blocks: List[np.ndarray] = []

    target = df[TARGET_COL].to_numpy()
    if include_lags:
        lag_mat = _lag_matrix(target, lags)  # (n_total, lags)
        x_lags = lag_mat[idx]                # (len(idx), lags)
        if np.isnan(x_lags).any():
            raise ValueError(
                f"lags NaN en idx: primer indice={int(idx[0])}, lags={lags}. "
                "Los indices deben cumplir t >= lags - 1."
            )
        blocks.append(x_lags)
        feature_names.extend([f"lag_{k}" for k in range(lags)])

    if include_features:
        x_feat = df[features].to_numpy()[idx]  # (len(idx), n_features)
        blocks.append(x_feat)
        feature_names.extend(list(features))

    X = np.concatenate(blocks, axis=1)
    y = target[idx + horizon]
    return X.astype(np.float32, copy=False), y.astype(np.float32, copy=False), feature_names


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_xgboost_h(
    df: pd.DataFrame,
    horizon: int,
    lags: int = 12,
    features: Optional[List[str]] = None,
    include_lags: bool = True,
    include_features: bool = True,
    xgb_params: Optional[Dict] = None,
    early_stopping_rounds: Optional[int] = DEFAULT_EARLY_STOPPING_ROUNDS,
    seq_length: int = SEQ_LENGTH,
    verbose: bool = False,
) -> Dict:
    """Entrena XGBoost para un horizonte y devuelve modelo + predicciones test.

    Devuelve un dict con:
      - model: XGBRegressor entrenado
      - y_pred_test, y_true_test: arrays en MGD (sin clipping)
      - y_pred_val, y_true_val: idem sobre val
      - timestamps_test: pd.Series alineada con y_*_test (para panel)
      - feature_names: nombres de las columnas del input
      - fit_seconds: tiempo de fit
      - best_iteration: iteracion seleccionada (si early stopping; None si no)
      - config: dict con horizon, lags, include_*, n_features_used
    """
    if xgb_params is None:
        xgb_params = DEFAULT_XGB_PARAMS
    splits = get_split_indices(df, horizon, seq_length=seq_length)
    idx_train, idx_val, idx_test = splits["train"], splits["val"], splits["test"]

    X_train, y_train, feat_names = build_features_with_lags(
        df, idx_train, horizon, lags, features, include_lags, include_features
    )
    X_val, y_val, _ = build_features_with_lags(
        df, idx_val, horizon, lags, features, include_lags, include_features
    )
    X_test, y_test, _ = build_features_with_lags(
        df, idx_test, horizon, lags, features, include_lags, include_features
    )

    params = dict(xgb_params)
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds

    model = XGBRegressor(**params)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)] if early_stopping_rounds is not None else None,
        verbose=bool(verbose),
    )
    fit_seconds = time.time() - t0

    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)

    timestamps_test = df.iloc[idx_test + horizon]["timestamp"].reset_index(drop=True)

    best_iter: Optional[int]
    try:
        best_iter = int(model.best_iteration) if early_stopping_rounds is not None else None
    except AttributeError:
        best_iter = None

    return {
        "model": model,
        "y_pred_test": np.asarray(y_pred_test, dtype=float),
        "y_true_test": np.asarray(y_test, dtype=float),
        "y_pred_val": np.asarray(y_pred_val, dtype=float),
        "y_true_val": np.asarray(y_val, dtype=float),
        "timestamps_test": timestamps_test,
        "feature_names": feat_names,
        "fit_seconds": float(fit_seconds),
        "best_iteration": best_iter,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "config": {
            "horizon": int(horizon),
            "lags": int(lags) if include_lags else 0,
            "include_lags": bool(include_lags),
            "include_features": bool(include_features),
            "n_features_input": int(len(feat_names)),
            "features": list(features) if (include_features and features is not None) else
                        (FEATURES_10 if include_features else []),
        },
    }
