"""Clasificador binario XGBoost para alertas de stormflow en iter18.

Este modulo implementa la reformulacion operativa del problema para
horizontes largos: en lugar de predecir un valor puntual de stormflow,
predice si en la ventana futura `t+1..t+h` ocurrira al menos una muestra
con `stormflow_mgd >= U`.

Diseno:
- Reutiliza exactamente el split cronologico oficial de iter17.
- Reutiliza los mismos 6 lags del target y las mismas 10 features exogenas.
- Trabaja en MGD reales, sin depender del pipeline legado de normalizacion.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.xgboost_baseline import (
    FEATURES_10,
    SEQ_LENGTH,
    TARGET_COL,
    _lag_matrix,
    aligned_indices,
    get_split_indices,
)


# Estos hiperparametros replican el punto de partida pedido para iter18.
DEFAULT_XGB_CLASSIFIER_PARAMS: Dict[str, object] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": 42,
}

# Se mantiene el mismo early stopping que el regresor para comparar limpio.
DEFAULT_EARLY_STOPPING_ROUNDS = 20


def build_binary_target(
    df: pd.DataFrame,
    horizon: int,
    threshold_mgd: float,
    idx: np.ndarray,
) -> np.ndarray:
    """Construye `y_bin(t)` usando `max(stormflow[t+1..t+h]) >= U`.

    La funcion usa el dataframe completo para que la ventana futura quede
    bien definida incluso si el target cruza la frontera entre splits.
    """
    # Extraemos el target completo como array porque la construccion del
    # maximo futuro es mas simple y mas rapida en NumPy.
    target = df[TARGET_COL].to_numpy(dtype=float, copy=False)
    # Reservamos una columna por desplazamiento futuro dentro de la ventana.
    future_blocks: List[np.ndarray] = []
    for step_ahead in range(1, horizon + 1):
        # En la posicion t colocamos el valor observado en t+step_ahead.
        shifted = np.full(target.shape[0], np.nan, dtype=float)
        # Solo llenamos las posiciones donde existe ese futuro en el array.
        shifted[:-step_ahead] = target[step_ahead:]
        future_blocks.append(shifted)

    # Apilamos las columnas futuras para poder tomar el maximo por fila.
    future_matrix = np.column_stack(future_blocks)
    # `idx` ya garantiza que t+h existe, asi que aqui no deberian quedar NaN.
    future_max = np.nanmax(future_matrix[idx], axis=1)
    # El target binario es 1 si cualquier muestra futura rebasa el umbral U.
    return (future_max >= float(threshold_mgd)).astype(np.int32, copy=False)


def _build_classifier_features(
    df: pd.DataFrame,
    idx: np.ndarray,
    lags: int,
    features: List[str],
) -> tuple[np.ndarray, List[str]]:
    """Construye el input del clasificador con el mismo contrato de iter17."""
    # Reutilizamos el mismo target crudo para construir los lags explicitos.
    target = df[TARGET_COL].to_numpy(dtype=float, copy=False)
    # `_lag_matrix` ya esta validada en iter17 y mantiene el orden esperado.
    lag_matrix = _lag_matrix(target, lags)
    # Seleccionamos las filas temporales exactas del split alineado.
    x_lags = lag_matrix[idx]
    if np.isnan(x_lags).any():
        raise ValueError(
            "Se encontraron NaN en la matriz de lags. "
            "Esto indica un desalineamiento entre `idx` y `lags`."
        )

    # Tomamos las exogenas en el mismo tiempo t para no dar privilegios extra.
    x_features = df[features].to_numpy(dtype=float, copy=False)[idx]
    # Concatenamos primero lags y luego exogenas para mantener trazabilidad.
    X = np.concatenate([x_lags, x_features], axis=1)
    # Generamos nombres claros para poder inspeccionar importancia luego.
    feature_names = [f"lag_{lag_index}" for lag_index in range(lags)] + list(features)
    return X.astype(np.float32, copy=False), feature_names


def _compute_scale_pos_weight(y_train_bin: np.ndarray) -> float:
    """Calcula `neg/pos` usando solo train, como exige la especificacion."""
    # Contamos positivos y negativos en train para ponderar el desbalance real.
    positives = int(np.sum(y_train_bin == 1))
    negatives = int(np.sum(y_train_bin == 0))
    if positives == 0:
        raise ValueError(
            "El split de train no contiene positivos para esta variante. "
            "No es posible entrenar un clasificador binario util."
        )
    if negatives == 0:
        raise ValueError(
            "El split de train no contiene negativos para esta variante. "
            "La variante esta mal definida para clasificacion binaria."
        )
    return float(negatives / positives)


def select_operational_threshold(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.85,
) -> float:
    """Elige el mayor umbral que cumple un recall minimo en validacion.

    Elegir el mayor umbral posible entre los que cumplen el recall reduce
    falsos positivos sin sacrificar la restriccion operativa principal.
    """
    # Ordenamos umbrales unicos de mayor a menor para priorizar precision.
    candidate_thresholds = np.unique(np.asarray(y_prob, dtype=float))
    candidate_thresholds = np.sort(candidate_thresholds)[::-1]

    # Recorremos de mayor a menor: el primero que cumpla recall es el optimo
    # bajo la regla "maximizar threshold sujeto a recall minimo".
    for threshold in candidate_thresholds:
        y_pred_bin = (y_prob >= threshold).astype(np.int32, copy=False)
        true_positives = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        false_negatives = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        recall = true_positives / max(true_positives + false_negatives, 1)
        if recall >= float(min_recall):
            return float(threshold)

    # Si ningun umbral alcanza el recall deseado, usamos 0.0 para disparar
    # siempre alerta y dejar explicito en resultados que el modelo no separa.
    return 0.0


def train_xgboost_classifier(
    df: pd.DataFrame,
    horizon: int,
    threshold_mgd: float,
    lags: int = 6,
    features: List[str] = FEATURES_10,
    xgb_params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    seq_length: int = SEQ_LENGTH,
    verbose: bool = False,
) -> Dict[str, object]:
    """Entrena un clasificador binario XGBoost para una variante `(h, U)`."""
    # Respetamos los hiperparametros por defecto del prompt salvo override.
    params = dict(DEFAULT_XGB_CLASSIFIER_PARAMS if xgb_params is None else xgb_params)

    # `aligned_indices` se importa por contrato explicito del prompt; esta
    # llamada adicional sirve como sanity check de que seguimos el mismo split.
    _ = aligned_indices(0, len(df), horizon, len(df), seq_length)

    # Obtenemos exactamente los mismos indices cronologicos de iter17.
    splits = get_split_indices(df, horizon=horizon, seq_length=seq_length)
    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    # Armamos el input con la misma informacion disponible para el regresor.
    X_train, feature_names = _build_classifier_features(df, idx_train, lags, features)
    X_val, _ = _build_classifier_features(df, idx_val, lags, features)
    X_test, _ = _build_classifier_features(df, idx_test, lags, features)

    # Construimos el target binario usando la ventana completa confirmada.
    y_train_bin = build_binary_target(df, horizon, threshold_mgd, idx_train)
    y_val_bin = build_binary_target(df, horizon, threshold_mgd, idx_val)
    y_test_bin = build_binary_target(df, horizon, threshold_mgd, idx_test)

    # Calculamos el peso positivo solo con train para evitar leakage.
    scale_pos_weight = _compute_scale_pos_weight(y_train_bin)
    params["scale_pos_weight"] = scale_pos_weight
    # Early stopping se pasa en el constructor para mantener el patron de iter17.
    params["early_stopping_rounds"] = int(early_stopping_rounds)

    # Instanciamos el clasificador con la configuracion ya cerrada.
    model = XGBClassifier(**params)

    # Medimos el tiempo real de ajuste para reportarlo en artefactos.
    fit_start = time.time()
    model.fit(
        X_train,
        y_train_bin,
        # Validamos solo en val para seleccionar la mejor iteracion sin tocar test.
        eval_set=[(X_val, y_val_bin)],
        # El notebook controla la verbosidad global; aqui la dejamos limpia.
        verbose=bool(verbose),
    )
    fit_seconds = float(time.time() - fit_start)

    # Extraemos probabilidades de la clase positiva para umbralizacion posterior.
    y_prob_val = model.predict_proba(X_val)[:, 1].astype(float, copy=False)
    y_prob_test = model.predict_proba(X_test)[:, 1].astype(float, copy=False)

    # `timestamps_test` apunta al instante t de decision, no al futuro.
    timestamps_test = pd.to_datetime(df.iloc[idx_test]["timestamp"]).reset_index(drop=True)
    # Guardamos el stormflow actual en t para reconstruir el primer rebasamiento real.
    y_true_raw_test = df.iloc[idx_test][TARGET_COL].to_numpy(dtype=float, copy=False)

    # Capturamos la mejor iteracion si el modelo la expone tras early stopping.
    try:
        best_iteration = int(model.best_iteration)
    except AttributeError:
        best_iteration = None

    # Calculamos la prevalencia real en train para dejar trazabilidad en JSON.
    prevalence_train = float(np.mean(y_train_bin))

    return {
        "model": model,
        "y_true_bin_val": y_val_bin.astype(int, copy=False),
        "y_prob_val": y_prob_val,
        "y_true_bin_test": y_test_bin.astype(int, copy=False),
        "y_prob_test": y_prob_test,
        "y_true_raw_test": y_true_raw_test,
        "timestamps_test": timestamps_test,
        "feature_names": feature_names,
        "fit_seconds": fit_seconds,
        "best_iteration": best_iteration,
        "prevalence_train": prevalence_train,
        "scale_pos_weight": scale_pos_weight,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "config": {
            "horizon": int(horizon),
            "threshold_mgd": float(threshold_mgd),
            "lags": int(lags),
            "features": list(features),
            "seq_length": int(seq_length),
            "n_features_input": int(len(feature_names)),
        },
    }
