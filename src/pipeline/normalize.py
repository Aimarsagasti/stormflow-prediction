"""Normalization utilities with train-only statistics for MSD pipeline."""

from __future__ import annotations  # Permite anotaciones modernas de tipos en versiones soportadas

from typing import Dict, List, Tuple  # Define estructuras de retorno claras para parametros de normalizacion

import numpy as np  # Aporta operaciones numericas robustas para escalado
import pandas as pd  # Provee DataFrames para transformar cada split


EPSILON = 1e-12  # Evita divisiones por cero cuando una columna es constante


def _get_log_columns(
    feature_columns: List[str],
    target_col: str,
    apply_log1p_to_target: bool,
) -> List[str]:
    """Return skewed columns that should receive log1p before Min-Max."""
    log_columns = []  # Acumula columnas sesgadas para transformacion logaritmica
    for column_name in feature_columns:  # Recorre cada feature candidata de entrada
        if column_name == "rain_in" or column_name.startswith("rain_sum_"):  # Selecciona lluvia base y acumulados segun proposal.md
            log_columns.append(column_name)  # Guarda columna para aplicar log1p antes del escalado
    if apply_log1p_to_target:  # Permite comprimir tambien la cola extrema del target cuando asi se configure
        log_columns.append(target_col)  # Agrega target a la lista de columnas transformadas con log1p
    return log_columns  # Devuelve lista final de columnas transformadas


def _apply_log_transform(df_split: pd.DataFrame, log_columns: List[str]) -> pd.DataFrame:
    """Apply log1p to selected columns using non-negative clipping."""
    df_out = df_split.copy()  # Trabaja sobre copia para no mutar entradas originales
    for column_name in log_columns:  # Recorre columnas sesgadas seleccionadas
        if column_name in df_out.columns:  # Verifica existencia para evitar errores por columnas ausentes
            df_out[column_name] = np.log1p(df_out[column_name].clip(lower=0.0))  # Aplica log1p sobre valores no negativos para estabilidad numerica
    return df_out  # Devuelve DataFrame transformado en el espacio logaritmico


def _minmax_scale(df_split: pd.DataFrame, columns: List[str], stats_min: Dict[str, float], stats_range: Dict[str, float]) -> pd.DataFrame:
    """Scale columns with precomputed Min-Max statistics."""
    df_out = df_split.copy()  # Crea copia para no tocar el DataFrame original
    for column_name in columns:  # Itera por todas las columnas a normalizar
        if column_name in df_out.columns:  # Evita fallo si una columna no esta disponible en el split
            df_out[column_name] = (df_out[column_name] - stats_min[column_name]) / stats_range[column_name]  # Aplica formula Min-Max con stats de train
    return df_out  # Regresa DataFrame normalizado


def normalize_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "stormflow_mgd",
    apply_log1p_to_target: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Normalize train/val/test with train-only Min-Max stats and optional log1p."""
    if target_col not in df_train.columns:  # Valida que el target exista en train para calcular sus estadisticas
        raise ValueError(f"Target column '{target_col}' not found in train split")  # Da mensaje explicito para depuracion rapida

    log_columns = _get_log_columns(  # Define columnas sesgadas que se transforman con log1p
        feature_columns=feature_columns,  # Pasa features del modelo para detectar lluvia/acumulados sesgados
        target_col=target_col,  # Pasa nombre del target por si tambien se debe comprimir
        apply_log1p_to_target=apply_log1p_to_target,  # Indica si el target entra o no en transformacion logaritmica
    )

    train_transformed = _apply_log_transform(df_train, log_columns)  # Transforma train antes de calcular min/max segun el pipeline propuesto
    val_transformed = _apply_log_transform(df_val, log_columns)  # Aplica misma transformacion en validacion para mantener consistencia
    test_transformed = _apply_log_transform(df_test, log_columns)  # Aplica misma transformacion en test para inferencia coherente

    all_norm_columns = list(feature_columns) + [target_col]  # Construye conjunto final de columnas a escalar en Min-Max
    stats_min: Dict[str, float] = {}  # Almacena minimos por columna calculados solo en train
    stats_max: Dict[str, float] = {}  # Almacena maximos por columna calculados solo en train
    stats_range: Dict[str, float] = {}  # Almacena rangos por columna para usar en escalado y desnormalizacion

    for column_name in all_norm_columns:  # Recorre features y target para extraer estadisticas base
        if column_name not in train_transformed.columns:  # Valida esquema esperado antes de continuar
            raise ValueError(f"Column '{column_name}' not found in train split")  # Falla de forma clara si falta alguna columna critica
        col_min = float(train_transformed[column_name].min())  # Toma minimo de train para la columna
        col_max = float(train_transformed[column_name].max())  # Toma maximo de train para la columna
        col_range = max(col_max - col_min, EPSILON)  # Evita rango cero para no dividir por cero en columnas constantes
        stats_min[column_name] = col_min  # Guarda minimo en diccionario de parametros
        stats_max[column_name] = col_max  # Guarda maximo en diccionario de parametros
        stats_range[column_name] = col_range  # Guarda rango util para normalizar y revertir

    df_train_norm = _minmax_scale(train_transformed, all_norm_columns, stats_min, stats_range)  # Escala train usando stats de train
    df_val_norm = _minmax_scale(val_transformed, all_norm_columns, stats_min, stats_range)  # Escala val sin leakage de informacion futura
    df_test_norm = _minmax_scale(test_transformed, all_norm_columns, stats_min, stats_range)  # Escala test con mismas reglas de entrenamiento

    norm_params: Dict[str, object] = {  # Empaqueta toda la informacion necesaria para reproducir transformacion e inversion
        "feature_columns": list(feature_columns),  # Conserva orden de features para reconstruir tensores luego
        "target_col": target_col,  # Guarda nombre del target normalizado
        "log1p_columns": log_columns,  # Lista columnas que recibieron transformacion log1p
        "apply_log1p_to_target": apply_log1p_to_target,  # Guarda bandera explicita para trazabilidad del target
        "min": stats_min,  # Diccionario de minimos por columna
        "max": stats_max,  # Diccionario de maximos por columna
        "range": stats_range,  # Diccionario de rangos por columna
    }

    print(f"[normalize] Columnas con log1p: {log_columns}")  # Reporta columnas sesgadas transformadas antes del Min-Max
    print(f"[normalize] Shape train_norm: {df_train_norm.shape}")  # Reporta dimensiones finales del split train normalizado
    print(f"[normalize] Shape val_norm: {df_val_norm.shape}")  # Reporta dimensiones finales de validacion normalizada
    print(f"[normalize] Shape test_norm: {df_test_norm.shape}")  # Reporta dimensiones finales de test normalizado

    return df_train_norm, df_val_norm, df_test_norm, norm_params  # Devuelve splits normalizados y parametros para inferencia/desnormalizacion


def normalize_target_values(values: np.ndarray, norm_params: Dict[str, object]) -> np.ndarray:
    """Convert raw target values in MGD into the normalized training scale."""
    target_col = str(norm_params["target_col"])  # Recupera nombre del target para acceder a sus parametros de transformacion
    values_array = np.asarray(values, dtype=float)  # Convierte valores entrantes a arreglo numpy para transformar en bloque
    if target_col in norm_params.get("log1p_columns", []):  # Revisa si el target se comprimio con log1p al normalizar
        values_array = np.log1p(np.clip(values_array, a_min=0.0, a_max=None))  # Reproduce la misma transformacion sobre valores reales no negativos
    target_min = float(norm_params["min"][target_col])  # Extrae minimo usado en Min-Max del target
    target_range = float(norm_params["range"][target_col])  # Extrae rango usado en Min-Max del target
    normalized_values = (values_array - target_min) / target_range  # Lleva los valores al mismo espacio de entrenamiento del modelo
    return normalized_values  # Devuelve arreglo listo para comparar con y_true/y_pred normalizados


def denormalize_target(y_norm: np.ndarray, norm_params: Dict[str, object]) -> np.ndarray:
    """Convert normalized target values back to real-world units."""
    target_col = str(norm_params["target_col"])  # Recupera nombre del target para buscar sus parametros de escala
    target_min = float(norm_params["min"][target_col])  # Extrae minimo de train usado en la normalizacion del target
    target_range = float(norm_params["range"][target_col])  # Extrae rango de train usado en la normalizacion del target

    y_norm_array = np.asarray(y_norm, dtype=float)  # Convierte entrada a arreglo numpy para operar de forma vectorizada
    y_real = (y_norm_array * target_range) + target_min  # Revierte Min-Max al espacio transformado previo al escalado

    if target_col in norm_params.get("log1p_columns", []):  # Verifica si el target tambien usa log1p para revertir completamente
        y_real = np.expm1(y_real)  # Revierte log1p y devuelve valores reales en unidades originales

    return y_real  # Devuelve target en escala fisica para interpretacion y metricas
