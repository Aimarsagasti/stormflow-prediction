"""Chronological split utilities for train/validation/test partitions."""

from __future__ import annotations  # Permite usar anotaciones de tipos modernas con compatibilidad amplia

from typing import Tuple  # Define tipos de retorno explicitos para el split

import pandas as pd  # Provee DataFrame y funciones de orden temporal


def _print_split_diagnostics(split_name: str, split_df: pd.DataFrame) -> None:
    """Print basic diagnostics for a split."""
    if split_df.empty:  # Evita errores de min/max cuando un split queda vacio
        print(f"[split] {split_name}: 0 registros | rango: N/A | % evento: N/A")  # Reporta split vacio para depuracion
        return  # Corta la funcion porque no hay mas estadisticas utiles

    time_min = split_df["timestamp"].min()  # Obtiene inicio temporal del split para validar orden cronologico
    time_max = split_df["timestamp"].max()  # Obtiene fin temporal del split para validar cobertura temporal
    event_pct = split_df["is_event"].mean() * 100.0 if "is_event" in split_df.columns else float("nan")  # Calcula porcentaje de evento si la columna existe
    event_text = f"{event_pct:.2f}%" if "is_event" in split_df.columns else "N/A"  # Formatea texto de porcentaje de evento para imprimir
    print(f"[split] {split_name}: {len(split_df)} registros | rango: {time_min} -> {time_max} | % evento: {event_text}")  # Imprime resumen pedido por el pipeline


def split_chronological(
    df_feat: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split features DataFrame into train/val/test in strict chronological order."""
    if "timestamp" not in df_feat.columns:  # Valida que exista la columna temporal requerida por el usuario
        raise ValueError("df_feat must contain a 'timestamp' column for chronological splitting")  # Lanza error explicito para facilitar debugging
    if train_ratio <= 0 or val_ratio <= 0:  # Evita ratios no validos que rompen el esquema de entrenamiento
        raise ValueError("train_ratio and val_ratio must be positive")  # Mensaje claro para corregir configuracion
    if train_ratio + val_ratio >= 1.0:  # Garantiza que quede un segmento para test
        raise ValueError("train_ratio + val_ratio must be < 1.0")  # Restringe configuracion a un split valido

    df_sorted = df_feat.copy()  # Trabaja sobre copia para no mutar el DataFrame recibido
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], errors="coerce")  # Asegura tipo datetime para ordenar correctamente
    df_sorted = df_sorted.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)  # Elimina timestamps invalidos y ordena en el tiempo

    total_rows = len(df_sorted)  # Obtiene total de registros disponibles para repartir entre splits
    train_end = int(total_rows * train_ratio)  # Calcula indice de corte para train segun el ratio solicitado
    val_end = int(total_rows * (train_ratio + val_ratio))  # Calcula indice de corte acumulado para train+val

    df_train = df_sorted.iloc[:train_end].copy().reset_index(drop=True)  # Extrae tramo inicial para entrenamiento sin mezclar futuro
    df_val = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)  # Extrae tramo intermedio para validacion temporal
    df_test = df_sorted.iloc[val_end:].copy().reset_index(drop=True)  # Extrae tramo final para prueba sobre datos mas recientes

    print(f"[split] Total registros: {total_rows}")  # Reporta tamano global antes de detallar cada split
    _print_split_diagnostics("train", df_train)  # Imprime diagnosticos de train como pide el requerimiento
    _print_split_diagnostics("val", df_val)  # Imprime diagnosticos de validacion
    _print_split_diagnostics("test", df_test)  # Imprime diagnosticos de test

    return df_train, df_val, df_test  # Devuelve los tres DataFrames para el resto del pipeline
