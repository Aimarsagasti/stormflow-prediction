"""Data cleaning utilities for MSD time series."""

from __future__ import annotations  # Permite usar anotaciones modernas de tipos en todas las versiones soportadas

import numpy as np  # Aporta operaciones vectorizadas para construir mascaras eficientes
import pandas as pd  # Provee el DataFrame y utilidades de limpieza temporal


def _build_event_mask(timestamps: pd.Series, df_events: pd.DataFrame) -> pd.Series:
    """Build a boolean mask indicating whether each timestamp is within an event interval."""
    valid_events = df_events.dropna(subset=["event_start", "event_end"]).copy()  # Asegura eventos con inicio y fin validos
    if valid_events.empty:  # Evita trabajo extra cuando no hay eventos utilizables
        return pd.Series(False, index=timestamps.index)  # Devuelve una mascara en falso para todo el periodo

    starts = np.sort(valid_events["event_start"].to_numpy(dtype="datetime64[ns]"))  # Ordena inicios para busqueda binaria
    ends = np.sort(valid_events["event_end"].to_numpy(dtype="datetime64[ns]"))  # Ordena finales para busqueda binaria
    ts_values = timestamps.to_numpy(dtype="datetime64[ns]")  # Convierte timestamps a arreglo nativo para acelerar calculos

    started_count = np.searchsorted(starts, ts_values, side="right")  # Cuenta eventos iniciados en o antes de cada timestamp
    ended_before_count = np.searchsorted(ends, ts_values, side="left")  # Cuenta eventos finalizados estrictamente antes del timestamp
    is_event_mask = started_count > ended_before_count  # Marca activo cuando hay al menos un evento abierto

    return pd.Series(is_event_mask, index=timestamps.index)  # Regresa la mascara alineada con el DataFrame original


def _infer_resolution_minutes(timestamps: pd.Series) -> int:
    """Infer sampling resolution in minutes from timestamp differences."""
    deltas = timestamps.sort_values().diff().dropna()  # Calcula diferencias entre marcas de tiempo consecutivas
    if deltas.empty:  # Maneja casos de una sola fila para no dividir entre cero
        return 5  # Usa 5 minutos por defecto segun la resolucion del proyecto
    resolution = int(round(deltas.dt.total_seconds().median() / 60.0))  # Estima la resolucion usando la mediana robusta
    return max(resolution, 1)  # Garantiza al menos 1 minuto para evitar parametros invalidos


def clean_timeseries(df_timeseries: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """Clean MSD time series and append is_event flag."""
    df_clean = df_timeseries.copy()  # Trabaja sobre una copia para no mutar el DataFrame de entrada
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")  # Fuerza tipo datetime para orden e interpolacion
    initial_rows = len(df_clean)  # Guarda el tamano inicial para diagnosticos globales
    df_clean = df_clean.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)  # Elimina timestamps invalidos y ordena cronologicamente
    print(f"[clean] Registros iniciales: {initial_rows}")  # Reporta el punto de partida antes de limpiar
    print(f"[clean] Registros tras validar timestamp: {len(df_clean)}")  # Reporta cuantas filas permanecen tras validar tiempo

    negative_stormflow_count = int((df_clean["stormflow_mgd"] < 0).sum())  # Cuenta stormflow negativo identificado como artefacto
    df_clean["stormflow_mgd"] = df_clean["stormflow_mgd"].clip(lower=0)  # Corrige stormflow negativo a cero por restriccion fisica
    print(f"[clean] Stormflow negativo clipeado a 0: {negative_stormflow_count}")  # Informa cuantas filas fueron afectadas en el paso 1

    negative_flow_count = int((df_clean["flow_total_mgd"] < 0).sum())  # Cuenta valores negativos de caudal total por posible error de sensor
    df_clean.loc[df_clean["flow_total_mgd"] < 0, "flow_total_mgd"] = np.nan  # Convierte valores negativos a NaN para reparacion controlada
    resolution_minutes = _infer_resolution_minutes(df_clean["timestamp"])  # Infere resolucion temporal real para calcular limite de huecos
    max_gap_steps = max(int(30 / resolution_minutes), 1)  # Traduce 30 minutos al numero de pasos de la serie
    nan_before_interp = int(df_clean["flow_total_mgd"].isna().sum())  # Cuenta huecos antes de interpolar para diagnostico
    df_clean = df_clean.set_index("timestamp")  # Usa timestamp como indice para interpolacion temporal basada en reloj real
    df_clean["flow_total_mgd"] = df_clean["flow_total_mgd"].interpolate(  # Interpola solo huecos cortos y internos
        method="time",  # Interpolacion temporal que considera la distancia real entre marcas
        limit=max_gap_steps,  # Limita relleno a huecos de hasta 30 minutos
        limit_area="inside",  # Evita extrapolar al inicio o al final de la serie
    )
    nan_after_interp = int(df_clean["flow_total_mgd"].isna().sum())  # Cuenta huecos que quedaron sin reparar tras interpolacion
    large_gap_rows_removed = nan_after_interp  # Interpreta NaN remanente como hueco mayor al limite permitido
    df_clean = df_clean.dropna(subset=["flow_total_mgd"]).reset_index()  # Elimina filas en huecos largos y recupera timestamp como columna
    print(f"[clean] flow_total negativo convertido a NaN: {negative_flow_count}")  # Reporta cuantas lecturas se marcaron como faltantes
    print(f"[clean] NaN en flow_total antes de interpolar: {nan_before_interp}")  # Reporta magnitud de huecos antes del relleno
    print(f"[clean] NaN en flow_total tras interpolar: {nan_after_interp}")  # Reporta cuantos huecos no se pudieron reparar
    print(f"[clean] Filas eliminadas por huecos > 30 min: {large_gap_rows_removed}")  # Reporta eliminaciones por huecos largos

    storm_above_flow_count = int((df_clean["stormflow_mgd"] > df_clean["flow_total_mgd"]).sum())  # Cuenta casos fisicamente inconsistentes
    df_clean["stormflow_mgd"] = df_clean["stormflow_mgd"].clip(upper=df_clean["flow_total_mgd"])  # Limita stormflow para no superar caudal total
    print(f"[clean] Registros con stormflow > flow clipeados: {storm_above_flow_count}")  # Reporta correcciones de consistencia hidrologica

    df_clean["baseflow_mgd"] = df_clean["flow_total_mgd"] - df_clean["stormflow_mgd"]  # Recalcula baseflow desde variables corregidas
    negative_baseflow_count = int((df_clean["baseflow_mgd"] < 0).sum())  # Cuenta baseflow negativo antes del clip final
    df_clean["baseflow_mgd"] = df_clean["baseflow_mgd"].clip(lower=0)  # Refuerza restriccion fisica de baseflow no negativo
    print(f"[clean] Baseflow recalculado y clipeado a 0 (negativos previos): {negative_baseflow_count}")  # Informa ajustes en baseflow

    df_clean["is_event"] = _build_event_mask(df_clean["timestamp"], df_events).astype(bool)  # Marca cada fila segun pertenencia a un evento
    valid_event_count = int(df_clean["is_event"].sum())  # Cuenta cuantas filas quedan dentro de ventanas de evento
    print(f"[clean] Registros marcados como evento: {valid_event_count}")  # Reporta cobertura temporal de eventos en la serie final
    print(f"[clean] Shape final: {df_clean.shape}")  # Resume dimensiones finales tras toda la limpieza

    return df_clean  # Devuelve serie limpia con columna auxiliar is_event para etapas posteriores
