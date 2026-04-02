"""Feature engineering utilities for MSD stormflow modeling."""

from __future__ import annotations  # Permite anotaciones modernas de tipos con compatibilidad amplia

import numpy as np  # Aporta operaciones vectorizadas para calculos de series temporales
import pandas as pd  # Ofrece estructura tabular y funciones rolling/datetime


API_K_BASE = 0.90  # Define la persistencia base del API en condiciones termicas neutras
API_ALPHA = 0.002  # Define cuanto cambia K por cada grado Fahrenheit respecto a la referencia
API_TEMP_REF_F = 50.0  # Usa 50 F como referencia termica neutral para el decaimiento del API
API_K_MIN = 0.80  # Evita que K baje demasiado y vuelva inestable la memoria hidrologica
API_K_MAX = 0.98  # Evita que K suba demasiado y acumule memoria excesiva por muchos dias


def _infer_resolution_minutes(timestamps: pd.Series) -> int:
    """Infer sampling resolution in minutes from timestamp differences."""
    deltas = timestamps.sort_values().diff().dropna()  # Calcula deltas temporales para estimar la resolucion real
    if deltas.empty:  # Evita errores cuando hay muy pocos datos para estimar diferencias
        return 5  # Usa 5 minutos por defecto segun especificacion del dataset
    resolution = int(round(deltas.dt.total_seconds().median() / 60.0))  # Usa mediana para robustez frente a gaps
    return max(resolution, 1)  # Garantiza un valor positivo para evitar divisiones invalidas


def _add_rain_features(df_feat: pd.DataFrame, resolution_minutes: int) -> pd.DataFrame:
    """Create rainfall rolling and recency features."""
    rolling_sum_minutes = [10, 15, 30, 60, 120, 180, 360]  # Define ventanas de acumulado pedidas en la propuesta con un nivel intermedio extra para humedad reciente
    rolling_max_minutes = [10, 30, 60]  # Define ventanas de maximo pedidas para lluvia intensa

    for window_minutes in rolling_sum_minutes:  # Recorre cada ventana de acumulado solicitada
        window_steps = max(int(window_minutes / resolution_minutes), 1)  # Convierte minutos a pasos de la serie
        column_name = f"rain_sum_{window_minutes}m"  # Construye nombre consistente para la feature
        df_feat[column_name] = df_feat["rain_in"].rolling(window=window_steps, min_periods=1).sum()  # Calcula suma movil causal

    for window_minutes in rolling_max_minutes:  # Recorre cada ventana de maximo solicitada
        window_steps = max(int(window_minutes / resolution_minutes), 1)  # Convierte minutos a pasos segun resolucion
        column_name = f"rain_max_{window_minutes}m"  # Define nombre de columna para maximo movil
        df_feat[column_name] = df_feat["rain_in"].rolling(window=window_steps, min_periods=1).max()  # Calcula maximo movil causal

    rain_positive_mask = df_feat["rain_in"] > 0  # Marca instantes con lluvia para medir recencia
    index_values = np.arange(len(df_feat))  # Crea indice numerico para computar distancia en pasos
    last_rain_index = np.where(rain_positive_mask.to_numpy(), index_values, -1)  # Guarda indice actual solo cuando llueve
    last_rain_index = np.maximum.accumulate(last_rain_index)  # Propaga el ultimo indice con lluvia hacia adelante
    steps_since_rain = index_values - last_rain_index  # Calcula cuantos pasos pasaron desde la ultima lluvia
    has_previous_rain = last_rain_index >= 0  # Identifica filas donde ya hubo lluvia previa
    cap_minutes = 24 * 60  # Define tope de 24 horas segun requerimiento
    minutes_since_rain = np.where(has_previous_rain, steps_since_rain * resolution_minutes, cap_minutes)  # Convierte a minutos y asigna tope inicial
    df_feat["minutes_since_last_rain"] = np.minimum(minutes_since_rain, cap_minutes).astype(float)  # Aplica cap de 24h y tipa a float

    return df_feat  # Devuelve DataFrame con bloque de features de lluvia agregado


def _prepare_temperature_feature(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Ensure daily temperature is available as a numeric feature without NaN."""
    if "temp_daily_f" not in df_feat.columns:  # Mantiene compatibilidad si el loader no agrego aun la temperatura
        df_feat["temp_daily_f"] = np.nan  # Crea la columna para poder aplicar fallback neutral mas adelante

    df_feat["temp_daily_f"] = pd.to_numeric(  # Fuerza la columna de temperatura a formato numerico estable
        df_feat["temp_daily_f"],  # Columna de temperatura diaria que llega desde el merge en carga
        errors="coerce",  # Convierte valores raros a NaN para tratarlos de forma uniforme
    )
    missing_temperature_count = int(df_feat["temp_daily_f"].isna().sum())  # Cuenta filas sin temperatura valida antes del fallback
    if missing_temperature_count > 0:  # Solo informa y rellena cuando realmente hay faltantes
        print(f"[features] Temperatura diaria faltante; se usara fallback de {API_TEMP_REF_F:.1f} F en {missing_temperature_count} filas")  # Explica el fallback neutral aplicado
        df_feat["temp_daily_f"] = df_feat["temp_daily_f"].fillna(API_TEMP_REF_F)  # Usa 50 F para no sesgar el decaimiento del API ni introducir NaN

    return df_feat  # Devuelve el DataFrame con temperatura diaria lista para usar como feature directa


def _compute_dynamic_api(rain_values: np.ndarray, temp_values: np.ndarray) -> np.ndarray:
    """Compute sequential API with temperature-modulated decay."""
    api_values = np.zeros(shape=rain_values.shape[0], dtype=float)  # Reserva el arreglo de salida para el API secuencial
    previous_api = 0.0  # Inicializa el API previo en cero al inicio de la serie

    for row_index, rain_value in enumerate(rain_values):  # Recorre secuencialmente porque cada paso depende del anterior
        temperature_value = temp_values[row_index]  # Toma la temperatura diaria ya alineada con el timestamp actual
        if np.isfinite(temperature_value):  # Usa la temperatura real cuando esta disponible
            k_value = API_K_BASE - (API_ALPHA * (temperature_value - API_TEMP_REF_F))  # Modula K para acelerar secado con calor y frenarlo con frio
        else:  # Mantiene un comportamiento estable si aun asi apareciera algun NaN residual
            k_value = API_K_BASE  # Aplica el K base como fallback neutral pedido
        k_value = float(np.clip(k_value, API_K_MIN, API_K_MAX))  # Clampea K dentro del rango estable especificado por el usuario
        current_api = float(rain_value) + (k_value * previous_api)  # Aplica la recurrencia API(t) = rain(t) + K(t) * API(t-1)
        api_values[row_index] = current_api  # Guarda el valor actual del API para este timestamp
        previous_api = current_api  # Propaga el estado al siguiente paso de la secuencia

    return api_values  # Devuelve la serie completa del API dinamico ya calculada


def create_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Create model features, target, and auxiliary columns from cleaned time series."""
    required_columns = ["timestamp", "rain_in", "flow_total_mgd", "stormflow_mgd", "is_event"]  # Define columnas estrictamente necesarias para construir el set de features
    missing_columns = [column for column in required_columns if column not in df_clean.columns]  # Detecta columnas faltantes en entrada
    if missing_columns:  # Valida esquema para fallar con mensaje claro en Colab
        raise ValueError(f"Missing required columns for feature engineering: {missing_columns}")  # Lanza error explicito para facilitar depuracion

    df_feat = df_clean.copy()  # Trabaja sobre copia para no mutar la salida de limpieza
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], errors="coerce")  # Asegura tipo datetime para derivar senales temporales
    df_feat = df_feat.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)  # Elimina timestamps invalidos y ordena cronologicamente
    resolution_minutes = _infer_resolution_minutes(df_feat["timestamp"])  # Estima resolucion para convertir ventanas en minutos a pasos

    df_feat = _add_rain_features(df_feat, resolution_minutes)  # Agrega rolling sums, rolling max y recencia de lluvia
    df_feat = _prepare_temperature_feature(df_feat)  # Asegura que la temperatura diaria este disponible como feature directa sin NaN

    rain_values = pd.to_numeric(  # Convierte lluvia a vector float para el calculo recursivo del API
        df_feat["rain_in"],  # Serie de lluvia incremental a 5 minutos
        errors="coerce",  # Protege frente a valores inesperados aunque no deberian existir
    ).fillna(0.0).to_numpy(dtype=float)  # Lleva cualquier faltante eventual a cero para no romper la recurrencia
    temp_values = df_feat["temp_daily_f"].to_numpy(dtype=float)  # Extrae temperatura diaria ya alineada para modular K en cada paso
    df_feat["api_dynamic"] = _compute_dynamic_api(rain_values=rain_values, temp_values=temp_values)  # Reintroduce el API dinamico con memoria secuencial dependiente de temperatura

    steps_5m = max(int(5 / resolution_minutes), 1)  # Traduce 5 minutos al numero de pasos de la serie
    steps_10m = max(int(10 / resolution_minutes), 1)  # Traduce 10 minutos al numero de pasos de la serie
    steps_15m = max(int(15 / resolution_minutes), 1)  # Traduce 15 minutos al numero de pasos de la serie
    steps_30m = max(int(30 / resolution_minutes), 1)  # Traduce 30 minutos al numero de pasos de la serie
    df_feat["delta_flow_5m"] = df_feat["flow_total_mgd"].diff(periods=steps_5m).fillna(0.0)  # Calcula variacion de flow_total en 5 minutos
    df_feat["delta_flow_15m"] = df_feat["flow_total_mgd"].diff(periods=steps_15m).fillna(0.0)  # Calcula variacion de flow_total en 15 minutos
    df_feat["delta_rain_10m"] = df_feat["rain_in"].diff(periods=steps_10m).fillna(0.0)  # Expone si la lluvia reciente se esta intensificando o debilitando a muy corto plazo
    df_feat["delta_rain_30m"] = df_feat["rain_in"].diff(periods=steps_30m).fillna(0.0)  # Resume cambio de lluvia en una ventana un poco mas estable para distinguir ascensos sostenidos

    hour_fraction = (df_feat["timestamp"].dt.hour + (df_feat["timestamp"].dt.minute / 60.0)) / 24.0  # Convierte hora del dia a fase continua
    df_feat["hour_sin"] = np.sin(2.0 * np.pi * hour_fraction)  # Codifica fase horaria en seno para capturar periodicidad
    df_feat["hour_cos"] = np.cos(2.0 * np.pi * hour_fraction)  # Codifica fase horaria en coseno para cerrar el ciclo

    month_fraction = (df_feat["timestamp"].dt.month - 1) / 12.0  # Convierte mes a fase anual normalizada
    df_feat["month_sin"] = np.sin(2.0 * np.pi * month_fraction)  # Codifica estacionalidad mensual en seno
    df_feat["month_cos"] = np.cos(2.0 * np.pi * month_fraction)  # Codifica estacionalidad mensual en coseno

    feature_columns = [  # Enumera columnas de entrada que iran al modelo
        "rain_in",  # Mantiene lluvia base como feature primaria causal
        "flow_total_mgd",  # Mantiene flujo total por su alta correlacion con stormflow
        "temp_daily_f",  # Agrega temperatura diaria directa porque modula respuesta hidrologica y evaporacion
        "api_dynamic",  # Reincorpora el API con memoria de humedad antecedente sensible a temperatura
        "rain_sum_10m",  # Acumulado de lluvia en ventana corta de respuesta rapida
        "rain_sum_15m",  # Acumulado de lluvia en 15 minutos
        "rain_sum_30m",  # Acumulado de lluvia en media hora
        "rain_sum_60m",  # Acumulado de lluvia en una hora
        "rain_sum_120m",  # Acumulado de lluvia en dos horas
        "rain_sum_180m",  # Acumulado de lluvia intermedio para distinguir saturacion reciente sin depender solo de 120 o 360 min
        "rain_sum_360m",  # Acumulado de lluvia en seis horas para memoria de humedad
        "rain_max_10m",  # Maximo de lluvia reciente de 10 minutos
        "rain_max_30m",  # Maximo de lluvia reciente de 30 minutos
        "rain_max_60m",  # Maximo de lluvia reciente de 60 minutos
        "minutes_since_last_rain",  # Recencia de lluvia con cap de 24 horas
        "delta_flow_5m",  # Tendencia corta de flow_total en 5 minutos
        "delta_flow_15m",  # Tendencia corta de flow_total en 15 minutos
        "delta_rain_10m",  # Cambio inmediato de lluvia para distinguir intensificacion rapida antes del pico
        "delta_rain_30m",  # Cambio de lluvia un poco mas estable para separar pulsos breves de episodios crecientes
        "hour_sin",  # Componente ciclica senoidal horaria
        "hour_cos",  # Componente ciclica cosenoidal horaria
        "month_sin",  # Componente ciclica senoidal mensual
        "month_cos",  # Componente ciclica cosenoidal mensual
    ]

    output_columns = ["timestamp", *feature_columns, "stormflow_mgd", "is_event"]  # Define salida con features, target y auxiliar
    df_output = df_feat.loc[:, output_columns].copy()  # Selecciona solo columnas finales para evitar leakage accidental
    df_output["is_event"] = df_output["is_event"].astype(bool)  # Fuerza tipo booleano para columna auxiliar

    feature_count = len(feature_columns)  # Cuenta features efectivas de entrada en el DataFrame final
    nan_count = int(df_output.isna().sum().sum())  # Calcula NaN global para diagnostico rapido en Colab
    print(f"[features] Numero de features de entrada: {feature_count}")  # Reporta cuantas features se generaron
    print(f"[features] Shape final: {df_output.shape}")  # Reporta dimensiones de la tabla resultante
    print(f"[features] NaN count total: {nan_count}")  # Reporta faltantes remanentes tras ingenieria de features

    return df_output  # Devuelve DataFrame listo para pipeline de split/secuencias
