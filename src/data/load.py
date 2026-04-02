"""Load MSD time series and event data from TSF and DAT files."""

from __future__ import annotations  # Permite anotaciones de tipos como texto en Python 3.7+

from pathlib import Path  # Manejo seguro de rutas locales o de Google Drive
from typing import Any, Dict, List, Tuple  # Tipos para hacer el contrato de funciones explicito

import pandas as pd  # DataFrames para tablas de tiempo y eventos
import yaml  # Lectura de YAML con rutas configurables


DEFAULT_TEMP_F = 50.0  # Define una temperatura neutral para fallback cuando el archivo no exista localmente


def _read_config(config_path: str | Path) -> Dict[str, Any]:
    """Read YAML config and return it as a dict."""
    config_path = Path(config_path)  # Normaliza la ruta para operar igual en Windows o Colab
    raw_text = config_path.read_text(encoding="utf-8")  # Lee todo el archivo para manejar basura al final
    try:  # Intenta parsear el YAML completo primero
        config = yaml.safe_load(raw_text)  # Usa safe_load para evitar ejecucion de codigo
    except yaml.YAMLError:  # Si hay texto no-YAML al final, aplica limpieza
        cleaned_text = raw_text.split("```", maxsplit=1)[0]  # Conserva solo el bloque YAML previo al fence
        config = yaml.safe_load(cleaned_text)  # Reintenta el parseo con el contenido limpio
    return config  # Devuelve el diccionario para extraer rutas y nombres


def _read_tsf_file(file_path: Path, value_name: str) -> pd.DataFrame:
    """Read a TSF file with 3 header lines and return a DataFrame."""
    data = pd.read_csv(  # Lee el archivo tabulado en una sola pasada por eficiencia
        file_path,  # Ruta del .tsf
        sep="\t",  # Separador tab para columnas datetime y valor
        header=None,  # No hay cabecera util en los datos
        names=["timestamp", value_name],  # Define nombres consistentes en el DataFrame
        skiprows=3,  # Omite las 3 lineas de cabecera segun el formato
        parse_dates=[0],  # Convierte la columna datetime a tipo fecha para joins limpios
    )
    return data  # Devuelve el DataFrame con timestamp y variable


def _read_daily_temperature_file(file_path: Path) -> pd.DataFrame:
    """Read the daily temperature TSF file and return one row per day."""
    data = pd.read_csv(  # Lee el archivo tabulado de temperatura diaria respetando el formato TSF indicado
        file_path,  # Ruta al archivo de temperatura diaria en Drive
        sep="\t",  # Usa tab porque el archivo esta separado por tabulaciones
        header=None,  # No usa cabecera porque las 3 primeras filas son metadata
        names=["date", "temp_daily_f"],  # Asigna nombres explicitos para merge y feature engineering
        skiprows=3,  # Omite las 3 lineas de cabecera propias del TSF
    )
    data["date"] = pd.to_datetime(  # Convierte la fecha diaria en datetime para alinear con la serie principal
        data["date"],  # Columna de fecha en formato M/d/yyyy
        format="%m/%d/%Y",  # Usa formato explicito para parseo estable en Colab y local
        errors="coerce",  # Convierte fechas invalidas en NaT para descartarlas limpiamente
    )
    data["temp_daily_f"] = pd.to_numeric(  # Convierte la temperatura a float por si llega como texto
        data["temp_daily_f"],  # Columna de temperatura diaria en Fahrenheit
        errors="coerce",  # Manda a NaN cualquier valor inesperado para filtrarlo luego
    )
    data = data.dropna(subset=["date", "temp_daily_f"])  # Conserva solo filas completas listas para merge
    data["date"] = data["date"].dt.normalize()  # Lleva cada fecha a medianoche para usar llave diaria consistente
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")  # Garantiza una sola temperatura por dia

    print(f"[load] Registros de temperatura diaria: {len(data)}")  # Reporta cuantos dias de temperatura quedaron validos
    if not data.empty:  # Evita imprimir rangos invalidos si el DataFrame quedo vacio
        print(f"[load] Rango temperatura diaria: {data['date'].min()} -> {data['date'].max()}")  # Reporta cobertura temporal del dataset diario

    return data  # Devuelve serie diaria lista para fusionarse con la serie de 5 minutos


def _merge_daily_temperature(df_timeseries: pd.DataFrame, df_temperature: pd.DataFrame) -> pd.DataFrame:
    """Merge daily temperature into the 5-minute time series using calendar date."""
    df_merged = df_timeseries.copy()  # Trabaja sobre copia para no mutar la serie original fuera de la funcion
    df_merged["date"] = pd.to_datetime(  # Deriva la fecha diaria desde cada timestamp de 5 minutos
        df_merged["timestamp"],  # Columna temporal base del dataset principal
        errors="coerce",  # Protege frente a timestamps invalidos aunque no se esperan aqui
    ).dt.normalize()  # Normaliza a medianoche para compartir la misma llave diaria del dataset de temperatura

    if df_temperature.empty:  # Maneja localmente el caso en que el archivo de Drive no exista
        df_merged["temp_daily_f"] = DEFAULT_TEMP_F  # Usa 50 F como valor neutral para no propagar NaN al pipeline
        print(f"[load] Temperatura diaria no disponible; se usara fallback de {DEFAULT_TEMP_F:.1f} F")  # Explica el fallback aplicado
        return df_merged.drop(columns=["date"])  # Elimina llave auxiliar antes de devolver la tabla

    df_merged = df_merged.merge(  # Une temperatura diaria a cada fila de 5 minutos por fecha de calendario
        df_temperature,  # DataFrame diario con una sola fila por fecha
        on="date",  # Usa la fecha diaria como llave de union
        how="left",  # Conserva toda la serie principal aunque falte alguna temperatura puntual
    )
    missing_temperature_count = int(df_merged["temp_daily_f"].isna().sum())  # Cuenta filas sin temperatura tras el merge
    if missing_temperature_count > 0:  # Solo aplica fallback cuando existen huecos reales en temperatura
        df_merged["temp_daily_f"] = df_merged["temp_daily_f"].fillna(DEFAULT_TEMP_F)  # Rellena con 50 F para mantener neutralidad del API
    print(f"[load] Filas sin temperatura diaria tras merge: {missing_temperature_count}")  # Reporta cobertura efectiva del merge diario

    return df_merged.drop(columns=["date"])  # Retira la llave auxiliar porque no es feature final del modelo


def _read_events_file(file_path: Path) -> pd.DataFrame:
    """Read the events .dat file with a trailing empty column."""
    event_columns = [  # Define los 32 nombres esperados en el archivo de eventos
        "event_id",  # Identificador numerico del evento
        "is_valid",  # Bandera de validez como texto True/False
        "event_start",  # Fecha y hora de inicio del evento
        "event_end",  # Fecha y hora de fin del evento
        "failure_description",  # Descripcion de fallo si existe
        "precond_24h_volume",  # Volumen precondicionante 24h
        "volume_1h",  # Volumen acumulado a 1h
        "volume_2h",  # Volumen acumulado a 2h
        "volume_4h",  # Volumen acumulado a 4h
        "volume_8h",  # Volumen acumulado a 8h
        "volume_24h",  # Volumen acumulado a 24h
        "total_volume",  # Volumen total del evento
        "duration_hours",  # Duracion en horas
        "total_rainfall",  # Lluvia total del evento
        "peak_wwf",  # Pico de flujo en wet weather flow
        "reactive_wwf",  # Variable de respuesta a wwf
        "precond_24h_base_vol",  # Volumen base 24h previo
        "precond_24h_base_vol_str",  # Campo string asociado a baseflow previo
        "max_hourly_flow",  # Maximo flujo horario
        "avg_sewage_at_max",  # Promedio de sewage al maximo
        "event_end_index",  # Indice de fin del evento
        "max_hourly_storm_intensity",  # Maxima intensidad horaria de tormenta
        "precond_24h_vol_str",  # Campo string asociado a volumen 24h
        "total_rain",  # Lluvia total alternativa
        "total_baseflow_vol",  # Volumen total de baseflow
        "total_storm_vol",  # Volumen total de stormflow
        "ignore_has_impact",  # Bandera de ignorar con impacto
        "ignore_no_impact",  # Bandera de ignorar sin impacto
        "runoff_coefficient",  # Coeficiente de escorrentia
        "event_start_index",  # Indice de inicio del evento
        "use_prev_24h_baseflow",  # Bandera para usar baseflow 24h previo
        "event_id_str",  # Identificador del evento como string
    ]
    data = pd.read_csv(  # Lee el archivo de eventos en bloque para mantener velocidad
        file_path,  # Ruta del .dat
        sep="|",  # Separador pipe segun el formato de eventos
        header=None,  # No hay cabecera en el archivo
        engine="python",  # Motor tolerante a separadores irregulares
    )
    if data.shape[1] > 0:  # Verifica que haya columnas antes de recortar
        data = data.iloc[:, :-1]  # Descarta la columna vacia generada por el pipe final
    data = data.iloc[:, : len(event_columns)]  # Asegura exactamente 32 columnas esperadas
    data.columns = event_columns  # Asigna nombres de columnas segun el esquema

    data["event_start"] = pd.to_datetime(  # Convierte inicio a datetime para filtros temporales
        data["event_start"],  # Columna original con texto de fecha
        errors="coerce",  # Convierte errores en NaT para filtrado seguro
    )
    data["event_end"] = pd.to_datetime(  # Convierte fin a datetime para filtros temporales
        data["event_end"],  # Columna original con texto de fecha
        errors="coerce",  # Convierte errores en NaT para filtrado seguro
    )

    data["is_valid"] = (  # Convierte bandera de texto a booleano real
        data["is_valid"].astype(str)  # Asegura tipo string para poder normalizar
        .str.strip()  # Elimina espacios alrededor
        .str.lower()  # Normaliza mayusculas/minusculas
        .map({"true": True, "false": False})  # Mapea texto a booleano
    )

    numeric_columns = [  # Lista de columnas numericas clave a convertir
        "total_volume",  # Volumen total del evento
        "duration_hours",  # Duracion en horas
        "total_rainfall",  # Lluvia total del evento
        "peak_wwf",  # Pico de flujo
        "total_rain",  # Lluvia total alternativa
        "total_storm_vol",  # Volumen total de stormflow
        "runoff_coefficient",  # Coeficiente de escorrentia
    ]
    for column_name in numeric_columns:  # Recorre columnas para conversion numerica
        data[column_name] = pd.to_numeric(  # Convierte texto a float de forma segura
            data[column_name],  # Columna a convertir
            errors="coerce",  # Convierte valores invalidos en NaN
        )

    data = data[  # Filtra eventos validos y con fechas correctas
        (data["is_valid"] == True)  # Solo eventos marcados como validos
        & data["event_start"].notna()  # Excluye inicios invalidos
        & data["event_end"].notna()  # Excluye finales invalidos
    ]

    return data  # Devuelve eventos limpios y tipados


def _load_part(base_path: Path, tsf_files: Dict[str, str], events_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a single part (1parte or 2parte) and return time series and events."""
    rain_path = base_path / tsf_files["rain"]  # Construye la ruta del archivo de lluvia
    flow_path = base_path / tsf_files["flow"]  # Construye la ruta del archivo de caudal total
    stormflow_path = base_path / tsf_files["stormflow"]  # Construye la ruta del archivo de stormflow
    events_path = base_path / events_filename  # Construye la ruta del archivo de eventos

    rain_df = _read_tsf_file(rain_path, "rain_in")  # Carga lluvia con nombre objetivo
    flow_df = _read_tsf_file(flow_path, "flow_total_mgd")  # Carga flujo total con nombre objetivo
    stormflow_df = _read_tsf_file(stormflow_path, "stormflow_mgd")  # Carga stormflow con nombre objetivo

    merged_df = rain_df.merge(  # Une lluvia con flujo total por timestamp
        flow_df,  # Segundo DataFrame para el join
        on="timestamp",  # Join exacto por tiempo para alinear mediciones
        how="inner",  # Inner join para quedarnos solo con timestamps comunes
    )
    merged_df = merged_df.merge(  # Une el resultado con stormflow por timestamp
        stormflow_df,  # DataFrame de stormflow
        on="timestamp",  # Alinea por la misma columna de tiempo
        how="inner",  # Inner join para evitar filas incompletas
    )

    merged_df["baseflow_mgd"] = (  # Calcula baseflow por definicion fisica
        merged_df["flow_total_mgd"] - merged_df["stormflow_mgd"]  # Resta para separar baseflow
    )
    merged_df["baseflow_mgd"] = merged_df["baseflow_mgd"].clip(  # Evita baseflow negativo no fisico
        lower=0  # Limite inferior en cero
    )

    events_df = _read_events_file(events_path)  # Carga eventos con el formato especifico

    return merged_df, events_df  # Devuelve ambas tablas para concatenar luego


def load_msd_data(config_path: str | Path = "configs/default.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MSD time series and event data from both parts defined in the config."""
    config = _read_config(config_path)  # Lee la configuracion para rutas y nombres
    data_cfg = config["data"]  # Extrae el bloque de datos para evitar indices largos
    base_paths = [Path(path) for path in data_cfg["base_paths"]]  # Normaliza rutas de partes
    tsf_files = data_cfg["tsf_files"]  # Diccionario con nombres de archivos TSF
    events_filename = data_cfg["events_filename"]  # Nombre del archivo de eventos
    temperature_daily_path = data_cfg.get("temperature_daily_path")  # Lee la ruta opcional del archivo diario de temperatura

    all_timeseries: List[pd.DataFrame] = []  # Acumula DataFrames de tiempo de cada parte
    all_events: List[pd.DataFrame] = []  # Acumula DataFrames de eventos de cada parte

    for base_path in base_paths:  # Itera por cada parte (1parte y 2parte)
        part_timeseries, part_events = _load_part(  # Carga series y eventos de la parte
            base_path,  # Ruta de la parte
            tsf_files,  # Nombres de archivos TSF
            events_filename,  # Nombre del archivo de eventos
        )
        all_timeseries.append(part_timeseries)  # Guarda la serie de tiempo para concatenar
        all_events.append(part_events)  # Guarda eventos para concatenar

        print(  # Diagnostico del tamano por parte
            f"Registros en {base_path.name}: {len(part_timeseries)}"  # Reporta cantidad de filas
        )
        print(  # Diagnostico del rango temporal por parte
            f"Rango en {base_path.name}: {part_timeseries['timestamp'].min()} -> {part_timeseries['timestamp'].max()}"  # Reporta minimo y maximo
        )

    df_timeseries = pd.concat(all_timeseries, ignore_index=True)  # Concatena partes en una sola tabla
    df_events = pd.concat(all_events, ignore_index=True)  # Concatena eventos de ambas partes

    duplicate_count = df_timeseries.duplicated(subset=["timestamp"]).sum()  # Cuenta duplicados antes de borrar
    df_timeseries = df_timeseries.drop_duplicates(  # Elimina solapamientos por timestamp
        subset=["timestamp"],  # Usa timestamp como clave temporal
        keep="first",  # Conserva el primer registro de cada tiempo
    )
    print(  # Diagnostico de duplicados eliminados
        f"Duplicados temporales eliminados: {duplicate_count}"  # Reporta cantidad eliminada
    )
    df_timeseries = df_timeseries.sort_values("timestamp").reset_index(drop=True)  # Ordena cronologicamente para consistencia y reindexa

    if temperature_daily_path:  # Solo intenta leer temperatura si la ruta esta declarada en config
        temperature_path = Path(temperature_daily_path)  # Normaliza la ruta diaria de temperatura para operar de forma portable
        if temperature_path.exists():  # En Colab deberia existir; localmente probablemente no
            df_temperature = _read_daily_temperature_file(temperature_path)  # Carga la tabla diaria de temperatura desde Drive
            df_timeseries = _merge_daily_temperature(df_timeseries, df_temperature)  # Replica la temperatura diaria sobre cada bloque de 5 minutos
        else:  # Maneja el caso local donde la ruta hardcodeada del Drive no esta montada
            print(f"[load] Archivo de temperatura no encontrado: {temperature_path}")  # Informa por que no pudo leerse el dataset externo
            df_timeseries = _merge_daily_temperature(  # Mantiene la columna disponible aunque no exista el archivo real
                df_timeseries,  # Serie principal ya concatenada y ordenada
                pd.DataFrame(columns=["date", "temp_daily_f"]),  # DataFrame vacio para activar el fallback neutral
            )
    else:  # Mantiene compatibilidad si alguien borra la ruta del YAML en una sesion experimental
        df_timeseries = _merge_daily_temperature(  # Garantiza que la columna exista incluso sin config explicita
            df_timeseries,  # Serie principal ya concatenada y ordenada
            pd.DataFrame(columns=["date", "temp_daily_f"]),  # DataFrame vacio para usar el fallback neutral
        )

    df_events = df_events.drop_duplicates()  # Elimina eventos duplicados exactos
    print(  # Diagnostico de eventos validos
        f"Eventos validos: {len(df_events)}"  # Reporta total de eventos validos
    )

    nan_count = df_timeseries.isna().sum().sum()  # Cuenta NaN totales en la tabla final
    print(  # Diagnostico de estadisticas basicas del DataFrame final
        f"Timeseries final shape: {df_timeseries.shape}, NaN count: {nan_count}"  # Reporta shape y NaN
    )

    return df_timeseries, df_events  # Retorna ambos DataFrames segun el contrato
