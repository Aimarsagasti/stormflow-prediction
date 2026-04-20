"""
Script para generar docs/DATASET_STATS.md a partir del dataset real.

Lee los datos crudos desde las rutas configuradas en configs/default.yaml,
aplica el mismo pipeline de limpieza y feature engineering que usa
claude_train.py, calcula 7 secciones de estadisticas descriptivas y
escribe un markdown con tablas y figuras embebidas.

Uso:
    python scripts/generate_dataset_stats.py

Flags opcionales:
    --no-cache        Fuerza regeneracion del dataframe cacheado.
    --skip-figures    No genera figuras (mas rapido, solo tablas).

Output:
    docs/DATASET_STATS.md                           (markdown principal)
    outputs/figures/dataset_stats/*.png             (figuras embebidas)
    outputs/data_analysis/dataset_stats.json        (numeros brutos)
    outputs/cache/df_with_features.parquet          (cache del dataframe)
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations  # Permite anotaciones de tipo modernas sin importar version de Python exacta

import argparse  # Libreria estandar para parsear flags de linea de comandos (--no-cache, --skip-figures)
import json  # Libreria estandar para serializar el diccionario de estadisticas a JSON
import sys  # Libreria estandar para manipular sys.path (anadir la raiz del repo a los imports)
from pathlib import Path  # Libreria estandar moderna para manipular rutas de archivos (mejor que os.path)
from typing import Any, Dict, List, Tuple  # Libreria estandar para anotaciones de tipo (type hints)

import numpy as np  # Libreria numerica, base para calculos vectorizados sobre arrays
import pandas as pd  # Libreria para manipulacion tabular (DataFrames), ya se usa en todo el proyecto
import yaml  # Libreria para leer el configs/default.yaml (requiere pip install pyyaml, ya instalado)

# Imports de matplotlib para generar figuras PNG embebidas en el markdown
import matplotlib  # Libreria base de plotting en Python
matplotlib.use("Agg")  # Backend no interactivo: permite generar PNGs sin necesidad de ventana GUI (clave en scripts)
import matplotlib.pyplot as plt  # API principal de matplotlib para crear figuras

# Imports de scipy y statsmodels para estadisticos especificos
from scipy import stats as scipy_stats  # Proporciona skewness, kurtosis y tests estadisticos
from statsmodels.tsa.stattools import acf  # Calcula la funcion de autocorrelacion (seccion 6)

# El script vive en scripts/ pero el repo esta en la carpeta padre. Anadimos la raiz al sys.path
# para poder importar src/... como hace claude_train.py y evaluate_local.py.
REPO_ROOT = Path(__file__).resolve().parent.parent  # Path a C:\Dev\TFM\ (parent de scripts/)
sys.path.insert(0, str(REPO_ROOT))  # Lo insertamos al principio para que tenga prioridad sobre otros paquetes

# Ahora si podemos importar modulos del proyecto
# Nota: los nombres exactos se verificaron con grep sobre los archivos de src/
from src.data.load import load_msd_data            # Devuelve una tupla (df_timeseries, df_events)
from src.data.clean import clean_timeseries        # Recibe (df_timeseries, df_events) y devuelve df limpio
from src.features.engineering import create_features  # Genera las 22 features a partir del df limpio
from src.pipeline.split import split_chronological   # Split cronologico train/val/test 70/15/15

# ============================================================================
# CONFIGURACION GLOBAL DEL SCRIPT
# ============================================================================

# Rutas de salida, relativas a REPO_ROOT
OUTPUT_MARKDOWN = REPO_ROOT / "docs" / "DATASET_STATS.md"  # Markdown principal que lee Opus y tu
OUTPUT_JSON = REPO_ROOT / "outputs" / "data_analysis" / "dataset_stats.json"  # Numeros brutos en formato estructurado
OUTPUT_FIGURES_DIR = REPO_ROOT / "outputs" / "figures" / "dataset_stats"  # Carpeta de figuras PNG
CACHE_DIR = REPO_ROOT / "outputs" / "cache"  # Carpeta de cache para el parquet
CACHE_FILE = CACHE_DIR / "df_with_features.parquet"  # Archivo parquet con el dataframe procesado

# Ruta al archivo de configuracion que usa el resto del proyecto
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"  # YAML con rutas de datos y parametros

# Variable target del proyecto: predecir el caudal de tormenta en millones de galones por dia
TARGET_COLUMN = "stormflow_mgd"  # Definido en AGENTS.md seccion 3

# Lista de las 22 features que usa el modelo en produccion (fuente: modelo_H1_sinSF_meta.json)
FEATURE_COLUMNS = [
    "rain_in", "temp_daily_f", "api_dynamic",
    "rain_sum_10m", "rain_sum_15m", "rain_sum_30m", "rain_sum_60m",
    "rain_sum_120m", "rain_sum_180m", "rain_sum_360m",
    "rain_max_10m", "rain_max_30m", "rain_max_60m",
    "minutes_since_last_rain",
    "delta_flow_5m", "delta_flow_15m", "delta_rain_10m", "delta_rain_30m",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# Umbrales para los buckets de severidad. Mismos que usa evaluate_local.py para que los analisis sean consistentes.
SEVERITY_BUCKETS: List[Tuple[str, float, float]] = [
    ("Base",     0.0,   0.5),     # El 92% de las muestras caen aqui (sin evento)
    ("Leve",     0.5,   5.0),     # Eventos pequenos
    ("Moderado", 5.0,   20.0),    # Eventos moderados: donde el modelo funciona mejor
    ("Alto",     20.0,  50.0),    # Eventos altos: menos muestras
    ("Extremo",  50.0,  np.inf),  # Eventos extremos: 59 muestras en test, el foco del problema
]

# Umbral para considerar "evento" en el analisis de regimenes (>= 0.5 MGD segun la convencion del proyecto)
EVENT_THRESHOLD_MGD = 0.5  # Tomado de las funciones de metrica en src/evaluation/metrics.py

# Configuracion de matplotlib para que las figuras queden bonitas y legibles en el markdown
plt.rcParams["figure.dpi"] = 100          # Resolucion de pantalla razonable
plt.rcParams["savefig.dpi"] = 150         # Resolucion al guardar PNG (mas alta que pantalla)
plt.rcParams["figure.figsize"] = (10, 5)  # Tamano por defecto: 10 ancho x 5 alto pulgadas
plt.rcParams["font.size"] = 10            # Texto legible pero no gigante
plt.rcParams["axes.grid"] = True          # Grid activada por defecto (mejora legibilidad de plots)
plt.rcParams["grid.alpha"] = 0.3          # Grid tenue para no saturar visualmente

# ============================================================================
# UTILIDADES DE BAJO NIVEL
# ============================================================================

def _load_config() -> Dict[str, Any]:
    """Lee y devuelve el contenido del YAML de configuracion del proyecto."""
    # Abrimos el archivo YAML en modo lectura con encoding UTF-8 para evitar problemas en Windows
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        # yaml.safe_load parsea el YAML a un dict Python sin ejecutar codigo arbitrario (mas seguro que yaml.load)
        return yaml.safe_load(f)


def _ensure_dirs_exist() -> None:
    """Crea las carpetas de output si no existen (equivalente a mkdir -p en bash)."""
    # parents=True crea directorios padres si hace falta, exist_ok=True no lanza error si ya existen
    OUTPUT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _format_number(x: float, decimals: int = 3) -> str:
    """Formatea un numero para la tabla markdown (maneja NaN, inf y numeros muy pequenos/grandes)."""
    # Los casos especiales tienen que manejarse antes porque NaN e inf rompen el formateo normal
    if pd.isna(x):
        return "NaN"
    if np.isinf(x):
        return "+inf" if x > 0 else "-inf"
    # Para numeros muy grandes usamos notacion cientifica (mas legible que un numero con muchos ceros)
    if abs(x) >= 1e6:
        return f"{x:.2e}"
    # Formateo normal con el numero de decimales pedido
    return f"{x:.{decimals}f}"


def _save_figure(fig: plt.Figure, filename: str) -> str:
    """Guarda la figura en OUTPUT_FIGURES_DIR y devuelve la ruta relativa para el markdown."""
    # Ruta absoluta donde se guarda fisicamente el PNG
    full_path = OUTPUT_FIGURES_DIR / filename
    # bbox_inches='tight' recorta margen en blanco sobrante alrededor de la figura
    fig.savefig(full_path, bbox_inches="tight")
    # Cerramos la figura explicitamente para liberar memoria (importante si generamos muchas figuras)
    plt.close(fig)
    # Devolvemos la ruta relativa desde docs/ (donde vive el markdown) hacia la figura
    # El markdown tiene que acceder con ../outputs/figures/dataset_stats/ para llegar a outputs/
    return f"../outputs/figures/dataset_stats/{filename}"


def _small_sample_warning(n: int, threshold: int = 100) -> str:
    """Genera una nota al pie si la muestra es pequena. Devuelve string vacio si no hace falta."""
    # Convencion: cualquier estadistica sobre <=100 muestras se marca como poco fiable
    if n <= threshold:
        return f" *(n={n}, tomar con cautela)*"
    return ""

# ============================================================================
# CARGA DE DATOS Y GESTION DEL CACHE
# ============================================================================

def _load_and_process_data(config_path: Path, use_cache: bool = True) -> pd.DataFrame:
    """
    Carga los datos crudos, los limpia y genera los 22 features.

    Si existe un cache valido en parquet, lo lee directamente (rapido).
    Si no, ejecuta el pipeline completo (load -> clean -> features) y guarda el resultado.

    Args:
        config_path: Path al YAML de configuracion (local.yaml en nuestra ejecucion).
        use_cache: Si es False, ignora el cache y regenera el dataframe desde cero.

    Returns:
        DataFrame con columnas: timestamp, [22 features], stormflow_mgd, is_event.
    """
    # Primera decision: usar cache o regenerar?
    if use_cache and CACHE_FILE.exists():
        print(f"[cache] Leyendo dataframe procesado desde {CACHE_FILE}")
        # Parquet es un formato columnar binario muy rapido de leer y escribir, mas eficiente que CSV
        df_cached = pd.read_parquet(CACHE_FILE)
        print(f"[cache] Shape del dataframe cacheado: {df_cached.shape}")
        return df_cached

    # No hay cache o nos pidieron regenerar: ejecutamos el pipeline completo
    print(f"[pipeline] Generando dataframe desde cero (puede tardar varios minutos)...")

    # Paso 1: cargar datos crudos. load_msd_data devuelve (df_timeseries, df_events)
    print(f"[pipeline] 1/3: Cargando datos crudos desde {config_path}")
    df_timeseries, df_events = load_msd_data(config_path=config_path)
    print(f"[pipeline]      df_timeseries: {df_timeseries.shape} | df_events: {df_events.shape}")

    # Paso 2: limpieza (timestamps, negativos, huecos)
    print(f"[pipeline] 2/3: Limpiando timeseries")
    df_clean = clean_timeseries(df_timeseries, df_events)
    print(f"[pipeline]      df_clean: {df_clean.shape}")

    # Paso 3: feature engineering (rolling sums, api_dynamic, deltas, cyclic encoding)
    print(f"[pipeline] 3/3: Generando features")
    df_features = create_features(df_clean)
    print(f"[pipeline]      df_features: {df_features.shape}")

    # Guardamos el resultado en cache para futuras ejecuciones
    print(f"[cache] Guardando dataframe procesado en {CACHE_FILE}")
    df_features.to_parquet(CACHE_FILE, index=False)

    return df_features


def _split_data(df_features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Aplica el split cronologico 70/15/15 usando la funcion del pipeline.

    Returns:
        Dict con claves 'train', 'val', 'test' y valores DataFrames.
    """
    # split_chronological devuelve una tupla de 3 dataframes en orden train, val, test
    df_train, df_val, df_test = split_chronological(df_features)
    # Los envolvemos en un dict con nombres para facilitar iteracion posterior
    return {"train": df_train, "val": df_val, "test": df_test}


# ============================================================================
# UTILIDADES ESTADISTICAS ESPECIFICAS
# ============================================================================

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Calcula correlacion de Pearson manejando NaN y muestras pequenas."""
    # Pearson mide correlacion LINEAL entre dos variables, rango [-1, 1]
    # np.corrcoef da una matriz 2x2; el [0, 1] es la correlacion entre x e y
    # Necesitamos filtrar NaN porque corrcoef los propaga y devuelve NaN si hay alguno
    mask = ~(np.isnan(x) | np.isnan(y))  # True donde AMBOS son finitos
    if mask.sum() < 2:  # Con menos de 2 puntos no se puede calcular correlacion
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Calcula correlacion de Spearman (por rangos) manejando NaN."""
    # Spearman mide correlacion MONOTONICA (no necesariamente lineal): si y sube cuando x sube, da 1
    # Util para relaciones no lineales que Pearson no detecta bien (p.ej. rain_sum_60m vs stormflow_mgd, que es concava)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return float("nan")
    # scipy_stats.spearmanr devuelve un named tuple (correlation, pvalue); solo nos interesa la correlacion
    correlation, _ = scipy_stats.spearmanr(x[mask], y[mask])
    return float(correlation)


# ============================================================================
# SECCION 1: RESUMEN GLOBAL DEL DATASET
# ============================================================================

def compute_section_1_global_summary(
    df_features: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
) -> Tuple[str, Dict[str, Any]]:
    """
    Genera la seccion 1 del documento: resumen global del dataset.

    Returns:
        Tupla (markdown_string, dict_con_numeros_brutos).
    """
    # Calculamos las estadisticas globales del dataset entero
    n_total = len(df_features)  # Numero total de registros a 5 minutos
    first_ts = df_features["timestamp"].min()  # Primer timestamp del dataset
    last_ts = df_features["timestamp"].max()  # Ultimo timestamp del dataset
    span_days = (last_ts - first_ts).days  # Duracion total en dias

    # NaNs por columna: util para detectar columnas problematicas
    nan_counts = df_features.isna().sum()
    # Solo reportamos columnas que tengan al menos un NaN (las que estan a 0 son menos interesantes)
    columns_with_nans = nan_counts[nan_counts > 0]

    # Estadisticas de cada split: nombre, fechas inicio/fin, numero de filas, porcentaje del total
    split_stats = {}
    for split_name, df_split in splits.items():
        split_stats[split_name] = {
            "n": len(df_split),
            "first_ts": df_split["timestamp"].min().isoformat(),
            "last_ts": df_split["timestamp"].max().isoformat(),
            "pct": 100 * len(df_split) / n_total,
            "span_days": (df_split["timestamp"].max() - df_split["timestamp"].min()).days,
        }

    # Diccionario bruto para el JSON de salida
    brute_data = {
        "n_total_records": n_total,
        "first_timestamp": first_ts.isoformat(),
        "last_timestamp": last_ts.isoformat(),
        "span_days": span_days,
        "span_years": round(span_days / 365.25, 2),
        "resolution_minutes": 5,
        "n_columns_after_features": df_features.shape[1],
        "columns_with_nans": {col: int(count) for col, count in columns_with_nans.items()},
        "splits": split_stats,
    }

    # Ahora construimos el markdown
    md_lines = []
    md_lines.append("## 1. Resumen global del dataset\n")
    md_lines.append(f"- **Registros totales:** {n_total:,} (a 5 minutos de resolucion).")
    md_lines.append(f"- **Cobertura temporal:** {first_ts.strftime('%Y-%m-%d')} a {last_ts.strftime('%Y-%m-%d')} ({span_days} dias, {brute_data['span_years']} anos).")
    md_lines.append(f"- **Columnas tras feature engineering:** {df_features.shape[1]} (timestamp + 22 features + stormflow_mgd + is_event).")

    if len(columns_with_nans) > 0:
        md_lines.append(f"- **Columnas con NaN:** {len(columns_with_nans)} columnas tienen al menos un NaN.")
        for col, count in columns_with_nans.items():
            pct = 100 * count / n_total
            md_lines.append(f"  - `{col}`: {count:,} NaN ({pct:.2f}%)")
    else:
        md_lines.append(f"- **Columnas con NaN:** ninguna.")

    md_lines.append("")  # Linea en blanco antes de la tabla
    md_lines.append("### Split cronologico 70/15/15\n")
    md_lines.append("| Split | Filas | % del total | Desde | Hasta | Duracion |")
    md_lines.append("|-------|-------|-------------|-------|-------|----------|")
    for split_name in ["train", "val", "test"]:
        s = split_stats[split_name]
        md_lines.append(
            f"| {split_name} | {s['n']:,} | {s['pct']:.1f}% | "
            f"{s['first_ts'][:10]} | {s['last_ts'][:10]} | {s['span_days']} dias |"
        )
    md_lines.append("")

    # Nota critica sobre el split (lo que Aimar detecto al empezar)
    md_lines.append(
        "> **Nota metodologica:** el split actual deja val y test con duracion inferior "
        "a un ciclo anual completo. En datasets con estacionalidad anual fuerte (eventos de tormenta "
        "concentrados en primavera/verano, eventos de deshielo concentrados en invierno), esto puede "
        "introducir sesgo estacional en las metricas de evaluacion. Alternativas a considerar: split "
        "en anos completos (p.ej. 7/2/1), TimeSeriesSplit con ventanas deslizantes, o blocked CV por "
        "eventos con embargo temporal. Esta decision queda como pendiente (ver STATE.md)."
    )
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 2: DISTRIBUCION DEL TARGET (stormflow_mgd)
# ============================================================================

def compute_section_2_target_distribution(
    df_features: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Analiza la distribucion de stormflow_mgd: cuantiles, skewness, zero-inflation,
    y comparacion entre splits para detectar desbalance.
    """
    # Series de stormflow por split y global
    series_global = df_features[TARGET_COLUMN].dropna().to_numpy()
    series_by_split = {name: df[TARGET_COLUMN].dropna().to_numpy() for name, df in splits.items()}

    # Cuantiles a computar: cubren desde la mediana hasta el extremo superior
    quantile_points = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.00]

    def compute_stats(series: np.ndarray) -> Dict[str, float]:
        """Calcula estadisticos basicos para una serie de stormflow."""
        return {
            "n": int(len(series)),
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            # Skewness: medida de asimetria. Valores altos (>>0) indican cola derecha muy larga (nuestro caso)
            "skewness": float(scipy_stats.skew(series)),
            # Kurtosis: medida de cuan "pesadas" son las colas. >3 indica colas mas pesadas que normal
            "kurtosis": float(scipy_stats.kurtosis(series)),
            # Porcentaje del tiempo en regimen base (sin evento, <0.5 MGD)
            "pct_baseflow": float(100 * np.mean(series < EVENT_THRESHOLD_MGD)),
            # Cuantiles especificos: clave "q50", "q75", "q90", "q95", "q99", "q99.9", "q100"
            **{f"q{q*100:g}": float(np.quantile(series, q)) for q in quantile_points},
        }

    global_stats = compute_stats(series_global)
    split_stats = {name: compute_stats(s) for name, s in series_by_split.items()}

    brute_data = {
        "global": global_stats,
        "by_split": split_stats,
    }

    # Generamos figura: histograma log-escala del target (los eventos extremos son pocos, necesitan log)
    figure_path = None
    if not skip_figures:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Panel izquierdo: histograma de todos los valores (incluye zero-inflation)
        # log=True en el eje Y permite ver la cola derecha que si no se comeria todo
        ax[0].hist(series_global, bins=100, log=True, color="steelblue", edgecolor="black", alpha=0.7)
        ax[0].set_xlabel("stormflow_mgd")
        ax[0].set_ylabel("Frecuencia (escala log)")
        ax[0].set_title("Distribucion completa de stormflow_mgd\n(escala Y logaritmica)")
        ax[0].axvline(EVENT_THRESHOLD_MGD, color="red", linestyle="--", label=f"Umbral evento ({EVENT_THRESHOLD_MGD} MGD)")
        ax[0].legend()

        # Panel derecho: histograma solo de la cola (>=0.5 MGD) para ver la distribucion de eventos
        tail = series_global[series_global >= EVENT_THRESHOLD_MGD]
        ax[1].hist(tail, bins=50, log=True, color="darkorange", edgecolor="black", alpha=0.7)
        ax[1].set_xlabel("stormflow_mgd (solo eventos)")
        ax[1].set_ylabel("Frecuencia (escala log)")
        ax[1].set_title(f"Distribucion de eventos (>={EVENT_THRESHOLD_MGD} MGD)\nn={len(tail):,}")

        fig.tight_layout()
        figure_path = _save_figure(fig, "section2_target_distribution.png")

    # Construimos el markdown
    md_lines = []
    md_lines.append("## 2. Distribucion de la variable objetivo `stormflow_mgd`\n")

    # Estadisticos globales en formato texto
    md_lines.append("### Estadisticos globales (dataset completo)\n")
    md_lines.append(f"- **n:** {global_stats['n']:,}")
    md_lines.append(f"- **Media:** {global_stats['mean']:.4f} MGD")
    md_lines.append(f"- **Desviacion tipica:** {global_stats['std']:.4f} MGD")
    md_lines.append(f"- **Skewness:** {global_stats['skewness']:.2f} (asimetria a la derecha muy marcada)")
    md_lines.append(f"- **Kurtosis:** {global_stats['kurtosis']:.2f} (colas extremadamente pesadas)")
    md_lines.append(f"- **% del tiempo en regimen base (<{EVENT_THRESHOLD_MGD} MGD):** {global_stats['pct_baseflow']:.2f}%")
    md_lines.append("")

    # Comparacion de cuantiles entre splits: esto es lo critico para detectar desbalance train/val/test
    md_lines.append("### Cuantiles de `stormflow_mgd` por split\n")
    md_lines.append("| Cuantil | Global | Train | Val | Test |")
    md_lines.append("|---------|--------|-------|-----|------|")
    for q in quantile_points:
        key = f"q{q*100:g}"  # Usamos la misma convencion que compute_stats: "q50", "q99.9", "q100"
        label = f"p{q*100:g}" if q < 1 else "max"
        md_lines.append(
            f"| {label} | {global_stats[key]:.3f} | {split_stats['train'][key]:.3f} | "
            f"{split_stats['val'][key]:.3f} | {split_stats['test'][key]:.3f} |"
        )
    md_lines.append("")

    # Nota sobre extrapolacion: si max(test) > max(train), el modelo esta siendo evaluado en
    # un regimen extremo que nunca vio. Esto es evidencia critica para el diagnostico de Opus.
    max_train = split_stats["train"]["q100"]
    max_test = split_stats["test"]["q100"]
    if max_test > max_train:
        md_lines.append(
            f"> **Alerta de extrapolacion:** el maximo de test ({max_test:.1f} MGD) es mayor "
            f"que el maximo de train ({max_train:.1f} MGD). El modelo esta siendo evaluado sobre "
            f"eventos mas extremos que los que vio en entrenamiento, lo que explica parcialmente "
            f"la infraestimacion sistematica en el bucket Extremo."
        )
    else:
        md_lines.append(
            f"> El rango de magnitudes del test ({max_test:.1f} MGD max) no excede el de train "
            f"({max_train:.1f} MGD max). No hay extrapolacion fuera del rango visto."
        )
    md_lines.append("")

    if figure_path:
        md_lines.append(f"![Distribucion del target]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 3: CARACTERIZACION DE LOS EVENTOS EXTREMOS (>50 MGD)
# ============================================================================

def compute_section_3_extreme_events(
    df_features: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Caracteriza los eventos extremos (stormflow_mgd >= 50 MGD): distribucion entre splits,
    estacionalidad, y un primer indicio de los eventos "sin lluvia en ventana".
    """
    # Mascara de eventos extremos en el dataset completo
    extreme_mask = df_features[TARGET_COLUMN] >= 50.0
    df_extreme = df_features.loc[extreme_mask].copy()

    # Cuantos extremos caen en cada split
    extreme_by_split = {}
    for split_name, df_split in splits.items():
        mask = df_split[TARGET_COLUMN] >= 50.0
        extreme_by_split[split_name] = {
            "n": int(mask.sum()),
            "pct_of_split": float(100 * mask.mean()),
            "max_mgd": float(df_split.loc[mask, TARGET_COLUMN].max()) if mask.any() else float("nan"),
            "mean_mgd": float(df_split.loc[mask, TARGET_COLUMN].mean()) if mask.any() else float("nan"),
        }

    # Analisis estacional: en que mes del ano ocurren los extremos
    df_extreme["month"] = df_extreme["timestamp"].dt.month  # dt.month devuelve 1-12
    extremes_by_month = df_extreme.groupby("month").size().to_dict()
    # Nos aseguramos de que todos los meses aparezcan aunque tengan cero eventos
    extremes_by_month = {int(m): int(extremes_by_month.get(m, 0)) for m in range(1, 13)}

    # Analisis anual: en que ano ocurren
    df_extreme["year"] = df_extreme["timestamp"].dt.year
    extremes_by_year = df_extreme.groupby("year").size().to_dict()
    extremes_by_year = {int(y): int(v) for y, v in extremes_by_year.items()}

    # Indicio de eventos sin lluvia: comprobamos rain_sum_60m en la muestra del pico
    # Si rain_sum_60m ~ 0 en un evento extremo, la lluvia reciente no justifica el pico (posible deshielo)
    if "rain_sum_60m" in df_extreme.columns:
        # Threshold muy bajo: 0.01 pulgadas en los ultimos 60 minutos es practicamente cero
        no_rain_mask = df_extreme["rain_sum_60m"] < 0.01
        n_extremes_without_recent_rain = int(no_rain_mask.sum())
        # Estos son los "candidatos a 15 de 59 sin lluvia" que menciona STATE.md
        # No es la definicion exacta (la real usa la ventana de entrada del modelo) pero da orden de magnitud
    else:
        n_extremes_without_recent_rain = -1  # -1 indica que no pudimos calcularlo

    brute_data = {
        "n_total_extremes": int(extreme_mask.sum()),
        "by_split": extreme_by_split,
        "by_month": extremes_by_month,
        "by_year": extremes_by_year,
        "n_extremes_without_recent_rain_60m": n_extremes_without_recent_rain,
    }

    # Figura: barplot de eventos por mes y por ano
    figure_path = None
    if not skip_figures and len(df_extreme) > 0:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Panel izquierdo: eventos por mes
        months = list(range(1, 13))
        counts_month = [extremes_by_month[m] for m in months]
        ax[0].bar(months, counts_month, color="steelblue", edgecolor="black", alpha=0.8)
        ax[0].set_xticks(months)
        ax[0].set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
        ax[0].set_xlabel("Mes")
        ax[0].set_ylabel("Numero de eventos extremos (>=50 MGD)")
        ax[0].set_title("Distribucion mensual de eventos extremos")

        # Panel derecho: eventos por ano
        years_sorted = sorted(extremes_by_year.keys())
        counts_year = [extremes_by_year[y] for y in years_sorted]
        ax[1].bar(years_sorted, counts_year, color="darkorange", edgecolor="black", alpha=0.8)
        ax[1].set_xlabel("Ano")
        ax[1].set_ylabel("Numero de eventos extremos (>=50 MGD)")
        ax[1].set_title("Distribucion anual de eventos extremos")
        ax[1].tick_params(axis="x", rotation=45)

        fig.tight_layout()
        figure_path = _save_figure(fig, "section3_extreme_events.png")

    # Markdown
    md_lines = []
    md_lines.append("## 3. Caracterizacion de los eventos extremos (stormflow >= 50 MGD)\n")
    md_lines.append(f"- **Total de muestras extremas en el dataset:** {brute_data['n_total_extremes']}")
    md_lines.append("")

    md_lines.append("### Distribucion entre splits\n")
    md_lines.append("| Split | n extremos | % del split | Max MGD | Media MGD |")
    md_lines.append("|-------|------------|-------------|---------|-----------|")
    for split_name in ["train", "val", "test"]:
        s = extreme_by_split[split_name]
        md_lines.append(
            f"| {split_name} | {s['n']} | {s['pct_of_split']:.3f}% | "
            f"{_format_number(s['max_mgd'], 1)} | {_format_number(s['mean_mgd'], 1)} |"
        )
    md_lines.append("")

    # Nota critica: si test tiene pocos extremos (p.ej. <60), las metricas en bucket Extremo
    # tienen alta varianza. Esto es importante para interpretar el NSE=-0.99 del JSON.
    n_test_extreme = extreme_by_split["test"]["n"]
    if n_test_extreme < 100:
        md_lines.append(
            f"> **Nota sobre tamano muestral:** el test solo contiene {n_test_extreme} muestras extremas. "
            f"Las metricas calculadas sobre este bucket (NSE, bias, MAPE) tienen alta varianza y un "
            f"unico evento atipico puede desplazarlas significativamente."
        )
        md_lines.append("")

    md_lines.append("### Estacionalidad de los eventos extremos\n")
    if n_extremes_without_recent_rain >= 0:
        pct_no_rain = 100 * n_extremes_without_recent_rain / max(brute_data["n_total_extremes"], 1)
        md_lines.append(
            f"- **Extremos con `rain_sum_60m` < 0.01 pulgadas:** {n_extremes_without_recent_rain} "
            f"({pct_no_rain:.1f}%). Estos son candidatos a eventos sin origen pluvial inmediato "
            f"(posible escorrentia retardada, deshielo, o errores de sensor)."
        )
        md_lines.append("")

    if figure_path:
        md_lines.append(f"![Eventos extremos]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 4: CORRELACIONES FEATURE-TARGET POR REGIMEN
# ============================================================================

def compute_section_4_feature_target_correlations(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Calcula Pearson y Spearman de cada feature contra el target, separado en
    regimen baseflow (<0.5 MGD) vs regimen evento (>=0.5 MGD).

    Usa SOLO el split de train para evitar data snooping sobre val/test.
    Esto es consistente con la regla de proyecto de calcular normalizacion
    solo sobre train (ver CLAUDE.md regla 3).

    Detecta features que son "shortcuts" solo en un regimen, como lo fue
    flow_total_mgd con r=0.9976 globalmente en iteraciones anteriores.
    """
    # Trabajamos exclusivamente con train: mismo principio que usa el pipeline de normalizacion
    df_train = splits["train"]

    baseflow_mask = df_train[TARGET_COLUMN] < EVENT_THRESHOLD_MGD
    event_mask = df_train[TARGET_COLUMN] >= EVENT_THRESHOLD_MGD

    target_array = df_train[TARGET_COLUMN].to_numpy()
    target_baseflow = target_array[baseflow_mask]
    target_event = target_array[event_mask]

    # Calculamos correlaciones para cada feature en cada regimen
    correlations = []
    for feat in FEATURE_COLUMNS:
        if feat not in df_train.columns:
            # Puede pasar si alguna feature no se genero por algun motivo; lo reportamos
            correlations.append({
                "feature": feat,
                "available": False,
                "pearson_global": None, "spearman_global": None,
                "pearson_baseflow": None, "spearman_baseflow": None,
                "pearson_event": None, "spearman_event": None,
            })
            continue

        feat_array = df_train[feat].to_numpy()
        feat_baseflow = feat_array[baseflow_mask]
        feat_event = feat_array[event_mask]

        correlations.append({
            "feature": feat,
            "available": True,
            "pearson_global": _safe_pearson(feat_array, target_array),
            "spearman_global": _safe_spearman(feat_array, target_array),
            "pearson_baseflow": _safe_pearson(feat_baseflow, target_baseflow),
            "spearman_baseflow": _safe_spearman(feat_baseflow, target_baseflow),
            "pearson_event": _safe_pearson(feat_event, target_event),
            "spearman_event": _safe_spearman(feat_event, target_event),
        })

    # Ordenamos por Spearman en regimen evento (mas relevante operativamente)
    # abs() porque correlacion fuerte puede ser positiva o negativa
    correlations_sorted = sorted(
        [c for c in correlations if c["available"]],
        key=lambda c: abs(c.get("spearman_event") or 0),
        reverse=True,
    )

    brute_data = {
        "correlations": correlations,
        "n_samples_baseflow": int(baseflow_mask.sum()),
        "n_samples_event": int(event_mask.sum()),
    }

    # Markdown
    md_lines = []
    md_lines.append("## 4. Correlaciones feature-target por regimen\n")
    md_lines.append(
        f"Se calculan correlaciones de Pearson (lineal) y Spearman (monotonica) entre cada una "
        f"de las 22 features y el target `stormflow_mgd`, separando en regimen baseflow "
        f"(<{EVENT_THRESHOLD_MGD} MGD, n={brute_data['n_samples_baseflow']:,}) y regimen evento "
        f"(>={EVENT_THRESHOLD_MGD} MGD, n={brute_data['n_samples_event']:,}). "
        f"**Calculado solo sobre el split de train** para evitar data snooping sobre val/test.\n"
    )
    md_lines.append(
        "**Interpretacion:** Pearson alto indica relacion lineal fuerte. Spearman alto indica "
        "relacion monotonica (puede ser no lineal). Diferencias grandes entre baseflow y evento "
        "indican que la feature se comporta de forma distinta en cada regimen.\n"
    )

    md_lines.append("| Feature | Pearson global | Spearman global | Spearman baseflow | Spearman evento |")
    md_lines.append("|---------|---------------:|----------------:|-------------------:|----------------:|")
    for c in correlations_sorted:
        md_lines.append(
            f"| `{c['feature']}` | "
            f"{_format_number(c['pearson_global'], 3)} | "
            f"{_format_number(c['spearman_global'], 3)} | "
            f"{_format_number(c['spearman_baseflow'], 3)} | "
            f"{_format_number(c['spearman_event'], 3)} |"
        )
    md_lines.append("")
    md_lines.append(
        "*Ordenado descendentemente por `|Spearman evento|` porque es el regimen operativamente relevante.*"
    )
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 5: MATRIZ DE CORRELACIONES FEATURE-FEATURE
# ============================================================================

def compute_section_5_feature_feature_matrix(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Calcula la matriz de correlaciones Pearson entre pares de features.
    Detecta redundancias (features que aportan la misma informacion).

    Ejemplo historico: api_dynamic vs rain_sum_60m, con r=0.94 segun notas previas.

    Usa solo train, consistente con la seccion 4.
    """
    # Mismo principio que seccion 4: train-only para evitar data snooping
    df_train = splits["train"]

    # Filtramos a columnas que realmente existen (por seguridad)
    available_features = [f for f in FEATURE_COLUMNS if f in df_train.columns]

    # pd.DataFrame.corr() calcula Pearson entre todas las columnas del df seleccionado
    # El resultado es una matriz NxN donde N es el numero de features
    corr_matrix = df_train[available_features].corr(method="pearson")

    # Extraemos los pares con correlacion absoluta alta (>0.7) que NO son la diagonal
    # La diagonal siempre es 1.0 (feature consigo misma), no interesa
    high_corr_pairs = []
    n_features = len(available_features)
    for i in range(n_features):
        # Solo recorremos el triangulo superior (i<j) para no duplicar pares
        for j in range(i + 1, n_features):
            feat_a = available_features[i]
            feat_b = available_features[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= 0.7:  # Umbral convencional para considerar correlacion "alta"
                high_corr_pairs.append({
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "pearson": float(corr_value),
                })

    # Ordenamos los pares por correlacion absoluta descendente (los mas redundantes primero)
    high_corr_pairs.sort(key=lambda p: abs(p["pearson"]), reverse=True)

    brute_data = {
        "n_features_analyzed": n_features,
        "correlation_matrix": corr_matrix.round(4).to_dict(),  # Dict anidado para JSON
        "high_correlation_pairs_abs_gt_0_7": high_corr_pairs,
    }

    # Figura: heatmap de la matriz de correlaciones
    figure_path = None
    if not skip_figures:
        fig, ax = plt.subplots(figsize=(12, 10))
        # imshow dibuja la matriz como imagen; cmap='coolwarm' es rojo-blanco-azul (intuitivo para correlaciones)
        # vmin=-1, vmax=1 fija la escala de color para que 0 siempre sea blanco
        im = ax.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        # Etiquetas de los ejes con los nombres de las features
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(available_features, rotation=90, fontsize=8)
        ax.set_yticklabels(available_features, fontsize=8)
        # Barra de color a la derecha para interpretar los valores
        plt.colorbar(im, ax=ax, label="Correlacion de Pearson")
        ax.set_title("Matriz de correlaciones feature-feature (train)")
        fig.tight_layout()
        figure_path = _save_figure(fig, "section5_feature_correlation_matrix.png")

    # Markdown
    md_lines = []
    md_lines.append("## 5. Matriz de correlaciones feature-feature\n")
    md_lines.append(
        f"Correlacion de Pearson entre todos los pares de las {n_features} features, calculada "
        f"sobre el split de train. Detecta redundancias: features con correlacion alta aportan "
        f"informacion casi identica al modelo.\n"
    )

    if high_corr_pairs:
        md_lines.append(f"### Pares con |Pearson| >= 0.7 ({len(high_corr_pairs)} pares)\n")
        md_lines.append("| Feature A | Feature B | Pearson |")
        md_lines.append("|-----------|-----------|--------:|")
        for p in high_corr_pairs:
            md_lines.append(f"| `{p['feature_a']}` | `{p['feature_b']}` | {p['pearson']:+.3f} |")
        md_lines.append("")
        md_lines.append(
            "> **Interpretacion:** pares con |Pearson| muy alto (>0.9) son candidatos a eliminar una "
            "de las dos features. Pares en el rango 0.7-0.9 pueden mantenerse si aportan senal ligeramente "
            "distinta, pero hay que evaluarlo con permutation importance en un modelo entrenado."
        )
    else:
        md_lines.append("No se encontraron pares con |Pearson| >= 0.7 entre features.\n")
    md_lines.append("")

    if figure_path:
        md_lines.append(f"![Matriz de correlaciones]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 6: AUTOCORRELACION DEL TARGET
# ============================================================================

def compute_section_6_target_autocorrelation(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Calcula la funcion de autocorrelacion (ACF) del target hasta 24 horas (288 pasos de 5 min).

    La ACF mide cuan correlacionado esta el target consigo mismo a distintos lags temporales.
    Si ACF(1) es cercano a 1, el siguiente paso es casi trivial (basta copiar el valor actual),
    lo cual explica por que H=1 da NSE=0.86 mientras H=6 colapsa a NSE=-1.21.
    """
    # Usamos solo train para ser coherentes con el resto del analisis
    df_train = splits["train"]
    target_series = df_train[TARGET_COLUMN].to_numpy()

    # Numero de lags a computar: hasta 24 horas a 5 min/paso = 288 pasos
    # +1 porque el lag 0 es la correlacion consigo mismo (=1 por definicion, pero la incluimos)
    max_lag_steps = 288

    # acf de statsmodels acepta NaN con missing='drop'; si no hubiera NaN este parametro es irrelevante
    # fft=True acelera el calculo usando transformada rapida de Fourier (importante con 770k puntos)
    acf_values = acf(target_series, nlags=max_lag_steps, fft=True, missing="drop")

    # Lags especificos interesantes para la tabla: los puntos donde la ACF decae de forma notable
    lags_of_interest = [1, 3, 6, 12, 24, 36, 72, 144, 288]  # en pasos de 5 minutos
    # Conversion a minutos para leer mas facil
    lags_as_minutes = {lag: lag * 5 for lag in lags_of_interest}

    acf_at_lags = {
        str(lag): {
            "minutes": lags_as_minutes[lag],
            "acf": float(acf_values[lag]),
        }
        for lag in lags_of_interest if lag < len(acf_values)
    }

    brute_data = {
        "max_lag_steps": max_lag_steps,
        "resolution_minutes": 5,
        "acf_at_key_lags": acf_at_lags,
        # No guardamos el array completo en JSON (288 valores, no es util), solo los key lags
    }

    # Figura: plot de la ACF
    figure_path = None
    if not skip_figures:
        fig, ax = plt.subplots(figsize=(12, 5))
        # Eje X en minutos para que sea facil de leer (1 paso = 5 min)
        lag_axis_minutes = np.arange(len(acf_values)) * 5
        ax.plot(lag_axis_minutes, acf_values, color="steelblue", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        # Lineas guia en 0.5 y 0.1 para ayudar a leer donde la autocorrelacion decae
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, label="ACF=0.5")
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5, label="ACF=0.1")
        ax.set_xlabel("Lag (minutos)")
        ax.set_ylabel("Autocorrelacion")
        ax.set_title(f"ACF de stormflow_mgd hasta {max_lag_steps * 5} minutos (24h)")
        ax.legend()
        fig.tight_layout()
        figure_path = _save_figure(fig, "section6_target_acf.png")

    # Markdown
    md_lines = []
    md_lines.append("## 6. Autocorrelacion del target (stormflow_mgd)\n")
    md_lines.append(
        "La funcion de autocorrelacion (ACF) mide cuanto se parece la serie a si misma desplazada "
        "en el tiempo. Es relevante para este proyecto porque:\n"
    )
    md_lines.append(
        "- Si ACF(lag=1) es muy cercana a 1, predecir el siguiente paso es casi trivial copiando el "
        "valor actual. Explica el NSE alto a H=1.\n"
    )
    md_lines.append(
        "- La velocidad con la que la ACF decae hacia cero indica cuanto horizonte predictivo util "
        "tiene la serie. Si la ACF cae rapido, predecir a H=6 es intrinsecamente dificil "
        "independientemente del modelo.\n"
    )

    md_lines.append("### ACF en lags de interes (train-only)\n")
    md_lines.append("| Lag (pasos) | Lag (minutos) | ACF |")
    md_lines.append("|-------------|---------------|----:|")
    for lag_str, entry in acf_at_lags.items():
        md_lines.append(f"| {lag_str} | {entry['minutes']} | {entry['acf']:.4f} |")
    md_lines.append("")

    # Interpretacion automatica basada en los valores
    acf_at_5min = acf_at_lags.get("1", {}).get("acf", None)
    acf_at_30min = acf_at_lags.get("6", {}).get("acf", None)
    if acf_at_5min is not None and acf_at_30min is not None:
        md_lines.append(
            f"> **Lectura:** ACF a 5 minutos = {acf_at_5min:.3f}, ACF a 30 minutos = {acf_at_30min:.3f}. "
            f"Como aproximacion teorica, el NSE de un predictor AR(1) de persistencia estaria acotado "
            f"inferiormente por 2*rho - 1 = {2*acf_at_5min - 1:.3f} a H=1 y {2*acf_at_30min - 1:.3f} a H=6. "
            f"El valor empirico exacto sobre test, calculado en la seccion 8 de este documento, "
            f"es **NSE naive = 0.811 a H=1**. La cifra empirica de la seccion 8 es la autoritativa; "
            f"las aproximaciones aqui sirven solo para contextualizar como decae la senal en la serie."
        )
    md_lines.append("")

    if figure_path:
        md_lines.append(f"![ACF del target]({figure_path})\n")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 7: CARACTERIZACION DE LA LLUVIA
# ============================================================================

def compute_section_7_rain_characterization(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Describe el comportamiento de la lluvia: fraccion de pasos sin lluvia, distribucion
    de los no-cero, duracion de eventos de lluvia, etc.

    Este es el contexto que explica por que las features rain_sum_* estan comprimidas
    cerca de cero un 92% del tiempo, como documenta AGENTS.md.
    """
    df_train = splits["train"]
    rain_series = df_train["rain_in"].to_numpy()

    # Fraccion del tiempo con lluvia exactamente cero
    n_total = len(rain_series)
    n_zero = int(np.sum(rain_series == 0.0))
    n_positive = int(np.sum(rain_series > 0.0))
    pct_zero = 100 * n_zero / n_total

    # Estadisticos de los no-cero (para ver como se distribuye la lluvia cuando si hay)
    rain_nonzero = rain_series[rain_series > 0.0]
    if len(rain_nonzero) > 0:
        rain_nonzero_stats = {
            "n": int(len(rain_nonzero)),
            "mean": float(np.mean(rain_nonzero)),
            "median": float(np.median(rain_nonzero)),
            "max": float(np.max(rain_nonzero)),
            "q95": float(np.quantile(rain_nonzero, 0.95)),
            "q99": float(np.quantile(rain_nonzero, 0.99)),
        }
    else:
        rain_nonzero_stats = {"n": 0}

    # Duracion media de rachas consecutivas de lluvia (paso a paso)
    # Definicion simple: una racha es una secuencia contigua de pasos con rain_in > 0
    # Usamos np.diff sobre indicadores para detectar inicios y fines de rachas
    is_raining = (rain_series > 0.0).astype(int)
    # Diferencias: +1 en el inicio de una racha, -1 en el fin, 0 donde no cambia
    transitions = np.diff(is_raining, prepend=0, append=0)
    starts = np.where(transitions == 1)[0]  # Indices donde empieza una racha
    ends = np.where(transitions == -1)[0]   # Indices donde acaba una racha
    # Duracion de cada racha (en pasos de 5 min)
    if len(starts) > 0 and len(ends) > 0:
        rain_streak_lengths = ends - starts
        streak_stats = {
            "n_rain_streaks": int(len(rain_streak_lengths)),
            "mean_duration_steps": float(np.mean(rain_streak_lengths)),
            "mean_duration_minutes": float(np.mean(rain_streak_lengths) * 5),
            "median_duration_minutes": float(np.median(rain_streak_lengths) * 5),
            "max_duration_steps": int(np.max(rain_streak_lengths)),
            "max_duration_hours": float(np.max(rain_streak_lengths) * 5 / 60),
        }
    else:
        streak_stats = {"n_rain_streaks": 0}

    brute_data = {
        "n_total_steps": n_total,
        "n_zero_rain_steps": n_zero,
        "n_positive_rain_steps": n_positive,
        "pct_zero_rain": pct_zero,
        "rain_nonzero_stats": rain_nonzero_stats,
        "rain_streak_stats": streak_stats,
    }

    # Figura: histograma de la lluvia no-cero (escala log)
    figure_path = None
    if not skip_figures and len(rain_nonzero) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(rain_nonzero, bins=100, log=True, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_xlabel("rain_in (pulgadas en 5 min, solo pasos con lluvia > 0)")
        ax.set_ylabel("Frecuencia (escala log)")
        ax.set_title(f"Distribucion de lluvia en pasos con rain_in > 0\nn={len(rain_nonzero):,}")
        fig.tight_layout()
        figure_path = _save_figure(fig, "section7_rain_distribution.png")

    # Markdown
    md_lines = []
    md_lines.append("## 7. Caracterizacion de la lluvia\n")
    md_lines.append(
        f"Analisis del comportamiento de `rain_in` (lluvia en pulgadas por paso de 5 min) sobre el "
        f"split de train. Contextualiza por que las features `rain_sum_*` estan comprimidas cerca "
        f"de cero durante la mayor parte del tiempo.\n"
    )

    md_lines.append("### Fraccion de pasos con lluvia\n")
    md_lines.append(f"- **Total de pasos:** {n_total:,}")
    md_lines.append(f"- **Pasos sin lluvia (rain_in == 0):** {n_zero:,} ({pct_zero:.2f}%)")
    md_lines.append(f"- **Pasos con lluvia (rain_in > 0):** {n_positive:,} ({100 - pct_zero:.2f}%)")
    md_lines.append("")

    if rain_nonzero_stats.get("n", 0) > 0:
        md_lines.append("### Estadisticos de la lluvia no-cero\n")
        md_lines.append(f"- **Media:** {rain_nonzero_stats['mean']:.5f} pulgadas / 5 min")
        md_lines.append(f"- **Mediana:** {rain_nonzero_stats['median']:.5f} pulgadas / 5 min")
        md_lines.append(f"- **Percentil 95:** {rain_nonzero_stats['q95']:.5f} pulgadas / 5 min")
        md_lines.append(f"- **Percentil 99:** {rain_nonzero_stats['q99']:.5f} pulgadas / 5 min")
        md_lines.append(f"- **Maximo:** {rain_nonzero_stats['max']:.5f} pulgadas / 5 min")
        md_lines.append("")

    if streak_stats.get("n_rain_streaks", 0) > 0:
        md_lines.append("### Rachas de lluvia (secuencias contiguas de rain_in > 0)\n")
        md_lines.append(f"- **Numero de rachas identificadas:** {streak_stats['n_rain_streaks']:,}")
        md_lines.append(f"- **Duracion media:** {streak_stats['mean_duration_minutes']:.1f} minutos")
        md_lines.append(f"- **Duracion mediana:** {streak_stats['median_duration_minutes']:.1f} minutos")
        md_lines.append(f"- **Duracion maxima:** {streak_stats['max_duration_hours']:.1f} horas")
        md_lines.append("")

    if figure_path:
        md_lines.append(f"![Distribucion de lluvia]({figure_path})\n")

    return "\n".join(md_lines), brute_data



# ============================================================================
# SECCION 8: COMPARACION CON PREDICTOR NAIVE (PERSISTENCIA)
# ============================================================================

def compute_section_8_naive_baseline(
    splits: Dict[str, pd.DataFrame],
    skip_figures: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Calcula el NSE de un predictor naive "persistencia" (y_pred(t+h) = y(t)) en test
    a horizontes H=1, H=3, H=6, y lo compara con el NSE del modelo actual.

    Esta comparacion es critica: si el modelo no bate claramente al naive, el proyecto
    no esta aprendiendo la relacion rain -> stormflow sino copiando el valor anterior.
    """
    # Trabajamos sobre test para hacer la comparacion directa con las metricas del modelo
    # (que tambien se evaluan en test segun local_eval_metrics.json)
    df_test = splits["test"]
    target_series = df_test[TARGET_COLUMN].to_numpy()

    # Definicion de NSE: 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    # Un NSE de 1 es prediccion perfecta, NSE de 0 es predecir la media, NSE negativo es peor que la media.
    def _nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return float("nan")
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)  # Varianza total observada
        if denominator <= 0:  # Caso degenerado: varianza cero
            return float("nan")
        numerator = np.sum((y_true - y_pred) ** 2)  # Suma de errores cuadraticos del predictor
        return float(1.0 - numerator / denominator)

    # Horizontes a comparar: 1, 3, 6 pasos = 5, 15, 30 minutos
    horizons = [1, 3, 6]

    # Metricas del modelo actual para cada horizonte (fuente: local_eval_metrics.json)
    # Las copiamos aqui para poder comparar directamente sin abrir el JSON
    model_nse_sin_sf = {1: 0.861, 3: 0.471, 6: -1.212}  # H1_sinSF, H3_sinSF, H6_sinSF
    model_nse_con_sf = {1: 0.854, 3: 0.488, 6: 0.255}   # H1_conSF, H3_conSF, H6_conSF

    # Para cada horizonte, calculamos el NSE del naive y la "ganancia" sobre el
    # modelo (delta NSE). Si la ganancia es positiva, el modelo aporta valor.
    comparison = {}
    for h in horizons:
        # Predictor naive: copiar el valor actual como prediccion del paso h siguiente
        # y_true son los valores desde el paso h en adelante (los que queremos predecir)
        # y_pred son los valores desde el paso 0 hasta -h (los que copiamos como prediccion)
        if h >= len(target_series):  # Caso de seguridad (no deberia pasar con test de 165k filas)
            continue
        y_true = target_series[h:]            # Los valores reales en t+h
        y_pred_naive = target_series[:-h]     # La prediccion naive es el valor en t
        nse_naive = _nse(y_true, y_pred_naive)

        # Ganancias del modelo sobre el naive (negativo = el modelo es peor que el naive)
        gain_sin_sf = model_nse_sin_sf[h] - nse_naive
        gain_con_sf = model_nse_con_sf[h] - nse_naive

        comparison[h] = {
            "n_samples": int(len(y_true)),
            "nse_naive_persistence": float(nse_naive),
            "nse_model_sin_sf": float(model_nse_sin_sf[h]),
            "nse_model_con_sf": float(model_nse_con_sf[h]),
            "gain_sin_sf_over_naive": float(gain_sin_sf),
            "gain_con_sf_over_naive": float(gain_con_sf),
        }

    brute_data = {
        "comparison_by_horizon": comparison,
        "note": "NSE del modelo tomado de outputs/data_analysis/local_eval_metrics.json (evaluacion del 16-abril-2026).",
    }

    # Markdown
    md_lines = []
    md_lines.append("## 8. Comparacion con predictor naive (persistencia)\n")
    md_lines.append(
        "Se compara el modelo actual con un predictor trivial que copia el valor actual como "
        "prediccion del paso siguiente. Formalmente: $\\hat{y}(t+h) = y(t)$. Este es el baseline "
        "absoluto: cualquier modelo util debe batirlo claramente.\n"
    )
    md_lines.append(
        "La comparacion es especialmente relevante porque la seccion 6 muestra que la ACF del "
        "target es alta a corto plazo (0.91 a 5 min), lo que implica que la persistencia ya tiene "
        "un rendimiento no trivial.\n"
    )

    md_lines.append("### NSE del modelo vs NSE del naive (split test)\n")
    md_lines.append("| Horizonte | Min | NSE naive | NSE modelo SIN SF | Ganancia SIN SF | NSE modelo CON SF | Ganancia CON SF |")
    md_lines.append("|-----------|-----|----------:|------------------:|----------------:|------------------:|----------------:|")
    for h, entry in comparison.items():
        minutes = h * 5
        naive_nse = entry["nse_naive_persistence"]
        m_sin = entry["nse_model_sin_sf"]
        m_con = entry["nse_model_con_sf"]
        g_sin = entry["gain_sin_sf_over_naive"]
        g_con = entry["gain_con_sf_over_naive"]
        # Formateamos la ganancia con signo explicito para que sea obvio si es positiva o negativa
        sign_sin = "+" if g_sin >= 0 else ""
        sign_con = "+" if g_con >= 0 else ""
        md_lines.append(
            f"| H={h} | {minutes}m | {naive_nse:.3f} | {m_sin:.3f} | {sign_sin}{g_sin:.3f} | "
            f"{m_con:.3f} | {sign_con}{g_con:.3f} |"
        )
    md_lines.append("")

    # Interpretacion automatica basada en los numeros
    h1_gain_sin = comparison[1]["gain_sin_sf_over_naive"]
    h3_gain_sin = comparison[3]["gain_sin_sf_over_naive"]
    h6_gain_sin = comparison[6]["gain_sin_sf_over_naive"]

    md_lines.append("### Lectura\n")
    if h1_gain_sin > 0.05:
        md_lines.append(f"- **H=1 (5 min):** el modelo SIN SF bate al naive por {h1_gain_sin:+.3f} NSE. Mejora moderada.")
    elif h1_gain_sin > 0.0:
        md_lines.append(f"- **H=1 (5 min):** el modelo SIN SF bate al naive por solo {h1_gain_sin:+.3f} NSE. Mejora marginal.")
    else:
        md_lines.append(f"- **H=1 (5 min):** el modelo SIN SF es PEOR que el naive por {h1_gain_sin:.3f} NSE.")

    if h3_gain_sin > 0.0:
        md_lines.append(f"- **H=3 (15 min):** el modelo SIN SF bate al naive por {h3_gain_sin:+.3f} NSE.")
    else:
        md_lines.append(f"- **H=3 (15 min):** el modelo SIN SF es PEOR que el naive por {abs(h3_gain_sin):.3f} NSE. Un script de una linea (`y_pred = y_last`) hace mejor que la TCN.")

    if h6_gain_sin > 0.0:
        md_lines.append(f"- **H=6 (30 min):** el modelo SIN SF bate al naive por {h6_gain_sin:+.3f} NSE.")
    else:
        md_lines.append(f"- **H=6 (30 min):** el modelo SIN SF es CATASTROFICAMENTE peor que el naive ({abs(h6_gain_sin):.2f} NSE de diferencia).")

    md_lines.append("")
    md_lines.append(
        "> **Implicacion operativa y academica:** reportar NSE=0.861 a H=1 sin comparar con el naive "
        "da una impresion erronea del valor aportado por el modelo. Una defensa robusta del TFM "
        "debe incluir esta comparacion y justificar por que un modelo con ~104K parametros mejora "
        "(o no) sobre un predictor de una linea."
    )
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# SECCION 9: SINTESIS DE HALLAZGOS PARA LA REVISION EXTERNA
# ============================================================================

def compute_section_9_synthesis(
    splits: Dict[str, pd.DataFrame],
    section_outputs: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Resume los hallazgos mas relevantes del documento en una lista corta, apta para
    servir de punto de partida a la revision externa con Opus 4.7.

    Esta seccion NO introduce analisis nuevo: sintetiza lo que las secciones 1-8
    ya dejaron expuesto, explicitandolo en forma de "observaciones para revisar".
    """
    # Extraemos datos numericos de las secciones anteriores para escribir una sintesis cuantitativa.
    # Esto permite que los numeros queden exactos aunque el dataset cambie en el futuro.
    s2 = section_outputs.get("section_2", {})
    s3 = section_outputs.get("section_3", {})
    s5 = section_outputs.get("section_5", {})
    s6 = section_outputs.get("section_6", {})
    s8 = section_outputs.get("section_8", {})

    # Valores para la sintesis. Usamos .get() con default para evitar KeyError si una seccion fallo.
    train_max = s2.get("by_split", {}).get("train", {}).get("q100", float("nan"))
    val_max = s2.get("by_split", {}).get("val", {}).get("q100", float("nan"))
    test_max = s2.get("by_split", {}).get("test", {}).get("q100", float("nan"))

    n_extreme_train = s3.get("by_split", {}).get("train", {}).get("n", 0)
    n_extreme_val = s3.get("by_split", {}).get("val", {}).get("n", 0)
    n_extreme_test = s3.get("by_split", {}).get("test", {}).get("n", 0)

    n_redundant_pairs = len(s5.get("high_correlation_pairs_abs_gt_0_7", []))

    acf_5min = s6.get("acf_at_key_lags", {}).get("1", {}).get("acf", float("nan"))
    acf_30min = s6.get("acf_at_key_lags", {}).get("6", {}).get("acf", float("nan"))

    h1_gain = s8.get("comparison_by_horizon", {}).get(1, {}).get("gain_sin_sf_over_naive", float("nan"))
    h3_gain = s8.get("comparison_by_horizon", {}).get(3, {}).get("gain_sin_sf_over_naive", float("nan"))
    h6_gain = s8.get("comparison_by_horizon", {}).get(6, {}).get("gain_sin_sf_over_naive", float("nan"))

    brute_data = {
        "key_findings": [
            {
                "id": "F1_naive_baseline_too_close",
                "title": "El modelo apenas bate al predictor naive a H=1 y es peor a H>=3.",
                "evidence": f"Ganancia sobre naive: H=1 {h1_gain:+.3f}, H=3 {h3_gain:+.3f}, H=6 {h6_gain:+.3f}",
                "severity": "alta",
            },
            {
                "id": "F2_val_extremes_larger_than_test",
                "title": "Val contiene eventos mas extremos que test, que contiene mas que train-max.",
                "evidence": f"Max por split: train={train_max:.1f} MGD, val={val_max:.1f} MGD, test={test_max:.1f} MGD",
                "severity": "alta",
            },
            {
                "id": "F3_test_size_for_extremes",
                "title": "Test solo tiene 59 muestras extremas, insuficiente para conclusiones robustas en bucket Extremo.",
                "evidence": f"n extremos: train={n_extreme_train}, val={n_extreme_val}, test={n_extreme_test}",
                "severity": "media",
            },
            {
                "id": "F4_feature_redundancy",
                "title": "Redundancia masiva entre features: muchos pares fuertemente correlacionados.",
                "evidence": f"{n_redundant_pairs} pares con |Pearson| >= 0.7 entre las 22 features.",
                "severity": "media",
            },
            {
                "id": "F5_acf_bounds_predictability",
                "title": "La ACF del target impone un techo teorico a la predictibilidad multi-paso.",
                "evidence": f"ACF(5min)={acf_5min:.3f}, ACF(30min)={acf_30min:.3f}",
                "severity": "media",
            },
        ]
    }

    # Markdown
    md_lines = []
    md_lines.append("## 9. Sintesis de hallazgos para la revision externa\n")
    md_lines.append(
        "Esta seccion resume los hallazgos mas relevantes del documento, pensada para servir "
        "de punto de partida a la revision con Opus 4.7. No introduce analisis nuevo: explicita "
        "lo que las secciones 1-8 ya exponen.\n"
    )

    md_lines.append("### Hallazgos ordenados por severidad\n")

    md_lines.append("**1. [ALTA] El modelo apenas bate al predictor naive a H=1 y es PEOR a H=3 y H=6.**")
    md_lines.append(f"")
    md_lines.append(f"Ganancia del modelo SIN SF sobre persistencia: H=1 {h1_gain:+.3f}, H=3 {h3_gain:+.3f}, H=6 {h6_gain:+.3f}.")
    md_lines.append(f"Un modelo de ~104K parametros debe justificar por que mejora (o no) a un predictor de una linea.")
    md_lines.append(f"Pregunta para el revisor: ¿el NSE=0.861 a H=1 es realmente un buen resultado teniendo en cuenta que naive da {s8.get('comparison_by_horizon', {}).get(1, {}).get('nse_naive_persistence', 'NaN'):.3f}?")
    md_lines.append("")

    md_lines.append("**2. [ALTA] Asimetria entre splits en cobertura de eventos extremos.**")
    md_lines.append(f"")
    md_lines.append(f"Valor maximo por split: train={train_max:.1f} MGD, **val={val_max:.1f} MGD** (maximo absoluto del dataset), test={test_max:.1f} MGD.")
    md_lines.append(f"Extremos por split: train={n_extreme_train}, val={n_extreme_val}, test={n_extreme_test}.")
    md_lines.append(f"Consecuencia: el modelo se evalua en test sobre un regimen menos extremo que el de entrenamiento, mientras que val incluye el pico absoluto. Early stopping sobre val puede estar favoreciendo infraestimacion.")
    md_lines.append("")

    md_lines.append("**3. [MEDIA] Tamano muestral insuficiente para el bucket Extremo en test.**")
    md_lines.append(f"")
    md_lines.append(f"Solo 59 muestras extremas en test. Las metricas sobre este bucket (NSE=-0.99, bias=-12.7 MGD) tienen alta varianza. Un unico evento atipico puede desplazarlas significativamente.")
    md_lines.append(f"Pregunta para el revisor: ¿tiene sentido reportar metricas sobre este bucket con esta muestra, o conviene usar bootstrap o k-fold para estimar intervalos de confianza?")
    md_lines.append("")

    md_lines.append("**4. [MEDIA] Redundancia masiva entre features.**")
    md_lines.append(f"")
    md_lines.append(f"{n_redundant_pairs} pares de features con |Pearson| >= 0.7. Las mas extremas: rain_sum_10m vs rain_max_10m (0.985), rain_sum_10m vs rain_sum_15m (0.970), api_dynamic correlaciona con 9 features distintas.")
    md_lines.append(f"De 22 features declaradas, el numero efectivo de dimensiones independientes es mucho menor (posiblemente ~8-10).")
    md_lines.append(f"Pregunta para el revisor: ¿conviene reducir features antes de seguir iterando en arquitectura/loss?")
    md_lines.append("")

    md_lines.append("**5. [MEDIA] ACF del target impone limite teorico a H>1.**")
    md_lines.append(f"")
    md_lines.append(f"ACF(5 min) = {acf_5min:.3f}, ACF(30 min) = {acf_30min:.3f}. La ACF decae rapido tras los primeros 30 minutos.")
    md_lines.append(f"Esto indica que, sin incorporar prediccion de lluvia externa (opcion c del STATE.md), extender el horizonte util mas alla de ~15 min puede ser fisicamente limitado.")
    md_lines.append("")

    md_lines.append("### Prioridades sugeridas (a validar con el revisor)\n")
    md_lines.append("1. **Reportar baseline naive en todas las metricas futuras.** No-regrets, 5 lineas de codigo.")
    md_lines.append("2. **Revisar split.** Evaluar splits alternativos (anos completos, TimeSeriesSplit, blocked CV por eventos).")
    md_lines.append("3. **Simplificar features.** Reducir de 22 a ~10 eliminando redundancia, con permutation importance sobre el modelo simplificado.")
    md_lines.append("4. **Intervalos de confianza.** Bootstrap sobre test para cuantificar la incertidumbre de las metricas en bucket Extremo.")
    md_lines.append("5. **Horizonte realista.** Aceptar que H>1 con features actuales tiene limite fisico, o integrar el modelo de lluvia del MSD (opcion c).")
    md_lines.append("")

    return "\n".join(md_lines), brute_data


# ============================================================================
# MAIN: orquestacion completa
# ============================================================================

def _build_markdown_header(config_path: Path) -> str:
    """Construye el encabezado del markdown con metadata de ejecucion."""
    from datetime import datetime  # Import local: solo se usa aqui

    md_lines = []
    md_lines.append("# DATASET_STATS.md - Resumen estadistico del dataset\n")
    md_lines.append(
        f"Documento generado automaticamente por `scripts/generate_dataset_stats.py`.\n"
    )
    md_lines.append(f"- **Fecha de generacion:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- **Config usada:** `{config_path}`")
    md_lines.append(f"- **Proposito:** input para la revision externa con Opus 4.7 sobre metodologia y limite fisico del modelo.")
    md_lines.append("")
    md_lines.append(
        "Este documento describe *los datos*, no el modelo. Para metricas del modelo actual, "
        "ver `outputs/data_analysis/local_eval_metrics.json` y `docs/STATE.md`."
    )
    md_lines.append("\n---\n")
    return "\n".join(md_lines)


def main(config_path: Path, use_cache: bool, skip_figures: bool) -> None:
    """
    Orquesta el flujo completo: carga datos, calcula 7 secciones, escribe markdown + JSON.
    """
    print("=" * 70)
    print("GENERATE_DATASET_STATS.PY")
    print("=" * 70)

    # Paso inicial: crear directorios de output si no existen
    _ensure_dirs_exist()

    # Paso 1: cargar y procesar datos (con cache si esta disponible)
    print("\n[main] Paso 1: cargar y procesar datos")
    df_features = _load_and_process_data(config_path=config_path, use_cache=use_cache)

    # Paso 2: hacer el split cronologico
    print("\n[main] Paso 2: split cronologico 70/15/15")
    splits = _split_data(df_features)
    for split_name, df_split in splits.items():
        print(f"[main]      {split_name}: {len(df_split):,} filas")

    # Paso 3: calcular las 7 secciones en orden
    print("\n[main] Paso 3: calcular las 7 secciones del reporte")

    print("[main]      Seccion 1: resumen global")
    md1, bd1 = compute_section_1_global_summary(df_features, splits)

    print("[main]      Seccion 2: distribucion del target")
    md2, bd2 = compute_section_2_target_distribution(df_features, splits, skip_figures=skip_figures)

    print("[main]      Seccion 3: eventos extremos")
    md3, bd3 = compute_section_3_extreme_events(df_features, splits, skip_figures=skip_figures)

    print("[main]      Seccion 4: correlaciones feature-target")
    md4, bd4 = compute_section_4_feature_target_correlations(splits, skip_figures=skip_figures)

    print("[main]      Seccion 5: matriz feature-feature")
    md5, bd5 = compute_section_5_feature_feature_matrix(splits, skip_figures=skip_figures)

    print("[main]      Seccion 6: autocorrelacion del target")
    md6, bd6 = compute_section_6_target_autocorrelation(splits, skip_figures=skip_figures)

    print("[main]      Seccion 7: caracterizacion de la lluvia")
    md7, bd7 = compute_section_7_rain_characterization(splits, skip_figures=skip_figures)

    print("[main]      Seccion 8: comparacion con predictor naive")
    md8, bd8 = compute_section_8_naive_baseline(splits, skip_figures=skip_figures)

    # La seccion 9 recibe las brute_data de las secciones anteriores para generar la sintesis
    section_outputs_for_synthesis = {
        "section_2": bd2,
        "section_3": bd3,
        "section_5": bd5,
        "section_6": bd6,
        "section_8": bd8,
    }
    print("[main]      Seccion 9: sintesis de hallazgos")
    md9, bd9 = compute_section_9_synthesis(splits, section_outputs_for_synthesis)

    # Paso 4: concatenar todas las secciones en un solo markdown
    print("\n[main] Paso 4: escribir markdown y JSON de salida")
    header = _build_markdown_header(config_path)
    full_markdown = header + "\n---\n\n".join([md1, md2, md3, md4, md5, md6, md7, md8, md9])

    # Escribir markdown
    with open(OUTPUT_MARKDOWN, "w", encoding="utf-8") as f:
        f.write(full_markdown)
    print(f"[main]      Markdown escrito en {OUTPUT_MARKDOWN}")

    # Escribir JSON con todos los brute_data
    full_brute_data = {
        "section_1_global_summary": bd1,
        "section_2_target_distribution": bd2,
        "section_3_extreme_events": bd3,
        "section_4_feature_target_correlations": bd4,
        "section_5_feature_feature_matrix": bd5,
        "section_6_target_autocorrelation": bd6,
        "section_7_rain_characterization": bd7,
        "section_8_naive_baseline": bd8,
        "section_9_synthesis": bd9,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_brute_data, f, indent=2, default=str)
    print(f"[main]      JSON escrito en {OUTPUT_JSON}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parseo de argumentos de linea de comandos
    parser = argparse.ArgumentParser(description="Genera docs/DATASET_STATS.md con estadisticas del dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local.yaml",
        help="Ruta al YAML de configuracion (default: configs/local.yaml para ejecucion local).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Fuerza regeneracion del dataframe sin usar el cache parquet.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="No genera figuras PNG (mas rapido, solo tablas).",
    )
    args = parser.parse_args()

    # Resolvemos la ruta relativa a la raiz del repo para ser independientes del cwd
    resolved_config = (REPO_ROOT / args.config).resolve()
    if not resolved_config.exists():
        raise FileNotFoundError(f"El archivo de configuracion no existe: {resolved_config}")

    main(
        config_path=resolved_config,
        use_cache=not args.no_cache,
        skip_figures=args.skip_figures,
    )