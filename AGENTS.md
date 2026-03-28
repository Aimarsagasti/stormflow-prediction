# AGENTS.md — Proyecto Stormflow Prediction

## 1. El Problema

El Metropolitan Sewer District of Greater Cincinnati (MSD) gestiona ~3,000 millas
de alcantarillado, del cual 40-45% es combinado (aguas residuales + pluviales).
Cuando llueve intensamente, el sistema se desborda y descarga agua sin tratar a
los ríos — Combined Sewer Overflows (CSOs). El MSD opera bajo un Consent Decree
federal que exige reducir estos desbordamientos.

El sistema actual es REACTIVO. El objetivo de este proyecto es hacerlo PREDICTIVO:
anticipar picos de stormflow (caudal de tormenta) ANTES de que ocurran, dando
tiempo para activar medidas preventivas.

**Requisito crítico**: infraestimar un pico es MUCHO peor que sobreestimarlo.
Una infraestimación significa no activar prevención → CSO real → consecuencias
legales y ambientales. Una sobreestimación solo significa precaución innecesaria.

## 2. Los Datos

### 2.1 Ubicación
Google Drive (montado en Colab como /content/drive):
- Parte 1: `/content/drive/MyDrive/Proyecto de capstone/Archivos del proyecto/Largos/MC-CL-005/1parte/`
- Parte 2: `/content/drive/MyDrive/Proyecto de capstone/Archivos del proyecto/Largos/MC-CL-005/2parte/`

Los datos NUNCA se suben a GitHub. Solo código y resultados de análisis.

### 2.2 Archivos disponibles (misma estructura en cada parte)
- `rain_list.tsf` → Lluvia incremental cada 5 min
- `flow_list.tsf` → Caudal total cada 5 min
- `lstStormTs.tsf` → Stormflow cada 5 min
- `lstEventsGenerated Events 0.dat` → Eventos de tormenta identificados

### 2.3 Formato de archivos
**.tsf**: 3 líneas de cabecera + datos tabulados (datetime\tvalor). Separador: tab.
Línea 1 = IDs sensor, Línea 2 = nombre variable, Línea 3 = unidades.

**.dat (eventos)**: Pipe-separated '|', sin cabecera, 32 campos + trailing pipe
vacío que genera una columna extra (33 total, descartar la última).

### 2.4 Problemas conocidos en los datos
- Stormflow puede ser negativo (artefacto de cálculo storm = flow - base).
  Físicamente imposible → requiere tratamiento.
- Flow total puede tener valores negativos (error de sensor).
- El dataset es ~10 años, ~1.1M registros a 5 minutos de resolución.
- Aproximadamente 92% del tiempo es flujo base (sin tormenta). El dataset
  está muy desbalanceado.

## 3. Variable Objetivo
**stormflow_mgd** — caudal de tormenta en millones de galones por día (MGD).
El objetivo es predecir sus PICOS con la mayor precisión posible.

## 4. Lo que el agente debe hacer

### Fase 1: Análisis Exploratorio
- Cargar y explorar los datos de ambas partes
- Diagnosticar calidad (NaN, negativos, huecos temporales, outliers)
- Analizar distribuciones, correlaciones, estacionalidad
- Entender la relación temporal lluvia → stormflow
- Guardar un resumen estadístico completo en outputs/data_analysis/

### Fase 2: Propuesta de Solución
Basándose en el análisis de los datos, el agente debe PROPONER:
- Qué features crear y por qué
- Qué arquitectura de modelo usar y por qué
- Qué función de pérdida usar y por qué
- Cómo manejar el desbalance del dataset
- Qué hiperparámetros iniciales usar y por qué

La propuesta debe guardarse en outputs/proposal.md con justificación
para cada decisión.

### Fase 3: Implementación
Implementar la solución propuesta en los módulos de src/.

### Fase 4: Evaluación
- Métricas obligatorias: NSE, RMSE, MAE, error en pico por evento
- Análisis de timing: ¿el modelo acierta CUÁNDO viene el pico?
- Análisis de magnitud: ¿el modelo acierta CUÁNTO sube el pico?
- Desglose por tamaño de evento (pequeño/moderado/grande/extremo)

## 5. Restricciones Técnicas

- **Framework**: PyTorch
- **Ejecución**: Google Colab con GPU T4. El código local NO tiene GPU.
- **Datos en Drive**: toda ruta a datos se lee de configs/default.yaml
- **Split SIEMPRE cronológico**: NUNCA aleatorio en series temporales
- **Normalización con stats de train**: NUNCA calcular min/max con val o test

## 6. Estructura del Proyecto
```
stormflow-prediction/
├── AGENTS.md              ← Este archivo
├── configs/
│   └── default.yaml       ← Hiperparámetros y rutas
├── src/
│   ├── data/              ← Carga, limpieza, eventos
│   ├── features/          ← Feature engineering
│   ├── pipeline/          ← Split, normalización, DataLoaders
│   ├── models/            ← Arquitectura del modelo
│   ├── training/          ← Loop de entrenamiento
│   └── evaluation/        ← Métricas y diagnósticos
├── notebooks/             ← Notebooks para Colab
├── outputs/               ← Resultados de análisis y métricas
└── requirements.txt
```

## 7. Reglas para el Agente

1. Datos NUNCA en GitHub.
2. Rutas configurables desde configs/default.yaml.
3. Código debe funcionar en Google Colab con GPU T4.
4. Comentarios en español. Variables y funciones en inglés (snake_case).
5. Type hints en funciones públicas.
6. Cada decisión de diseño debe estar JUSTIFICADA (en comentarios o en proposal.md).
7. Priorizar captura de picos sobre precisión en flujo base.
8. El agente tiene LIBERTAD TOTAL para elegir arquitectura, features, y estrategia.
```

---

Dile a Codex que reemplace el `AGENTS.md` actual con ese contenido. Después haz commit y push:
```
git add AGENTS.md
git commit -m "Replantear AGENTS.md - enfoque desde cero"
git push