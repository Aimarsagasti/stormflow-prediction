Soy un ingeniero con base teorica fuerte pero poca experiencia practica en programacion. Estoy aprendiendo.

Reglas:

* Comenta cada linea no trivial de codigo explicando que hace y por que
* Comentarios en espanol, codigo en ingles
* Explica brevemente cada cambio antes de hacerlo
* Usa type hints en funciones Python
* Prioriza codigo limpio y legible sobre codigo clever o compacto

--- project-doc ---

# AGENTS.md - Proyecto Stormflow Prediction

## 1. El Problema

El Metropolitan Sewer District of Greater Cincinnati (MSD) gestiona ~3,000 millas
de alcantarillado, del cual 40-45% es combinado (aguas residuales + pluviales).
Cuando llueve intensamente, el sistema se desborda y descarga agua sin tratar a
los rios - Combined Sewer Overflows (CSOs). El MSD opera bajo un Consent Decree
federal que exige reducir estos desbordamientos.

El sistema actual es REACTIVO. El objetivo de este proyecto es hacerlo PREDICTIVO:
anticipar picos de stormflow (caudal de tormenta) ANTES de que ocurran, dando
tiempo para activar medidas preventivas.

**Requisito critico**: infraestimar un pico es MUCHO peor que sobreestimarlo.
Una infraestimacion significa no activar prevencion -> CSO real -> consecuencias
legales y ambientales. Una sobreestimacion solo significa precaucion innecesaria.

## 2. Los Datos

### 2.1 Ubicacion

Google Drive (montado en Colab como /content/drive):

* Parte 1: `/content/drive/MyDrive/Proyecto de capstone/Archivos del proyecto/Largos/MC-CL-005/1parte/`
* Parte 2: `/content/drive/MyDrive/Proyecto de capstone/Archivos del proyecto/Largos/MC-CL-005/2parte/`

Los datos NUNCA se suben a GitHub. Solo codigo y resultados de analisis.

### 2.2 Archivos disponibles (misma estructura en cada parte)

* `rain_list.tsf` -> Lluvia incremental cada 5 min
* `flow_list.tsf` -> Caudal total cada 5 min
* `lstStormTs.tsf` -> Stormflow cada 5 min
* `lstEventsGenerated Events 0.dat` -> Eventos de tormenta identificados

### 2.3 Formato de archivos

**.tsf**: 3 lineas de cabecera + datos tabulados (datetime\tvalor). Separador: tab.
Linea 1 = IDs sensor, Linea 2 = nombre variable, Linea 3 = unidades.

**.dat (eventos)**: Pipe-separated '|', sin cabecera, 32 campos + trailing pipe
vacio que genera una columna extra (33 total, descartar la ultima).

### 2.4 Problemas conocidos en los datos

* Stormflow puede ser negativo (artefacto de calculo storm = flow - base).
  Fisicamente imposible -> requiere tratamiento.
* Flow total puede tener valores negativos (error de sensor).
* El dataset es ~10 anos, ~1.1M registros a 5 minutos de resolucion.
* Aproximadamente 92% del tiempo es flujo base (sin tormenta). El dataset
  esta muy desbalanceado (zero-inflated).
* Solo 59 eventos extremos (>50 MGD) en todo el dataset.
* 15 de 59 eventos extremos no tienen lluvia en la ventana de entrada
  (escorrentia retardada / posible deshielo). Son fisicamente impredecibles
  con los datos actuales.

## 3. Variable Objetivo

**stormflow_mgd** - caudal de tormenta en millones de galones por dia (MGD).
El objetivo es predecir sus PICOS con la mayor precision posible.

## 4. Fases del proyecto (referencia historica)

El proyecto paso por las siguientes fases (completadas):

### Fase 1: Analisis Exploratorio (completada)
Carga y exploracion de datos, diagnostico de calidad, distribuciones y
correlaciones. Resultados en `outputs/data_analysis/`.

### Fase 2: Propuesta de Solucion (completada)
Documentada en `outputs/proposal.md`.

### Fase 3: Implementacion (completada)
Modulos en `src/`.

### Fase 4: Iteraciones de mejora (en curso)
Ver `docs/EXPERIMENTS.md` para historial completo. Estado actual en `docs/STATE.md`.

## 5. Restricciones Tecnicas

* **Framework**: PyTorch
* **Ejecucion**: Google Colab con GPU T4. El codigo local NO tiene GPU.
* **Datos en Drive**: toda ruta a datos se lee de configs/default.yaml
* **Split SIEMPRE cronologico**: NUNCA aleatorio en series temporales
* **Normalizacion con stats de train**: NUNCA calcular min/max/mean/std con val o test
* **Local es SOLO para evaluacion**: los entrenamientos se hacen siempre en Colab

## 6. Estructura del Proyecto

Ver `CLAUDE.md` para mapa completo. Estructura resumida:

stormflow-prediction/
├── CLAUDE.md              <- Punto de entrada para Claude Code
├── AGENTS.md              <- Este archivo (contexto permanente)
├── docs/                  <- STATE, EXPERIMENTS, WORKFLOW
├── configs/               <- default.yaml
├── src/                   <- Codigo fuente modular
├── notebooks/             <- Notebooks Colab descargados como .py
├── outputs/               <- Resultados, plots, iteraciones historicas
└── MC-CL-005/             <- Datos brutos (NO en GitHub)

## 7. Reglas para el Agente

1. Datos NUNCA en GitHub.
2. Rutas configurables desde configs/default.yaml.
3. Codigo debe funcionar en Google Colab con GPU T4.
4. Comentarios en espanol. Variables y funciones en ingles (snake_case).
5. Type hints en funciones publicas.
6. Cada decision de diseno debe estar JUSTIFICADA (en comentarios o en proposal.md).
7. Priorizar captura de picos sobre precision en flujo base.
8. NO modificar archivos que no se mencionan explicitamente en el prompt.
9. Maximo 3 archivos modificados por tarea.

## 8. Lecciones aprendidas permanentes

Estas son conclusiones firmes del proyecto. Son permanentes y no deben re-cuestionarse
sin evidencia experimental que las contradiga.

### 8.1 Decisiones confirmadas con evidencia experimental

**flow_total_mgd FUERA de FEATURE_COLUMNS (confirmado iter 11):**
flow_total_mgd tiene r=0.9976 con stormflow_mgd (porque stormflow = flow - base).
El modelo la usaba como atajo en vez de aprender lluvia -> stormflow.
Al quitarla: NSE global empeoro ligeramente pero error en picos mejoro de -60% a -41%.
NUNCA re-incluir flow_total_mgd como feature.

**api_dynamic es redundante (confirmado iters 10-12):**
Correlacion r=0.94 con rain_sum_60m. PI negativo en TODAS las iteraciones.
Es una version suavizada de lluvia acumulada, no aporta informacion nueva.
Sin embargo, podria ser util si se reformula con evapotranspiracion (futuro).

**Los acumulados largos (rain_sum_60m a 360m) son NECESARIOS (confirmado iter 12):**
Al quitarlos, rain_max_10m domino con PI=138% y el modelo sobreestimo picos (+114%).
Sin contexto temporal largo, el modelo no puede calibrar magnitud de respuesta.
NUNCA quitar todos los rain_sum largos a la vez.

**log1p es necesario para el target (confirmado iter 6):**
Sin log1p, NSE cayo a -26.6. Catastrofico. No quitar.

**Normalizacion actual: log1p + z-score (desde iter 10):**
z-score mejor que MinMax. Target normalizado tiene picos en z-scores de 30-70.
Features de lluvia siguen comprimidas (92% < 0.05) porque 92% del tiempo no llueve.
Ningun scaler resuelve esto - es la naturaleza de los datos.

**Two-Stage (Hurdle Model) es la arquitectura correcta (confirmado iter 13):**
Separa clasificacion de evento (cls head) y regresion de magnitud (reg head) sobre
un backbone compartido. Inferencia con switch duro: si p(evento) >= 0.3 -> regresor,
si no -> 0. Fue el cambio arquitectonico mas impactante del proyecto.

**Cosine Annealing empeora (confirmado iter 14):**
Probado con CosineAnnealingWarmRestarts. Degrado resultados. Revertido en iter 14b.
Usar ReduceLROnPlateau (factor=0.5, patience=4).

**El error en pico del ~83% es sistematico, no aleatorio (confirmado iters varias):**
Baja varianza entre runs. No es un problema de inicializacion aleatoria. Es una
limitacion estructural dada la rareza de eventos extremos y la ausencia de senal
de lluvia en 15 de 59 extremos.

**El modelo funciona bien en eventos moderados a H=1 (confirmado iter final):**
MAPE 17-34% en rango 5-50 MGD. Operativamente util. Juzgar el modelo solo por
extremos sin separar los 15 casos sin lluvia es injusto.

### 8.2 Errores que NO repetir

* **Cambios masivos en una sola iteracion**: Siempre empeoran o dan resultados
  no interpretables. UN cambio por iteracion, con hipotesis clara.
* **Gating (multiplicar probabilidad x magnitud)**: Comprime picos porque p nunca
  satura a 1.0. Probado en iter 2-3. Usar SWITCH DURO (si p > umbral -> regresor,
  si no -> 0), NO multiplicacion.
* **Sampler estratificado agresivo**: Causa drift train/val. Mejor distribucion
  natural + loss ponderada.
* **Quitar features por PI negativo en modelo malo**: PI en modelo con NSE < 0
  NO es fiable. Solo es fiable para features claramente redundantes (api_dynamic)
  o consistentemente daninas.
* **Reducir datos de entrenamiento para ahorrar RAM**: Pierde eventos extremos
  raros. Resolver RAM con subsampling en diagnosticos, no en datos.
* **Ponderar la loss del regresor por magnitud (alpha grande)**: No resuelve la
  infraestimacion de extremos. Probado en iter 15. Revertido con alpha=0.05 y
  luego revertido del todo.
* **Usar stormflow como feature para "mejorar picos"**: Mejora NSE global pero
  NO ayuda en picos extremos. A H=1 sobreestima el pico +59%. A H=3 no ayuda.
  Util como experimento comparativo, no como solucion al problema de extremos.

## 9. Arquitectura actual del modelo (referencia)

**TwoStageTCN** con backbone compartido:
- Conv1x1 inicial para proyectar a 32 canales
- 5 bloques residuales [32, 64, 64, 64, 32] con dilations [1, 2, 4, 8, 16]
- GroupNorm, CausalConv1d, kernel_size=3, dropout=0.2
- Campo receptivo: 125 timesteps = 10.4 horas

**Classifier head**:
- Linear(32, 64) -> ReLU -> Dropout -> Linear(64, 1) -> Sigmoid
- Entrenada con BCE, pos_weight dinamico.

**Regressor head**:
- Linear(32, 128) -> ReLU -> Dropout -> Linear(128, 64) -> ReLU -> Linear(64, 1)
- Entrenada con Huber loss SOLO en muestras con evento.
- Penalizacion asimetrica: x3 cuando y_pred < y_true.

**Inferencia**: switch duro. Si p_evento >= 0.3 -> salida = regresor. Si no -> salida = 0.

**Parametros entrenables**: ~104K.

## 10. Features actuales (22 sin stormflow, 23 con)

```python
FEATURE_COLUMNS = [
    'rain_in',
    'temp_daily_f', 'api_dynamic',
    'rain_sum_10m', 'rain_sum_15m', 'rain_sum_30m',
    'rain_sum_60m', 'rain_sum_120m', 'rain_sum_180m', 'rain_sum_360m',
    'rain_max_10m', 'rain_max_30m', 'rain_max_60m',
    'minutes_since_last_rain',
    'delta_flow_5m', 'delta_flow_15m',
    'delta_rain_10m', 'delta_rain_30m',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
]
# Version "con stormflow" anade 'stormflow_mgd' al final (autoregresivo).
# flow_total_mgd esta DEFINIDA en engineering.py pero EXCLUIDA de FEATURE_COLUMNS.
```

## 11. Bug critico conocido y corregido

Cuando `stormflow_mgd` aparece tanto en FEATURES como en TARGET_COL,
`normalize_splits` lo normaliza DOS veces (in-place sobre el mismo df).
Correccion obligatoria:

```python
FEATURES_FOR_NORM = [f for f in FEATURES if f != 'stormflow_mgd']
df_tn, df_vn, df_tsn, npar = normalize_splits(df_tr, df_va, df_te, FEATURES_FOR_NORM, TARGET_COL)
tl, vl, tstl = create_dataloaders(df_tn, df_vn, df_tsn, FEATURES, TARGET_COL, AUX_COL, ...)
```

Sin este parche, el target desnormalizado daba valores absurdos (52,466 MGD en
vez de 135 MGD) y todas las metricas CON stormflow estaban corruptas.