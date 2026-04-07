<INSTRUCTIONS>
Soy un ingeniero con base teorica fuerte pero poca experiencia practica en programacion. Estoy aprendiendo.

Reglas:
- Comenta cada linea no trivial de codigo explicando que hace y por que
- Comentarios en espanol, codigo en ingles
- Explica brevemente cada cambio antes de hacerlo
- Usa type hints en funciones Python
- Prioriza codigo limpio y legible sobre codigo clever o compacto

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
- Parte 1: `/content/drive/MyDrive/Proyecto de capstone/Archivos del proyecto/Largos/MC-CL-005/1parte/`
- Parte 2: `/content/drive/MyDrive/Proyecto de capstone/Archivos del proyecto/Largos/MC-CL-005/2parte/`

Los datos NUNCA se suben a GitHub. Solo codigo y resultados de analisis.

### 2.2 Archivos disponibles (misma estructura en cada parte)
- `rain_list.tsf` -> Lluvia incremental cada 5 min
- `flow_list.tsf` -> Caudal total cada 5 min
- `lstStormTs.tsf` -> Stormflow cada 5 min
- `lstEventsGenerated Events 0.dat` -> Eventos de tormenta identificados

### 2.3 Formato de archivos
**.tsf**: 3 lineas de cabecera + datos tabulados (datetime\tvalor). Separador: tab.
Linea 1 = IDs sensor, Linea 2 = nombre variable, Linea 3 = unidades.

**.dat (eventos)**: Pipe-separated '|', sin cabecera, 32 campos + trailing pipe
vacio que genera una columna extra (33 total, descartar la ultima).

### 2.4 Problemas conocidos en los datos
- Stormflow puede ser negativo (artefacto de calculo storm = flow - base).
  Fisicamente imposible -> requiere tratamiento.
- Flow total puede tener valores negativos (error de sensor).
- El dataset es ~10 anos, ~1.1M registros a 5 minutos de resolucion.
- Aproximadamente 92% del tiempo es flujo base (sin tormenta). El dataset
  esta muy desbalanceado (zero-inflated).

## 3. Variable Objetivo
**stormflow_mgd** - caudal de tormenta en millones de galones por dia (MGD).
El objetivo es predecir sus PICOS con la mayor precision posible.

## 4. Lo que el agente debe hacer

### Fase 1: Analisis Exploratorio
- Cargar y explorar los datos de ambas partes
- Diagnosticar calidad (NaN, negativos, huecos temporales, outliers)
- Analizar distribuciones, correlaciones, estacionalidad
- Entender la relacion temporal lluvia -> stormflow
- Guardar un resumen estadistico completo en outputs/data_analysis/

### Fase 2: Propuesta de Solucion
Basandose en el analisis de los datos, el agente debe PROPONER:
- Que features crear y por que
- Que arquitectura de modelo usar y por que
- Que funcion de perdida usar y por que
- Como manejar el desbalance del dataset
- Que hiperparametros iniciales usar y por que

La propuesta debe guardarse en outputs/proposal.md con justificacion
para cada decision.

### Fase 3: Implementacion
Implementar la solucion propuesta en los modulos de src/.

### Fase 4: Evaluacion
- Metricas obligatorias: NSE, RMSE, MAE, error en pico por evento
- Analisis de timing: el modelo acierta CUANDO viene el pico?
- Analisis de magnitud: el modelo acierta CUANTO sube el pico?
- Desglose por tamano de evento (pequeno/moderado/grande/extremo)

## 5. Restricciones Tecnicas

- **Framework**: PyTorch
- **Ejecucion**: Google Colab con GPU T4. El codigo local NO tiene GPU.
- **Datos en Drive**: toda ruta a datos se lee de configs/default.yaml
- **Split SIEMPRE cronologico**: NUNCA aleatorio en series temporales
- **Normalizacion con stats de train**: NUNCA calcular min/max con val o test

## 6. Estructura del Proyecto
```
stormflow-prediction/
├── AGENTS.md              <- Este archivo
├── configs/
│   └── default.yaml       <- Hiperparametros y rutas
├── src/
│   ├── data/              <- Carga, limpieza, eventos
│   ├── features/          <- Feature engineering
│   ├── pipeline/          <- Split, normalizacion, DataLoaders
│   ├── models/            <- Arquitectura del modelo
│   ├── training/          <- Loop de entrenamiento
│   └── evaluation/        <- Metricas y diagnosticos
├── notebooks/             <- Notebooks para Colab
├── outputs/               <- Resultados de analisis y metricas
└── requirements.txt
```

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

## 8. Lecciones Aprendidas (iteraciones 0-12) - NO IGNORAR

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

**hour_sin y hour_cos perjudican (confirmado iters 10-12):**
PI negativo consistente. La hora del dia no afecta la hidrodinamica de escorrentia.

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

### 8.2 Errores que NO repetir

- **Cambios masivos en una sola iteracion**: Siempre empeoran o dan resultados
  no interpretables. UN cambio por iteracion, con hipotesis clara.
- **Gating (multiplicar probabilidad x magnitud)**: Comprime picos porque p nunca
  satura a 1.0. Probado en iter 2-3. Si se usa modelo two-headed, usar SWITCH
  duro (si p > umbral -> usar regresion directa), NO multiplicacion.
- **Sampler estratificado agresivo**: Causa drift train/val. Mejor distribucion
  natural + loss ponderada.
- **Quitar features por PI negativo en modelo malo**: PI en modelo con NSE < 0
  NO es fiable. Solo es fiable para features claramente redundantes (api_dynamic)
  o consistentemente daninas (hour_sin/cos).
- **Reducir datos de entrenamiento para ahorrar RAM**: Pierde eventos extremos
  raros. Resolver RAM con subsampling en diagnosticos, no en datos.

### 8.3 Estado actual del modelo (post-iter 12)

Arquitectura: TCN con regresion directa, backbone [32,64,64,64,32],
dilations [1,2,4,8,16], GroupNorm, kernel_size=3, dropout=0.2.
Campo receptivo: 125 timesteps (10.4 horas). Parametros: ~102K.

Loss: CompositeLoss con 4 componentes (Huber + asimetrica + peak MSE + base penalty).

Mejor resultado: NSE=-0.55 (iter 10), error pico=-60%.
Problema principal: zero-inflation (92% datos son stormflow ~0).
El modelo aprende "predecir bajo" en 2-3 epochs y se estanca.

### 8.4 Siguiente paso: Modelo Two-Stage (Hurdle Model)

El problema raiz es que el modelo intenta resolver dos problemas distintos
con un solo conjunto de pesos:
- Problema A: hay evento o no? (clasificacion, 92% vs 8%)
- Problema B: si si, cuanto? (regresion, 0.1-135+ MGD)

Solucion: separar en dos etapas con backbone compartido:
- Etapa 1 (clasificador): cabeza binaria con sigmoid, entrenada con BCE
- Etapa 2 (regresor): cabeza de regresion, entrenada SOLO con muestras de evento
- Inferencia: switch duro. Si clasificador < umbral -> output=0. Si >= umbral -> regresor.
- IMPORTANTE: NO usar gating (multiplicar). Usar switch duro.
</INSTRUCTIONS>