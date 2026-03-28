# Propuesta de solución para predicción de stormflow

## Contexto del problema

El dataset contiene 1,105,056 registros a resolución de 5 minutos entre 2015-08-01 y 2026-01-31, equivalentes a 10.5 años de operación y 491 eventos válidos. El objetivo no es solo minimizar error promedio, sino capturar picos de `stormflow_mgd` con suficiente anticipación para activar medidas preventivas.

Los números más importantes del análisis son:

- `stormflow_mgd` está extremadamente desbalanceado: P50 = 0.1091 MGD, P95 = 1.9714 MGD, P99 = 12.8159 MGD, P99.9 = 50.9801 MGD y máximo = 225.3305 MGD.
- El máximo es 114.3 veces el P95 y 2065.4 veces la mediana, por lo que una pérdida estándar tenderá a optimizar la masa de valores pequeños e ignorar los picos operativamente críticos.
- La cuenca responde muy rápido: el lag óptimo lluvia -> stormflow es 10 minutos con correlación máxima 0.6372; a 30 minutos cae a 0.4522, a 1 hora a 0.3781 y a 2 horas a 0.2295.
- `flow_total_mgd` y `stormflow_mgd` son casi idénticos en términos lineales (`r = 0.998`), mientras que `baseflow_mgd` casi no aporta señal directa sobre lluvia (`r = 0.016`) ni sobre stormflow (`r = 0.05`).
- La estacionalidad existe, pero es moderada: los eventos mensuales van de 29 a 47; febrero tiene la mayor media de stormflow y abril el mayor pico absoluto.

Con base en esto, propongo un enfoque orientado explícitamente a extremos, con contexto corto, arquitectura convolucional temporal y entrenamiento sesgado hacia subestimaciones en picos.

## 1. Features

### 1.1 Variables base

Usar como entradas principales:

- `rain_in`
- `flow_total_mgd`
- `baseflow_mgd`
- `stormflow_mgd_lagged` solo dentro de la ventana de entrada, nunca en el horizonte futuro

Justificación:

- `flow_total_mgd` tiene correlación 0.998 con `stormflow_mgd`, así que es la señal instantánea más informativa. Sería un error no aprovecharla.
- `rain_in` es la señal causal primaria y tiene correlación cruzada máxima con 10 minutos de desfase, por lo que debe estar muy representada en lags cortos.
- `baseflow_mgd` tiene baja correlación global, pero sigue siendo útil para separar caudal sanitario/base de respuesta de tormenta en transiciones y eventos de baja intensidad.
- Incluir historia reciente de `stormflow_mgd` ayuda a modelar inercia, ascenso y recesión del hidrograma. En este problema importa mucho la forma local del ascenso al pico.

### 1.2 Features de memoria corta de lluvia

Crear acumulados y resúmenes causales de lluvia:

- rolling sum de 10, 15, 30, 60, 120 y 360 minutos
- rolling max de 10, 30 y 60 minutos
- intensidad reciente: `rain_in / 5min`, `max_10min`, `max_30min`
- tiempo desde la última lluvia significativa (`rain_in > umbral pequeño`)

Justificación:

- Como el lag óptimo es 10 minutos y la correlación se degrada claramente a 30, 60 y 120 minutos, la cuenca responde con memoria corta. Los acumulados de 10-120 minutos capturan casi toda la dinámica útil.
- Incluir 360 minutos no busca correlación lineal alta, sino representar condiciones previas de humedad/saturación sin obligar al modelo a memorizar 72 pasos crudos.
- El tiempo desde lluvia reciente ayuda a distinguir entre un pico nuevo y la cola de un evento previo.

### 1.3 Features de tendencia hidrológica

Crear derivadas y relaciones:

- `delta_flow_5m`, `delta_flow_15m`
- `delta_stormflow_5m`, `delta_stormflow_15m`
- `flow_minus_baseflow` como chequeo redundante de consistencia
- ratio `stormflow / max(flow_total, eps)`
- slope de lluvia y slope de flujo en ventanas cortas

Justificación:

- El objetivo operativo depende mucho de detectar el ascenso al pico antes del máximo. Las derivadas capturan pendiente y aceleración, algo que una correlación estática no resume.
- Cuando el máximo es 114.3 veces el P95, detectar transición de régimen importa más que refinar décimas alrededor de 0.1 MGD.

### 1.4 Features temporales cíclicas

Crear:

- `hour_sin`, `hour_cos`
- `month_sin`, `month_cos`
- opcional: `day_of_week_sin`, `day_of_week_cos`

Justificación:

- La estacionalidad mensual es moderada, no extrema: eventos por mes entre 29 y 47. Esto sugiere que sí conviene informar el calendario al modelo, pero como señal secundaria, no dominante.
- Codificación cíclica evita discontinuidades artificiales entre 23:55 y 00:00 o entre diciembre y enero.

### 1.5 Features basadas en eventos

Crear durante entrenamiento/evaluación:

- máscara `is_event_period` derivada de los eventos válidos
- distancia temporal al inicio del evento si el timestamp cae dentro de evento
- bucket de magnitud de evento según `total_storm_vol` o `peak_wwf`

Justificación:

- Hay 491 eventos válidos, suficiente para usar la estructura por evento en entrenamiento y métricas.
- Esto permite hacer muestreo inteligente y evaluación por severidad sin contaminar el pipeline inferencial.

### 1.6 Limpieza de target

Para la variable objetivo:

- clipear `stormflow_mgd` a 0 antes de entrenar

Justificación:

- El 26.4% de los valores son negativos y el análisis los identifica como artefacto físico imposible. Si no se corrigen, el modelo desperdicia capacidad intentando reproducir ruido no físico.

## 2. Arquitectura

### 2.1 Modelo propuesto: TCN residual multiescala

Propongo una **Temporal Convolutional Network (TCN)** con bloques residuales dilatados, salida many-to-one o many-to-few según el horizonte.

Configuración inicial:

- 4 a 6 bloques residuales
- dilataciones: 1, 2, 4, 8, 16, 32
- kernel size = 3
- canales por bloque: 32 -> 64 -> 64 -> 64 -> 32
- dropout moderado: 0.1 a 0.2
- normalización por canal o weight norm
- cabeza final densa para predecir el horizonte

Justificación:

- La respuesta lluvia -> stormflow es muy corta. Un TCN captura relaciones locales y multiescala muy bien sin el coste secuencial de LSTM.
- Con resolución de 5 minutos, 2 horas son 24 pasos. Una TCN con dilataciones cubre sobradamente ese rango y también contexto extendido de varias horas.
- En Colab con T4, la TCN entrena más rápido y estable que LSTM/CNN-LSTM para 1.1M registros.
- Los picos son eventos locales con ascenso rápido. Las convoluciones temporales son especialmente fuertes detectando patrones de subida, ráfagas de lluvia y respuestas abruptas.

### 2.2 Por qué no LSTM como primera opción

- LSTM funciona, pero aquí la memoria realmente útil es corta: el lag óptimo es 10 minutos y a 2 horas la correlación cae a 0.2295.
- Su ventaja en dependencias largas no compensa el mayor coste y menor paralelización.
- En datasets muy desbalanceados, la estabilidad y velocidad de iteración son clave para poder ajustar pérdidas, sampling y thresholds con rapidez.

### 2.3 Por qué no Transformer como primera opción

- El problema no parece dominado por dependencias muy largas ni por patrones de atención global.
- Para 1.1M puntos a 5 minutos, un Transformer puro añade coste y sensibilidad de tuning sin evidencia de que la estructura temporal larga sea el cuello de botella.
- Primero conviene explotar bien la dinámica corta, donde están los mayores retornos.

### 2.4 Extensión recomendada

Si una TCN base no captura bien los extremos > P99.9, la siguiente iteración sería un **híbrido TCN + cabeza de clasificación de evento extremo**:

- salida 1: regresión de `stormflow_mgd`
- salida 2: probabilidad de superar umbral, por ejemplo 10, 25 o 50 MGD

Esto ayuda porque los picos > 50 MGD están en torno al 0.1% o menos del dataset, así que una tarea auxiliar de clasificación fuerza al backbone a reconocer precursores de evento severo.

## 3. Loss function

### 3.1 Pérdida propuesta

Usar una pérdida compuesta:

`Loss = 0.55 * WeightedHuber + 0.30 * AsymmetricUnderpredictionLoss + 0.15 * PeakClassificationLoss`

#### a) Weighted Huber

- Huber en lugar de MSE para no dejar que unos pocos outliers dominen totalmente el gradiente
- peso por magnitud del target:
  - 1x si `y < P95` (1.9714)
  - 4x si `P95 <= y < P99` (12.8159)
  - 10x si `P99 <= y < P99.9` (50.9801)
  - 20x si `y >= P99.9`

Justificación:

- El salto entre P95 y máximo es enorme. Si todos los errores pesan igual, el modelo se optimiza para la región 0-2 MGD.
- Huber mantiene robustez numérica, mientras los pesos reequilibran la importancia operativa.

#### b) Penalización asimétrica por infraestimación

Aplicar un factor extra cuando `y_pred < y_true`:

- 1x si no hay infraestimación
- 2x si hay infraestimación en valores < P99
- 4x si hay infraestimación en `y >= P99`
- 6x si hay infraestimación en `y >= P99.9`

Forma sugerida:

`asym_error = ((y_true - y_pred).clamp(min=0) ** 2) * peak_weight`

Justificación:

- El requisito de negocio dice explícitamente que infraestimar es mucho peor que sobreestimar.
- Esta asimetría debe crecer con la magnitud, porque fallar 5 MGD en un valor de 0.2 no tiene el mismo coste que fallar 20 MGD cerca de un CSO.

#### c) Cabeza auxiliar de clasificación

Clasificar si el target supera umbrales críticos:

- `stormflow >= P99` (12.8159)
- opcionalmente `stormflow >= 50 MGD` como evento extremo

Pérdida:

- BCE focal o BCE ponderada

Justificación:

- Con menos del 0.1% de datos por encima de 50 MGD, la regresión sola puede suavizar picos. La cabeza auxiliar introduce una presión explícita para distinguir régimen normal vs severo.

## 4. Estrategia de desbalance

### 4.1 Limpieza y redefinición del target

- Reemplazar negativos por 0 en `stormflow_mgd` antes de entrenar
- Mantener `flow_total_mgd` y `baseflow_mgd` como features

Justificación:

- 26.4% negativo es demasiado alto como para dejarlo intacto. Son artefactos, no señal física.

### 4.2 Muestreo por ventanas estratificadas

No entrenar con ventanas uniformes puras. Usar sampler mixto:

- 40% ventanas centradas en timestamps de evento
- 30% ventanas donde `stormflow >= P95`
- 20% ventanas donde `stormflow >= P99`
- 10% ventanas de baseflow/no evento

Si estas categorías se solapan, priorizar la más extrema.

Justificación:

- El régimen de interés operativo es rarísimo. Si se muestrea uniforme sobre 1.1M registros, casi todas las ventanas serán de valores pequeños y el modelo aprenderá a predecir casi cero.
- Aun así, mantener 10% de no evento preserva calibración del flujo base y evita una avalancha de falsos positivos.

### 4.3 Curriculum training

Entrenamiento en dos fases:

1. Fase 1: sampler moderado, pesos más suaves, para aprender dinámica general.
2. Fase 2: oversampling más agresivo en `>= P99` y `>= P99.9`, con mayor penalización a infraestimación.

Justificación:

- Saltar directamente a extremos puede volver inestable el entrenamiento inicial.
- El currículo ayuda a que primero aprenda forma del hidrograma y luego se especialice en colas.

### 4.4 Evaluación separada por severidad

Reportar métricas por buckets:

- baseflow / casi cero: `< 0.5 MGD`
- evento pequeño: `0.5 - 2 MGD`
- moderado: `2 - 12.8 MGD`
- grande: `12.8 - 50.98 MGD`
- extremo: `> 50.98 MGD`

Justificación:

- Un solo RMSE oculta si el modelo sirve o no donde importa.
- Estos cortes salen directamente de la distribución real: P95, P99 y P99.9.

## 5. Hiperparámetros iniciales

### 5.1 Horizonte de predicción

Propongo empezar con:

- `horizon = 6` pasos = 30 minutos

Justificación:

- La respuesta óptima es a 10 minutos, así que 30 minutos da margen operativo real sin pedir al modelo una extrapolación demasiado lejana.
- A 1 hora la correlación lluvia -> stormflow ya cae a 0.3781; por tanto 30 minutos es un compromiso razonable entre utilidad y dificultad.

### 5.2 Longitud de secuencia

Propongo:

- `seq_length = 72` pasos = 6 horas

Justificación:

- La señal más fuerte está en 10-120 minutos, pero 6 horas permite incluir memoria de saturación, ascenso, pico y recesión del evento.
- Es suficientemente larga para acumulados hidrológicos y suficientemente corta para entrenar bien en T4.

Valor alternativo a probar:

- `seq_length = 48` pasos = 4 horas si la TCN ya captura igual rendimiento con menor coste.

### 5.3 Batch size

Propongo:

- `batch_size = 256` para TCN en Colab T4

Justificación:

- TCN permite mejor paralelización que LSTM.
- Con secuencias de 72 pasos y pocas variables, 256 suele ser un buen punto de partida. Si aparece OOM, bajar a 128.

### 5.4 Learning rate y optimizador

Propongo:

- optimizador `AdamW`
- `learning_rate = 1e-3`
- `weight_decay = 1e-4`
- scheduler `ReduceLROnPlateau` con factor 0.5 y paciencia 3

Justificación:

- AdamW es una base robusta para series temporales multivariadas con pérdida compuesta.
- `1e-3` suele converger rápido; si la pérdida asimétrica genera inestabilidad, probar `5e-4`.

### 5.5 Regularización

Propongo:

- `dropout = 0.1`
- early stopping con paciencia `6`
- clip de gradiente `1.0`

Justificación:

- Hay 491 eventos válidos, suficiente para entrenar, pero no tantos extremos severos. Una regularización ligera ayuda a no sobreajustar los pocos picos extremos.

### 5.6 Arquitectura concreta inicial

- `num_channels = [32, 64, 64, 64, 32]`
- `kernel_size = 3`
- `dilations = [1, 2, 4, 8, 16]`
- `output_horizon = 6`

Justificación:

- Esta configuración cubre bien el rango temporal útil sin hacer el modelo innecesariamente grande para una T4.

## 6. Pipeline

### 6.1 Split cronológico

Usar split por tiempo, nunca aleatorio:

- train: 70%
- validation: 15%
- test: 15%

Con 10.5 años, esto equivale aproximadamente a:

- train: primeros 7.35 años
- validation: siguientes 1.58 años
- test: últimos 1.58 años

Justificación:

- Es obligatorio por la naturaleza temporal y por la regla del proyecto.
- Además evalúa cambio de régimen realista hacia datos futuros.

### 6.2 Normalización

- Min-Max o Robust scaling solo con estadísticas de train
- Recomendación práctica:
  - `rain_in`, acumulados y variables muy sesgadas: `log1p` seguido de Min-Max
  - `flow_total_mgd`, `baseflow_mgd`, lags y derivados: RobustScaler o Min-Max robusto usando percentiles de train

Justificación:

- La distribución tiene cola extrema. Escalar todo con min/max bruto puede comprimir en exceso casi todo el rango útil por culpa del máximo 225.33.
- `log1p` en lluvia y volúmenes ayuda a estabilizar la cola sin perder orden.

### 6.3 Construcción de secuencias

- sliding window causal
- entrada: `[t-seq_length+1, ..., t]`
- salida: `[t+1, ..., t+horizon]`
- no mezclar ventanas que crucen fronteras entre train/val/test

### 6.4 Data augmentation

No recomiendo augmentation agresivo tipo jitter aleatorio en esta fase.

Sí recomiendo:

- oversampling de ventanas extremas
- pequeña perturbación gaussiana solo en features derivadas o de lluvia si hace falta robustez, nunca en el target

Justificación:

- En hidrología temporal, augmentation fuerte puede destruir relaciones físicas entre lluvia y respuesta.
- Aquí el problema no es falta de volumen total de datos, sino falta de masa en la cola extrema.

### 6.5 Criterio principal de selección de modelo

Seleccionar checkpoint no por loss media global, sino por una combinación de:

- NSE en test/val
- RMSE global
- MAE global
- error relativo del pico por evento
- recall de eventos `>= P99` y `>= P99.9`
- error de timing del pico en minutos

Justificación:

- Si se selecciona solo por RMSE promedio, el modelo tenderá a favorecer el 99% de observaciones pequeñas y fallar exactamente donde el proyecto tiene más valor.

## Recomendación final

La solución inicial recomendada es:

1. Limpiar `stormflow_mgd` negativo a 0.
2. Construir features causales centradas en lluvia reciente, acumulados cortos, lags y derivadas.
3. Entrenar una TCN residual con `seq_length = 72` y `horizon = 6`.
4. Usar pérdida compuesta con ponderación por cuantiles y fuerte penalización a infraestimación.
5. Hacer oversampling estratificado de ventanas de evento y de cola extrema.
6. Evaluar explícitamente por severidad y por timing/magnitud del pico.

Este enfoque está alineado con los datos reales: memoria corta de cuenca, gran volumen histórico, 491 eventos válidos y una cola tan extrema que obliga a diseñar el entrenamiento alrededor de los picos, no alrededor del error promedio.
