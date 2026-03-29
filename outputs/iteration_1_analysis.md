# Analisis de la primera iteracion

## Nota sobre la fuente de resultados

El archivo solicitado `outputs/data_analysis/first_training_results.json` no existe en el repositorio actual. Para no bloquear la iteracion, tomo como proxy del primer entrenamiento el archivo `outputs/data_analysis/test_metrics.json`, que contiene las metricas del modelo ya entrenado y el checkpoint `outputs/models/tcn_best.pt`.

## Diagnostico: que salio mal y por que

### 1. El modelo queda mal calibrado en el regimen base

Los resultados muestran:

- NSE global = -9.9758
- RMSE global = 7.9654 MGD
- MAE global = 4.0561 MGD
- En bucket `base` (`< 0.5 MGD`), el bias es **+4.0626 MGD** sobre **152896 muestras**

Eso significa que el modelo predice stormflow positivo fuerte incluso cuando deberia estar cerca de cero. El fallo no es ruido pequeño: es un desplazamiento sistematico.

### 2. El modelo comprime la cola extrema

Los resultados muestran:

- Pico real = 135.1509 MGD
- Pico predicho = 114.7429 MGD
- Error de pico = -20.4079 MGD (-15.10%)
- En bucket `grande`, bias = -9.0352 MGD
- En bucket `extremo`, bias = -60.1255 MGD sobre 58 muestras

Esto indica que el modelo se va a una solucion "promedio": sobrepredice la base y aun asi subpredice los extremos. Esa combinacion es tipica de un modelo que pierde contraste temporal y de una funcion de perdida que no esta enfatizando correctamente la cola.

### 3. La loss actual no esta operando en la escala correcta

`src/models/loss.py` compara `y_true` contra `p95_threshold`, `p99_threshold` y `p999_threshold`, pero el pipeline entrena con targets normalizados. Si al constructor se le pasan los umbrales en MGD reales (1.9714, 12.8159, 50.9801), mientras `y_true` esta en escala normalizada, entonces:

- la componente `Peak MSE` casi nunca se activa
- la penalizacion asimetrica por tramos no distingue correctamente entre moderado, grande y extremo

En la practica, el comportamiento "peak-focused" de la loss queda muy debilitado o desactivado justo donde mas hacia falta.

### 4. El muestreo estratificado y BatchNorm introducen drift de distribucion

El pipeline actual:

- sobre-muestrea fuerte el regimen de evento/extremos
- usa `BatchNorm1d` en una red entrenada con batches artificialmente sesgados

Eso es una mala combinacion para series temporales hidrologicas. `BatchNorm1d` aprende estadisticas de batch que dejan de representar la distribucion real del test, que vuelve a ser mayoritariamente baseflow. Ese desajuste explica muy bien el bias positivo enorme en el bucket base.

### 5. El pooling global promedia demasiada historia para una cuenca de respuesta rapida

La propuesta y el resumen de datos dicen que:

- lag optimo lluvia -> stormflow = 10 minutos
- a 2 horas la correlacion cae a 0.2295

Sin embargo, `src/models/tcn.py` usa `Global Average Pooling` sobre toda la ventana. Para una tarea causal many-to-one, eso diluye la informacion mas importante del final de la secuencia, que es precisamente donde vive la señal para `t + horizon`. El resultado esperado es una prediccion mas suave y menos sensible a ascensos rapidos.

### 6. Val/Test estan siendo mezclados aleatoriamente en los DataLoaders

En `src/pipeline/sequences.py`, cuando no hay sampler estratificado se deja `shuffle=True`. Eso rompe el orden natural de validacion y test, complica el alineamiento con mascaras `is_event`, y explica que en el resultado disponible `event_only.nse` aparezca `NaN`: el pipeline de evaluacion no queda bien preparado para analisis temporal/event-based reproducibles.

### 7. La cola extrema esta siendo atacada de forma inconsistente

Actualmente:

- `sample_weight` se usa solo en `Weighted Huber`
- la componente asimetrica y la de picos no incorporan esos pesos

Eso deja tres objetivos parcialmente desacoplados. La parte de peak emphasis no recibe toda la senal de rebalanceo, asi que el entrenamiento no empuja con suficiente coherencia a los extremos.

## Cambios que voy a hacer y justificacion

### A. Normalizacion del target con `log1p`

Voy a modificar `src/pipeline/normalize.py` para permitir `log1p` tambien en el target `stormflow_mgd`.

Justificacion:

- la distribucion es extremadamente asimetrica: maximo 114.3x P95
- `log1p` comprime la cola sin perder orden
- hace mas facil optimizar base y extremos en la misma red

Resultado esperado:

- menos compresion del rango dinamico
- mejor estabilidad de entrenamiento
- menor tendencia a una prediccion "intermedia" para todo

### B. Umbrales de loss convertidos a la misma escala de entrenamiento

Voy a modificar `src/models/loss.py` para que los umbrales P95/P99/P99.9 puedan convertirse automaticamente desde MGD reales a la escala usada en el entrenamiento usando `norm_params`.

Justificacion:

- la loss debe decidir que es un pico en la misma escala de `y_true`
- si no, `Peak MSE` y la asimetria por severidad quedan mal calibradas

Resultado esperado:

- activacion real de la penalizacion de picos
- mejor separacion entre moderado, grande y extremo

### C. Aplicar `sample_weight` a todas las componentes relevantes de la loss

Voy a hacer que:

- `Weighted Huber` siga usando `sample_weight`
- la componente asimetrica tambien se module por `sample_weight`
- `Peak MSE` tambien respete `sample_weight`

Justificacion:

- hoy la cola extrema recibe una senal fuerte en una parte de la loss y debil en otras
- eso genera objetivos contradictorios

Resultado esperado:

- entrenamiento mas coherente respecto al desbalance
- menor infraestimacion en buckets grandes/extremos

### D. Sustituir `BatchNorm1d` por `GroupNorm`

Voy a modificar `src/models/tcn.py` para eliminar la dependencia de estadisticas de batch sesgadas por el sampler.

Justificacion:

- `BatchNorm1d` con batches artificialmente estratificados distorsiona la calibracion
- `GroupNorm` es mucho mas estable cuando la distribucion del batch no representa el test real

Resultado esperado:

- menos drift train/test
- reduccion del sesgo positivo en baseflow

### E. Sustituir `Global Average Pooling` por lectura del ultimo timestep causal

Voy a usar la representacion del ultimo timestep de la TCN en vez del promedio temporal global.

Justificacion:

- en un modelo causal, el ultimo estado ya resume la historia relevante
- para una cuenca de respuesta rapida, el final de la secuencia importa mucho mas que el promedio de toda la ventana

Resultado esperado:

- mayor sensibilidad a ascensos rapidos
- mejor captura de picos y menor suavizado excesivo

### F. Corregir `shuffle` en validacion y test

Voy a modificar `src/pipeline/sequences.py` para que solo train se mezcle, mientras val/test mantengan orden estable.

Justificacion:

- la evaluacion temporal debe ser reproducible
- hace posible alinear predicciones con `is_event` y con analisis de picos/timing

Resultado esperado:

- `event_only` evaluable de forma consistente
- diagnosticos mas fiables

### G. Rebalancear el sampler de entrenamiento y hacerlo configurable

Voy a suavizar el sesgo del sampler para recuperar mas cobertura del regimen base y dejar las proporciones configurables.

Justificacion:

- el sampler actual arranca demasiado agresivo para una primera iteracion
- la propia propuesta recomendaba curriculum training, no agresividad maxima desde epoch 1

Resultado esperado:

- menos sobreprediccion sistematica en base
- mejor compromiso entre calibracion general y sensibilidad a eventos

## Resultados esperados tras los cambios

No espero perfeccion en una sola iteracion, pero si deberian aparecer mejoras claras:

- NSE global menos negativo y, idealmente, acercandose a cero o positivo
- bias del bucket `base` muy por debajo de +4 MGD
- menor infraestimacion en `grande` y `extremo`
- `Peak MSE` realmente activa durante entrenamiento
- metricas por eventos calculables de forma consistente
- mejor estabilidad entre train y validacion

Si estos cambios funcionan como espero, la siguiente iteracion ya tendria sentido dedicarla a:

- curriculum training explicito por fases
- prediccion multi-horizonte
- evaluacion por evento/timing del pico
