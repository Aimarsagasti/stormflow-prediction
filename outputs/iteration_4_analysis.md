# Analisis de la iteracion 4

## Objetivo de esta iteracion

El objetivo central es eliminar la compresion sistematica de picos extremos observada en la iteracion 3, donde el sesgo en la categoria extrema fue cercano a `-50 MGD`.

El diagnostico operativo es claro:

`pred_final = magnitude * sigmoid(event_logit)`

Cuando `sigmoid(event_logit)` no satura cerca de 1.0 en eventos extremos, la salida final queda limitada aunque la rama de magnitud haya aprendido una amplitud alta. En la practica, esto convierte incertidumbre de clasificacion en infraestimacion de magnitud, que es exactamente el tipo de error mas costoso para MSD.

## Decision de arquitectura

Se reemplaza el esquema multitarea con gating por una regresion directa:

`TCN -> ultimo timestep -> MLP -> escalar`

Cambios concretos:

1. Quitar `TemporalAttentionPooling`.
2. Quitar cabeza de evento.
3. Quitar gating multiplicativo.
4. Mantener backbone TCN completo sin cambios estructurales:
   - `GroupNorm`
   - `CausalConv1d`
   - `TCNResidualBlock`
   - `num_channels=[32, 64, 64, 64, 32]`
   - `dilations=[1, 2, 4, 8, 16]`
   - `kernel_size=3`
   - `dropout=0.2`

Justificacion:

- Esta arquitectura elimina el cuello de botella probabilistico que recortaba amplitud.
- Conserva la capacidad temporal multiescala que ya venia mejorando NSE por iteracion.
- Hace que toda la capacidad del modelo se enfoque en magnitud continua, que es la variable objetivo real.

## Decision de funcion de perdida

La funcion de perdida se simplifica a tres componentes, todas definidas sobre `y_true`:

1. `Weighted Huber` como base robusta para toda la distribucion.
2. Penalizacion asimetrica de infraestimacion para muestras con `y_true >= P95`, escalando por severidad.
3. `Peak MSE` ponderado solo sobre muestras con `y_true >= P95`.

Componentes que se eliminan:

- Event BCE.
- Penalizacion de false positives separada.
- Supervision directa de magnitud auxiliar (ya no existe cabeza separada).

Justificacion:

- Eliminar terminos auxiliares evita objetivos en conflicto y reduce complejidad de optimizacion.
- Mantener Huber protege frente a ruido y outliers sin perder estabilidad.
- La penalizacion asimetrica y Peak MSE mantienen sesgo operativo explicito contra infraestimacion de cola alta.
- Usar umbrales sobre `y_true` evita depender de heuristicas indirectas basadas en `sample_weights`.

## Ajuste en entrenamiento

Se mantiene intacta la estrategia de early stopping y scheduler:

- `min_epochs=20`
- `patience=8`
- `min_delta=5e-5`
- `scheduler_patience=3`

Unico ajuste funcional en trainer:

- El modelo ahora devuelve `torch.Tensor` directo con shape `(batch, 1)`.
- La loss recibe `(y_pred, y_true, sample_weights)` sin `event_targets`.

Justificacion:

- Esto asegura continuidad experimental: solo cambia lo necesario para quitar el sesgo estructural de gating.
- Mantener regimen de entrenamiento evita mezclar efectos de arquitectura/loss con cambios de scheduler o early stopping.

## Hipotesis esperada

Si esta simplificacion corrige la causa raiz, se espera:

1. Menor infraestimacion en eventos extremos.
2. Mejor calibracion de amplitud en cola alta.
3. Menor dependencia de mecanismos auxiliares de clasificacion para producir magnitud.
4. Mejora en metricas de severidad extrema, incluso si la metrica global cambia de forma moderada.
