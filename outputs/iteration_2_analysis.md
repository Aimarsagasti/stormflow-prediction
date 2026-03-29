# Analisis de la iteracion 2

## Resumen ejecutivo

La iteracion 1 no fallo por un unico bug. Fallo por la combinacion de cuatro problemas de fondo:

1. El entrenamiento vio una distribucion muy distinta a validacion/test.
2. El modelo no tenia un mecanismo explicito para distinguir "hay evento" vs "no hay evento".
3. La arquitectura seguia resolviendo la tarea como una regresion unica y suave, lo que favorece soluciones promedio.
4. El pipeline seguia usando variables muy cercanas al target, lo que facilito memorizar patrones locales en lugar de aprender una respuesta hidrologica mas robusta.

Eso explica por que coexistieron tres sintomas aparentemente contradictorios:

- sesgo positivo fuerte cuando el real esta cerca de cero
- captura casi perfecta de un pico extremo aislado
- fallo catastrofico en otros picos de magnitud similar

El modelo no estaba realmente generalizando picos; estaba mezclando memorizacion parcial de ciertos patrones con una tendencia a "encenderse" demasiado facil fuera de evento.

## Que salio mal y por que

### 1. Hubo drift de distribucion entre train y validacion

Aunque la version actual ya habia suavizado el sampler, el problema base seguia ahi: el entrenamiento forzaba una frecuencia de eventos y extremos muy superior a la real.

Consecuencias:

- el modelo aprendio que un batch "tipico" contiene muchos eventos
- en validacion/test, donde domina el baseflow, sobrepredijo con facilidad
- el sesgo positivo en `base` (+1.66 MGD en test, +1.41 MGD en validacion) es exactamente el patron esperado cuando el modelo se calibra con demasiados positivos

El scatter con una columna vertical densa en `x ≈ 0` y predicciones hasta `130 MGD` es la evidencia visual mas clara de este problema.

### 2. La tarea real tenia dos preguntas y el modelo intentaba resolverlas como si fueran una sola

La prediccion operativa correcta no es solo "cuanto stormflow habra", sino:

1. ¿habra evento o no?
2. si si, ¿que magnitud tendra?

La iteracion 1 trataba ambas cosas con una sola salida de regresion. Ese enfoque castiga poco las falsas alarmas cuando el target es casi cero y, al mismo tiempo, obliga a la misma cabeza a modelar baseflow y extremos.

Consecuencias:

- cuando no hay evento, faltaba un mecanismo explicito para apagar la prediccion
- cuando si hay evento, la misma salida tenia que pasar de casi cero a colas extremas muy raras
- eso genera una solucion inestable: unas veces sobreenciende fuera de evento y otras veces colapsa un pico real

### 3. La arquitectura seguia siendo demasiado "promediadora"

La TCN actual mejoro frente a la primera version, pero seguia usando una sola representacion final para toda la decision. Eso no basta cuando hay que separar:

- presencia de evento
- magnitud del evento
- sensibilidad a la parte mas reciente de la secuencia

El hecho de acertar el pico maximo y fallar brutalmente el tercer pico indica que el modelo no tiene una representacion suficientemente discriminativa de las distintas formas de evento. No basta con "ver mucha historia"; hace falta decidir mejor que partes de la historia reciente importan.

### 4. El set de features seguia demasiado cerca del target

Las features `baseflow_mgd`, `delta_stormflow_5m`, `delta_stormflow_15m` y `stormflow_flow_ratio` dependen directa o indirectamente de `stormflow_mgd`.

Eso no es leakage temporal del target futuro, pero si introduce una dependencia muy fuerte del target actual/de su descomposicion. En la practica, el modelo queda demasiado autoregresivo y aprende patrones locales muy especificos de los eventos vistos en train.

Consecuencias:

- mejora aparente en algunos picos ya "familiares"
- peor generalizacion entre tipos de evento
- mayor sensibilidad a ruido o pequenas oscilaciones locales cerca de cero

Para un sistema predictivo operativo, conviene que el modelo se apoye mas en lluvia + flow total + contexto temporal, y menos en transformaciones directas del propio stormflow.

### 5. El control del sobreajuste seguia siendo insuficiente

La observacion de entrenamiento es clara:

- la `val_loss` se aplana cerca de epoch 8
- la `train_loss` sigue bajando hasta el early stop de epoch 25

Eso significa que el modelo seguia ganando capacidad para memorizar train despues de que la generalizacion se habia agotado. El early stopping actual detecta "no mejora", pero todavia permite demasiadas epocas adicionales de ajuste fino sobre patrones ya aprendidos.

## Decision de rediseño

No voy a hacer otro ajuste pequeño sobre el mismo enfoque. La iteracion 2 necesita un cambio estructural:

### A. Pasar de regresion simple a prediccion multitarea con gating por evento

Voy a separar la tarea en dos cabezas:

- una cabeza de clasificacion: probabilidad de evento
- una cabeza de magnitud: magnitud potencial de stormflow

La prediccion final sera una version "gateada":

- si la probabilidad de evento es baja, la salida final se suprime
- si la probabilidad de evento es alta, la magnitud puede expresarse

Justificacion:

- ataca directamente las falsas alarmas cuando el target real esta cerca de cero
- reutiliza `is_event`, que ya existe en el pipeline pero estaba infrautilizado
- obliga al modelo a aprender primero la activacion del evento y despues su magnitud

### B. Quitar por defecto el muestreo estratificado agresivo

Voy a dejar el entrenamiento en distribucion natural y usar el desbalance principalmente via loss ponderada, no via duplicacion frecuente de ventanas raras.

Justificacion:

- reduce drift train/val/test
- reduce memorizacion de ventanas extremas repetidas
- mantiene la prioridad operacional de picos dentro de la loss, que es donde debe estar el sesgo de negocio

### C. Simplificar las features para evitar dependencia excesiva del target

Voy a quitar del input del modelo las features mas cercanas a `stormflow_mgd`:

- `baseflow_mgd`
- `delta_stormflow_5m`
- `delta_stormflow_15m`
- `stormflow_flow_ratio`

Justificacion:

- reduce el caracter autoregresivo y memoristico del modelo
- mejora la coherencia con un escenario operativo real
- obliga al backbone a aprender mejor la relacion lluvia/flow/contexto -> stormflow futuro

### D. Mejorar la representacion temporal

Voy a reemplazar la lectura simple de una sola representacion final por una combinacion de:

- ultimo timestep causal
- pooling temporal con atencion aprendible

Justificacion:

- el ultimo timestep conserva el estado mas reciente
- la atencion permite rescatar fragmentos previos relevantes cuando el pico se esta gestando pero aun no exploto
- esto deberia mejorar consistencia entre eventos de forma similar pero con distinta evolutiva temporal

### E. Rediseñar la loss para reflejar mejor el problema operativo

La loss nueva va a combinar:

- regresion robusta ponderada
- penalizacion fuerte de infraestimacion en eventos severos
- penalizacion explicita de falsas alarmas fuera de evento
- BCE para clasificacion de evento

Justificacion:

- la infraestimacion de picos sigue siendo el error mas caro
- pero ahora tambien se penaliza de forma explicita el patron que vimos en el scatter: predicciones altas cuando el real es casi cero
- la clasificacion de evento deja de ser una variable latente y pasa a supervisarse directamente

### F. Endurecer el control de sobreajuste

Voy a hacer el early stopping mas estricto mediante `min_delta` y dejar el entrenamiento preparado para cortar antes cuando la mejora sea marginal.

Justificacion:

- la observacion empirica ya muestra que despues de ~epoch 8 la generalizacion deja de crecer
- seguir hasta epoch 25 solo da mas oportunidad de memorizar

## Cambios concretos que voy a implementar

1. `src/features/engineering.py`
   - retirar features derivadas directamente de stormflow del set de entrada

2. `src/pipeline/sequences.py`
   - alinear y devolver `is_event` a nivel de target horizon
   - quitar el sampler agresivo por defecto
   - recalibrar pesos de muestra usando magnitud + evento

3. `src/models/tcn.py`
   - convertir el modelo en backbone compartido + dos cabezas
   - agregar pooling temporal con atencion
   - gatear la salida de magnitud con la probabilidad de evento

4. `src/models/loss.py`
   - aceptar salida multitarea
   - agregar BCE de evento
   - agregar penalizacion explicita de falsas alarmas cerca de cero

5. `src/training/trainer.py`
   - entrenar con batches que incluyen `event_target`
   - soportar salida multitarea
   - endurecer early stopping con `min_delta`
   - permitir inferencia con metadata de eventos

6. `src/evaluation/metrics.py`
   - mantener compatibilidad con el nuevo flujo de evaluacion usando mascara de evento alineada

7. `configs/default.yaml`
   - limpiar el archivo para que vuelva a ser YAML valido sin texto instruccional incrustado

## Resultado esperado de esta iteracion

No espero resolver todo en una sola vuelta, pero si deberian verse mejoras muy concretas:

- menos predicciones grandes cuando el real esta cerca de cero
- menor sesgo positivo en `base`
- validacion menos divergente respecto a train
- mayor consistencia entre picos, incluso si el maximo absoluto ya estaba bien
- `event_only` evaluable de forma coherente usando la mascara alineada al horizonte

Si despues de esta iteracion la captura de timing sigue floja, la siguiente mejora natural seria pasar a prediccion multi-horizonte o incorporar una perdida especifica de pico por evento. Pero antes habia que corregir la estructura del problema, no solo ajustar hiperparametros.
