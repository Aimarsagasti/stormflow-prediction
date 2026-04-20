# STATE.md - Estado actual del proyecto

**Última actualización:** 2026-04-20 (añadidos hallazgos de revisión con Opus 4.7)
**Mantenido por:** Aimar (actualizar al final de cada sesión de trabajo significativa).

---

## Modelo actual en producción

**Nombre:** `modelo_H1_sinSF` (v1)
**Fecha de entrenamiento:** 2026-04-13
**Ubicación de pesos:** `MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_weights.pt`

### Arquitectura
- TwoStageTCN (clasificador + regresor sobre backbone compartido).
- Backbone: Conv1x1 -> 5 bloques residuales [32, 64, 64, 64, 32] con dilations [1, 2, 4, 8, 16].
- GroupNorm, CausalConv1d, kernel_size=3, dropout=0.2.
- Classifier head: Linear(32,64) -> ReLU -> Dropout -> Linear(64,1) -> Sigmoid.
- Regressor head: Linear(32,128) -> ReLU -> Dropout -> Linear(128,64) -> ReLU -> Linear(64,1).
- Inferencia: switch duro, threshold=0.3.
- Campo receptivo: 125 timesteps = 10.4 horas.
- Parámetros entrenables: ~104K.

### Hiperparámetros
- Horizonte de predicción: 1 paso (5 minutos).
- Sequence length: 72 pasos (6 horas de contexto).
- Batch size: 256.
- Optimizador: AdamW.
- Learning rate: 5e-4.
- Weight decay: 1e-4.
- Max epochs: 100.
- Early stopping patience: 10.
- Early stopping min_delta: 1e-5.
- Grad clip max norm: 1.0.
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=4).
- Mejor época en entrenamiento: 6.

### Features (22)
rain_in, temp_daily_f, api_dynamic,
rain_sum_10m, rain_sum_15m, rain_sum_30m, rain_sum_60m,
rain_sum_120m, rain_sum_180m, rain_sum_360m,
rain_max_10m, rain_max_30m, rain_max_60m,
minutes_since_last_rain,
delta_flow_5m, delta_flow_15m, delta_rain_10m, delta_rain_30m,
hour_sin, hour_cos, month_sin, month_cos

### Loss
TwoStageLoss (componente 0.3 BCE para clasificador + componente 0.7 Huber para regresor, con penalización asimétrica x3 cuando y_pred < y_true).

### Normalización
log1p + z-score. Aplicado al target y a features de lluvia.
Parámetros guardados en `modelo_H1_sinSF_norm_params.json`.

---

## Resultados del último eval (16-abril-2026)

Generado por `evaluate_local.py`. Fuente autoritativa: `outputs/data_analysis/local_eval_metrics.json`.

### Comparación global: SIN stormflow como feature

| Horizonte | Min  | NSE     | RMSE | MAE  | Error Pico | Bias Base | Bias Extremo |
|-----------|------|---------|------|------|------------|-----------|--------------|
| H=1       | 5m   | 0.861   | 0.89 | 0.29 | -21.0%     | +0.21     | -12.7        |
| H=3       | 15m  | 0.471   | 1.74 | 0.38 | -52.1%     | +0.25     | -15.9        |
| H=6       | 30m  | -1.212  | 3.57 | 1.21 | -78.2%     | +0.94     | -50.5        |

### Comparación global: CON stormflow como feature (autoregresivo)

| Horizonte | Min  | NSE     | RMSE | MAE  | Error Pico | Bias Base | Bias Extremo |
|-----------|------|---------|------|------|------------|-----------|--------------|
| H=1       | 5m   | 0.853   | 0.92 | 0.24 | +59.4%     | +0.18     | -8.8         |
| H=3       | 15m  | 0.488   | 1.72 | 0.35 | -48.5%     | +0.24     | -20.4        |
| H=6       | 30m  | 0.255   | 2.08 | 0.49 | -88.2%     | +0.28     | -57.5        |

### Diagnóstico por rangos de magnitud (H=1 SIN SF, el modelo en producción)

| Rango MGD          | Muestras | MAPE    | NSE      |
|---------------------|----------|---------|----------|
| Base (<0.5)        | 152,904  | 73,606% | -14.51   |
| Leve (0.5-5)       | 9,518    | 59.8%   | -0.26    |
| Moderado (5-20)    | 2,305    | 25.8%   | +0.19    |
| Alto (20-50)       | 437      | 19.9%   | +0.02    |
| Extremo (>50)      | 59       | 30.3%   | -0.99    |

### Diagnóstico por rangos de magnitud (H=1 CON SF)

| Rango MGD          | Muestras | MAPE    | NSE      |
|---------------------|----------|---------|----------|
| Base (<0.5)        | 152,901  | 66.7%   | -9.8     |
| Leve (0.5-5)       | 9,520    | 32.4%   | 0.41     |
| Moderado (5-20)    | 2,303    | 17.1%   | 0.51     |
| Alto (20-50)       | 440      | 18.6%   | 0.03     |
| Extremo (>50)      | 59       | 39.8%   | -2.4     |

### Versión v2 (modelo_H1_sinSF_v2, 16-abril-2026)

Reentrenamiento del mismo modelo H1_sinSF con ligeros ajustes. Captura ligeramente mejor los picos que v1 pero su NSE global es inferior. **v1 sigue siendo el modelo en producción** por NSE global. Plots de comparación en `outputs/figures/local_eval/v1vs_v2_*.png`.

---

## Qué funciona

- **Eventos moderados (5-20 MGD) a H=1 SIN SF:** MAPE 25.8%, NSE=0.19 (modesto pero positivo). Con SF sube a NSE=0.52.
- **Eventos leves (0.5-5 MGD) a H=1 CON SF:** MAPE 32%, NSE=0.41. Sin SF el NSE es negativo (-0.26).
- **Predicción base (flujo sin tormenta):** sesgo pequeño y estable (bias ≈ +0.21 MGD).
- **Timing del pico:** el modelo acierta bastante bien CUÁNDO ocurre el pico a H=1.

---

## Qué NO funciona

### Infraestimación sistemática de picos extremos
- Error en pico del ~83% es estructural, no aleatorio.
- Baja varianza entre runs repetidos con diferentes seeds.
- El regresor TIENE capacidad de predecir >100 MGD, pero NO discrimina magnitudes correctamente en extremos.

### 15 de 59 eventos extremos sin lluvia en la ventana de entrada
- Probablemente escorrentía retardada o deshielo.
- Con las features actuales son físicamente impredecibles.
- Documentado en `outputs/data_analysis/extreme_events_no_rain.json`.

### Degradación brutal con el horizonte
- De H=1 a H=3: NSE baja de 0.82 a 0.54 (~-34%).
- De H=3 a H=6: NSE colapsa a -0.45.
- H=6 (30 min) NO es operativamente útil con los datos actuales.

### Dominancia del flujo base en las métricas globales
- 92% de las muestras son base (<0.5 MGD).
- NSE y RMSE globales están dominados por el régimen donde el modelo funciona bien.
- Las métricas por rangos son más informativas para el caso operativo.

---


## Hallazgos del análisis del dataset (2026-04-20)

Generado por `scripts/generate_dataset_stats.py`. Documento completo: `docs/DATASET_STATS.md`. Números brutos: `outputs/data_analysis/dataset_stats.json`.

### [ALTA] El modelo apenas bate al predictor naive

Un predictor trivial `y_pred(t+h) = y(t)` (persistencia) obtiene sobre test:

| Horizonte | NSE naive | NSE modelo SIN SF | Ganancia |
|-----------|----------:|------------------:|---------:|
| H=1       | ~0.826    | 0.861             | +0.035   |
| H=3       | ~0.512    | 0.471             | -0.041   |
| H=6       | ~0.309    | -1.212            | -1.521   |

El modelo mejora marginalmente al naive a H=1 y es peor a H≥3. Un script de una línea (`y_pred = y_last`) supera a la TCN a H=3.

**Implicación:** todas las métricas reportadas deben incluir el baseline naive como referencia. Reportar NSE=0.861 a H=1 sin contextualizar es engañoso.

### [ALTA] Asimetría de extremos entre splits

| Split | n extremos (>50 MGD) | Max MGD |
|-------|---------------------:|--------:|
| train | 677                  | 199.4   |
| val   | 97                   | **225.3** (máximo absoluto del dataset) |
| test  | 59                   | 135.2   |

El modelo se evalúa en test sobre un régimen menos extremo que el de entrenamiento, mientras que val incluye el pico absoluto. El early stopping sobre val puede estar favoreciendo infraestimación de picos grandes.

### [MEDIA] Redundancia masiva entre features

36 pares de features con |Pearson| ≥ 0.7 sobre 22 features totales. Casos más extremos:
- `rain_sum_10m` vs `rain_max_10m`: 0.985
- `rain_sum_10m` vs `rain_sum_15m`: 0.970
- `api_dynamic` correlaciona >0.7 con 9 features distintas

Número efectivo de dimensiones independientes entre las 22 features: probablemente ~8-10.

### [MEDIA] ACF del target limita predictibilidad a H>1

ACF(5 min) = 0.909. ACF(30 min) = 0.556. La ACF decae rápido tras los primeros 30 minutos. Sin incorporar predicción de lluvia externa (opción c), extender horizonte útil más allá de ~15 min tiene límite físico.

### [MEDIA] Tamaño muestral insuficiente para bucket Extremo

Solo 59 muestras extremas en test. Las métricas sobre ese bucket (NSE=-0.99, bias=-12.7 MGD) tienen alta varianza. Un único evento atípico puede desplazarlas.

---

### Hallazgos adicionales de revisión externa con Opus 4.7 (2026-04-20)

Revisión completa archivada en `docs/OPUS_REVIEW_2026-04-20.md`. Los hallazgos que se añaden a los de sección anterior son:

#### [ALTO] Inconsistencia train/inference del regresor en TwoStageTCN

`src/models/loss.py:182-188` entrena el regresor solo en muestras con `y_true_real > 0.5 MGD` (~10% del batch). Durante inferencia, el switch duro con threshold=0.3 ejecuta el regresor también en muestras que el clasificador marca como evento por falso positivo. El regresor nunca ha visto esas regiones, produce salida OOD, y contamina el bucket Base (bias=+0.21 MGD no es ruido, es firma OOD).

**Acción pendiente:** entrenar el regresor en TODAS las muestras manteniendo los pesos por magnitud. Mantener switch duro en inferencia.

#### [MODERADO] `delta_flow_5m` y `delta_flow_15m` son puerta trasera al atajo de flow_total_mgd

Como `flow_total ≈ baseflow + stormflow` y baseflow varía lento, `delta_flow[t] ≈ delta_stormflow[t]`. Estas features transportan de facto la derivada del target en t, reintroduciendo el atajo que se cerró en iter 11 al quitar `flow_total_mgd`.

**Hipótesis a verificar empíricamente:** si al eliminar estas dos features el NSE cae al nivel del naive (~0.82), el modelo está haciendo AR(1) disfrazado en lugar de aprender lluvia→stormflow.

#### [ALTO] Pregunta crítica sin verificar: ¿la ganancia sobre naive viene de los delta_flow o de la lluvia?

La ganancia del modelo sobre naive es solo +0.050 NSE a H=1. Si esa ganancia proviene de `delta_flow_*`, el aparato de TwoStageTCN + 22 features es teatro sobre un AR(1). Si proviene de la lluvia, el modelo sí aprende hidrología pero marginalmente.

**Esta es la pregunta que decide la dirección del proyecto.** Se responde con el ablation de la iteración 16.

#### [PENDIENTE DE VERIFICAR] Afirmación sin medición: "el modelo acierta cuándo ocurre el pico"

Actualmente se afirma en la sección "Qué funciona" sin métrica asociada. Hace falta implementar `peak_lag_minutes` en `evaluate_local.py` para cuantificar el error de timing separado del error de magnitud.

#### [PENDIENTE DE VERIFICAR] Afirmación sin evidencia: "15 extremos sin lluvia son físicamente impredecibles"

Esta defensa es usada para justificar el error en bucket Extremo pero no se ha cruzado con `temp_daily_f` ni con `rain_sum_*` de 24-72h previas. Si los eventos se concentran en enero-marzo con temperaturas bajas + nieve acumulada, son predecibles con las features actuales y la afirmación es falsa.

**Acción pendiente:** análisis cruzado que genere `docs/EXTREME_EVENTS_ANALYSIS.md`.

#### Cotas de NSE alcanzable (derivadas del análisis de Opus)

Descomposición del denominador de NSE por bucket sobre test:

- Bucket Extremo contribuye ~18% del denominador.
- Bucket Alto contribuye ~24%.
- Buckets Moderado+Leve contribuyen ~32%.
- Bucket Base contribuye ~26%.

Cota superior estimada a H=1 con features actuales: **NSE alcanzable 0.88-0.92** (predicción perfecta de extremos con lluvia + mejora moderada del resto). Cota para peak_err_pct en bucket Extremo filtrado a los 44 con lluvia: **-5% con regresor óptimo**. Por encima de 0.92 se requieren señales externas (radar de lluvia futura, saturación del suelo, deshielo explícito).

## Siguientes pasos priorizados

**Prioridad actual:** mejorar el modelo. El TFM se redactará después (margen hasta septiembre 2026).

Plan de iteraciones tras la revisión con Opus 4.7 (2026-04-20). Cada iteración es independiente en atribución: se ejecutan y evalúan por separado para saber qué aportó cada cambio.

### Iteración 16 — Ablation de `delta_flow_*` [INMEDIATA]

Eliminar `delta_flow_5m` y `delta_flow_15m` de `FEATURE_COLUMNS` en `claude_train.py` (pasa de 22 a 20 features). Reentrenar `modelo_H1_sinSF`. Evaluar con `evaluate_local.py`.

**Responde:** ¿la ganancia de +0.050 NSE sobre el naive viene de estas features o de la lluvia?

**Decide:** si NSE cae a 0.81-0.82 (nivel naive), el proyecto cambia de enfoque radical. Si cae a 0.84-0.85, seguir con iteración 17. Si se mantiene o mejora, las features eran redundantes/dañinas y seguimos.

### Iteración 17 — Regresor no-condicional en TwoStageLoss

Modificar `src/models/loss.py` para que el regresor se entrene sobre TODAS las muestras, no solo las de evento. Mantener switch duro en inferencia. Mantener los pesos por magnitud.

**Responde:** ¿el bias del bucket Base (+0.21 MGD) es OOD del regresor o ruido real?

**Decide:** si bias_base baja claramente (≤+0.05) y NSE no empeora, la hipótesis A2 de Opus es correcta. Si empeora, se revierte.

### Iteración 18 — Añadir `peak_lag_minutes` a evaluación

No es reentrenamiento. Añadir métrica de error de timing a `evaluate_local.py` y re-evaluar el mejor modelo que tengamos a estas alturas.

**Responde:** ¿el modelo realmente acierta "cuándo" ocurre el pico, o es una afirmación sin verificar?

### Análisis paralelo — Los 15 extremos sin lluvia (D3 de Opus)

Notebook local (no Colab, no GPU). Cruzar los 15 eventos sin lluvia con `temp_daily_f` y acumulados de 24-72h previas. Generar `docs/EXTREME_EVENTS_ANALYSIS.md`.

**Responde:** ¿son realmente "físicamente impredecibles" o tienen patrón (deshielo, temperaturas bajas precedidas de nieve)?

### Iteración 19 — Cambio de split a años completos

Modificar `src/pipeline/split.py` para cortar en años calendario (8 train / 1 val / 1.5 test) en lugar de 70/15/15 por filas. Reentrenar con el mejor modelo disponible.

**Responde:** ¿la evaluación actual sobre-estima o sub-estima el rendimiento real?

### Iteración 20 — Reducción de features por permutation importance

Con el modelo ya limpio (iteraciones 16-17 aplicadas), calcular PI de forma fiable. Eliminar redundancias según `DATASET_STATS.md` sección 5. Objetivo: 10 features sin perder NSE.

### Condiciones de parada y bloqueos

- Si iteración 16 revela NSE=0.82, se detiene la secuencia y se reevalúa todo.
- Opción (c) original (modelo de lluvia externo del MSD) sigue bloqueada esperando reunión con el jefe. Si se desbloquea, salta al primer puesto.

---

## Bloqueos actuales

Ninguno técnico. El proyecto avanza. La decisión pendiente es qué iteración probar a continuación (ver "Siguientes pasos").

---

## Estado del TFM (documento académico)

**No iniciada la redacción extensa.** El plan es:
- Estructura definida (6 secciones). Ver abajo.
- Estado del arte parcialmente estructurado en NotebookLM (3 notebooks temáticos con 22 papers).
- Redacción extensa: a partir de que el modelo esté cerrado. Margen hasta septiembre 2026.

### Estructura actual del TFM
1. Introducción y contexto institucional (MSD, CSOs, Consent Decree).
2. Marco teórico / Estado del Arte (TCN, LSTM, modelos hidrológicos, zero-inflated models).
3. Metodología (datos, features, arquitectura, loss, entrenamiento).
4. Resultados y análisis:
   - Curva de predicibilidad por horizonte (SIN SF y CON SF).
   - Diagnóstico por rangos de magnitud.
   - Análisis de eventos sin lluvia vs con lluvia.
   - Visualización de eventos moderados donde el modelo funciona bien.
5. Discusión (limitaciones, por qué los extremos son difíciles, valor operativo).
6. Conclusiones y trabajo futuro (integración con modelo de lluvia, sistema de alertas, TimeSeriesSplit).

---

## Artefactos generados más recientes

- **Plots locales:** 56 archivos en `outputs/figures/local_eval/` (16-abril-2026).
- **Métricas:** `outputs/data_analysis/local_eval_metrics.json` (16-abril-2026).
- **Análisis de extremos sin lluvia:** `outputs/data_analysis/extreme_events_no_rain.json` (16-abril-2026).
- **Threshold sweeps:** `threshold_sweep_H1_conSF.json` y `threshold_sweep_H1_sinSF.json`.
- **Pesos de los 7 modelos entrenados:** `MC-CL-005/Pesos 13-04-2026/`.

---

## Reuniones y comunicación externa

- **Pendiente:** reunión con jefe del MSD para preguntar por el modelo de predicción de lluvia (formato, resolución temporal/espacial, horizonte, precisión).