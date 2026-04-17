# STATE.md - Estado actual del proyecto

**Última actualización:** 2026-04-17
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
| H=1       | 5m   | 0.819   | 1.02 | 0.31 | +12.8%     | +0.22     | -3.7         |
| H=3       | 15m  | 0.536   | 1.64 | 0.39 | -48.2%     | +0.26     | -25.8        |
| H=6       | 30m  | -0.447  | 2.89 | 0.81 | -92.6%     | +0.53     | -53.3        |

### Comparación global: CON stormflow como feature (autoregresivo)

| Horizonte | Min  | NSE     | RMSE | MAE  | Error Pico | Bias Base | Bias Extremo |
|-----------|------|---------|------|------|------------|-----------|--------------|
| H=1       | 5m   | 0.853   | 0.92 | 0.24 | +59.4%     | +0.18     | -8.8         |
| H=3       | 15m  | 0.488   | 1.72 | 0.35 | -48.5%     | +0.24     | -20.4        |
| H=6       | 30m  | 0.255   | 2.08 | 0.49 | -88.2%     | +0.28     | -57.5        |

### Diagnóstico por rangos de magnitud (H=1 SIN SF, el modelo en producción)

| Rango MGD          | Muestras | MAPE    | NSE      |
|---------------------|----------|---------|----------|
| Base (<0.5)        | 152,896  | 183.2%  | -115.9   |
| Leve (0.5-5)       | 9,520    | 65.4%   | -2.58    |
| Moderado (5-20)    | 2,303    | 34.4%   | -1.09    |
| Alto (20-50)       | 440      | 31.5%   | -4.14    |
| Extremo (>50)      | 59       | 44.3%   | -3.23    |

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

- **Eventos moderados (5-20 MGD) a H=1:** MAPE 17-34% (modelo con SF), NSE positivo. Operativamente útil.
- **Eventos leves (0.5-5 MGD) a H=1 con SF:** MAPE 32%, NSE=0.41.
- **Predicción base (flujo sin tormenta):** sesgo pequeño y estable.
- **Timing del pico:** el modelo acierta bastante bien CUÁNDO ocurre el pico (a H=1).

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

## Siguientes pasos priorizados

**Prioridad actual:** mejorar el modelo. El TFM se redactará después (margen hasta septiembre 2026).

### 1. Decidir siguiente experimento
No definido aún. Opciones sobre la mesa:

**a) Probar `api_dynamic` reformulado con evapotranspiración.** La feature actual es redundante con `rain_sum_60m` (r=0.94). Podría ser útil si se reformula como índice de saturación del suelo incorporando evapotranspiración real.

**b) Modelo multi-cabeza con clasificación de niveles (5 niveles: Normal/Leve/Moderado/Alto/Severo).** Cross-entropy en vez de MSE. Operativamente puede ser más útil que predecir MGD exactos.

**c) Integrar modelo de predicción de lluvia del MSD.** Los jefes mencionaron que tienen uno. Usar lluvia predicha como input extendería el horizonte efectivo. PENDIENTE de reunión con el jefe para detalles (formato, resolución, precisión).

**d) Cerrar el modelo aquí y empezar análisis por severidad + valor operativo.** Defensa académicamente sólida: el modelo funciona bien donde importa operativamente (moderados a H=1), los extremos sin lluvia son una limitación de los datos no del modelo.

### 2. Decisiones pendientes (no bloqueantes)
- Sistema de alerta por niveles: pendiente de decisión tras ver si se implementa la opción (b).
- Reunión con el jefe sobre modelo de lluvia (opción c).

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