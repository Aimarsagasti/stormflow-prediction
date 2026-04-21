# STATE.md - Estado actual del proyecto

**Última actualización:** 2026-04-21 (iter16 ejecutado, resultado catastrófico, secuencia 16-20 detenida)
**Mantenido por:** Aimar (actualizar al final de cada sesión de trabajo significativa).

---

## Modelo actual en producción

**Nombre:** `modelo_H1_sinSF` (v1)
**Fecha de entrenamiento:** 2026-04-13
**Ubicación de pesos:** `MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_weights.pt`

> **Aviso crítico (21-abril):** v1 sigue siendo el "mejor modelo disponible" por NSE, pero el diagnóstico de iter16 confirma que su ganancia sobre el predictor naive (+0.050 NSE a H=1) venía **en su totalidad** del atajo `delta_flow_5m` + `delta_flow_15m`. El modelo no aprende la dinámica lluvia-stormflow. Es un AR(1) disfrazado. Ver sección "Diagnóstico de iter16" abajo.

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
- Batch size: 256. Optimizador: AdamW. Learning rate: 5e-4. Weight decay: 1e-4.
- Max epochs: 100. Early stopping patience: 10. Early stopping min_delta: 1e-5.
- Grad clip max norm: 1.0. Scheduler: ReduceLROnPlateau (factor=0.5, patience=4).
- Mejor época en entrenamiento: 6.

### Features (22)
rain_in, temp_daily_f, api_dynamic,
rain_sum_10m, rain_sum_15m, rain_sum_30m, rain_sum_60m,
rain_sum_120m, rain_sum_180m, rain_sum_360m,
rain_max_10m, rain_max_30m, rain_max_60m,
minutes_since_last_rain,
**delta_flow_5m, delta_flow_15m** (atajo AR(1) confirmado en iter16),
delta_rain_10m, delta_rain_30m,
hour_sin, hour_cos, month_sin, month_cos

### Loss
TwoStageLoss (componente 0.3 BCE para clasificador + componente 0.7 Huber para regresor, con penalización asimétrica x3 cuando y_pred < y_true).

### Normalización
log1p + z-score. Aplicado al target y a features de lluvia.

---

## Resultados del último eval (21-abril-2026)

Generado por `evaluate_local.py`. Incluye iter16.

### Comparación global (ordenado por NSE descendente)

| Modelo | NSE | RMSE | MAE | ErrPico | BiasBase | BiasExt |
|---|---:|---:|---:|---:|---:|---:|
| H1_sinSF (v1) | **0.861** | 0.89 | 0.29 | -21.0% | +0.21 | -12.7 |
| H1_conSF (v1) | 0.854 | 0.92 | 0.24 | +58.8% | +0.18 | -8.9 |
| H1_sinSF_v2 | 0.798 | 1.08 | 0.31 | +33.4% | +0.20 | -5.5 |
| H3_conSF | 0.488 | 1.72 | 0.35 | -48.6% | +0.24 | -20.4 |
| H3_sinSF | 0.471 | 1.74 | 0.38 | -52.1% | +0.25 | -15.9 |
| H6_conSF | 0.255 | 2.07 | 0.49 | -88.2% | +0.28 | -57.4 |
| **H1_sinSF_iter16** | **-0.169** | **2.60** | **0.63** | **+358.6%** | **+0.48** | **+39.2** |
| H6_sinSF | -1.212 | 3.57 | 1.21 | -78.2% | +0.94 | -50.5 |

### NSE vs naive (DATASET_STATS §8)

| Horizonte | NSE naive | NSE v1 sinSF | Ganancia |
|-----------|----------:|-------------:|---------:|
| H=1 | 0.811 | 0.861 | **+0.050** (toda ella del atajo delta_flow, ver iter16) |
| H=3 | 0.409 | 0.471 | +0.062 |
| H=6 | 0.081 | -1.212 | -1.293 |

---

## Diagnóstico de iter16 (21-abril)

Detalle completo en `EXPERIMENTS.md` entrada "Iteración 16".

### Qué reveló iter16

- Quitar `delta_flow_5m` y `delta_flow_15m` (pasando de 22 a 20 features) colapsa el modelo: NSE=-0.169, error pico +358%, bias extremo +39 MGD, predicciones de hasta 625 MGD (extrapolación x3 sobre el máximo del train).
- La ganancia de +0.050 NSE del modelo v1 sobre el predictor naive **venía entera** de esas dos features. No había aprendizaje hidrológico real.
- `H1_conSF` (con `stormflow_mgd` como feature directa) da NSE=0.854, casi idéntico a v1 sinSF. Confirma que toda la señal útil del modelo era autoregresiva, no hidrológica.
- El aparato de TwoStageTCN (backbone dilatado, clasificador+regresor, loss asimétrica, switch duro) no aportaba señal; era infraestructura sobre un AR(1) que funcionaba por las dos features de atajo.

### Hallazgo colateral que invalida documentación previa

El eval reporta **0 eventos extremos sin lluvia en ventana de 72 pasos**, no 15 como se documentaba. Queda invalidado el argumento de "impredecibilidad física" que se usaba para justificar el error en bucket Extremo. Los 59/59 extremos del test tienen lluvia detectable en las 6h previas.

### Implicaciones

1. La métrica NSE global es engañosa en este problema: 92% de las muestras son baseflow y dominan el denominador.
2. Horizonte H=1 (5 min) es operativamente poco útil para el MSD (los operadores necesitan más margen). H=3 y H=6 son los que importan operativamente, y ahí el modelo es mucho peor o directamente peor que naive.
3. Sin modelo de lluvia externo disponible (decisión del 21-abril: no se usará en esta fase), el techo físico estimado por Opus está en NSE 0.88-0.92 a H=1, pero requiere predicción real de extremos con lluvia, no AR(1) disfrazado.

---

## Iteraciones 17-20 descartadas

El plan de iteraciones post-Opus (16-20) se detiene aquí. Motivos:

- **iter17 (regresor no-condicional) descartada**: asumía que el modelo aprendía hidrología pero el regresor generaba OOD en baseflow. Con NSE global negativo tras quitar el atajo, no hay señal que estabilizar.
- **iter18 (peak_lag_minutes) queda pendiente** como métrica adicional útil para cualquier modelo nuevo, pero no como iteración en sí.
- **iter19 (split por años) queda pendiente** como ajuste metodológico para el futuro modelo.
- **iter20 (reducción de features por PI) queda pendiente** hasta tener un modelo que realmente use las features de lluvia.

---

## Fase de diagnóstico profundo (en curso desde 21-abril)

Antes de decidir el camino nuevo (cambio de target, cambio de arquitectura, reformulación del problema, etc.), se lanza una auditoría completa del proyecto con Claude Code usando subagentes especializados.

**Objetivo:** obtener un `DIAGNOSTIC_REPORT.md` con baselines rigurosos, auditoría de código, análisis de extremos, y **una recomendación única del mejor camino a seguir** con los datos actuales (sin modelo de lluvia externo).

**Restricciones fijadas:**
- No se usará modelo de lluvia externo del MSD.
- Claude Code trabaja en rama `diagnostico`, no toca `main`.
- Claude Code NO implementa la solución final; solo diagnostica y recomienda.
- Decisión final sobre arquitectura, target y enfoque la toma Aimar tras revisar el reporte.

**Prompt maestro:** `PROMPT_CLAUDE_CODE.md` (generado 21-abril).

**Subagentes planificados (6):**
1. Auditoría de pipeline y búsqueda de leakages adicionales.
2. Baselines rigurosos (naive, AR(1), AR(k), XGBoost, Random Forest, regresión física).
3. Análisis de los 59 eventos extremos del test, uno a uno.
4. Evaluación de techo físico por horizonte y por bucket de magnitud.
5. Análisis de redundancia de features y permutation importance.
6. Revisión de arquitecturas alternativas (LSTM hidrológica, TFT, atención temporal, etc.) sin entrenar.

**Artefacto final esperado:** `outputs/diagnostic/DIAGNOSTIC_REPORT.md` con resumen ejecutivo, hallazgos por área, caminos posibles y **recomendación única** con justificación técnica.

---

## Bloqueos actuales

Ninguno técnico para lanzar el diagnóstico. La decisión pendiente es la del camino nuevo, que se tomará tras el reporte de Claude Code.

---

## Estado del TFM (documento académico)

**No iniciada la redacción extensa.** Margen hasta septiembre 2026.

Lo aprendido en iter16 (que el modelo v1 era AR(1) disfrazado) es un hallazgo de alto valor académico si se presenta bien. La sección de resultados del TFM puede estructurarse en torno a: planteamiento inicial, fallo detectado mediante baseline naive, replanteamiento, modelo final. Es un arco narrativo honesto y publicable.

### Estructura actual del TFM
1. Introducción y contexto institucional (MSD, CSOs, Consent Decree).
2. Marco teórico / Estado del Arte (TCN, LSTM, modelos hidrológicos, zero-inflated models).
3. Metodología (datos, features, arquitectura, loss, entrenamiento).
4. Resultados y análisis (incluir ahora el hallazgo del AR(1) disfrazado).
5. Discusión (limitaciones, por qué los extremos son difíciles, valor operativo).
6. Conclusiones y trabajo futuro.

---

## Artefactos generados más recientes

- **Plots locales:** `outputs/figures/local_eval/` (actualizado 21-abril con iter16).
- **Métricas:** `outputs/data_analysis/local_eval_metrics.json` (21-abril).
- **Análisis de extremos sin lluvia:** `outputs/data_analysis/extreme_events_no_rain.json` (21-abril, actualizado: 0 extremos sin lluvia en ventana de 72 pasos).
- **Pesos de los 8 modelos entrenados:** `MC-CL-005/Pesos 13-04-2026/` (añadido iter16).

---

## Reuniones y comunicación externa

- **Decisión del 21-abril:** no se usará modelo de predicción de lluvia externo del MSD en esta fase. Primero hay que lograr un modelo que funcione con las features actuales. La reunión con el jefe queda despriorizada hasta tener un modelo mejor o decidir explícitamente que necesitamos esa palanca.