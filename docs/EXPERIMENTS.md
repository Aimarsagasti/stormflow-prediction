# EXPERIMENTS.md - Historial de iteraciones

Log cronológico de todas las iteraciones del modelo de stormflow. Se añade una entrada al final cada vez que se completa un experimento.

---

## Convenciones

Formato de cada entrada:

Iteración N (YYYY-MM-DD)

Hipótesis: por qué se probó este cambio.
Cambio: qué se modificó exactamente.
Resultado: NSE, RMSE, error pico, y otras métricas relevantes.
Lección: qué se aprendió.
Análisis detallado: (opcional) referencia al iteration_N_analysis.md si existe.

Las iteraciones 1-10 tienen análisis detallados completos en `outputs/iteration_N_analysis.md`.
Las iteraciones 11-15 y v2 tienen documentación más reducida porque se hicieron directamente en Colab sin commits detallados de código. La documentación completa vive en los pesos guardados (`MC-CL-005/Pesos 13-04-2026/*_meta.json`) y los mensajes de commit.

---

## Iteración 1 (2026-03-28)

- **Hipótesis:** baseline de TCN con regresión directa, CompositeLoss de 4 componentes y sampler estratificado.
- **Cambio:** arquitectura base.
- **Resultado:** NSE global bajo, bias positivo severo en base (sobreestimación masiva fuera de eventos).
- **Lección:** el sampler estratificado agresivo causaba drift train/val.
- **Análisis detallado:** `outputs/iteration_1_analysis.md`.

---

## Iteración 2 (2026-03-29)

- **Hipótesis:** reducir dependencia del target quitando features autoregresivas fuertes.
- **Cambio:** simplificación de features, eliminando deltas de stormflow y stormflow_flow_ratio.
- **Resultado:** reducción del bias autoregresivo pero empeora capacidad de distinguir intensificación rápida.
- **Lección:** el modelo usaba features directamente derivadas del target como atajo.
- **Análisis detallado:** `outputs/iteration_2_analysis.md`.

---

## Iteración 3 (2026-03-30)

- **Hipótesis:** multitarea con gating multiplicativo (prob evento × magnitud).
- **Cambio:** introducir cabeza de clasificación + multiplicación con magnitud.
- **Resultado:** compresión severa de picos (el gate nunca satura a 1.0).
- **Lección:** **NO usar gating multiplicativo. Usar switch duro (if/else).**
- **Análisis detallado:** `outputs/iteration_3_analysis.md`.

---

## Iteración 4 (2026-03-31)

- **Hipótesis:** simplificar a regresión directa sin multitarea.
- **Cambio:** eliminar cabeza de clasificación, volver a regresión pura.
- **Resultado:** NSE=-7.64, peor que iter 1. Empeoró.
- **Lección:** sin una señal fuerte que distinga evento/no-evento, el modelo colapsa en el desbalance 92% base.
- **Análisis detallado:** `outputs/iteration_4_analysis.md`.

---

## Iteración 5 (2026-04-01)

- **Hipótesis:** penalizar explícitamente la sobreestimación en base.
- **Cambio:** añadir componente de loss "base overprediction penalty".
- **Resultado:** NSE=-1.72. Mejoró el bias base pero no las métricas generales.
- **Lección:** la penalización ayuda pero no ataca el problema raíz (zero-inflation).
- **Análisis detallado:** `outputs/iteration_5_analysis.md`.

---

## Iteración 6 (2026-04-01)

- **Hipótesis:** probar sin log1p para ver si la transformación distorsiona el aprendizaje.
- **Cambio:** quitar log1p del target.
- **Resultado:** **NSE=-26.6. CATASTRÓFICO.**
- **Lección:** **log1p en el target es NECESARIO. Nunca quitarlo.** La distribución heavy-tailed del target requiere esa transformación para que el gradiente no sea inútil.
- **Análisis detallado:** `outputs/iteration_6_analysis.md`.

---

## Iteración 7 (2026-04-01)

- **Hipótesis:** reactivar log1p y añadir tail focus en la loss.
- **Cambio:** log1p de nuevo + peso extra a samples de cola alta.
- **Resultado:** NSE=-2.58, error pico -45.9%. Recuperó lo que iter 6 rompió.
- **Lección:** log1p es innegociable. El tail focus ayuda modestamente.
- **Análisis detallado:** `outputs/iteration_7_analysis.md`.

---

## Iteración 8 (2026-04-02)

- **Hipótesis:** añadir API dinámico (Antecedent Precipitation Index) y temperatura diaria.
- **Cambio:** nuevas features temp_daily_f y api_dynamic.
- **Resultado:** NSE=-1.94, pero bias en extremos empeoró a -63%.
- **Lección:** api_dynamic parecía redundante con rain_sum_*. Temp_daily_f correlaciona con month_cos. Añadir features sin análisis previo no siempre ayuda.
- **Análisis detallado:** `outputs/iteration_8_analysis.md`.

---

## Iteración 9 (2026-04-02)

- **Hipótesis:** learning rate más conservador (5e-4) para estabilizar el entrenamiento.
- **Cambio:** lr reducido de 1e-3 a 5e-4.
- **Resultado:** NSE=-0.86 (mejor que iter 8) pero error pico empeoró a -71%.
- **Lección:** lr más bajo ayuda al NSE global pero hace al modelo más "conservador" en los picos.
- **Análisis detallado:** `outputs/iteration_9_analysis.md`.

---

## Iteración 10 (2026-04-06)

- **Hipótesis:** cambiar normalización de MinMax a z-score para descomprimir el rango de features.
- **Cambio:** log1p + z-score en vez de log1p + MinMax.
- **Resultado:** **NSE=-0.55 (mejor hasta la fecha). RMSE=2.99, error pico -60%.**
- **Lección:** z-score da más resolución al modelo en picos (target normalizado llega a z-scores de 30-70 en vez de 0.7-1.0 con MinMax). Aun así, 92% de features de lluvia siguen comprimidas porque la distribución original está dominada por ceros.
- **Análisis detallado:** `outputs/iteration_10_analysis.md`.

---

## Iteración 11 (2026-04-XX)

- **Hipótesis:** eliminar `flow_total_mgd` como feature. Tenía correlación r=0.9976 con el target, sospechosa de ser un atajo.
- **Cambio:** quitar `flow_total_mgd` de FEATURE_COLUMNS.
- **Resultado:** NSE global empeoró ligeramente pero el error en picos mejoró de -60% a -41%.
- **Lección:** **`flow_total_mgd` era un atajo. NUNCA re-incluirla.** El modelo la usaba para copiar el target en vez de aprender lluvia -> stormflow.
- **Análisis detallado:** no disponible en `iteration_11_analysis.md`. Documentación parcial en mensajes de commit y en los chats de Claude del 6-9 de abril.

---

## Iteración 12 (2026-04-XX)

- **Hipótesis:** `api_dynamic` es redundante con `rain_sum_60m`. `hour_sin/cos` tienen PI negativo. Quitarlas podría simplificar el modelo.
- **Cambio:** analizar permutation importance exhaustivo. Confirmación de que `rain_sum_60m` a `rain_sum_360m` son necesarios.
- **Resultado:** al quitar los `rain_sum_*` largos en pruebas, `rain_max_10m` dominó con PI=138% y el modelo sobreestimó picos +114%. NO eliminar los rolling sums largos a la vez.
- **Lección:** las features de contexto temporal largo son necesarias para calibrar magnitud. `hour_sin/cos` tienen PI negativo consistente (no ayudan). `api_dynamic` es redundante pero no perjudicial.
- **Análisis detallado:** no disponible como archivo. Documentación parcial en chats y en `project_status.md`.

---

## Iteración 13 (2026-04-XX)

- **Hipótesis:** separar el problema en dos etapas (Hurdle Model). Un clasificador decide si hay evento, un regresor solo predice magnitud cuando sí.
- **Cambio:** **TwoStageTCN** con backbone compartido + classifier head (BCE) + regressor head (Huber asimétrica). Inferencia con switch duro, threshold=0.3.
- **Resultado:** **CAMBIO ARQUITECTÓNICO MÁS IMPACTANTE DEL PROYECTO.** NSE y error pico mejoraron significativamente. Picos capturados con anomalía de -2.6% (luego se confirmó que fue excepcional).
- **Lección:** la arquitectura two-stage es la correcta para zero-inflated. Mantener esta arquitectura como base desde ahora. El ~2.6% de error fue anómalo (baja varianza confirmada en iters posteriores, el valor sistemático es ~-60/-83%).
- **Commit:** `fea90be Iter 13: Modelo Two-Stage (Hurdle) para zero-inflated stormflow`.
- **Análisis detallado:** no disponible como archivo.

---

## Iteración 14 (2026-04-XX)

- **Hipótesis:** un scheduler más sofisticado (CosineAnnealingWarmRestarts) puede mejorar la convergencia.
- **Cambio:** reemplazar ReduceLROnPlateau por CosineAnnealingWarmRestarts. También fix en diagnostics para que funcione con TwoStageTCN.
- **Resultado:** empeoró los resultados. El scheduler cíclico no ayuda en este problema.
- **Lección:** **NO usar Cosine Annealing en este proyecto. Usar ReduceLROnPlateau.**
- **Commit:** `7c368e8 Iter 14: CosineAnnealingWarmRestarts + fix diagnostics para TwoStageTCN`.

---

## Iteración 14b (2026-04-XX)

- **Hipótesis:** revertir el scheduler para volver a ReduceLROnPlateau.
- **Cambio:** revert del scheduler de iter 14.
- **Resultado:** métricas vuelven al nivel de iter 13.
- **Lección:** confirmación de que ReduceLROnPlateau es la opción correcta para este problema.
- **Commit:** `fdc4399 Iter 14b: Revertir a ReduceLROnPlateau (Cosine empeoraba)`.

---

## Iteración 15 (2026-04-XX)

- **Hipótesis:** ponderar la loss del regresor por magnitud (alpha=0.05) para forzar mejor predicción de extremos.
- **Cambio:** introducir magnitude-weighted regression loss con alpha=0.05.
- **Resultado:** alpha=0.05 demasiado agresivo, degradó resultados. Se revirtió.
- **Lección:** **la ponderación por magnitud no resuelve la infraestimación de extremos.** El problema es estructural (pocos samples de extremos + 15 de 59 sin lluvia en la ventana), no se arregla con pesos de loss.
- **Commits:** `d72bff9 Iter 15: magnitude-weighted regression loss (alpha=0.05)` y `7e256da Revert alpha=0.05, demasiado agresivo`.

---

## Iteración 15b (2026-04-XX)

- **Hipótesis:** una versión más suave de la ponderación por magnitud + penalización explícita por sobreestimación excesiva.
- **Cambio:** loss gradual por magnitud + penalización por sobreestimación.
- **Resultado:** mejora marginal. No resuelve el problema de fondo.
- **Lección:** confirma que la infraestimación de picos extremos es estructural, no ajustable con tuning de loss.
- **Commit:** `1fbaaf1 feat: loss gradual por magnitud y penalizacion por sobreestimacion excesiva`.

---

## Experimento: comparación de horizontes (2026-04-13)

Notebook `horizon_comparison.py`. Entrenamiento de 6 modelos: H={1,3,6} × {con stormflow, sin stormflow}.

- **Hipótesis:** caracterizar cómo degrada el modelo con el horizonte de predicción y si stormflow como feature ayuda.
- **Cambio:** 6 entrenamientos controlados, misma arquitectura, solo cambia horizonte y set de features.
- **Resultado (SIN stormflow):**
  - H=1: NSE=0.819, error pico +12.8%.
  - H=3: NSE=0.536, error pico -48.2%.
  - H=6: NSE=-0.447, error pico -92.6%.
- **Resultado (CON stormflow):**
  - H=1: NSE=0.853, error pico +59.4%.
  - H=3: NSE=0.488, error pico -48.5%.
  - H=6: NSE=0.255, error pico -88.2%.
- **Lección:**
  - H=1 SIN SF es el mejor modelo rain-only (el más relevante operativamente).
  - Stormflow como feature mejora NSE global en H=1 pero sobreestima picos severamente (+59.4%).
  - Degradación brutal con el horizonte: H=6 (30 min) no es útil.
  - 15 de 59 eventos extremos no tienen lluvia en la ventana (físicamente impredecibles).
- **Commit:** `f9c5ea8 feat: evaluate_local, metricas, analisis extremos, threshold sweep`.

---

## Iteración v2 (2026-04-16)

- **Hipótesis:** reentrenar H1_sinSF con ajustes para ver si se consigue mejor captura de picos sin perder NSE global.
- **Cambio:** reentrenamiento de modelo_H1_sinSF con configuración ligeramente modificada (detalles en `modelo_H1_sinSF_v2_meta.json`).
- **Resultado:** captura ligeramente mejor los picos que v1 pero NSE global inferior. **v1 sigue siendo el modelo en producción por NSE global.**
- **Lección:** trade-off claro entre NSE global y captura de picos extremos. La decisión de cuál modelo elegir depende del caso operativo: si se valora más el MAPE medio, v1. Si se valora más no infraestimar picos, v2.
- **Plots de comparación:** `outputs/figures/local_eval/v1vs_v2_*.png`.

---

## Resumen de conclusiones firmes (abril 2026)

1. Cambios masivos = resultados impredecibles. UN cambio por iteración.
2. Gating multiplicativo comprime picos. Usar switch duro.
3. log1p en el target es innegociable.
4. z-score mejor que MinMax (pero no resuelve compresión de features de lluvia).
5. Permutation Importance en modelo malo NO es fiable.
6. Sampler estratificado agresivo causa drift. Mejor distribución natural + loss ponderada.
7. Two-Stage (Hurdle) es el cambio más impactante.
8. Cosine Annealing empeora. Usar ReduceLROnPlateau.
9. flow_total_mgd es un atajo. NUNCA re-incluir.
10. No quitar todos los rain_sum largos a la vez.
11. La ponderación por magnitud en la loss NO resuelve la infraestimación de extremos.
12. El error en pico ~83% es SISTEMÁTICO, no aleatorio.
13. 15 de 59 eventos extremos no tienen señal de lluvia en la ventana (físicamente impredecibles).
14. El regresor TIENE capacidad (puede predecir >100 MGD) pero no discrimina magnitudes.
15. Cuando stormflow es feature Y target, normalize_splits lo normaliza DOS veces. Parche obligatorio.
16. Stormflow como feature mejora métricas globales pero NO ayuda con picos extremos.
17. El modelo funciona bien en eventos moderados (5-50 MGD) a H=1. Juzgarlo solo por extremos es injusto.
18. Colab SOLO para entrenar. Todo análisis y plots en VS Code local.

# EXPERIMENTS.md - Registro de iteraciones

Registro de cada iteración del modelo con formato estándar: hipótesis, cambio aplicado, resultado, decisión.

---

## Iteración 16 - Ablation de `delta_flow_5m` y `delta_flow_15m`

**Fecha:** 2026-04-21
**Modelo:** `modelo_H1_sinSF_iter16`
**Features:** 20 (v1 tenía 22, se eliminaron `delta_flow_5m` y `delta_flow_15m`)

### Hipótesis

Basada en la revisión externa de Opus 4.7 (ver `docs/OPUS_REVIEW_2026-04-20.md` §A1 y §D1):

> Como $\text{flow\_total} = \text{baseflow} + \text{stormflow}$ y baseflow varía lentamente, se cumple que $\Delta\text{flow}(t) \approx \Delta\text{stormflow}(t)$. Estas features transportan de facto la derivada del target en $t$, reintroduciendo el atajo que se cerró en iter11 al quitar `flow_total_mgd`. Si al eliminarlas el NSE cae al nivel del naive (~0.811), el modelo está haciendo AR(1) disfrazado en lugar de aprender lluvia $\to$ stormflow.

### Cambio aplicado

Único cambio respecto a v1: eliminación de `delta_flow_5m` y `delta_flow_15m` de `FEATURE_COLUMNS`. Arquitectura (TwoStageTCN), loss (TwoStageLoss), hiperparámetros, split y normalización idénticos a v1.

### Resultado

Entrenamiento completado en 14 épocas (early stopping), mejor época = 4.

| Métrica | v1 (22 feat) | iter16 (20 feat) | Delta |
|---|---:|---:|---:|
| NSE | +0.861 | **-0.169** | -1.030 |
| RMSE | 0.89 MGD | 2.60 MGD | +1.71 |
| MAE | 0.29 MGD | 0.63 MGD | +0.34 |
| Error pico | -21.0% | **+358.6%** | cambio de signo |
| Bias base | +0.21 MGD | +0.48 MGD | +0.27 |
| Bias extremo | -12.7 MGD | **+39.2 MGD** | cambio de signo |
| Pico predicho máx | ~106 MGD | **625 MGD** | extrapolación x3 fuera del rango del train (max=199 MGD) |

Meta.json verificado: `n_features=20`, `delta_flow_*` confirmadas fuera de la lista. No hay bug.

### Interpretación

La hipótesis de Opus queda confirmada con mayor severidad de la esperada. No se trata de que la ganancia de +0.050 NSE sobre naive viniera de `delta_flow_*`. Se trata de que **toda la estabilidad numérica del modelo** dependía de esas dos features. Sin ellas:

- El regresor no aprende lluvia $\to$ stormflow.
- Extrapola sin cota en extremos (predice 625 MGD con máximo de train de 199 MGD).
- Sobreestima masivamente incluso el baseflow (bias base x2).
- `H1_conSF` (con `stormflow_mgd` como feature autoregresiva directa) sigue dando NSE=0.854, confirmando que toda la señal útil es autoregresiva.

### Hallazgo colateral

La evaluación local de iter16 reporta **0 eventos extremos sin lluvia en ventana de 72 pasos**, no 15 como se documentaba previamente en STATE.md §"Qué NO funciona" y en `extreme_events_no_rain.json`. Los 59/59 extremos del test tienen lluvia detectable en las 6h previas. Queda invalidado el argumento de "extremos físicamente impredecibles" que se usaba para justificar el error en bucket Extremo.

### Decisión

**Parar la secuencia de iteraciones 16-20 planificada tras la revisión de Opus.** El resultado de iter16 invalida la premisa de iter17 (regresor no-condicional para reducir bias base), que asumía que el modelo aprendía hidrología pero el regresor generaba OOD. Con NSE negativo global, no hay señal que estabilizar: no hay hidrología aprendida, es un AR(1) con features de adorno.

**Próximo paso:** auditoría profunda con Claude Code (rama `diagnostico`) antes de decidir el camino nuevo. No se ejecutarán más iteraciones sobre el planteamiento actual hasta obtener el `DIAGNOSTIC_REPORT.md`.

### Referencias

- Datos brutos: `MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_iter16_{weights.pt, norm_params.json, meta.json}`
- Evaluación completa: `outputs/data_analysis/local_eval_metrics.json`
- Prompt maestro para Claude Code: `PROMPT_CLAUDE_CODE.md`