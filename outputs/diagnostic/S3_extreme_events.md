# S3 - Analisis de los 59 eventos extremos del test

Diseccion uno a uno de los 59 picos con `stormflow_mgd >= 50 MGD` en el test set (ventana cronologica 2024-07-07 -> 2026-01-31, `iloc[936669:]`). Objetivo: identificar patrones, resolver la contradiccion documental sobre extremos sin lluvia, y cuantificar que subgrupo es predecible.

## 1. Inventario

- **Muestras extremas** (stormflow>=50 MGD en test): **59**.
- **Eventos fisicos** (muestras agrupadas por gap <= 48 pasos = 4h): **24**.

Criterio de agrupacion: dos muestras consecutivas pertenecen al mismo evento si estan separadas por <=48 pasos de 5 min (4 horas). Con este criterio, las 59 muestras se agrupan en tormentas fisicas independientes (ver tabla).

| event_id | ts_pico | n_samples | peak_mgd |
|---:|---|---:|---:|
| 0 | 2024-07-10 00:15:00 | 1 | 57.64 |
| 1 | 2024-07-28 13:10:00 | 9 | 135.15 |
| 2 | 2024-08-16 17:10:00 | 2 | 53.30 |
| 3 | 2024-09-27 16:40:00 | 5 | 71.32 |
| 4 | 2025-04-02 23:55:00 | 2 | 87.84 |
| 5 | 2025-04-03 22:05:00 | 1 | 51.27 |
| 6 | 2025-04-14 15:00:00 | 1 | 72.62 |
| 7 | 2025-04-24 14:20:00 | 2 | 59.13 |
| 8 | 2025-04-25 16:35:00 | 1 | 55.55 |
| 9 | 2025-04-29 18:20:00 | 3 | 70.29 |
| 10 | 2025-05-01 12:55:00 | 2 | 54.45 |
| 11 | 2025-06-17 02:00:00 | 2 | 97.90 |
| 12 | 2025-06-18 18:35:00 | 2 | 61.45 |
| 13 | 2025-07-15 14:20:00 | 1 | 60.86 |
| 14 | 2025-07-17 20:10:00 | 3 | 81.27 |
| 15 | 2025-07-19 15:20:00 | 1 | 72.12 |
| 16 | 2025-07-20 14:35:00 | 5 | 104.80 |
| 17 | 2025-07-27 17:05:00 | 1 | 52.37 |
| 18 | 2025-07-28 10:20:00 | 3 | 91.35 |
| 19 | 2025-07-31 08:45:00 | 2 | 73.66 |
| 20 | 2025-08-12 19:35:00 | 2 | 79.86 |
| 21 | 2025-08-19 16:20:00 | 4 | 104.86 |
| 22 | 2025-11-18 11:20:00 | 2 | 61.70 |
| 23 | 2025-11-18 18:20:00 | 2 | 67.45 |

## 2. Resolucion de la contradiccion "sin lluvia"

La documentacion previa afirmaba que 15 de 59 extremos no tenian lluvia en la ventana de entrada. El eval de iter16 reportaba 0. Aplico tres criterios sobre el test actual (`iloc[936669:]`):

| Criterio | Definicion | Cuenta / 59 |
|---|---|---:|
| A | `rain_sum_360m(t) < 0.01 in` (sin lluvia detectable en 6h) | 0 |
| B | `rain_sum_60m(t) < 0.01 in` (sin lluvia en 1h) | 0 |
| C | `sum(rain_in)` en ventana 72 pasos `< 0.01 in` | 0 |

**Veredicto**: los **59 extremos tienen lluvia detectable** en la ventana de 6h y de 72 pasos previos al pico. La afirmacion vieja de "15 sin lluvia" NO aplica al test actual. La cifra operativa de iter16 (0/59 sin lluvia) es la correcta. La documentacion previa debe actualizarse.

Hipotesis para la discrepancia historica: la cifra 15/59 probablemente corresponde a un split o test set anterior (por ejemplo cuando el test era mas corto o incluia muestras con rain_in=0 en la exacta muestra t pero lluvia en la ventana). Ya no se reproduce.

## 3. Patrones temporales (resumen estadistico)

Estadisticas agregadas de las 59 muestras extremas (no eventos fisicos):

| feature | mean | std | min | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| lag_pico_lluvia_min | 19.32 | 35.10 | 5.00 | 10.00 | 28.00 | 250.00 |
| rain_intensity_max_in | 0.15 | 0.07 | 0.05 | 0.14 | 0.28 | 0.29 |
| rain_total_window_in | 0.60 | 0.42 | 0.09 | 0.44 | 1.34 | 1.62 |
| rain_duration_steps | 14.42 | 16.29 | 3.00 | 9.00 | 47.20 | 62.00 |
| api_pico | 0.35 | 0.18 | 0.06 | 0.30 | 0.58 | 0.84 |
| temp_daily_pico_f | 71.87 | 9.87 | 41.60 | 76.00 | 80.82 | 81.50 |
| time_since_last_rain_min | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| month | 6.93 | 1.83 | 4.00 | 7.00 | 9.00 | 11.00 |

**Distribucion mensual** (mes:n_muestras): {4: 10, 5: 2, 6: 4, 7: 26, 8: 8, 9: 5, 11: 4}. Picos concentrados en meses de tormentas primavera-verano (julio = 26/59).

## 4. Clustering (K-Means, k=3)

Features estandarizadas: ['lag_pico_lluvia_min', 'rain_intensity_max_in', 'rain_total_window_in', 'rain_duration_steps', 'api_pico', 'time_since_last_rain_min', 'temp_daily_pico_f']. K=3 justificado por hipotesis hidrologica (convectivo vs estratiforme vs atipico). Inercia = 162.15.

| cluster | etiqueta | n | lag_min | rain_max_in | rain_total_in | duracion | api | tslr_min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Convectivo (intenso, corto, lag pequeno) | 12 | 10.8 | 0.251 | 0.924 | 10.7 | 0.6322 | 0 |
| 1 | Mixto | 38 | 11.1 | 0.129 | 0.345 | 7.6 | 0.2601 | 0 |
| 2 | Estratiforme (total alto, duracion larga) | 9 | 65.6 | 0.103 | 1.221 | 48.3 | 0.3588 | 0 |

## 5. Predicciones por cluster (v1 vs naive)

`y_pred_v1` cargado desde `modelo_H1_sinSF_weights.pt` con switch duro (threshold=0.3). `y_pred_naive = stormflow(t-1)`. Error % = `(pred-real)/real*100`.

| cluster | label | n | NSE v1 | RMSE v1 | ErrPico% v1 (med) | under50% v1 | NSE naive |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Convectivo (intenso, corto, lag pequeno) | 12 | -0.269 | 30.99 | -3.4 | 0 | -0.842 |
| 1 | Mixto | 38 | -3.048 | 29.60 | -32.8 | 11 | -7.455 |
| 2 | Estratiforme (total alto, duracion larga) | 9 | -2.651 | 11.63 | -11.8 | 0 | -7.500 |

### Metricas globales sobre los 59 extremos

- NSE v1 (59 puntos aislados): **-0.990**
- RMSE v1: **27.93 MGD**
- MAE v1: **21.83 MGD**
- Error pico % medio v1: **-18.0%**
- NSE naive (referencia): **-2.853**
- RMSE naive: **38.86 MGD**

Nota: NSE calculado sobre solo los 59 puntos aislados del bucket Extremo no es directamente comparable con el NSE global del bucket en `local_eval_metrics.json`, porque aquel usa todas las muestras del bucket como conjunto.

## 6. Cota optimista: oraculo en cluster predecible

- Cluster mas predecible por v1: **#0** (Convectivo (intenso, corto, lag pequeno), n=12).
- NSE actual (v1) sobre los 59 extremos: **-0.990**.
- NSE si v1 fuera perfecto en ese cluster y mantuviera su error actual en el resto: **-0.492** (delta = +0.498).

Lectura: esta cota muestra cuanta mejora maxima se puede esperar si resolvemos SOLO el cluster mas predecible. Para mejorar mas alla habria que trabajar tambien los clusters atipicos (que por hipotesis son fisicamente menos predecibles con las features actuales).

## 7. Veredicto

### Respuestas a las preguntas clave

**P1. Subconjunto predecible con features actuales** (alto rain_total_window, lag corto, alta API):
- Si: el **cluster #0** (Convectivo (intenso, corto, lag pequeno), n=12/59) es el mas predecible: NSE_v1=-0.269, error pico mediano -3.4%, con 0 infraestimaciones >50%. Cumple el patron esperado: lluvia intensa, lag corto, API alta. Aproximadamente el 20% de las muestras extremas.

**P2. Extremos estructuralmente impredecibles** (sin lluvia, lag enorme, fuera de patron):
- 0/59 estan "sin lluvia" con cualquier criterio. **No hay extremos fisicamente ciegos** en el test actual.
- Sin embargo, el cluster mas problematico es el **#1** (Mixto, n=38): NSE_v1=-3.048, error pico mediano -32.8%, y 11 muestras con infraestimacion >50%. Aqui esta concentrada la mayor parte del fallo.
- El cluster Estratiforme (lluvia larga, lag mayor) tiene RMSE bajo pero tampoco lo predice bien: el v1 acierta orden de magnitud pero subestima.

**P3. v1 vs cluster:**
- v1 acierta mas en el cluster Convectivo (alta API + lluvia intensa).
- v1 falla sistematicamente en el cluster Mixto/baja-API: lluvia moderada sobre suelo poco saturado da picos altos que el modelo no anticipa.
- En todos los clusters el v1 mejora al naive (NSE_v1 > NSE_naive), pero ningun cluster supera NSE_v1=0 sobre los 59 puntos aislados (esto es esperable: 59 puntos con varianza enorme penalizan mucho el denominador).

**P4. Cota optimista (oraculo en cluster predecible):**
- NSE actual sobre los 59 = **-0.990**.
- Con oraculo perfecto en el cluster #0 (n=12): NSE = **-0.492** (delta = +0.498).
- Conclusion: resolver SOLO el cluster predecible aporta una ganancia limitada porque ese cluster ya es el mejor predicho. El margen real esta en el cluster Mixto (n=38, ~64% de los extremos), que requiere features o arquitectura nuevas para mejorar.

### Hallazgos transversales

- **Sin lluvia = 0/59** con cualquier criterio razonable. La cifra vieja "15/59" NO es valida para el test actual; actualizar documentacion (`AGENTS.md`, `CLAUDE.md`, `docs/STATE.md`).

- Los 59 picos forman **24 tormentas fisicas** distintas. Varias tormentas contribuyen con multiples muestras consecutivas al bucket Extremo: la sobre-representacion del bucket Extremo en metricas no indica diversidad de eventos sino picos prolongados.

- Los extremos tienen senal de lluvia consistente: el problema de infraestimacion **no viene de ausencia de input**, sino de la capacidad del regresor para calibrar magnitud en la cola. Coherente con S1/S2: el TCN no extrae mas informacion temporal que XGBoost y el atajo `delta_flow_*` domina la varianza baja-media pero no ayuda en la cola alta.
