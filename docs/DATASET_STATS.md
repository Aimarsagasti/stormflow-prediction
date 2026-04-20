# DATASET_STATS.md - Resumen estadistico del dataset

Documento generado automaticamente por `scripts/generate_dataset_stats.py`.

- **Fecha de generacion:** 2026-04-20 11:48:31
- **Config usada:** `C:\Dev\TFM\configs\local.yaml`
- **Proposito:** input para la revision externa con Opus 4.7 sobre metodologia y limite fisico del modelo.

Este documento describe *los datos*, no el modelo. Para metricas del modelo actual, ver `outputs/data_analysis/local_eval_metrics.json` y `docs/STATE.md`.

---
## 1. Resumen global del dataset

- **Registros totales:** 1,101,964 (a 5 minutos de resolucion).
- **Cobertura temporal:** 2015-08-01 a 2026-01-31 (3836 dias, 10.5 anos).
- **Columnas tras feature engineering:** 26 (timestamp + 22 features + stormflow_mgd + is_event).
- **Columnas con NaN:** ninguna.

### Split cronologico 70/15/15

| Split | Filas | % del total | Desde | Hasta | Duracion |
|-------|-------|-------------|-------|-------|----------|
| train | 771,374 | 70.0% | 2015-08-01 | 2022-12-11 | 2689 dias |
| val | 165,295 | 15.0% | 2022-12-11 | 2024-07-07 | 573 dias |
| test | 165,295 | 15.0% | 2024-07-07 | 2026-01-31 | 573 dias |

> **Nota metodologica:** el split actual deja val y test con duracion inferior a un ciclo anual completo. En datasets con estacionalidad anual fuerte (eventos de tormenta concentrados en primavera/verano, eventos de deshielo concentrados en invierno), esto puede introducir sesgo estacional en las metricas de evaluacion. Alternativas a considerar: split en anos completos (p.ej. 7/2/1), TimeSeriesSplit con ventanas deslizantes, o blocked CV por eventos con embargo temporal. Esta decision queda como pendiente (ver STATE.md).

---

## 2. Distribucion de la variable objetivo `stormflow_mgd`

### Estadisticos globales (dataset completo)

- **n:** 1,101,964
- **Media:** 0.4901 MGD
- **Desviacion tipica:** 3.0622 MGD
- **Skewness:** 19.81 (asimetria a la derecha muy marcada)
- **Kurtosis:** 606.89 (colas extremadamente pesadas)
- **% del tiempo en regimen base (<0.5 MGD):** 89.80%

### Cuantiles de `stormflow_mgd` por split

| Cuantil | Global | Train | Val | Test |
|---------|--------|-------|-----|------|
| p50 | 0.060 | 0.069 | 0.054 | 0.038 |
| p75 | 0.181 | 0.211 | 0.138 | 0.112 |
| p90 | 0.513 | 0.589 | 0.320 | 0.321 |
| p95 | 1.305 | 1.442 | 0.889 | 0.950 |
| p99 | 9.920 | 10.512 | 7.543 | 9.056 |
| p99.9 | 43.513 | 47.058 | 39.450 | 31.633 |
| max | 225.331 | 199.395 | 225.331 | 135.151 |

> El rango de magnitudes del test (135.2 MGD max) no excede el de train (199.4 MGD max). No hay extrapolacion fuera del rango visto.

![Distribucion del target](../outputs/figures/dataset_stats/section2_target_distribution.png)

---

## 3. Caracterizacion de los eventos extremos (stormflow >= 50 MGD)

- **Total de muestras extremas en el dataset:** 833

### Distribucion entre splits

| Split | n extremos | % del split | Max MGD | Media MGD |
|-------|------------|-------------|---------|-----------|
| train | 677 | 0.088% | 199.4 | 75.8 |
| val | 97 | 0.059% | 225.3 | 79.8 |
| test | 59 | 0.036% | 135.2 | 69.5 |

> **Nota sobre tamano muestral:** el test solo contiene 59 muestras extremas. Las metricas calculadas sobre este bucket (NSE, bias, MAPE) tienen alta varianza y un unico evento atipico puede desplazarlas significativamente.

### Estacionalidad de los eventos extremos

- **Extremos con `rain_sum_60m` < 0.01 pulgadas:** 44 (5.3%). Estos son candidatos a eventos sin origen pluvial inmediato (posible escorrentia retardada, deshielo, o errores de sensor).

![Eventos extremos](../outputs/figures/dataset_stats/section3_extreme_events.png)

---

## 4. Correlaciones feature-target por regimen

Se calculan correlaciones de Pearson (lineal) y Spearman (monotonica) entre cada una de las 22 features y el target `stormflow_mgd`, separando en regimen baseflow (<0.5 MGD, n=683,417) y regimen evento (>=0.5 MGD, n=87,957). **Calculado solo sobre el split de train** para evitar data snooping sobre val/test.

**Interpretacion:** Pearson alto indica relacion lineal fuerte. Spearman alto indica relacion monotonica (puede ser no lineal). Diferencias grandes entre baseflow y evento indican que la feature se comporta de forma distinta en cada regimen.

| Feature | Pearson global | Spearman global | Spearman baseflow | Spearman evento |
|---------|---------------:|----------------:|-------------------:|----------------:|
| `api_dynamic` | 0.730 | 0.370 | 0.160 | 0.706 |
| `rain_sum_120m` | 0.691 | 0.379 | 0.104 | 0.682 |
| `rain_sum_180m` | 0.642 | 0.397 | 0.127 | 0.675 |
| `rain_sum_60m` | 0.670 | 0.341 | 0.069 | 0.654 |
| `minutes_since_last_rain` | -0.265 | -0.408 | -0.188 | -0.639 |
| `rain_sum_360m` | 0.546 | 0.416 | 0.157 | 0.621 |
| `rain_sum_30m` | 0.671 | 0.305 | 0.048 | 0.618 |
| `rain_max_60m` | 0.581 | 0.340 | 0.069 | 0.611 |
| `rain_max_30m` | 0.599 | 0.304 | 0.048 | 0.599 |
| `rain_sum_15m` | 0.630 | 0.275 | 0.038 | 0.570 |
| `rain_sum_10m` | 0.585 | 0.262 | 0.034 | 0.546 |
| `rain_max_10m` | 0.566 | 0.262 | 0.034 | 0.544 |
| `rain_in` | 0.517 | 0.247 | 0.029 | 0.519 |
| `delta_flow_5m` | 0.213 | 0.103 | 0.171 | -0.120 |
| `delta_rain_10m` | -0.100 | -0.027 | 0.003 | -0.117 |
| `temp_daily_f` | 0.026 | -0.063 | -0.054 | 0.106 |
| `delta_rain_30m` | 0.066 | -0.037 | 0.005 | -0.099 |
| `delta_flow_15m` | 0.376 | 0.092 | 0.173 | -0.078 |
| `month_sin` | 0.019 | -0.001 | -0.060 | -0.073 |
| `hour_cos` | -0.003 | -0.041 | -0.043 | 0.033 |
| `month_cos` | 0.005 | 0.118 | 0.089 | -0.026 |
| `hour_sin` | -0.010 | -0.002 | 0.008 | -0.008 |

*Ordenado descendentemente por `|Spearman evento|` porque es el regimen operativamente relevante.*

---

## 5. Matriz de correlaciones feature-feature

Correlacion de Pearson entre todos los pares de las 22 features, calculada sobre el split de train. Detecta redundancias: features con correlacion alta aportan informacion casi identica al modelo.

### Pares con |Pearson| >= 0.7 (36 pares)

| Feature A | Feature B | Pearson |
|-----------|-----------|--------:|
| `rain_sum_10m` | `rain_max_10m` | +0.985 |
| `rain_sum_10m` | `rain_sum_15m` | +0.970 |
| `rain_sum_15m` | `rain_max_10m` | +0.954 |
| `rain_in` | `rain_sum_10m` | +0.952 |
| `rain_in` | `rain_max_10m` | +0.942 |
| `api_dynamic` | `rain_sum_60m` | +0.940 |
| `rain_sum_30m` | `rain_max_30m` | +0.934 |
| `rain_sum_120m` | `rain_sum_180m` | +0.911 |
| `api_dynamic` | `rain_sum_30m` | +0.906 |
| `rain_sum_60m` | `rain_max_60m` | +0.903 |
| `rain_sum_15m` | `rain_sum_30m` | +0.895 |
| `rain_in` | `rain_sum_15m` | +0.885 |
| `temp_daily_f` | `month_cos` | -0.870 |
| `api_dynamic` | `rain_sum_120m` | +0.864 |
| `rain_sum_30m` | `rain_sum_60m` | +0.861 |
| `rain_sum_15m` | `rain_max_30m` | +0.844 |
| `api_dynamic` | `rain_max_60m` | +0.844 |
| `rain_sum_60m` | `rain_sum_120m` | +0.842 |
| `api_dynamic` | `rain_max_30m` | +0.841 |
| `rain_max_30m` | `rain_max_60m` | +0.840 |
| `rain_sum_10m` | `rain_sum_30m` | +0.821 |
| `api_dynamic` | `rain_sum_15m` | +0.813 |
| `rain_sum_180m` | `rain_sum_360m` | +0.812 |
| `rain_max_10m` | `rain_max_30m` | +0.812 |
| `rain_sum_30m` | `rain_max_10m` | +0.806 |
| `rain_sum_60m` | `rain_max_30m` | +0.803 |
| `rain_sum_10m` | `rain_max_30m` | +0.796 |
| `rain_sum_30m` | `rain_max_60m` | +0.787 |
| `api_dynamic` | `rain_sum_180m` | +0.776 |
| `api_dynamic` | `rain_sum_10m` | +0.759 |
| `rain_sum_120m` | `rain_max_60m` | +0.747 |
| `api_dynamic` | `rain_max_10m` | +0.743 |
| `rain_in` | `rain_max_30m` | +0.737 |
| `rain_in` | `rain_sum_30m` | +0.731 |
| `rain_sum_60m` | `rain_sum_180m` | +0.721 |
| `rain_sum_15m` | `rain_sum_60m` | +0.715 |

> **Interpretacion:** pares con |Pearson| muy alto (>0.9) son candidatos a eliminar una de las dos features. Pares en el rango 0.7-0.9 pueden mantenerse si aportan senal ligeramente distinta, pero hay que evaluarlo con permutation importance en un modelo entrenado.

![Matriz de correlaciones](../outputs/figures/dataset_stats/section5_feature_correlation_matrix.png)

---

## 6. Autocorrelacion del target (stormflow_mgd)

La funcion de autocorrelacion (ACF) mide cuanto se parece la serie a si misma desplazada en el tiempo. Es relevante para este proyecto porque:

- Si ACF(lag=1) es muy cercana a 1, predecir el siguiente paso es casi trivial copiando el valor actual. Explica el NSE alto a H=1.

- La velocidad con la que la ACF decae hacia cero indica cuanto horizonte predictivo util tiene la serie. Si la ACF cae rapido, predecir a H=6 es intrinsecamente dificil independientemente del modelo.

### ACF en lags de interes (train-only)

| Lag (pasos) | Lag (minutos) | ACF |
|-------------|---------------|----:|
| 1 | 5 | 0.9087 |
| 3 | 15 | 0.7157 |
| 6 | 30 | 0.5559 |
| 12 | 60 | 0.4207 |
| 24 | 120 | 0.2751 |
| 36 | 180 | 0.2261 |
| 72 | 360 | 0.1404 |
| 144 | 720 | 0.0937 |
| 288 | 1440 | 0.0431 |

> **Lectura:** ACF a 5 minutos = 0.909, ACF a 30 minutos = 0.556. Esto da una referencia cuantitativa del limite teorico de un predictor naive (persistencia): su NSE a H=1 estaria proximo a 0.826 y a H=6 proximo a 0.309, lo que contextualiza los NSE del modelo actual (0.861 a H=1, -1.21 a H=6) en la tabla del STATE.md.

![ACF del target](../outputs/figures/dataset_stats/section6_target_acf.png)

---

## 7. Caracterizacion de la lluvia

Analisis del comportamiento de `rain_in` (lluvia en pulgadas por paso de 5 min) sobre el split de train. Contextualiza por que las features `rain_sum_*` estan comprimidas cerca de cero durante la mayor parte del tiempo.

### Fraccion de pasos con lluvia

- **Total de pasos:** 771,374
- **Pasos sin lluvia (rain_in == 0):** 740,754 (96.03%)
- **Pasos con lluvia (rain_in > 0):** 30,620 (3.97%)

### Estadisticos de la lluvia no-cero

- **Media:** 0.01163 pulgadas / 5 min
- **Mediana:** 0.00600 pulgadas / 5 min
- **Percentil 95:** 0.03820 pulgadas / 5 min
- **Percentil 99:** 0.10210 pulgadas / 5 min
- **Maximo:** 0.38600 pulgadas / 5 min

### Rachas de lluvia (secuencias contiguas de rain_in > 0)

- **Numero de rachas identificadas:** 4,514
- **Duracion media:** 33.9 minutos
- **Duracion mediana:** 10.0 minutos
- **Duracion maxima:** 11.1 horas

![Distribucion de lluvia](../outputs/figures/dataset_stats/section7_rain_distribution.png)

---

## 8. Comparacion con predictor naive (persistencia)

Se compara el modelo actual con un predictor trivial que copia el valor actual como prediccion del paso siguiente. Formalmente: $\hat{y}(t+h) = y(t)$. Este es el baseline absoluto: cualquier modelo util debe batirlo claramente.

La comparacion es especialmente relevante porque la seccion 6 muestra que la ACF del target es alta a corto plazo (0.91 a 5 min), lo que implica que la persistencia ya tiene un rendimiento no trivial.

### NSE del modelo vs NSE del naive (split test)

| Horizonte | Min | NSE naive | NSE modelo SIN SF | Ganancia SIN SF | NSE modelo CON SF | Ganancia CON SF |
|-----------|-----|----------:|------------------:|----------------:|------------------:|----------------:|
| H=1 | 5m | 0.811 | 0.861 | +0.050 | 0.854 | +0.043 |
| H=3 | 15m | 0.409 | 0.471 | +0.062 | 0.488 | +0.079 |
| H=6 | 30m | 0.081 | -1.212 | -1.293 | 0.255 | +0.174 |

### Lectura

- **H=1 (5 min):** el modelo SIN SF bate al naive por +0.050 NSE. Mejora moderada.
- **H=3 (15 min):** el modelo SIN SF bate al naive por +0.062 NSE.
- **H=6 (30 min):** el modelo SIN SF es CATASTROFICAMENTE peor que el naive (1.29 NSE de diferencia).

> **Implicacion operativa y academica:** reportar NSE=0.861 a H=1 sin comparar con el naive da una impresion erronea del valor aportado por el modelo. Una defensa robusta del TFM debe incluir esta comparacion y justificar por que un modelo con ~104K parametros mejora (o no) sobre un predictor de una linea.

---

## 9. Sintesis de hallazgos para la revision externa

Esta seccion resume los hallazgos mas relevantes del documento, pensada para servir de punto de partida a la revision con Opus 4.7. No introduce analisis nuevo: explicita lo que las secciones 1-8 ya exponen.

### Hallazgos ordenados por severidad

**1. [ALTA] El modelo apenas bate al predictor naive a H=1 y es PEOR a H=3 y H=6.**

Ganancia del modelo SIN SF sobre persistencia: H=1 +0.050, H=3 +0.062, H=6 -1.293.
Un modelo de ~104K parametros debe justificar por que mejora (o no) a un predictor de una linea.
Pregunta para el revisor: ¿el NSE=0.861 a H=1 es realmente un buen resultado teniendo en cuenta que naive da 0.811?

**2. [ALTA] Asimetria entre splits en cobertura de eventos extremos.**

Valor maximo por split: train=199.4 MGD, **val=225.3 MGD** (maximo absoluto del dataset), test=135.2 MGD.
Extremos por split: train=677, val=97, test=59.
Consecuencia: el modelo se evalua en test sobre un regimen menos extremo que el de entrenamiento, mientras que val incluye el pico absoluto. Early stopping sobre val puede estar favoreciendo infraestimacion.

**3. [MEDIA] Tamano muestral insuficiente para el bucket Extremo en test.**

Solo 59 muestras extremas en test. Las metricas sobre este bucket (NSE=-0.99, bias=-12.7 MGD) tienen alta varianza. Un unico evento atipico puede desplazarlas significativamente.
Pregunta para el revisor: ¿tiene sentido reportar metricas sobre este bucket con esta muestra, o conviene usar bootstrap o k-fold para estimar intervalos de confianza?

**4. [MEDIA] Redundancia masiva entre features.**

36 pares de features con |Pearson| >= 0.7. Las mas extremas: rain_sum_10m vs rain_max_10m (0.985), rain_sum_10m vs rain_sum_15m (0.970), api_dynamic correlaciona con 9 features distintas.
De 22 features declaradas, el numero efectivo de dimensiones independientes es mucho menor (posiblemente ~8-10).
Pregunta para el revisor: ¿conviene reducir features antes de seguir iterando en arquitectura/loss?

**5. [MEDIA] ACF del target impone limite teorico a H>1.**

ACF(5 min) = 0.909, ACF(30 min) = 0.556. La ACF decae rapido tras los primeros 30 minutos.
Esto indica que, sin incorporar prediccion de lluvia externa (opcion c del STATE.md), extender el horizonte util mas alla de ~15 min puede ser fisicamente limitado.

### Prioridades sugeridas (a validar con el revisor)

1. **Reportar baseline naive en todas las metricas futuras.** No-regrets, 5 lineas de codigo.
2. **Revisar split.** Evaluar splits alternativos (anos completos, TimeSeriesSplit, blocked CV por eventos).
3. **Simplificar features.** Reducir de 22 a ~10 eliminando redundancia, con permutation importance sobre el modelo simplificado.
4. **Intervalos de confianza.** Bootstrap sobre test para cuantificar la incertidumbre de las metricas en bucket Extremo.
5. **Horizonte realista.** Aceptar que H>1 con features actuales tiene limite fisico, o integrar el modelo de lluvia del MSD (opcion c).
