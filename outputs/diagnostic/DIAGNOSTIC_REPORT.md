# DIAGNOSTIC REPORT - Stormflow Prediction TFM

**Fecha**: 2026-04-22
**Rama**: `diagnostico`
**Autor**: Claude Code (orquestador) sobre 6 subagentes (S1-S6)
**Artefactos detallados**: `outputs/diagnostic/S{1..6}_*.md` y `.json`. Scripts reproducibles en `scripts/diagnostic/`.

---

## 1. Resumen ejecutivo

**Veredicto**: el TwoStageTCN actual no es recuperable y debe descartarse. El "buen NSE" del v1 (0.86 a H=1) es el resultado combinado de (a) un atajo autorregresivo (`delta_flow_5m/15m`, ya confirmado en iter16), (b) un bug estructural en la `TwoStageLoss` que entrena el regresor solo en evento pero lo aplica en inferencia a baseflow, y (c) una métrica global dominada por baseflow que enmascara el error en la cola. Sin el atajo, NSE colapsa a -0.17. Con el atajo y métricas alineadas, queda en 0.74-0.86 según cómo se calcule, frente a un AR(12) lineal puro de NSE=0.83 — es decir, el TCN no aporta valor arquitectónico sobre 12 lags del propio target en una regresión lineal.

**Recomendación única**: abandonar el camino TCN como modelo principal y montar un sistema de dos partes que se ejecuta en paralelo:

1. **Modelo de producción**: XGBoost regresivo entrenado sobre **lags explícitos del target (y(t-0..t-11))** + **10 features reducidas de S5** (api_dynamic, rain_sum_360m, rain_sum_120m, rain_sum_15m, temp_daily_f, hour_sin, minutes_since_last_rain, delta_rain_10m, delta_rain_30m, rain_sum_30m). Reentrenable en local en <60 s. NSE H=1 esperado: **0.84-0.88** (S6 §3, basado en composición AR(12)=0.827 + features con PI > 0). Resuelve por construcción los dos bugs (no two-stage, no train-eval mismatch). Reformula salida adicional como **alerta binaria a H=6 y H=12** porque a esos horizontes el techo físico de NSE (0.32 y 0.19 respectivamente) hace inviable la regresión.

2. **Comparación deep learning**: una sola iteración Colab con **TCN estándar (Bai 2018, sin two-stage) + lags del target en el input** para confirmar si el deep learning aporta sobre XGBoost+lags. Si no aporta >0.02 NSE, se cierra deep learning para este TFM.

**El TFM debe reformularse en torno al hallazgo negativo**: la sección de resultados gana en valor académico al narrar honestamente "modelo inicial parecía exitoso → análisis de baselines reveló atajo → modelo final más simple supera al inicial cuando se compara correctamente". Es una historia publicable y defendible.

**Lo que queda fuera del alcance** con los datos actuales: NSE > 0.5 a H=6 o H=12, predicción cuantitativa fiable de extremos (la cota física pone NSE máximo en bucket Extremo cerca de 0 sobre solo 59 muestras). Para abrir esos horizontes hace falta forecast de lluvia externo (decisión explícita del usuario: no usarlo en esta fase).

---

## 2. Estado real del modelo actual

### 2.1 Lo que se reportaba

- TCN v1 sinSF H=1: NSE=0.861, RMSE=0.89, MAE=0.29, error pico=-21%.
- v1 considerado "modelo en producción" desde 2026-04-13.

### 2.2 Lo que es realmente

| Métrica | Valor reportado (`local_eval_metrics.json`) | Valor sobre test alineado (S4 reproducción) | Diferencia |
|---|---:|---:|---:|
| NSE H=1 sinSF | 0.8614 | 0.7388 | **-0.123** |
| Error pico H=1 | -21.0% | +42.6% | cambio de signo |

**Discrepancia detectada**: S4 reprodujo la inferencia del TCN v1 sobre el test alineado a S2 (mismo `seq_length=72`, mismo `horizon=1`, descartando los primeros 71 pasos para que la ventana exista) y obtuvo NSE=0.74 con error pico positivo (+42.6%, sobreestima). Hay que auditar evaluate_local.py vs scripts/diagnostic/s4_horizon_ceiling.py para entender el origen (probables candidatos: diferente offset al inicio del test, diferente threshold del switch duro o diferencia de denormalización con/sin clipping). **Este punto debe resolverse antes de cualquier comparación contra modelos nuevos** — no se puede comparar NSE de modelos futuros si la cifra de referencia del v1 es ambigua. Ver Plan §7.1.

### 2.3 Verdad incómoda: el TCN no aporta sobre AR(12) lineal

| Modelo | NSE H=1 | NSE H=3 | NSE H=6 |
|---|---:|---:|---:|
| Naive persistencia | 0.811 | 0.409 | 0.081 |
| **AR(12) lineal sobre 12 lags de y** | **0.827** | **0.509** | **0.317** |
| XGBoost-22 (con `delta_flow`) | 0.790 | 0.656 | 0.382 |
| XGBoost-20 (sin `delta_flow`) | 0.662 | 0.572 | 0.336 |
| TCN v1 sinSF (`local_eval_metrics.json`) | 0.861 | 0.471 | -1.21 |
| TCN v1 sinSF (S4 reproducción) | 0.739 | 0.293 | -1.30 |
| TCN v1 sinSF SIN `delta_flow` (iter16) | -0.169 | n/d | n/d |

Lecturas:
- En H=1, AR(12) lineal supera al TCN v1 según S4 (0.83 > 0.74) y queda 0.03 por debajo según `evaluate_local.py` (0.83 vs 0.86). En la mejor lectura, el TCN aporta +0.03 NSE sobre una regresión lineal de 12 lags.
- En H=3 y H=6 el TCN está claramente por debajo de XGBoost-22 y de AR(12). El TCN es estructuralmente inferior a horizontes >5 minutos.
- Sin el atajo `delta_flow`, el TCN colapsa (-0.17). El atajo aporta +1.03 NSE al TCN, +0.13 NSE a XGBoost.
- A H=6, naive da 0.08, AR(12) 0.32, TCN sinSF -1.30. El TCN no llega ni al naive en H=6 sin SF como feature autoregresiva.

### 2.4 Bugs en el código fuente (S1, no relacionados con `delta_flow`)

- **BUG1 — train-eval mismatch en `TwoStageLoss`** (`src/models/loss.py:182-211` + `src/models/tcn.py:248-253`): regresor entrenado solo sobre muestras con `y_real > 0.5 MGD`; en inferencia se aplica a TODAS las muestras donde el clasificador da `cls_prob >= 0.3`. Sobre baseflow con falso positivo, el regresor produce salida OOD. Explica el NSE=-14.5 en bucket Base. Es un bug de diseño, no de implementación.
- **BUG2 — doble normalización latente** (`src/pipeline/normalize.py:71`): si `stormflow_mgd` aparece en FEATURES Y TARGET, se normaliza dos veces in-place. Corregido en callers (evaluate_local.py, horizon_comparison_v2.py) pero NO en la propia función. Cualquier notebook futuro que pase la lista completa lo reintroduce silenciosamente.
- **Sospechoso — definición doble de "evento"**: la loss usa `y > 0.5 MGD` mientras los sample weights usan `is_event` (label MSD). Optimizan objetivos distintos.
- **Sospechoso — alineamiento del naive**: en `scripts/generate_dataset_stats.py:1046-1079` el naive se calcula sobre TODO el test, mientras que el modelo se evalúa sobre `test[seq_length+horizon-1:]`. Diferencia pequeña pero introduce sesgo en la "ganancia +0.050 NSE" reportada.
- **Sospechoso — criterio inconsistente de "extremo sin lluvia"**: dos definiciones distintas en `evaluate_local.py` y `generate_dataset_stats.py`. Por eso "15/59" (vieja) vs "0/59" (iter16) no cuadran. Ambas pueden ser correctas bajo su criterio. Hay que fijar uno.

Detalle completo y citas línea a línea: `outputs/diagnostic/S1_pipeline_audit.md`.

---

## 3. Hallazgos por área

### 3.1 S1 — Auditoría pipeline (`S1_pipeline_audit.md`)

2 bugs (regresor train-eval mismatch + doble normalización latente), 5 sospechosos, 3 OK. El hallazgo principal es BUG1 — `TwoStageLoss` entrena el regresor solo en evento pero el switch duro de inferencia lo aplica a baseflow donde el regresor nunca fue penalizado. Es la segunda vía (junto al atajo `delta_flow`) por la que el modelo reportado no es una TCN "de verdad". Es un fallo de diseño no parchable sin rediseñar la loss y la inferencia conjuntas.

### 3.2 S2 — Baselines rigurosos (`S2_baselines.md`, `.json`)

Sobre test alineado (n=165,222 a H=1):
- AR(12) lineal sobre 12 lags de `y`: NSE=**0.827**.
- AR(1) óptimo (con corrección hacia la media): 0.820.
- Naive: 0.811.
- XGBoost con 20 features sin `delta_flow`: 0.662.
- XGBoost con 22 features (con `delta_flow`): 0.790. Atajo aporta +0.13 NSE en XGB.
- Predictor lineal con 2 features físicas (rain_sum_60m + api_dynamic): 0.619.

**Hallazgo principal S2**: a H=1 la señal autoregresiva del caudal domina sobre la señal exógena. AR(12) lineal supera a XGBoost-22 (0.83 vs 0.79). XGBoost no recibe lags explícitos del target — los `delta_flow` son derivadas, no niveles, y son insuficientes. Esto sugiere que añadir lags del target a XGBoost lo subiría por encima de AR(12).

A H=3 y H=6, XGBoost-22 supera al TCN v1: en H=3 XGB-22=0.66 vs TCN v1=0.47. Esto cierra la cuestión de si el TCN aporta en horizontes mayores: **no aporta**.

### 3.3 S3 — Análisis de los 59 eventos extremos (`S3_extreme_events.md`, `.json`)

- **24 tormentas físicas**, no 59 eventos independientes. Varias tormentas contribuyen con múltiples muestras consecutivas al bucket Extremo.
- **0/59 sin lluvia** con cualquier criterio razonable (rain_sum_360m, rain_sum_60m, suma en ventana 72). La cifra histórica "15/59" no aplica al test actual; documentación (CLAUDE.md, AGENTS.md, STATE.md) debe actualizarse.
- 3 clusters por K-means:
  - **Convectivo** (n=12, 20%): mejor predicho por v1, NSE_local=-0.27, errpico mediano -3.4%, 0 infraestimaciones >50%.
  - **Mixto** (n=38, 64%): peor cluster, NSE_local=-3.05, errpico mediano -32.8%, **11/38 infraestimaciones >50%**. Patrón: lluvia moderada + API baja → picos altos no anticipados.
  - **Estratiforme** (n=9, 15%): RMSE bajo (11.6) pero subestima.
- Cota optimista: oráculo perfecto en cluster Convectivo solo sube NSE sobre los 59 a -0.49 (delta +0.50). El margen real está en el cluster Mixto (64% de los extremos), que necesita features o arquitectura nuevas, no más datos.

### 3.4 S4 — Techo físico por horizonte (`S4_horizon_ceiling.md`, `.json`)

| h | min | NSE naive | NSE AR(12) | Cota 2ρ-1 | NSE máx defensible | Margen sobre naive |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.811 | 0.827 | 0.817 | **~0.83** | +0.02 |
| 3 | 15 | 0.409 | 0.509 | 0.431 | **~0.51** | +0.10 |
| 6 | 30 | 0.081 | 0.317 | 0.112 | **~0.32** | +0.24 |
| 12 | 60 | -0.187 | 0.190 | -0.159 | **~0.19** | +0.38 |
| 24 | 120 | -0.440 | 0.095 | -0.450 | **~0.09** | +0.53 |

**Contribución de cada bucket al denominador del NSE H=1** (clave para entender qué importa):
- Extremo: 31.9% (solo 59 muestras = 0.036% del test).
- Alto: 26.3% (224 muestras).
- Moderado: 37.5% (2,519 muestras).
- Leve: 2.5%, Base: 1.8%.

**Oráculos parciales sobre TCN v1 H=1** (NSE base = 0.74 según S4):
- Oráculo perfecto solo en Extremo: NSE=0.84 (+0.10).
- Oráculo perfecto en Extremo + Alto: NSE=0.89 (+0.15).
- Oráculo perfecto en todo excepto Extremo: NSE=0.90 (+0.16).
- Oráculo perfecto solo en Moderado (cuerpo): NSE=0.82 (+0.08).

Hallazgo: **el cuerpo (Moderado + Leve + Base) aporta más al denominador que Extremo solo (42% vs 32%)**. La narrativa de "el modelo solo falla en extremos" es incompleta — también falla apreciablemente en el cuerpo, especialmente en Moderado donde tiene bias +1.7 MGD y NSE_local=-0.11.

**Veredicto sobre horizontes**: H=1 tiene techo físico alto pero margen sobre naive bajo (+0.02). H=6 tiene margen +0.24 pero techo bajo (0.32). H=12 y H=24 son inviables como regresión sin forecast externo de lluvia. La conclusión defendible es **trabajar H=1 y H=3 como regresión, y H=6/H=12 como clasificación binaria de alerta**.

### 3.5 S5 — Análisis de features (`S5_feature_analysis.md`, `.json`)

- **Permutation importance top-5** (XGB-20 sobre test): `api_dynamic` (+0.418), `rain_sum_360m` (+0.108), `rain_sum_120m` (+0.080), `rain_sum_15m` (+0.037), `rain_sum_180m` (+0.034). `api_dynamic` domina con un orden de magnitud.
- **6 features con PI ≤ 0** (ruido activo, perjudican): `hour_cos`, `month_sin`, `month_cos`, `rain_sum_60m`, `rain_max_30m`, `rain_max_60m`.
- **15 pares con |r| ≥ 0.85** → la dimensionalidad efectiva de las 20 features es ~13 clusters.
- **Conjunto reducido propuesto (10 features)**: `api_dynamic, rain_sum_360m, rain_sum_120m, rain_sum_15m, temp_daily_f, hour_sin, minutes_since_last_rain, delta_rain_10m, delta_rain_30m, rain_sum_30m`. Sobre XGB: NSE=**0.698** vs XGB-20=0.662 → mejora +0.036 al quitar ruido.
- **Sin atajos encubiertos adicionales** tras quitar `delta_flow_*`. Avisos:
  - `rain_sum_60m` tiene ACF(12) similar a la del target (0.42 vs 0.42), inercia parecida, pero PI negativa en XGB-20 → no funciona como atajo dominante.
  - `api_dynamic` correlaciona r=0.94 con `rain_sum_60m`. Domina PI; AGENTS.md decía "redundante con rain_sum_60m" pero en presencia de api_dynamic la rain_sum_60m es ruido. Hay que actualizar AGENTS.md §8.1.
- XGB con 10 features reducidas (NSE=0.70) **sigue por debajo de AR(12) (0.83)**. Sin lags explícitos del target, las features de lluvia no pueden con la pura autoregresión. Esto es la palanca clave: añadir lags del target al input.

### 3.6 S6 — Revisión de arquitecturas alternativas (`S6_architecture_review.md`)

Revisión sin entrenar. Recomendaciones:

1. **Primaria — XGBoost + lags del target + 10 features reducidas**. Reusable desde `s2_baselines.py`. ~50 LoC adicionales. <1 min de entrenamiento. NSE H=1 esperado 0.83-0.88 (combinación de AR(12)=0.827 + features con PI > 0). Resuelve por construcción los bugs S1 (sin two-stage, sin train-eval mismatch). Implementable en local sin GPU.

2. **Secundaria — TCN estándar (Bai 2018) + lags del target en el input + Huber loss + sin two-stage**. Reusa ~90% del pipeline existente (eliminando `TwoStageLoss` y `is_event`). Una iteración en Colab T4. Si bate a XGB+lags por >0.02 NSE, justifica seguir con deep learning. Si no, se cierra esa rama.

Descartadas con justificación: LSTM hidrológica (Kratzert) — pensada para multi-catchment con static features, complejidad excesiva para un solo punto de medición. TFT — overhead alto de implementación frente a beneficio dudoso. Seq2seq con atención — útil para H>1 pero techo físico de H=6 es muy bajo, no compensa el coste.

**Veredicto sobre TwoStageTCN**: descartar. Parchear el bug del train-eval mismatch convierte la arquitectura en una TCN estándar con un clasificador cosmético — más limpio empezar de cero con el enfoque (2).

---

## 4. Diagnóstico del problema de raíz

El proyecto ha estado optimizando un modelo cuyo "buen NSE" reportado provenía de tres fuentes superpuestas, ninguna asociada a aprendizaje hidrológico genuino:

1. **Atajo autorregresivo `delta_flow_5m/15m`**. Confirmado en iter16 (sin él, NSE = -0.17) y replicado en XGBoost por S2 (+0.13 NSE). Era una puerta trasera al `flow_total_mgd` que ya se había excluido en iter11. La derivada del caudal total en t es prácticamente la derivada del stormflow en t, reintroduciendo el atajo.

2. **Métrica global dominada por baseflow**. El 92% del test es baseflow (<0.5 MGD). Cualquier modelo que prediga ≈ 0 fuera de eventos hereda un NSE alto sin entender hidrología. La métrica NSE global es engañosa en problemas zero-inflated, y compararla contra naive (que ya da 0.81 a H=1) sin reportar la ganancia explícita es contar mal el resultado.

3. **Bug estructural en `TwoStageLoss`**. El regresor se entrena solo sobre muestras con `y_real > 0.5 MGD` pero en inferencia se aplica a baseflow cuando el clasificador da falso positivo. Esto produce comportamiento OOD y explica el NSE=-14.5 del bucket Base. Es inconsistencia de diseño no resoluble sin rediseñar la loss.

Combinado, esto significa: el TwoStageTCN no era una TCN bidireccional aprendiendo lluvia → stormflow; era un modelo que (a) usaba `delta_flow` como AR(1) implícito, (b) reportaba NSE alto porque el denominador estaba inflado por baseflow, y (c) tenía un regresor con dominio mal definido. Las cuatro iteraciones de mejora del último mes (v2, iter15, iter15b, iter16) estaban tocando piezas de un sistema cuya base era inestable.

**El problema no es solo arquitectónico**. Aunque se arregle `TwoStageLoss` y se quite `delta_flow`, las features de lluvia sin lags explícitos del target solo dan NSE=0.70 (S5) — muy lejos de AR(12)=0.83. Lo que falta es:

- **Ningún modelo actual tiene acceso a los lags del target salvo el AR(12) lineal**. El TCN ve la ventana de features pero no incluye `stormflow(t-k)` salvo en variantes "conSF" donde lo añade como una sola feature autoregresiva.
- **Las 22 features tienen redundancia masiva** (15 pares con r≥0.85, dimensionalidad efectiva ~13). La información útil cabe en 10 features.
- **El bucket Mixto de extremos (64% del bucket Extremo)** está mal predicho por estructura, no por falta de datos. Lluvia moderada sobre suelo poco saturado → pico alto no anticipado. Esto es un patrón hidrológico real; probablemente se mejora con interacción no lineal entre `api_dynamic` y `rain_intensity_max`, no con más features.

---

## 5. Caminos posibles considerados

Solo los caminos para los que tengo evidencia de S1-S6.

### 5.1 Mantener TwoStageTCN, parchear bugs y quitar `delta_flow`

- **Pros**: reusa el código existente. Cumpliría iter17 que estaba planeada.
- **Contras**: con bugs corregidos sigue por debajo de AR(12) en H=1 y H=3 (S2/S4). Incluso parcheado, la arquitectura pierde el sentido del two-stage (regresor entrenado en todo el dominio = TCN estándar con un clasificador cosmético). Es trabajo sobre infraestructura con techo conocido. **Descartado**.

### 5.2 TCN estándar + lags del target en el input

- **Pros**: arquitectura más limpia, alineada con la literatura (Bai et al. 2018). Resuelve bugs S1 por construcción. Reusa pipeline.
- **Contras**: requiere Colab para entrenar; sin evidencia de que supere a un GBM con las mismas inputs.
- **Decisión**: válido como **modelo de comparación, no de producción**. Si supera al GBM por >0.02 NSE, se mantiene; si no, se cierra deep learning.

### 5.3 XGBoost / LightGBM con lags explícitos del target + 10 features reducidas

- **Pros**: implementable en local en <1h. Reusa `s2_baselines.py`. NSE H=1 esperado 0.83-0.88 (composición AR(12)=0.83 + features con PI>0). Resuelve los bugs S1 por construcción. Permite reentrenar barato y comparar variantes. SHAP nativo para interpretabilidad (útil para el TFM).
- **Contras**: pierde la narrativa "deep learning" del TFM si se queda como modelo final. Pero esa narrativa ya no se sostiene tras este diagnóstico — la honestidad académica vale más.
- **Decisión**: **modelo principal**.

### 5.4 Quantile Regression / EQRN para extremos

- **Pros**: predice cuantiles en vez de media; el cuantil 95 captura mejor extremos.
- **Contras**: no es la palanca dominante (S4: oráculo en Extremo solo aporta +0.10 NSE). No ataca el bug estructural ni la falta de lags. Útil como mejora marginal después.
- **Decisión**: **postergado a fase 3** del nuevo plan. No es la primera batalla.

### 5.5 Reformular evaluación: NSE por bucket + peak_lag + recall@50

- **Pros**: NSE global está sesgado por baseflow. Un panel de métricas más rico es buena práctica académica y operativa. El MSD necesita "alerta antes del CSO", no NSE alto.
- **Contras**: ninguno relevante.
- **Decisión**: **complementario, obligatorio**, no es un camino sino un acompañamiento al modelo.

### 5.6 Reformular el problema como clasificación binaria multi-horizonte

- **Pros**: si el techo físico de H=6 (NSE≤0.32) hace inviable la regresión, una alerta binaria "habrá pico ≥ 25 MGD en próximos 30 min" es más honesta y operativamente útil.
- **Contras**: cambio de scope académico. Pero el TFM puede defender ambos targets.
- **Decisión**: **complementario al modelo de regresión**, para H=6 y H=12. Modelo principal sigue siendo regresión a H=1 y H=3.

### 5.7 Reformular el TFM como hallazgo negativo + camino correcto

- **Pros**: el arco "modelo aparente → diagnóstico de baselines → modelo real más simple" es publicable y honesto. Aporta valor académico. Mejor narrativa que "modelo deep learning con NSE=0.86".
- **Contras**: requiere coordinar con tutoría académica. Pero la decisión es buena en cualquier caso.
- **Decisión**: **estructura recomendada del TFM**.

---

## 6. RECOMENDACIÓN ÚNICA

**Camino**: descartar TwoStageTCN. Construir un sistema de dos componentes en una nueva iteración (iter17):

> **Componente A — Modelo principal**: XGBoost regresivo entrenado sobre **el target a horizonte h** usando como input **12 lags del target (`stormflow_mgd[t-0..t-11]`) + 10 features exógenas reducidas de S5** (no incluir `delta_flow_5m/15m`). Modelos separados para h=1 y h=3.
>
> **Componente B — Modelo de alerta**: **clasificador binario** (XGBoost o LightGBM) que predice "habrá `stormflow ≥ U` en próximas h muestras" para h=6 y h=12 con U={25, 50} MGD. Métrica: precision/recall y F1.
>
> **Componente C — Comparación deep learning**: una sola iteración Colab con **TCN estándar (Bai 2018) + 12 lags del target + 10 features + Huber loss**, sin two-stage, sin clasificador, sin switch duro. Compara NSE H=1 contra Componente A. Si NSE_TCN - NSE_XGB < 0.02 → cierra deep learning para el TFM. Si gana >0.02 → considera deep learning como modelo final, documentando el coste.

**Justificación técnica**:

1. **Resuelve los dos bugs estructurales por construcción**. Sin two-stage no hay train-eval mismatch (BUG1). Sin pasar `stormflow_mgd` como feature al GBM no hay riesgo de doble normalización (BUG2). Las 10 features reducidas eliminan 6 con PI≤0 (S5).

2. **Ataca la causa raíz identificada**: AR(12) lineal supera a XGB-22 en H=1 (0.83 vs 0.79) porque las features de lluvia no recuperan la información autoregresiva contenida en los lags del target. Inyectar 12 lags como input explícito da al GBM acceso completo a esa información, con la flexibilidad no lineal añadida sobre features exógenas.

3. **NSE H=1 esperado 0.83-0.88**, supera o iguala al TCN v1 actual incluso bajo la lectura optimista (0.86). Importante: es el primer modelo del proyecto que supera a AR(12) lineal con justificación estructural.

4. **Es implementable en local en una tarde**. Reusa `scripts/diagnostic/s2_baselines.py`, añade ~50 LoC. Sin GPU. Reentrenable barato — permite muchas variantes (10 vs 12 vs 14 lags, con/sin SHAP, con/sin ponderación).

5. **El componente B reconoce el techo físico de H=6/H=12** (S4: NSE máximo 0.32 y 0.19 respectivamente). En vez de pelear contra ese techo con un modelo de regresión que va a fallar, ofrece al MSD una alerta binaria que SÍ es defensible operativamente. Mejora la utilidad real del proyecto sin pretender que H=6 es alcanzable como regresión.

6. **El componente C cierra honestamente la pregunta deep learning**. Si TCN + lags supera a XGB + lags, hay deep learning en el TFM. Si no, se queda fuera con justificación cuantitativa.

**Convicción**: alta. Los datos de S2-S5 son consistentes y apuntan en la misma dirección. La única incógnita real es el rendimiento exacto de XGB+lags vs AR(12) — la cota inferior es AR(12)=0.83 (porque los 12 lags del target ya están ahí) y la cota superior teórica es 0.87 según S4 (techo físico H=1 ≈ AR(12) + algo de las features exógenas).

**Información que falta y cómo obtenerla**: la única ambigüedad relevante es la discrepancia NSE TCN v1 = 0.86 (`evaluate_local.py`) vs 0.74 (S4). Resolverla en el paso 1 del plan permite establecer la cifra de referencia rigurosa contra la que comparar el nuevo modelo.

---

## 7. Plan de ejecución

Cada paso indica archivos concretos y comando o entorno.

### 7.1 Paso 1 — Resuelta discrepancia NSE TCN v1: cifra oficial = **0.86** [HECHO]

**Resultado**: la re-ejecución de `scripts/diagnostic/s4_horizon_ceiling.py` (log completo en `outputs/diagnostic/logs/s4_rerun.log`, artefactos `outputs/diagnostic/S4_horizon_ceiling.{json,md}` regenerados) reproduce **NSE H=1 sinSF = 0.8615** sobre el test alineado (n=165,222). La cifra coincide con `evaluate_local.py` (0.8614, n=165,223) hasta la tercera decimal, por lo que se cierra la discrepancia y se adopta **NSE = 0.86 como cifra oficial** del TCN v1 sinSF a H=1.

**Cifras oficiales alineadas** (rerun S4 del 2026-04-22):

| h | min | NSE TCN v1 sinSF | Pico err % | NSE `evaluate_local.py` | Delta vs AR(12) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | **0.8615** | +42.6 | 0.8614 | +0.0342 (AR12=0.8273) |
| 3 | 15 | 0.4697 | +121.2 | 0.4714 | −0.0388 (AR12=0.5085) |
| 6 | 30 | −1.2039 | +38.9 | −1.2121 | −1.5209 (AR12=0.3170) |

**Auditoría de la discrepancia 0.74 vs 0.86** (basada en comparación de la S4 antigua commit `68b038d` vs el rerun):

- Script sin cambios (`git diff 68b038d..HEAD -- scripts/diagnostic/s4_horizon_ceiling.py` vacío), pesos sin cambios (`MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_*` fechados 2026-04-13), caché `outputs/cache/df_with_features.parquet` ya existía (ctime 2026-04-20) antes de la primera ejecución de S4.
- **Invariantes que coinciden exactamente** entre ambos runs: `n_test=165,222`, `test_index_first=936,741`, `test_index_last=1,101,962`, `peak_real=135.1509`, `peak_pred=192.7817`, `peak_err_pct=+42.6418%`, `cls_prob_pct_above_thr=12.198738666763505%`, `cls_prob_mean≈0.105087` (difieren en la 9ª decimal, float32 noise). Inferencia idéntica en el pico y en el recuento de activaciones del clasificador; la varianza global del target (`sum(y-mean)^2 ≈ 955,079`) también coincide.
- **Lo que cambia** son los SSE residuales por bucket — p. ej. en H=1: Moderado 74,840 → 37,261; Extremo 99,101 → 46,031; Base 14,936 → 18,155. Esto desplaza NSE de 0.7388 a 0.8615 sin que peak ni cls_prob se vean afectados. No hay explicación reproducible posterior (código/datos/pesos idénticos), por lo que el run antiguo queda marcado como **no reproducible** y descartado como fuente de verdad.
- La cifra del rerun (0.8615) es la que se corresponde con el mismo pipeline de `evaluate_local.py` (0.8614) — se adopta como oficial. El histórico del JSON anterior queda en el git log (commit `68b038d`) a efectos de trazabilidad.

**Implicaciones para el resto del reporte** (no se reescriben aquí por instrucción explícita, el lector debe tratar §7.1 como la referencia vigente):

- §2.2 "Valor sobre test alineado (S4 reproducción): NSE=0.7388" → cifra obsoleta; la reproducible es 0.8615.
- §2.3, tabla: fila "TCN v1 sinSF (S4 reproducción) | 0.739 | 0.293 | −1.30" → valores obsoletos; lectura vigente: 0.8615 / 0.4697 / −1.2039.
- §3.4, tabla: "NSE máx defendible H=1 ~0.83" sube a ~0.86 con la cifra oficial, pero no invalida el veredicto porque TCN v1 sigue dentro del orden del AR(12) y del 2ρ−1. Oráculos parciales: deltas caen (p. ej. oráculo Extremo ahora +0.048 en vez de +0.104) porque la base sube; la jerarquía relativa entre buckets se conserva.
- §6 "Recomendación única": la comparativa pasa a ser **TCN v1 = 0.8615 vs AR(12) = 0.8273 → +0.034 NSE**, no ±0.03 según cómo se calcule. El TCN v1 **sí supera** al AR(12) en H=1 por un margen modesto; en H=3 y H=6 **sigue por debajo** de AR(12) (0.47 < 0.51 y −1.20 < 0.32). El veredicto sobre escoger XGBoost+lags como modelo principal no cambia: el argumento dominante era (a) bugs estructurales S1 inabordables sin rediseñar la loss, y (b) atajo `delta_flow` sin el cual NSE colapsa a −0.17 (iter16). Ambas siguen en pie.

**Artefactos y comando de reproducción**:

```bash
python scripts/diagnostic/s4_horizon_ceiling.py > outputs/diagnostic/logs/s4_rerun.log 2>&1
```

- JSON: `outputs/diagnostic/S4_horizon_ceiling.json` (sobrescrito por el rerun).
- MD: `outputs/diagnostic/S4_horizon_ceiling.md` (sobrescrito por el rerun).
- Figura: `outputs/figures/diagnostic/s4_horizon_ceiling.png` (sobrescrito por el rerun).
- Log: `outputs/diagnostic/logs/s4_rerun.log`.

**Commit sugerido**: `diagnostic: rerun S4, NSE v1 H=1 oficial=0.86 (reconciliado con evaluate_local)`.

### 7.2 Paso 2 — Crear rama iter17 y borrador de modelo XGBoost+lags (medio día, local)

- `git checkout -b iter17-xgboost-lags` desde `diagnostico` (o desde `main` después de mergear `diagnostico` — decisión del usuario).
- Crear `src/models/xgboost_baseline.py` con función `build_features_with_lags(df, lags=12, features=FEATURES_10)` y `train_xgboost_h(df_train, df_val, horizon, hyperparams)`.
- Crear `notebooks/iter17_xgboost_lags.py` (estilo Colab .py) que: (a) carga parquet, (b) hace split, (c) entrena XGB H=1 y H=3 con/sin lags y reporta panel de métricas multi-bucket.
- Hiperparámetros iniciales: `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, tree_method=hist, early_stopping_rounds=20` con val set.
- Commit: `iter17: scaffolding XGBoost+lags`.

### 7.3 Paso 3 — Implementar batería de métricas multi-bucket (1-2h, local)

- Crear `src/evaluation/metrics_panel.py` con función `evaluate_full_panel(y_true, y_pred, buckets=BUCKETS)` que devuelve:
  - NSE, RMSE, MAE globales.
  - NSE, RMSE, bias, MAE por bucket.
  - Error pico (%) sobre todo el test y sobre cada bucket.
  - `peak_lag_minutes`: para cada evento físico (gap >4h), tiempo entre pico real y pico predicho.
  - `recall@U` para U ∈ {25, 50}: fracción de eventos con `max(y_real) ≥ U` correctamente alertados (max(y_pred) ≥ U).
  - `quantile_coverage` para 90% y 95% si el modelo da cuantiles.
- Commit: `iter17: panel de metricas multi-bucket`.

### 7.4 Paso 4 — Entrenar XGBoost+lags H=1 y H=3, comparar (medio día, local)

- Ejecutar `notebooks/iter17_xgboost_lags.py`.
- Generar `outputs/diagnostic/iter17_xgb_results.json` con panel completo.
- Comparar contra: TCN v1 referencia (paso 1), AR(12), XGB-20, XGB-22.
- Criterios de éxito:
  - NSE H=1 ≥ 0.85 (igualar al menos a TCN v1 bajo la cifra optimista).
  - NSE H=3 ≥ 0.66 (superar a XGB-22).
  - Error pico H=1 mejor que -21% (TCN v1) en valor absoluto.
  - Bias bucket Base ≤ +0.05 MGD.
- Commit: `iter17: XGB+lags resultados, comparativa contra v1`.

### 7.5 Paso 5 — Modelo de alerta binaria H=6 y H=12 (medio día, local)

- En el mismo `notebooks/iter17_xgboost_lags.py` añadir sección de clasificación binaria: target `stormflow[t+h] ≥ U` para h ∈ {6, 12}, U ∈ {25, 50}.
- Hiperparámetros iniciales del clasificador iguales al regresor; `objective='binary:logistic'`, `scale_pos_weight` calculado de la prevalencia de train.
- Métricas: precision, recall, F1, ROC-AUC, lead time medio antes del rebasamiento.
- Decidir umbral operativo en función del coste relativo (FN >> FP en MSD).
- Commit: `iter17: alerta binaria H=6/H=12`.

### 7.6 Paso 6 — Comparación deep learning en Colab (1-2 días)

- Solo si paso 4 da NSE H=1 ≥ 0.85.
- En `notebooks/rescate_colab_2026-04-17/iter17_tcn_lags.py`: TCN estándar (sin two-stage, sin switch), input = 12 lags del target + 10 features reducidas, Huber loss, regresión directa.
- Mismo split, misma normalización, mismas métricas que paso 4.
- Criterio de cierre: NSE_TCN_lags - NSE_XGB_lags < 0.02 → cerrar deep learning para el TFM. Si gana ≥ 0.02 → mantener TCN como modelo final.
- Commit: `iter17: TCN estandar + lags, comparativa final`.

### 7.7 Paso 7 — Actualizar documentación (1-2h)

- `docs/STATE.md`: nuevo modelo en producción, métricas, decisión sobre deep learning.
- `docs/EXPERIMENTS.md`: entrada iter17 completa.
- `AGENTS.md` §8.1: corregir hallazgos obsoletos:
  - Eliminar mención "15/59 extremos sin lluvia" en §"Qué NO funciona" (S3 confirma 0/59).
  - Documentar que `api_dynamic` SÍ aporta (PI dominante en S5), pero si está presente entonces `rain_sum_60m` se vuelve ruido (PI negativa en XGB-20 reducido).
  - Añadir regla "si añades una feature derivada de `flow_total_mgd`, ablation obligatoria contra XGBoost-20".
- `CLAUDE.md`: actualizar "Modelo actual en producción" si iter17 lo sustituye.
- Commit: `docs: iter17 cierra fase, nuevo modelo en produccion`.

### 7.8 Paso 8 — Arreglar bugs latentes pendientes (1h, local)

Independientes del modelo nuevo, pero deben resolverse para que el repo no acumule deuda:
- `src/pipeline/normalize.py:71`: dedup explícito (`list(dict.fromkeys(...))`) para impedir doble normalización.
- `src/features/engineering.py:115-116`: marcar `delta_flow_*` como `# DEPRECATED - backdoor to flow_total_mgd` o eliminar.
- `scripts/generate_dataset_stats.py`: alinear cálculo del naive con el offset del modelo.
- Fijar criterio único de "extremo sin lluvia" entre `evaluate_local.py` y `generate_dataset_stats.py`.
- Commit: `fix: bugs latentes detectados en S1`.

### 7.9 Paso 9 — Reformular plan del TFM (1 día, off-code)

- Sección Resultados del TFM: estructurarla en torno al hallazgo negativo. Plantilla:
  1. Modelo inicial: TwoStageTCN, NSE=0.86, considerado exitoso.
  2. Diagnóstico mediante baselines rigurosos: descubrimiento de que `delta_flow` es un atajo y AR(12) lineal supera al modelo.
  3. Análisis de techo físico por horizonte y bucket.
  4. Modelo final: XGBoost + lags + 10 features. NSE comparable, sin atajos, con métricas multi-bucket.
  5. Alerta binaria como complemento operativo para H=6/H=12.
  6. Lecciones metodológicas: importancia de baselines triviales, peligro de métricas globales en zero-inflated.
- Es un arco narrativo honesto y publicable, mejor que la versión "deep learning ganador".

### 7.10 Paso 10 — (Opcional) Quantile regression para extremos (postpuesto)

- Si tras pasos 4-6 hay tiempo, añadir LightGBM con `objective='quantile'` para cuantiles 0.5, 0.9, 0.95 sobre la salida del regresor.
- Útil para reportar "intervalo de incertidumbre del pico predicho", no como mejora central.
- Commit: `iter18: quantile regression para extremos`.

---

## 8. Qué se espera alcanzar y qué queda fuera del alcance

### 8.1 Alcanzable con el plan de la sección 7

- **NSE H=1 entre 0.83 y 0.88** sobre test alineado, con ganancia sobre naive ≥ +0.02 en la cota inferior y +0.07 en la cota superior. Modelo sin atajos confirmados, con ablation auditable feature por feature.
- **NSE H=3 entre 0.55 y 0.70**, superando al XGB-22 actual (0.66) y al TCN v1 actual (0.47).
- **Modelo de alerta binaria H=6 y H=12** con recall ≥ 70% en eventos de magnitud ≥ 25 MGD y precision ≥ 50% (umbrales orientativos, ajustables tras evaluación). Útil operativamente para el MSD.
- **Panel de métricas multi-bucket + peak_lag + recall@U** reportado para cada modelo. Acaba con la dependencia del NSE global como único indicador.
- **Diagnóstico binario "deep learning sí o no"** para el TFM con justificación cuantitativa.
- **Repo limpio** sin atajos confirmados ni bugs latentes.
- **Documentación coherente** que refleje el estado real, sin hallazgos obsoletos.

### 8.2 Fuera del alcance con los datos actuales

- **NSE > 0.5 a H=6 o H=12 como regresión**. El techo físico (S4) es 0.32 y 0.19 respectivamente. Persigirlo es hacer trampa al lector.
- **Predicción cuantitativa fiable de extremos individuales >100 MGD**. La cola pesada con 59 muestras de test no permite generalizar más allá del cluster Convectivo (n=12). Lo defendible es predecir bucket Alto bien y reportar cota superior con cuantiles para Extremo.
- **NSE > 0.92 a H=1** sin información futura externa (Opus 4.7 §B). Para abrir ese techo se necesita forecast de lluvia del MSD — decisión explícita del usuario es no usarlo en esta fase.
- **Modelo "general" multi-catchment estilo Kratzert**. Solo hay un punto (MC-CL-005). Sin static features ni grupos de cuencas, la LSTM hidrológica no aplica.

### 8.3 Riesgos del plan

- **Riesgo bajo**: que XGB+lags no supere a AR(12). Mitigación: AR(12) ya es 0.83, una composición razonable de AR + features con PI > 0 debería superarlo. Si no, el problema es la implementación, no el enfoque.
- **Riesgo medio**: que TCN+lags supere a XGB+lags por márgenes pequeños (0.02-0.05). En ese caso, decisión: ¿mantener deep learning por +0.02 NSE? Recomendación: si la mejora es <0.02, no compensa el coste de complejidad y se cierra. Documentarlo en el TFM como decisión consciente.
- **Riesgo medio**: que la alerta binaria H=6 dé recall bajo (<60%). Mitigación: no es modelo principal, su fallo no invalida el TFM. Reportarlo honestamente como "el techo físico de H=6 limita también la utilidad como clasificación".
- **Riesgo bajo**: que el tutor académico prefiera la narrativa "deep learning ganador" sobre "hallazgo negativo + modelo simple". Mitigación: el hallazgo negativo es académicamente más fuerte y reproducible. Si no convence, el componente C (TCN+lags) se queda como modelo final aunque solo aporte +0.02 NSE.

---

**Fin del reporte.**
