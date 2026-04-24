# S1 - Pipeline audit

Auditoria del codigo fuente (rama `diagnostico`, commit c8bfe84). Objetivo: identificar bugs, leakages o decisiones cuestionables mas alla del atajo `delta_flow_*` ya confirmado en iter16.

## Resumen

- BUGS: 2
- SOSPECHOSOS: 5
- OK: 3
- Hallazgo principal: **`TwoStageLoss` (src/models/loss.py:163-220) entrena el regresor SOLO sobre muestras con `y_true_real > 0.5 MGD`, pero en inferencia (`TwoStageTCN.predict`, src/models/tcn.py:248-253) el switch duro lo aplica a TODAS las muestras con `p_evento >= 0.3`. Esto produce comportamiento indefinido sobre baseflow con falso positivo y explica el NSE=-14.5 en bucket Base. Ademas, la definicion de evento en la loss (`y > 0.5 MGD`) no coincide con la que usa el DataLoader (`is_event` booleano), por lo que los sample weights y la supervision del clasificador estan optimizando objetivos distintos.** Junto a la puerta trasera ya conocida de `delta_flow_*`, este es el segundo mecanismo por el que el sistema reportado no es una TCN "de verdad": es un AR(1) + un clasificador mal supervisado.

---

## 1. Features derivadas y atajos potenciales

**Verdict: BUG (conocido + agravante)**

Evidencia: `src/features/engineering.py:115-118`

```
df_feat["delta_flow_5m"] = df_feat["flow_total_mgd"].diff(periods=steps_5m).fillna(0.0)
df_feat["delta_flow_15m"] = df_feat["flow_total_mgd"].diff(periods=steps_15m).fillna(0.0)
df_feat["delta_rain_10m"] = df_feat["rain_in"].diff(periods=steps_10m).fillna(0.0)
df_feat["delta_rain_30m"] = df_feat["rain_in"].diff(periods=steps_30m).fillna(0.0)
```

Analisis:
- `delta_flow_5m` y `delta_flow_15m` derivan de `flow_total_mgd = baseflow + stormflow` que tiene r=0.9976 con el target. Es la puerta trasera ya confirmada en iter16 (NSE bajo de 0.861 -> -0.169 al retirarlas). Spearman en evento de `delta_flow_15m` = -0.078 (casi aleatorio), pero la Pearson global es 0.376; en el primer paso tras un salto del target la derivada carga casi toda la senal AR. `DATASET_STATS.md §4` lo refleja explicitamente.
- Ninguna otra feature derivada del target: `api_dynamic` se calcula solo de `rain_in` y `temp_daily_f` (engineering.py:70-86, recurrencia API = rain(t) + K*API(t-1), K modulado por temperatura). OK.
- `delta_rain_10m`/`delta_rain_30m` derivan solo de `rain_in`, no del target. OK.
- **Candidata adicional a vigilar**: `flow_total_mgd` esta presente como COLUMNA en `df_feat` (engineering.py se basa en ella para deltas y la mantiene en `output_columns` a traves de `feature_columns` + target, aunque explicitamente NO esta en la lista `feature_columns` de linea 128-152). Es decir, esta disponible en `df_feat` pero no se pasa al modelo via FEATURE_COLUMNS. Riesgo bajo siempre que los notebooks y `evaluate_local.py` no la vuelvan a anadir accidentalmente. Revisar al revisar iters nuevas.
- `minutes_since_last_rain` (engineering.py:40-48): causal, derivada solo de `rain_in`. OK.

Sigue siendo el mecanismo principal del atajo. Iter16 ya lo elimino de FEATURE_COLUMNS en el notebook, pero las columnas se siguen creando en `engineering.py` y pueden reaparecer facilmente.

---

## 2. Split cronologico y cruce de ventanas train/val/test

**Verdict: BUG (LEAKAGE MODERADO DE FEATURES, NO DEL TARGET)**

Evidencia:
- `src/pipeline/split.py:44-46`: `df_train = df_sorted.iloc[:train_end]; df_val = df_sorted.iloc[train_end:val_end]; df_test = df_sorted.iloc[val_end:]`. Corte estricto por indice (no por timestamp pero equivalente tras `sort_values("timestamp")`).
- `src/pipeline/sequences.py:65-70`: las ventanas se construyen DENTRO de cada split, usando solo `feature_matrix` y `target_array` del propio `df_split`.

Analisis:
- Bueno: ninguna ventana cruza fronteras entre splits, porque `_build_window_arrays` opera sobre un solo `df_split` a la vez. No hay leakage directo del target de un split al otro.
- Pero: las features acumuladas rolling de `engineering.py` (`rain_sum_*`, `rain_max_*`, `api_dynamic`, `minutes_since_last_rain`) se calculan en `create_features(df_clean)` ANTES del split. Por tanto, el primer registro de `df_val` tiene en sus `rain_sum_360m` valores que incluyen lluvia de las ultimas 6h de `df_train`, y equivalente en test. Esto NO es leakage del target, pero si "contaminacion" de features: los primeros ~72-360 pasos de cada split (dependiendo de la ventana mas larga) tienen informacion del split anterior.
- Consecuencia practica: minima para el modelo (es solo una continuidad temporal fisicamente razonable, el API es acumulativo real), pero invalida la idea de que `df_val` y `df_test` son "experimentos independientes". Si el TFM quiere reportar metricas de test limpias es facil arreglarlo: calcular features por split o descartar los primeros N pasos de val/test. Recomendable documentarlo.
- `split_chronological` no aplica gap/buffer entre splits. Si por algun motivo se replican timestamps (no es el caso aqui tras `drop_duplicates`/`sort`), habria riesgo; con los datos reales es limpio.

---

## 3. Normalizacion y posible doble normalizacion

**Verdict: BUG (latente, no siempre activo)**

Evidencia: `src/pipeline/normalize.py:71`

```
all_norm_columns = list(feature_columns) + [target_col]
```

Analisis:
- Las estadisticas (`mean`, `std`) se calculan SOLO sobre `train_transformed` (normalize.py:75-82). OK: no hay leakage de val/test a stats.
- Pero si `target_col` ya esta dentro de `feature_columns` (caso conSF, cuando se pasa `stormflow_mgd` como autoregresivo), la linea 71 lo incluye DOS VECES en la lista de columnas a escalar. `_zscore_scale` (normalize.py:42-46) itera y aplica `(x - mean) / std` sobre la misma columna in-place en la copia, por lo que se normaliza dos veces. Es exactamente el bug descrito en `AGENTS.md §11`.
- El codigo fuente de `normalize.py` NO esta arreglado. La correccion vive SOLO en los callers:
  - `evaluate_local.py:368`: `features_for_norm = [f for f in features if f != TARGET_COL]` (OK).
  - `notebooks/rescate_colab_2026-04-17/horizon_comparison_v2.py:270, 610, 830`: filtra antes de llamar (OK).
  - `notebooks/rescate_colab_2026-04-17/claude_train.py:181, 1028`: NO filtra, pero el iter16 actual no incluye `stormflow_mgd` en `FEATURE_COLUMNS`, asi que no se dispara. Si manana se reintroduce un experimento conSF desde ese notebook, vuelve el bug.
- Recomendacion: fijar el bug dentro de `normalize_splits` (dedup con `dict.fromkeys(all_norm_columns)` o aviso explicito) y no depender de que el caller se acuerde. Es el tipo de bug que la pipeline deberia prohibir por construccion.

Verificacion adicional OK:
- `normalize_target_values` (normalize.py:105-114) y `denormalize_target` (117-130) aplican log1p/expm1 condicionalmente segun `norm_params["log1p_columns"]`. Simetricos. OK.
- log1p se aplica antes de calcular mean/std (normalize.py:67), lo cual es el orden correcto.

---

## 4. Definicion de `is_event`

**Verdict: OK**

Evidencia: `src/data/clean.py:9-23, 76`.

Analisis:
- `_build_event_mask` usa timestamps con `searchsorted` sobre `event_start`/`event_end` provenientes del fichero de eventos del MSD (no calculado desde el target). Marca `is_event=True` si `t in [event_start, event_end)`.
- NO mira al futuro del TARGET: la ventana temporal viene de la definicion externa (ficheros de tormentas del MSD). No hay leakage de `stormflow_mgd` hacia `is_event`.
- Sin embargo, fisicamente `event_start`/`event_end` SI incluyen el momento del pico, y ese momento ES el target que el modelo intenta predecir. Esto significa que `is_event[t+h]` puede codificar "en t+h ya estamos dentro de una tormenta", que es lo que se intenta predecir. El Dataset (sequences.py:51, 70) usa `event_array[target_index]` como `event_target`, lo cual es legitimo como label de supervision multitarea, pero NO se puede usar `is_event` como FEATURE de la ventana de entrada sin introducir leakage.
- Revisar: no se usa como feature. En `engineering.py:154` se incluye en `output_columns` como columna auxiliar, y en `sequences.py:51` se extrae como `event_array` para el DataLoader (no entra en `feature_matrix`). OK.

---

## 5. Construccion de ventanas (leakage temporal)

**Verdict: OK**

Evidencia: `src/pipeline/sequences.py:65-70`.

```
for end_index in range(seq_length - 1, max_end_index + 1):
    start_index = end_index - seq_length + 1
    target_index = end_index + horizon
    windows_x.append(feature_matrix[start_index : end_index + 1])   # [t-71, ..., t]
    windows_y.append(float(target_array[target_index]))              # y(t + horizon)
```

Analisis:
- Ventana de input: `[end_index - seq_length + 1, end_index]` inclusive = 72 pasos.
- Target: `feature_matrix` cubre hasta el indice `end_index`; target se toma en `end_index + horizon` con `horizon >= 1`. El target NUNCA entra en la ventana de input. Correcto.
- El bucle empieza en `seq_length - 1` (ventana completa disponible) y acaba en `max_end_index = total_rows - horizon - 1`, garantizando `target_index` valido.
- Un comentario: con `horizon=1`, `end_index+1` es el target, y el ultimo elemento del input (`end_index`) es `t`. El modelo tiene acceso a `features[t]` (incluyendo `delta_flow_5m[t]` y en conSF a `stormflow_mgd[t]`). Para conSF esto es AR(1) explicito y `stormflow[t]` tiene autocorrelacion 0.909 con `stormflow[t+1]`, de donde salen los NSE altos reportados. No es leakage del futuro, pero si confirma por que el naive bate casi al modelo.

---

## 6. Loss del two-stage

**Verdict: BUG (A2 CONFIRMADO) + SOSPECHOSO (A3 codigo muerto) + SOSPECHOSO (doble definicion de evento)**

Evidencia: `src/models/loss.py:182-211`.

```
y_true_real = self._denormalize_target(y_true)
event_label = (y_true_real > self.event_threshold).float()  # threshold=0.5 MGD

event_mask = (event_label == 1.0).squeeze(1)
if event_mask.any():
    reg_pred = reg_value[event_mask]
    reg_true = y_true[event_mask]
    ...
    reg_loss = (huber_values * reg_factor * reg_weights).mean()
else:
    reg_loss = torch.zeros(...)
```

Analisis:

**A2 (BUG, ALTO)**: El regresor se entrena EXCLUSIVAMENTE sobre muestras con `y_true_real > 0.5`. Nunca ve baseflow. Pero en inferencia (`src/models/tcn.py:248-253`):

```
return torch.where(cls_prob >= threshold, reg_value, zeros)
```

Con `threshold=0.3` (trainer.py:247), cualquier falso positivo del clasificador sobre una muestra de baseflow dispara el regresor sobre un input OOD (out-of-distribution). El regresor tiene libertad total en esa region (nunca penalizado alli) y produce predicciones arbitrarias. Esto casa con `local_eval_metrics.json` que reporta bucket Base con NSE=-14.5 y bias +0.21 MGD: no es ruido, es comportamiento indefinido.

**Doble definicion de evento (SOSPECHOSO)**:
- En el loss (loss.py:183), `event_label = (y_true_real > 0.5 MGD)`.
- En el DataLoader (sequences.py:51, 70), `event_array = df_split[aux_col]` donde `aux_col="is_event"` proviene del fichero MSD (clean.py:76).
- `_compute_sample_weights` (sequences.py:100) usa `event_array` (is_event MSD) para dar +1.75x boost.
- Pero `TwoStageLoss` usa `y > 0.5 MGD` (loss.py:183). Son supervisiones distintas sobre el mismo clasificador: los pesos de muestra premian estar en ventana MSD, mientras que el BCE optimiza `y > 0.5 MGD`. Si `is_event` y `y>0.5` no coinciden 100%, los gradientes de clasificacion y los pesos de muestra estan tirando en direcciones distintas.

**A3 (SOSPECHOSO, codigo muerto)**: `CompositeLoss` (loss.py:14-110) existe pero no se usa en el training actual (`claude_train.py` importa `CompositeLoss, TwoStageLoss` pero solo instancia `TwoStageLoss` en lineas 382 y 1090). Es codigo antiguo de iteraciones previas. Su rama `thresholds_are_normalized` + `norm_params` (loss.py:34-42) facilita confundir al lector sobre si los umbrales viven en MGD o en espacio normalizado. En `TwoStageLoss` los umbrales NO existen porque la loss compara en espacio normalizado y desnormaliza internamente solo para definir `event_label`. Eliminar `CompositeLoss` o marcarlo deprecado.

**Sample weights en espacio normalizado (A3 de Opus, MENOR)**: `_compute_quantile_thresholds` (sequences.py:79-87) calcula p95/p99/p999 sobre `df_train[target_col]` cuando ya esta log1p + z-score (confirmado: `claude_train.py:180-181` llama `normalize_splits` ANTES de `create_dataloaders`). Los umbrales viven en espacio normalizado; `_compute_sample_weights` (sequences.py:95-103) compara `y_array` normalizado contra ellos. Internamente consistente, pero los valores impresos en consola (`"Weight thresholds(train): {p95: X, p99: Y}"`, sequences.py:228) NO son MGD reales, lo cual puede confundir al leer logs. No es bug, pero es trampa futura.

---

## 7. Doble normalizacion de `stormflow_mgd`

**Verdict: BUG latente** (ver tambien punto 3).

Evidencia:
- El bug VIVE en `src/pipeline/normalize.py:71` (no corregido).
- Corregido por WORKAROUND en callers:
  - `evaluate_local.py:368`: filtra antes de llamar.
  - `horizon_comparison_v2.py:270, 610, 830`: filtra antes de llamar.
- NO corregido en `notebooks/rescate_colab_2026-04-17/claude_train.py:181, 1028`: pasa FEATURE_COLUMNS completo. En iter16 actual `stormflow_mgd` no esta en `FEATURE_COLUMNS`, asi que no se dispara, pero el bug latente queda.

Evaluacion: el codigo fuente oficial (`normalize.py`) NO contiene la correccion. Cualquier refactor o notebook futuro que pase una lista con `stormflow_mgd` incluido reintroduce el bug silenciosamente. Es exactamente el tipo de bug que la pipeline deberia impedir por construccion. Arreglarlo en la propia funcion es trivial:

```python
all_norm_columns = list(dict.fromkeys(list(feature_columns) + [target_col]))
```

---

## 8. Alineamiento del predictor naive

**Verdict: SOSPECHOSO (A5 de Opus confirmado)**

Evidencia:
- `scripts/generate_dataset_stats.py:1046-1079`:

```
df_test = splits["test"]                 # df_test completo SIN descartar los 72 primeros
target_series = df_test[TARGET_COLUMN].to_numpy()
...
y_true = target_series[h:]               # [h, ..., N-1]
y_pred_naive = target_series[:-h]        # [0, ..., N-1-h]
nse_naive = _nse(y_true, y_pred_naive)
```

- Compara contra `model_nse_sin_sf = {1: 0.861, ...}` que vive en `outputs/data_analysis/local_eval_metrics.json`, calculado por `evaluate_local.py` sobre el rango `test[seq_length+horizon-1 : ]` (el modelo descarta los 72 primeros pasos del test, evaluate_local.py:755 `offset = SEQ_LENGTH + horizon - 1`).

Analisis:
- El NSE del naive se calcula sobre TODO el test set (~165k muestras), mientras que el NSE del modelo se calcula sobre test[72+h-1:] (~164.8k muestras, 72 pasos menos). La diferencia en muestras es muy pequena, pero el denominador NSE depende del rango y puede cambiar en el 3er decimal.
- Mas importante: los primeros 72 puntos del test caen justo en el arranque, que suele ser un periodo tranquilo (el test empieza en un limite cronologico cualquiera). La diferencia es minima pero el NSE reportado del naive (0.811 en `DATASET_STATS §8` vs 0.826 en `STATE.md`) sugiere que se esta calculando de dos formas distintas. Hay que alinear.
- Fix trivial: descartar los primeros `seq_length+horizon-1` puntos del test tambien en el calculo del naive. Sin ese fix, la comparacion "ganancia +0.050 NSE" no es rigurosa.

---

## 9. Inferencia (switch duro) y falsos positivos del clasificador

**Verdict: BUG (consecuencia directa de A2, mismo bug que el punto 6)**

Evidencia: `src/models/tcn.py:248-253` + `src/training/trainer.py:247`.

```
def predict(self, x, threshold=0.5):
    ...
    return torch.where(cls_prob >= threshold, reg_value, zeros)
```

En `predict()` de trainer.py se llama con `threshold=0.3`:

```
y_pred = model.predict(x_batch, threshold=0.3)
```

Analisis:
- Con threshold=0.3 la recall sube a costa de precision -> mas falsos positivos.
- Cada falso positivo activa el regresor, que solo vio muestras con `y_real > 0.5 MGD` en training -> prediccion OOD.
- En lugar del switch duro, un fix limpio es entrenar el regresor sobre TODAS las muestras (con peso de magnitud, que ya se calcula) y mantener el switch solo como filtro de baja magnitud. Asi el regresor tiene definicion valida en baseflow y no explota con falsos positivos. Es el fix C1-2 que sugiere Opus 4.7.

---

## 10. Criterio de "eventos extremos sin lluvia"

**Verdict: SOSPECHOSO (criterio cambio entre ejecuciones)**

Evidencia:
- `evaluate_local.py:606-655`: cuenta "extremos con/sin lluvia" iterando sobre `y_real > 50` en el vector de predicciones (tras el offset de 72 pasos), y para cada uno chequea si `rain_sum_60m > 0` en la ventana de 72 pasos anteriores a `df_idx = OFFSET + idx` de `df_test` (df_test YA NORMALIZADO porque el caller pasa el df normalizado a los plots... revisar). Criterio: "alguna muestra en las 6h previas tiene `rain_sum_60m > 0`".
- `scripts/generate_dataset_stats.py:502-507`: cuenta extremos donde `rain_sum_60m < 0.01` en el INSTANTE del pico (no ventana). Criterio diferente.
- El comentario en 507-509 admite que no son el mismo criterio: "No es la definicion exacta (la real usa la ventana de entrada del modelo) pero da orden de magnitud".

Analisis:
- Por eso el dato "15 de 59 extremos sin lluvia" de docs antiguos (instante del pico) no cuadra con "0 extremos sin lluvia" de iter16 (ventana de 72 pasos). El criterio se flexibilizo: una tormenta real casi siempre tiene ALGUNA lluvia en las 6h previas, aunque el pico suceda cuando ya dejo de llover. Ambos numeros pueden ser correctos, solo miden cosas diferentes.
- Hay una sutileza adicional: `evaluate_local.py:640` comprueba `rain_sum_60m > 0`, que es un acumulado rolling. En el dataset crudo `rain_sum_60m` en el minuto t recoge lluvia del intervalo [t-60min, t]; si hubo lluvia ligera hace 4h, `rain_sum_60m` ya lleva 3h en cero pero la ventana de 72 pasos todavia contiene el pulso antiguo al inicio. Eso hace que la deteccion sea muy permisiva: basta UN paso con `rain_sum_60m > 0` en 6h de ventana para marcar "con lluvia".
- Sub-bug adicional: line 635 `window_df = df_test.iloc[win_start:win_end]` usa el df_test del caller. Si ese df_test esta NORMALIZADO (log1p+z), `rain_sum_60m > 0` cambia de umbral fisico (log1p(0) es 0 y z-score(0) puede ser negativo, asi que "> 0" es una comparacion distinta en ese espacio). Revisar `main()` para ver que df_test pasa al plot.

Conclusion: la cifra "0 extremos sin lluvia" de iter16 puede ser correcta, pero bajo un criterio muy laxo. Para un TFM serio hay que:
1. Fijar un criterio unico (propongo: `rain_sum_180m > 0.05 pulgadas` en el instante del pico, o acumulado de la ventana >= 0.1).
2. Documentarlo.
3. Recalcular ambas estadisticas con el mismo criterio.

---

## Recomendaciones accionables (prioritarias)

1. **[ALTO] Arreglar A2 en `src/models/loss.py:182-211`**: entrenar el regresor sobre todas las muestras (no solo `event_mask`), manteniendo los pesos por magnitud. Mantener el switch duro solo como filtro de baja magnitud en inferencia. Elimina el comportamiento OOD sobre baseflow con falsos positivos y unifica la supervision. Prob. de mejorar bucket Base drasticamente (NSE=-14.5 -> algo razonable).

2. **[ALTO] Unificar la definicion de evento**: decidir si el clasificador debe predecir `is_event` (ventana MSD) o `y > 0.5 MGD` (umbral de stormflow). Hoy `TwoStageLoss` (loss.py:183) usa uno y los sample weights (sequences.py:100) usan el otro. Un solo criterio consistente.

3. **[MEDIO] Arreglar el bug de doble normalizacion DENTRO de `src/pipeline/normalize.py:71`**: dedup con `dict.fromkeys`. No depender de que el caller filtre. Bug latente que vuelve en cualquier experimento conSF futuro lanzado desde el notebook principal.

4. **[MEDIO] Alinear naive y modelo sobre el mismo rango de indices en `scripts/generate_dataset_stats.py:compute_section_8_naive_baseline`**: descartar `seq_length + horizon - 1` puntos al inicio del test antes de calcular NSE naive. Explica la discrepancia 0.811 vs 0.826.

5. **[MEDIO] Eliminar `delta_flow_5m` y `delta_flow_15m` de `src/features/engineering.py:115-116`** (o al menos marcarlos claramente como "deprecated - backdoor to flow_total"). Ya estan fuera de FEATURE_COLUMNS en iter16 pero siguen generandose.

6. **[MEDIO] Fijar un unico criterio de "extremo sin lluvia"** entre `evaluate_local.py:637-642` y `scripts/generate_dataset_stats.py:504-507`. Recalcular ambos outputs bajo el nuevo criterio. Asi la discrepancia 15/59 vs 0/59 se resuelve.

7. **[BAJO] Eliminar `CompositeLoss` (src/models/loss.py:14-110)** o marcarla como obsoleta para evitar confusion sobre umbrales normalizados vs MGD.

8. **[BAJO] Documentar que las features rolling (`rain_sum_*`, `api_dynamic`, `minutes_since_last_rain`) se calculan antes del split y heredan continuidad de train->val->test**. No es leakage del target pero si una dependencia cruzada que conviene documentar.

9. **[BAJO] Imprimir los thresholds p95/p99/p999 en MGD reales ademas de en espacio normalizado en `sequences.py:228`** para evitar confusion al leer logs.
