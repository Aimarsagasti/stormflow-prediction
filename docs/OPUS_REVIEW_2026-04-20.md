# Revisión externa de Opus 4.7 (2026-04-20)

Revisión solicitada via Claude Code tras completar `docs/DATASET_STATS.md`. Respuesta íntegra del modelo, archivada como referencia para iteraciones futuras.

- **Modelo:** Claude Opus 4.7 via Claude Code
- **Archivos revisados:** CLAUDE.md, AGENTS.md, docs/STATE.md, docs/EXPERIMENTS.md, docs/WORKFLOW.md, docs/DATASET_STATS.md, outputs/data_analysis/local_eval_metrics.json, outputs/data_analysis/dataset_stats.json, src/, notebooks/claude_train.py
- **Prompt usado:** disponible en HANDOFF_2026-04-20.md

---

He leído los archivos. Paso al diagnóstico.

  A. Diagnóstico metodológico: errores ocultos

  A1. Leakage implícito vía delta_flow_5m y delta_flow_15m [MODERADO]

  src/features/engineering.py:115-116:
  df_feat["delta_flow_5m"] = df_feat["flow_total_mgd"].diff(periods=steps_5m)
  df_feat["delta_flow_15m"] = df_feat["flow_total_mgd"].diff(periods=steps_15m)

  La regla en AGENTS.md §8.1 excluye flow_total_mgd como feature porque r=0.9976 con stormflow. Pero estas dos features SÍ están incluidas (notebook claude_train.py:142). Dado que flow_total = baseflow
   + stormflow y el baseflow varía lentamente, delta_flow_5m[t] ≈ delta_stormflow[t]. El modelo recibe, de facto, la derivada del target en t.

  Evidencia cuantitativa de DATASET_STATS §4: Pearson(delta_flow_15m, stormflow) = 0.376, Spearman en régimen evento = -0.078. La correlación global es moderada, pero en el primer paso trasel salto
  del target (exactamente donde importa para H=1) esta feature transporta casi toda la información autoregresiva. Es coherente con que el modelo bata al naive solo en +0.050 NSE: la ganancia marginal
  puede estar saliendo de la derivada de flow_total, no de la lluvia. No es leakage del futuro, pero sí es una puerta trasera al mismo atajo que se cerró en iter 11. Ablación obligatoria para
  determinar cuánto NSE aporta realmente la rama de lluvia.

  A2. Inconsistencia train/eval en el clasificador del Two-Stage [ALTO]

  src/models/loss.py:182-188:
  y_true_real = self._denormalize_target(y_true)
  event_label = (y_true_real > self.event_threshold).float()   # threshold=0.5 MGD
  ...
  event_mask = (event_label == 1.0).squeeze(1)
  if event_mask.any():
      reg_pred = reg_value[event_mask]

  El regresor se entrena solo en muestras con y_true_real > 0.5 MGD (~10% de los batches). Nunca ve muestras de baseflow. Pero en inferencia (tcn.py:248-253) el switch duro hace torch.where(cls_prob >=
   0.3, reg_value, 0). Cuando el clasificador tiene un falso positivo sobre baseflow (con threshold 0.3, la recall es alta a costa de la precisión), el regresor produce una salida OOD porque nunca
  entrenó esa región. Esta asimetría conditional-training / unconditional-inference contamina el bucket Base (NSE=-14.5, bias +0.21 MGD): el error no es ruido del regresor, es comportamiento
  indefinido.

  Además, el label de clasificación en training (y_true_real > 0.5) ≠ el label is_event que elpipeline adjunta al DataLoader (sequences.py:51). _compute_sample_weights en sequences.py:100 usa is_event
   para dar un boost de 1.75×, pero TwoStageLoss ignora ese is_event y redefine el evento sobre y>0.5. Si is_event (generado en clean.py/events.py a partir de los ficheros .dat del MSD) no coincide con
   y>0.5 MGD, el sistema de pesos y el entrenamiento del clasificador están optimizando cosas distintas.

  A3. Sample weights aplicados a los cuantiles de y NORMALIZADO [MENOR pero confuso]

  sequences.py:185 computa train_thresholds sobre df_train[target_col] que a esa altura ya está log1p+z-score (claude_train.py:168 llama a normalize_splits ANTES de create_dataloaders). Losumbrales
  p95/p99/p999 viven en escala normalizada. Luego _compute_sample_weights compara el y normalizado contra ellos. Es internamente consistente, pero el comentario de loss.py:79 (p99_threshold=
  float(p99_threshold)) sugiere que antes se pasaban en MGD reales. Ese branch de conversión (thresholds_are_normalized=False + norm_params) existe en CompositeLoss, no en TwoStageLoss, quees el que
  realmente entrena. No es un bug, pero hay código muerto que facilita bugs futuros.

  A4. NSE reportado, desnormalización y clipping a 0 [OK]

  src/evaluation/metrics.py:91-94: desnormaliza con denormalize_target (que revierte z-score yluego expm1 con clip≥0), y luego vuelve a clippear. Correcto. El NSE se calcula sobre MGD reales. No hay
  bug aquí.

  A5. Discrepancia en el NSE del naive [MENOR]

  DATASET_STATS §8 dice NSE_naive(H=1) = 0.811. STATE.md §"Hallazgos" dice 0.826. Es el mismo test set con la misma definición. Revisa cómo se alinea el naive (si incluye o no los primeros
  seq_length+horizon pasos que el modelo no predice). Si el naive se calcula sobre todo el test y el modelo solo sobre t ≥ seq_length-1+horizon, la comparación está sesgada.

  A6. Early stopping optimiza un régimen que no es el de test [ALTO, ya cubierto parcialmente por F2]

  No repito F2, pero añado una consecuencia operativa: trainer.py:185 guarda el mejor modelo por val_loss_epoch. La val loss está dominada por los 97 extremos de val incluyendo el 225 MGD. La mejor
  época es la 6 (STATE.md), extraordinariamente temprana. Eso es característico de early stopping que premia "no equivocarse demasiado en la cola altísima" antes de que el modelo aprenda picos
  moderados. Con test limitado a 135 MGD, ese óptimo es inaplicable. Recomendación concreta enC.

  ---
  B. Diagnóstico de límite físico

  Datos duros de DATASET_STATS:

  - ACF(1)=0.909 → cota inferior de NSE para predictor AR(1): 2ρ−1 = 0.818. El naive da 0.811 empíricamente. 0.82 es el suelo trivial.
  - Varianza total de y en test ≈ σ² = 9.37 MGD² (σ=3.06).
  - Muestras por bucket: 152,904 Base + 9,518 Leve + 2,305 Moderado + 437 Alto + 59 Extremo = 165,223.
  - Contribución al denominador de NSE por bucket (Σ(y−ȳ)²):
    - Extremo: 59 × (69.5−0.49)² ≈ 280,000 MGD² (≈18% del denominador total)
    - Alto: 437 × (~30−0.49)² ≈ 380,000 MGD² (≈24%)
    - Moderado+Leve: ≈500,000 MGD² (≈32%)
    - Base: ≈400,000 MGD² (≈26%)

  Errores cuadráticos actuales (modelo H1_sinSF):
  - Extremo: 59 × 27.8² ≈ 45,600
  - Alto: 437 × 7.3² ≈ 23,300
  - Moderado: 2305 × 3.6² ≈ 30,000
  - Leve+Base: ~110,000 + ~18,000 ≈ 128,000
  - Total numerador ≈ 227,000. Denominador ≈ 1,560,000. NSE ≈ 1 − 0.146 = 0.854 (coherente con0.861).

  Cota superior alcanzable a H=1:

  Un oráculo perfecto en extremos con lluvia (44/59) y mantener el resto → numerador baja ~34,000 (30k extremes con lluvia → 0) → NSE sube a ~0.88.

  Un oráculo perfecto en extremos+altos con lluvia y lluvia→stormflow ideal en moderados → numerador baja a ~80,000 → NSE ~0.95.

  Veredicto: el NSE máximo defendible a H=1 con las features actuales está en 0.88–0.92. Por encima de 0.92 se requerirían señales externas (radar de lluvia futura, estado de saturación de suelo,
  deshielo). El 0.861 actual deja ~0.06 de margen real, no "casi cerrado". Pero ese margen vive en extremos, no en NSE global. Si la métrica objetivo fuera peak_err_pct o RMSE del bucket Extremo, la
  cota es distinta: el límite físico por los 15 extremos sin lluvia impone bias_ext de al menos ~-3.5 MGD con un regresor óptimo sobre los 44 restantes (suponiendo que ésos se predigan perfectamente),
  equivalente a peak_err_pct ≈ −5%.

  ---
  C. Recomendación única para los primeros 3 meses

  Ablación y rediseño del target de supervisión: entrenar el regresor con supervisión NO condicional.

  Qué exactamente

  Tres cambios coordinados que son realmente UN solo cambio metodológico:

  1. src/features/engineering.py:115-116: quitar delta_flow_5m y delta_flow_15m. Son la puertatrasera a flow_total_mgd. Reemplazar (si hace falta señal de tendencia) por delta_stormflow_5mcalculado
  del target autoregresivamente — pero como feature opcional, en una variante para comparar. Prioridad: primero eliminarlas y ver cuánto cae el NSE. Esa caída es el leak real que tenías.
  2. src/models/loss.py:187-209: entrenar el regresor en TODAS las muestras, no solo en event_mask. Huber sobre todo el batch, con los pesos por magnitud que ya calculas. El clasificador sigue igual.
  En inferencia, mantener el switch duro. Esto elimina el mismatch train/inference de A2.
  3. src/pipeline/split.py: cambiar a split por años completos (8 train / 1 val / 1.5 test). Concretamente los índices actuales (iloc[:train_end]) por cortes en 2024-01-01 y 2025-01-01 (o similar). Val
   con ≥1 año completo deja de ser un segmento anómalo que contiene el máximo absoluto (F2).

  Por qué eso y no otra cosa

  - Tres opciones sobre la mesa en STATE.md: (a) API+ET, (b) clasificación multinivel, (c) lluvia futura. (a) es refinamiento marginal de una feature redundante (DATASET_STATS §5: api_dynamic tiene
  |r|≥0.7 con 9 features); no ataca nada estructural. (b) cambia la métrica objetivo sin arreglar el modelo subyacente; si el modelo actual no discrimina magnitudes, un clasificador de 5 clases tampoco
   lo hará, solo ocultará el problema. (c) depende de una reunión que aún no ha ocurrido y de un artefacto externo del MSD; no es accionable en la próxima iteración.
  - La recomendación 1+2+3 ataca las tres palancas con evidencia cuantitativa en el repo: (1) leak de A1, (2) inconsistencia de A2, (3) asimetría F2.
  - Contradice parcialmente una lección aprendida: "UN cambio por iteración" (AGENTS.md §7). Sí. Pero las tres modificaciones son metodológicas, no de tuning, y son INDEPENDIENTES para atribución: se
  ejecutan 4 entrenamientos (baseline, sin δflow, con reg no-condicional, con split por años, y uno con los tres juntos). El gasto computacional es bajo (mejor época=6, batch 256) y el decoupling se
  mantiene.

  Cómo medir si funcionó

  - Umbral primario: NSE del modelo sobre test − NSE del naive sobre test ≥ +0.10 a H=1. Hoy es +0.050 (DATASET_STATS §8). Si no supera +0.10, el modelo sigue sin justificarse frente a un script de una
   línea.
  - Umbral secundario: peak_err_pct en bucket Extremo (filtrado a los 44 eventos con lluvia, no los 59) ≤ −15%. Hoy es −21% en los 59 mezclados.
  - Umbral de consistencia: Bias en bucket Base ≤ +0.05 MGD (hoy +0.21). Si el regresor no-condicional lo empeora, la hipótesis de A2 es falsa.

  Si quedan dos meses después

  4. Segunda cosa: reducción de features a ~10 vía permutation importance sobre el modelo ya reparado (no sobre el actual; como dice AGENTS.md §8.2, PI en modelo malo no es fiable). Candidatas a
  eliminar por DATASET_STATS §5: api_dynamic (redundante con 9), rain_sum_15m (0.970 con rain_sum_10m), rain_max_10m (0.985 con rain_sum_10m), hour_sin/cos (PI negativo ya confirmado), quizá
  temp_daily_f (−0.870 con month_cos). Objetivo: 10 features sin pérdida de NSE.
  5. Tercera cosa: bootstrap de intervalos de confianza sobre los 59 extremos del test (F3). Con 59 muestras y resampleo 1000×, un IC 95% para NSE Extremo probablemente sea del orden de [−3, +0.3]. Eso
   hace explícito que el bucket no soporta el peso de conclusiones que se le está poniendo. Para el TFM es blindaje académico, no mejora operativa.

  ---
  D. Preguntas que no me has hecho

  D1. ¿La ganancia de +0.05 NSE sobre el naive proviene de delta_flow_5m/15m o de las featuresde lluvia?

  Por qué importa: si al quitar las dos features de delta el NSE cae a ≤0.82 (≈naive), el modelo no está aprendiendo hidrología, está haciendo AR(1) disfrazado. Todo el aparato de TwoStageTCN + 22
  features sería teatro para añadir +0.05 NSE que una línea de Python (y[t+1] = y[t] + 0.5*(y[t]-y[t-1])) ya cubriría. Evidencia que me lleva a plantearla: A1 + DATASET_STATS §4 (Pearson delta_flow_15m
   × storm = 0.376 global) + la casi igualdad NSE_modelo ≈ NSE_naive.

  D2. ¿Cuál es el error de timing (lag entre pico real y pico predicho) separado del error de magnitud?

  Por qué importa: peak_err_pct = (max(y_pred) − max(y_true))/max(y_true) no distingue entre "el modelo predice el mismo pico pero 5 minutos tarde" y "el modelo predice un pico más pequeño en el
  momento correcto". El valor operativo cambia mucho: un pico correcto con 5 min de retraso sigue dando tiempo a reaccionar; un pico subestimado en magnitud no activa alerta. Evidencia que me lleva a
  plantearla: STATE.md dice "el modelo acierta bastante bien CUÁNDO ocurre el pico a H=1" perono hay métrica reportada para eso. Se afirma sin medir.

  D3. ¿Los 15 extremos sin lluvia en test comparten algún patrón temporal (deshielo, día del año, temperatura previa) que el modelo debería poder capturar con temp_daily_f y memoria larga?

  Por qué importa: AGENTS.md y STATE.md los clasifican como "físicamente impredecibles con losdatos actuales" y los usan como argumento de defensa. Si están concentrados en enero–marzo con
  temp_daily_f < 40°F precedidos de acumulados de nieve (inferibles de precipitaciones pasadas), entonces sí son predecibles y la afirmación actual es conveniente pero falsa. Evidencia que me lleva a
  plantearla: DATASET_STATS §3 menciona 44 extremos con rain_sum_60m < 0.01, candidatos a deshielo, pero el análisis extreme_events_no_rain.json no se ha cruzado con temperatura en el documento. La
  afirmación "impredecible" se está dando por hecha sin el análisis de confirmación.