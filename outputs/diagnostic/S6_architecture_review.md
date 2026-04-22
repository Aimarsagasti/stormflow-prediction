# S6 - Revision de arquitecturas alternativas

Revision bibliografica + analisis arquitectonico de alternativas al TwoStageTCN actual para la prediccion de stormflow en MC-CL-005. Sin entrenamiento: solo criterios, referencias y evidencia previa (S1-S5) para elegir 1-2 candidatas.

Fecha: 2026-04-22. Commit base: `c8bfe84` (rama `diagnostico`).

---

## 1. Criterios de evaluacion

Un candidato arquitectonico para este caso se evalua en cinco ejes:

1. **Ataca el problema estructural diagnosticado en S1**:
   - Elimina el bug train-eval del `TwoStageLoss` (regresor entrenado solo sobre `y>0.5` y aplicado a todo lo clasificado positivo).
   - Permite usar lags explicitos del target como input (o los contiene por construccion recurrente). S5 y S2 muestran que sin lags del target, los baselines no autorregresivos se estancan en NSE~0.70 a H=1.
   - Trata zero-inflation (92% baseflow + cola con skewness 19.8, kurtosis 607) sin introducir OOD.

2. **Supera el techo fisico identificado en S4**. Los techos razonables con las features actuales son H=1 <= 0.83, H=3 <= 0.51, H=6 <= 0.32. Cualquier arquitectura que prometa mas sin info futura externa hay que cuestionarla.

3. **Implementable en Colab Pro T4 con el plazo TFM (hasta septiembre 2026)**. Tiempo de entrenamiento razonable (< 6h por corrida), codigo open-source canonico disponible (PyPI o repo GitHub de referencia), complejidad de implementacion acotada al perfil del alumno (< ~300 LoC nuevas + adaptacion del pipeline existente).

4. **Batiria AR(12) lineal = 0.827 en H=1**. S2 establece AR(12) como el baseline a batir; S5 confirma que XGB-20 sin lags queda en 0.70. Un candidato solo merece la pena si al menos empata AR(12).

5. **Aporta valor diferenciado** (algo que los baselines lineales no dan): intervalos de prediccion, mejor calibracion de cola (extremo), o mejor H>=3 donde el AR(12) tambien degrada (S2: AR(12) a H=3 = 0.51, a H=6 = 0.32).

Todos los candidatos asumen que, previamente, el pipeline se arregla segun S1:
- Entrenar el regresor sobre todas las muestras (no solo `y>0.5`).
- Unificar definicion de evento (`y>0.5` vs `is_event`).
- Arreglar doble normalizacion en `normalize.py`.
- Alinear naive y modelo sobre el mismo rango de indices en test.
- Feature set reducido (10 features segun S5) + lags explicitos del target (`y(t-1)..y(t-12)`).

Sin estos fixes el comparativo arquitectonico esta contaminado.

---

## 2. Enfoques considerados

### 2.1 LSTM hidrologica estilo Kratzert et al. (2018, 2019)

**Descripcion**. LSTM de 1-2 capas (64-256 hidden units) que recibe la secuencia de features + lags del target y predice `y(t+h)`. Kratzert demostro en CAMELS (241 cuencas) que una LSTM simple bate a SAC-SMA + Snow-17 conceptual. La libreria `neuralhydrology` (PyPI) implementa CudaLSTM y EA-LSTM listas para entrenar desde config YAML. Loss tipica: NSE basin-averaged o composite NSE + MSE sobre log(y).

**Ataca el problema estructural**:
- No hay switch duro: un solo regresor end-to-end. Elimina por construccion el bug A2 del S1.
- Soporta nativamente lags del target como entradas observadas.
- Para zero-inflation: con target en log1p (que ya usamos) + loss ponderada por magnitud, se trata razonablemente. No es tan explicito como hurdle pero es estable.

**Implementable**. Alto. `pip install neuralhydrology` + config YAML + dataset adapter (~150 LoC para adaptar MC-CL-005 al formato CAMELS-like). Tiempo Colab T4: ~1-3h para un experimento con 1.1M muestras y 72 pasos de contexto. Codigo de referencia: https://github.com/neuralhydrology/neuralhydrology.

**Rendimiento esperado H=1**. Literatura hidrologica consistente: LSTM empata o bate a AR simple cuando hay lags del target + features de lluvia. En estudios en cuencas con high-frequency data (hourly/sub-hourly) la LSTM aporta 0.02-0.10 NSE sobre AR. Para nuestro H=1 donde AR(12)=0.83 y el techo fisico es ~0.83, la mejora maxima esperable es marginal (+0.01-0.03 NSE). Donde la LSTM puede brillar es **H=3 y H=6**, porque su memoria selectiva permite descartar baseflow largo y enfocar lluvia reciente. AR(12) a H=3 = 0.51, AR(12) a H=6 = 0.32 son objetivos plausibles; batir AR(12) a H=6 por +0.05 seria un resultado publicable.

**Aporta sobre alternativas**. Implementacion probada, interpretable via cell states (papers de Kratzert 2019), valor didactico alto para un TFM (replicabilidad), y maneja multi-horizonte nativamente (output vector).

**Riesgos**.
- Sin static features (el problema es monocatchment, no CAMELS): EA-LSTM no aporta. Hay que usar CudaLSTM basico.
- Tendencia a sobre-suavizar picos extremos si la loss es MSE pura: repetir el problema de infraestimacion actual. Mitigable con loss NSE-like o Huber ponderado por magnitud.
- La ventaja empirica sobre AR(12) a H=1 puede ser <0.01 NSE, lo cual no justifica el TFM si es el unico resultado.

**Referencias**:
- Kratzert et al. 2018, HESS, "Rainfall-runoff modelling using Long Short-Term Memory (LSTM) networks" ([link](https://hess.copernicus.org/articles/22/6005/2018/)).
- Kratzert et al. 2019, "NeuralHydrology - Interpreting LSTMs in Hydrology".
- Libreria: https://github.com/neuralhydrology/neuralhydrology (PyPI `neuralhydrology`).

### 2.2 TCN estandar (Bai et al. 2018) con lags del target

**Descripcion**. TCN dilatada un solo stage (sin two-stage ni switch), regresion directa con input `[features(t-71..t), y(t-12..t-1)]`. Residual blocks con dilations 1,2,4,8,16,32 (cubre 72 pasos a 5 min = 6h). Output: `y(t+h)` regresion directa, sin clasificador ni gate. Loss: Huber ponderado por magnitud.

**Ataca el problema estructural**:
- Elimina el switch duro y el bug train-eval del S1: un solo regresor entrenado sobre todas las muestras.
- Inyeccion explicita de lags del target como canales adicionales resuelve la objecion de S2 ("XGB-20 sin lags < AR(12)").
- Zero-inflation: tratamiento implicito via log1p + Huber. Menos explicito que hurdle pero compatible.

**Implementable**. Muy alto. Ya existe `src/models/tcn.py` con la infraestructura. Modificar: (a) sustituir `TwoStageTCN` por `TCN` (backbone + head regresivo), (b) `trainer.py` pasa de `TwoStageLoss` a `HuberLoss` + sample_weights (ya calculados en sequences.py), (c) `FEATURE_COLUMNS` anade `y_lag_1..y_lag_12`. ~100-150 LoC de cambio. Tiempo Colab T4: similar al actual (5-20 min por iteracion).

**Rendimiento esperado H=1**. Una TCN con lags del target y las mismas 20 features deberia acercarse a AR(12) (0.827) por abajo, posiblemente llegar a 0.83-0.85 si extrae algo de no-linearidad sobre las features de lluvia + API (confirma S5: shap no-monotono entre base y extremo, hay interacciones aprovechables). En H=3 y H=6 es donde deberia aportar mas: la dilatacion multiscale captura mejor la cinetica lluvia->caudal que un AR lineal.

**Aporta sobre alternativas**.
- Minima friccion con el codigo actual (reusa pipeline, loader, normalizacion, plots).
- Paralelizable (ventaja sobre LSTM en inferencia, relevante si en el futuro se despliega online).
- Literatura (Liu 2021, Song 2021) muestra TCN >= LSTM en streamflow hourly, con entrenamiento mas rapido.

**Riesgos**.
- Si el TCN sin two-stage y con lags del target iguala a AR(12) pero no lo bate, el TFM se queda con un resultado negativo ("nada mejora lo lineal"). Hay que tener B-plan.
- Sobreajuste a los lags del target (reintroduce un tipo de atajo mas sutil que `delta_flow`): hay que monitorizar con ablations.
- La dilatacion asume que el field receptive de 72 pasos es suficiente. S5 muestra `rain_sum_360m` con PI=+0.11: la senal util llega a 6h. Parece que 72 pasos basta, pero habria que probar 144 si se ve que la extension ayuda.

**Referencias**:
- Bai, Kolter, Koltun 2018, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (arXiv:1803.01271). Repo: https://github.com/locuslab/TCN.
- Application of TCN for flood forecasting, IWA Hydrology Research 2021 (TCN > LSTM en hourly streamflow).

### 2.3 Temporal Fusion Transformer (Lim et al. 2021)

**Descripcion**. Arquitectura con variable selection networks, LSTM encoder-decoder, multi-head attention, y static/observed/known-future splits. Diseno para forecasting multi-horizonte con interpretabilidad.

**Ataca el problema estructural**. Parcial:
- No tiene switch duro, la salida es regresion cuantil directa (TFT original predice 10%/50%/90%). Esto resuelve A2 y ademas aporta intervalos de prediccion (valor para alerta CSO).
- Lags del target entran como "observed inputs".
- Zero-inflation: TFT quantile loss funciona bien con distribuciones asimetricas.

**Implementable**. Medio-bajo. PyTorch Forecasting tiene TFT listo (https://github.com/jdb78/pytorch-forecasting). Pero TFT es **sobredimensionado** para 1 cuenca + 20 features + sin static covariates + sin known-future inputs. La literatura hidrologica reciente (Nadarajah et al. 2024 arXiv:2506.20831, efficacy of TFTs for runoff simulation) confirma que TFT mejora LEVEMENTE a LSTM en CAMELS (>500 cuencas) con muchos static features. Para un monocatchment las ventajas desaparecen. Tiempo Colab T4: 3-10h por corrida, considerablemente mas lento que LSTM/TCN.

**Rendimiento esperado H=1**. Comparable a LSTM con lags del target (+0.02-0.05 NSE sobre AR(12) optimistamente). No hay evidencia de que TFT explote mejor un monocatchment.

**Riesgos**.
- Complejidad alta, curva de aprendizaje empinada para el alumno.
- Hiperparametros criticos (attention heads, hidden sizes, dropout) requieren busqueda dedicada.
- Sin static features ni known-future inputs, se pierde la mitad del valor del TFT.

**Veredicto**: interesante si ademas se inyecta **forecast externo de lluvia** (known-future input), pero el usuario ha descartado esa opcion explicitamente. Sin eso, el TFT no justifica su complejidad frente a una LSTM o TCN con quantile head.

**Referencias**:
- Lim et al. 2021, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting", Int. J. Forecast.
- Nadarajah et al. 2025 "Efficacy of Temporal Fusion Transformers for Runoff Simulation" (arXiv:2506.20831).

### 2.4 XGBoost / GBM puramente tabular + lags

**Descripcion**. Extension de XGB-20 (S2) anadiendo `y_lag_1..y_lag_12` como features + lags de `rain_sum_60m`, `api_dynamic` en pasos -3, -6, -12. Es la iteracion natural sobre lo que ya se probo en S2/S5.

**Ataca el problema estructural**. Completamente, desde otra direccion:
- No hay two-stage: un solo regresor. Sin switch, sin bug.
- Lags del target son inputs explicitos: por construccion iguala o bate a AR(12).
- Zero-inflation: XGBoost maneja heavy-tailed de forma robusta via gradient boosting con loss Tweedie o Gamma (ideal para stormflow no-negativo con masa en cero).

**Implementable**. Maximo. `pip install xgboost` (ya instalado). ~50 LoC de cambio sobre el script S2 actual. Tiempo Colab T4: 10-60 segundos por experimento (XGB-22 de S2 tardo 10s). Literalmente el experimento mas rapido de comparar.

**Rendimiento esperado H=1**. Si XGB-20 (sin lags) da 0.66 y AR(12) (solo lags) da 0.827, un XGB-30 (20 features + 12 lags del target) deberia salir **entre 0.83 y 0.88** en H=1 con alta probabilidad. Literatura general ML en series temporales y el benchmark propio (S2) soportan esta estimacion. Con Tweedie/Gamma loss en vez de squared, podria mejorar la cola.

**Aporta sobre alternativas**.
- Baseline robusto antes de complicar con deep learning.
- Entrenamiento trivial, interpretabilidad alta (SHAP ya disponible).
- En S2, a H=3 XGB-22 = 0.66 > TCN v1 H=3 = 0.47. Es decir, la XGB ya bate al TCN actual a horizontes > 1 con menos ingenieria.

**Riesgos**.
- XGB no captura dinamica secuencial de una ventana de 72 pasos tan bien como una red con convolucion temporal. Queda expuesto a dependencias no capturadas por lags discretos.
- Multi-output no natural: para H=3 y H=6 hay que entrenar modelos separados.
- Como resultado de TFM, XGBoost sin red neuronal puede leerse como "poco ambicioso". Justificable si bate consistentemente a la deep learning.

**Referencias**:
- Chen & Guestrin 2016, KDD (XGBoost).
- S2 de este mismo diagnostico: XGB-22 = 0.79 H=1, 0.66 H=3, 0.38 H=6 (ya bate al TCN a H>=3).

### 2.5 Quantile Regression (LightGBM multiquantile / LSTM quantile)

**Descripcion**. Predecir quantiles (p10, p50, p90, p95, p99) en vez de la media. Dos variantes: (a) LightGBM con `objective="quantile"` entrenado una vez por cuantil, (b) LSTM/TCN con quantile loss Pinball.

**Ataca el problema estructural**:
- Resuelve el problema del bias sobre la cola alta: p90/p99 tendra bias menor en extremos que el p50.
- Permite reportar intervalos de confianza (alerta CSO con prob >= X).
- Zero-inflation: quantile loss no asume distribucion simetrica, apta para cola larga.
- El bug two-stage desaparece (no hay switch).

**Implementable**. Alto. LightGBM quantile: `pip install lightgbm` + loop de 5 entrenamientos (1 por cuantil). ~80 LoC. Quantile LSTM/TCN: Pinball loss es 10 lineas de PyTorch.

**Rendimiento esperado H=1**. Para NSE puntual (media), la mediana quantile (p50) no supera sistematicamente a XGB con squared loss; tipicamente pierde 0.01-0.02 NSE. Pero las metricas operativas cambian: `peak_lag`, `exceedance recall at threshold 50 MGD` son mas relevantes para CSO. La literatura hidrologica (Pasche & Engelke 2022 EQRN, Chandra et al. 2024 Ensemble Quantile LSTM) muestra mejoras claras en flood forecasting cuando la metrica es "captura de extremos" en vez de NSE global.

**Aporta sobre alternativas**.
- **Reformula la metrica**. Si el objetivo operativo es alerta de CSO, el NSE global es la metrica equivocada. Un quantile model optimizado para recall@threshold puede ser estrictamente mejor que un TCN con NSE=0.86 pero err_pico=-21%.
- Intervalos de confianza nativos.
- Muy rapido de iterar en LightGBM.

**Riesgos**.
- Si el TFM insiste en NSE como metrica unica, QR no va a ganar.
- Requiere redefinir que significa "mejor" (ver seccion 4 veredicto).

**Referencias**:
- Pasche & Engelke 2022, "Neural Networks for Extreme Quantile Regression" (arXiv:2208.07590). Repo EQRN.
- Chandra et al. 2024 "Ensemble quantile-based deep learning framework for streamflow and flood prediction in Australian catchments" (arXiv:2407.15882).

### 2.6 Zero-inflated / Hurdle con componentes SEPARADOS (no shared backbone)

**Descripcion**. Reformular el two-stage actual: en vez de backbone compartido TCN -> clasificador + regresor (con bug), entrenar dos modelos **independientes** con sus propios inputs, features y losses. Combinar por regla determinista en inferencia.

Ejemplo concreto:
- Componente A (classifier): XGBoost clasificador binario `is_event_future = (y(t+h) > 0.5)`, entrenado con BCE + class_weight sobre TODAS las muestras.
- Componente B (regressor): TCN o XGBoost regresivo `y(t+h)` entrenado **solo sobre muestras con `is_event=True`** (subsample explicito), con Huber o Tweedie.
- Inferencia: `y_hat = p(event) * E[y | event, x] + (1-p(event)) * baseflow_mean_local`, o equivalentemente `if p(event) >= 0.5: y_hat = reg_out; else: y_hat = baseflow_persistence(t)`.

**Ataca el problema estructural**:
- A diferencia del TwoStageTCN actual, los componentes NO comparten pesos ni gradientes; la inconsistencia train-eval desaparece por construccion porque el regresor se aplica *solo* donde fue entrenado (event=True), y el baseflow se gestiona con un estimador trivial local (persistencia AR(1) o media rolling).
- Lags del target entran naturalmente en ambos componentes.

**Implementable**. Medio. ~200 LoC: un script de entrenamiento por componente, una regla de combinacion en el evaluador. Tiempo Colab T4: trivial (minutos), cada componente se entrena por separado y son modelos pequenos.

**Rendimiento esperado H=1**. Dependencia fuerte del clasificador. Si `p(event)` es bien calibrado, esta aproximacion iguala o bate al XGBoost-30 plano porque elimina el ruido baseflow del regresor. Literatura (Zhang et al. 2022 deep extreme mixture models, KDD) muestra mejoras de 0.02-0.05 NSE en tasks zero-inflated frente a modelos unicos. Para extremos, si el clasificador tiene recall > 0.9 en eventos reales, el regresor especializado deberia reducir el bias -21% actual a magnitudes <10%.

**Aporta sobre alternativas**.
- Disena el modelo en torno a la asimetria del target (92% zeros-ish + cola pesada).
- Interpretable por partes (puedes reportar precision/recall del clasificador y NSE del regresor condicionado).
- Corrige el bug A2 del S1 por construccion, no por parcheo.

**Riesgos**.
- Sensible a la definicion de "evento" (ver S1: `y>0.5` vs `is_event`). Hay que fijar un criterio y ceñirse.
- El umbral duro en inferencia (`p(event) >= 0.5`) crea discontinuidad. Suavizar con una mezcla ponderada (`y_hat = p * reg + (1-p) * baseflow`) puede ayudar pero pierde interpretabilidad.
- Si el clasificador falla (recall bajo en extremos), todo el sistema falla.

**Referencias**:
- Zhang, Rwebangira, Zhang, Tan 2022, KDD, "Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models".
- Hurdle model original: Cragg 1971. Para series temporales reciente, Wen et al. 2020 IJCAI "Deep Hurdle Networks".

### 2.7 Hibrido AR + boosting (AR first, XGB sobre residuos)

**Descripcion**. First-stage: AR(12) lineal predice `y_hat_ar(t+h)`. Second-stage: XGBoost sobre `r(t+h) = y(t+h) - y_hat_ar(t+h)` usando features de lluvia + API + estacionalidad (los 10 de S5). Prediccion final: `y_hat = y_hat_ar + y_hat_xgb_resid`.

**Ataca el problema estructural**:
- Sin switch duro. Dos modelos independientes.
- Explota la fortaleza del AR (memoria lineal del caudal) y deja la no-linealidad / efectos de lluvia al XGBoost sobre residuos.
- Zero-inflation: la AR ya modela bien la masa central; el XGB en residuos no tiene que aprender la diagonal, solo los picos.

**Implementable**. Maximo. ~80 LoC. Extremadamente rapido (AR(12) y XGB tardan < 30s cada uno). Literatura extensa en hybrid ARIMA-XGBoost (Qi et al. 2024, MDPI).

**Rendimiento esperado H=1**. Fuerte. AR(12) solo = 0.827. Si XGB captura un 10-20% de la varianza residual (lo cual es razonable dado que S5 muestra que las features de lluvia tienen PI +0.42 para api_dynamic), NSE final podria llegar a 0.85-0.87. En H=3 donde AR(12)=0.51 y XGB-22=0.66 (S2), la combinacion lineal de ambos puede acercarse a 0.70. En H=6 donde AR(12)=0.32 y XGB-22=0.38, el hibrido podria llegar a ~0.45.

**Aporta sobre alternativas**.
- Baseline hibrido fuerte antes de deep learning. Si bate al TCN actual, es indicio de que el TCN no aporta valor real.
- Separa cleanly la senal autorregresiva del modulo no-lineal. Interpretabilidad alta.
- Rapidisimo de iterar.

**Riesgos**.
- Asume que los residuos del AR son estacionarios, lo cual no esta garantizado en extremos (pico = residuo enorme). Mitigable modelando el log(residuo) o usando loss asimetrica en XGB.
- Si AR(12) ya extrae casi toda la senal aprovechable (confirmado por S4: techo H=1 ~ 0.83), el XGB sobre residuos solo arana 0.02-0.05 NSE.
- Peligro de doble contar si XGB ve lags del target que ya entraron en AR. Hay que excluirlos del input XGB.

**Referencias**:
- Qi et al. 2024 MDPI "A Hybrid ARIMA-LSTM-XGBoost Model with Linear Regression Stacking" (propuesta similar).
- Zhang 2003 "Time series forecasting using a hybrid ARIMA and neural network model" (clasico).

### 2.8 Seq2seq con atencion local (encoder-decoder)

**Descripcion**. Encoder RNN/TCN sobre ventana pasada, decoder RNN/TCN con atencion sobre el encoder que genera `y(t+1..t+H)` en vez de un solo paso. Util para multi-horizonte consistente (genera H=1, 3, 6 en un solo forward).

**Ataca el problema estructural**. Parcial:
- Sin switch duro, regresion directa.
- Atencion local puede enfocar en la parte relevante de la ventana (pulso de lluvia reciente vs baseflow lejano).
- Zero-inflation: similar a LSTM, implicito via log1p.

**Implementable**. Medio. Mas complejo que TCN simple (~300-400 LoC) pero hay ejemplos canonicos (pytorch-forecasting DeepAR, Luke Tonin seq2seq). Tiempo Colab T4: 2-5h por corrida.

**Rendimiento esperado**. Donde seq2seq deberia brillar es **H=6 y H=12**. AR(12) a H=12=0.19, a H=24=0.09. Si un encoder-decoder con atencion logra sostener la senal de lluvia y API a horizontes largos, podria subir H=12 a 0.25-0.35 (por debajo del techo S4 de 0.19 que es el de AR(12) solo). Pero **S4 indica que el techo fisico a H=12 es 0.19** con features actuales: sin info futura, ninguna arquitectura puede romperlo mucho.

**Riesgos**.
- Complejidad alta para un TFM sin ganancia clara sobre LSTM simple multi-output.
- El decoder con teacher forcing tiene exposure bias conocido.
- Techo fisico de S4 limita la ganancia real: ninguna arquitectura inventa informacion que no esta en las features.

**Veredicto**: solo tiene sentido si se combina con **forecast externo de lluvia como input conocido al decoder**. Sin eso, no aporta sobre LSTM vanilla. Descartar para este caso.

**Referencias**:
- Sutskever et al. 2014 NIPS (seq2seq original).
- Salinas et al. 2020 DeepAR (seq2seq probabilistico, PyTorch Forecasting).

---

## 3. Tabla comparativa

Cada eje 1-5 (1 = muy mal, 5 = excelente). El total sirve como ordenacion orientativa.

| # | Enfoque | Estructural (S1 fix) | Implementable Colab T4 | Bate AR(12) H=1 | Valor a H>=3 | Riesgo/complejidad | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| 2.1 | LSTM (neuralhydrology) | 5 | 4 | 3 | 4 | 3 | 19 |
| 2.2 | TCN estandar + lags | 5 | 5 | 4 | 4 | 4 | **22** |
| 2.3 | TFT | 4 | 2 | 3 | 3 | 2 | 14 |
| 2.4 | XGBoost + lags target | 5 | 5 | 5 | 4 | 5 | **24** |
| 2.5 | Quantile regression | 5 | 4 | 3 | 4 | 4 | 20 |
| 2.6 | Hurdle separado | 5 | 4 | 4 | 4 | 3 | 20 |
| 2.7 | AR + XGB residuos | 5 | 5 | 4 | 4 | 5 | **23** |
| 2.8 | Seq2seq attention | 4 | 3 | 3 | 3 | 2 | 15 |

Nota sobre la columna "Bate AR(12) H=1":
- 5 = muy probable (>0.85 esperado): XGB con lags.
- 4 = probable (0.83-0.85): TCN+lags, hurdle, AR+XGB.
- 3 = dudoso (0.82-0.84): LSTM, TFT, QR p50, seq2seq.

---

## 4. Recomendacion (1-2 arquitecturas)

### Recomendacion primaria: **XGBoost + lags del target (enfoque 2.4)** como baseline fuerte, y **TCN estandar + lags del target (enfoque 2.2)** como candidata de deep learning.

Argumento combinado:

1. **XGBoost + lags (2.4) debe ser el PRIMER experimento post-diagnostico**. Es el experimento mas barato (< 1 minuto en Colab), mas replicable, y con la probabilidad mas alta de batir al TCN actual y a AR(12) simultaneamente. Si XGBoost-30 (20 features reducidas a 10 por S5 + 12 lags del target) da NSE H=1 >= 0.85 y H=3 >= 0.60, el veredicto del TFM se reescribe: "el valor del TCN v1 era ficticio; un GBM tabular bien diseñado ya resuelve el H=1 y H=3". Esto es un hallazgo fuerte para el TFM.

2. **TCN estandar + lags (2.2) es la candidata deep learning mas razonable**. Reutiliza el 90% del pipeline actual (loader, normalizacion, sequences), elimina el TwoStageLoss bug, y deja el modelo a merced de la loss + features para el resto. Si supera a XGBoost + lags en al menos 0.01-0.02 NSE, justifica la deep learning; si no, confirma que para monocatchment 5-min el ML tabular ya exprime la senal.

3. **No se recomienda LSTM (2.1)** como candidata *unica* porque para este caso concreto (monocatchment, sin static features, sin many-catchment transfer learning, sin estacionalidad fuerte aprovechable) las ventajas de LSTM sobre TCN desaparecen y su entrenamiento es mas lento por falta de paralelizacion. Si hay tiempo para un tercer experimento, vale la pena probar CudaLSTM de neuralhydrology para tener comparativa limpia con la literatura hidrologica, pero NO como primera opcion.

4. **Recomendacion secundaria condicional**: si despues de XGB+lags y TCN+lags el usuario aun quiere explorar algo con "valor de TFM" diferenciado, usar **hurdle separado (2.6)** o **AR + XGB residuos (2.7)**. Ambos explican mejor la fisica del problema que un TCN monolitico y son capitulo narrativo interesante en la memoria del TFM.

5. **Se recomienda ademas complementar con quantile regression (2.5) como metrica auxiliar, no como arquitectura principal**. Entrenar una LightGBM quantile (p50, p90, p99) sobre las mismas features + lags permite reportar intervalos de prediccion y calibrar alerta CSO, sin cambiar el modelo principal.

### ¿Cual resuelve el problema estructural Y la senal debil?

- **Estructural (bug S1)**: cualquier candidato sin two-stage lo resuelve. Todos excepto el parche del TwoStageTCN actual cumplen.
- **Senal debil**: la clave es inyectar lags del target. XGB+lags, TCN+lags, LSTM, AR+XGB los incluyen por construccion. Sin lags explicitos ningun candidato bate a AR(12), como demuestran S2 (XGB-20=0.66) y S5 (reducido sin lags=0.70).

Los dos problemas se resuelven **simultaneamente** con XGB+lags o TCN+lags. Esos son los candidatos dominantes.

---

## 5. Veredicto sobre TwoStageTCN: parchear o descartar

**Recomendacion: DESCARTAR el TwoStageTCN actual como modelo principal**. Mantener el codigo en el repo como referencia historica (iter1-16) pero no seguir iterando sobre el. Motivos:

1. El bug train-eval del TwoStageLoss (S1 A2) NO es un parche trivial: arreglarlo equivale a reentrenar el regresor sobre todo el dataset, lo cual convierte al TwoStageTCN en una TCN estandar con un clasificador pegado. En ese caso, el clasificador ya no aporta nada estructural (porque el regresor ya esta definido en todo el dominio) y el switch duro se vuelve mera compuerta cosmetica. Es mas limpio descartar el two-stage y entrenar una TCN regresiva directa (enfoque 2.2) con sample_weights por magnitud.

2. La evidencia empirica de S2 muestra que el TwoStageTCN v1 (NSE H=1=0.861 en version con `delta_flow`; NSE=0.739 sobre test alineado sin truquillo) NO bate consistentemente a AR(12)=0.827 a H=1 una vez alineado correctamente, y pierde claramente a XGB-22=0.66 a H=3 y 0.38 a H=6. Es decir, la arquitectura actual no justifica su complejidad.

3. La arquitectura two-stage con backbone compartido + switch duro es una hipotesis hidrologica razonable (separar "regimen baseflow" de "regimen tormenta") pero en la practica introduce mas bugs de los que resuelve. La literatura hidrologica reciente (Kratzert 2018, Klotz 2022, Nadarajah 2025) ha convergido en **single-regressor con loss apropiada** (NSE-like, quantile, o Gaussian output con head Gaussiana para PIs), NO en two-stage. Perseguir two-stage aqui es ir contra la corriente.

4. Parchear mantiene la deuda tecnica (CompositeLoss deprecado, doble definicion de evento, doble normalizacion latente, etc.). Empezar limpio es mas productivo.

**Plan propuesto**: mantener `TwoStageTCN` como iteracion historica (congelada), y construir la siguiente generacion del modelo (iter17+) sobre TCN regresiva directa (enfoque 2.2) reutilizando pipeline. Si el usuario quiere conservar la logica two-stage por hipotesis hidrologica, implementarla como **hurdle separado (enfoque 2.6)** con componentes independientes, no como backbone compartido.

---

## ¿Superar AR(12) o reformular evaluacion?

Esta es la pregunta mas importante para el TFM. Mi recomendacion:

**Hacer las dos cosas, pero en este orden**:

1. **Primero, reformular la evaluacion**. El NSE global es engañoso con zero-inflation extrema (92% baseflow): cualquier modelo que acierte bien el baseflow tiene NSE alto aunque falle los picos. Reportar siempre, en paralelo:
   - NSE global (para comparabilidad con literatura).
   - NSE por bucket (Base, Leve, Moderado, Alto, Extremo) — ya se calcula en S2.
   - `peak_lag` y `peak_amplitude_error` por tormenta (hay 24 tormentas fisicas en test, S3).
   - `exceedance recall @ 50 MGD` y `@ 25 MGD` (operativo para CSO).
   - (Si se usa QR) Quantile coverage 80%/95%.

   Con esta bateria de metricas, la pregunta "¿bate al AR(12)?" se desdobla en "¿bate en NSE global? ¿bate en extremos? ¿bate en peak_lag?". Un modelo que empate al AR(12) en NSE global pero gane en recall@50 es un resultado valioso.

2. **Despues, intentar superar AR(12) en H=1 con XGB+lags y TCN+lags**. Si cualquiera de ambos bate a AR(12) por >=0.01 NSE globalmente *y* mejora en Extremo/Alto, hay historia de TFM solida.

3. **Aceptar que H=6 y H=12 tienen techo duro** (0.32 y 0.19 por S4). Documentar explicitamente que sin info futura externa (forecast de lluvia que el usuario descarta), estos horizontes no son operativos. Para CSO, reformular el problema como "alerta binaria con antelacion variable" usando el clasificador de hurdle (2.6) o quantile p95 (2.5) en vez de regresion de amplitud.

---

## Resumen de siguiente experimento concreto (iter17 propuesto)

Sobre la rama `diagnostico` o en una nueva:

1. Arreglar los 4 bugs del S1 (loss, normalize, alineamiento naive, definicion evento).
2. Ampliar `FEATURE_COLUMNS` con lags explicitos del target: `y_lag_1, y_lag_2, y_lag_3, y_lag_6, y_lag_12` (5 lags, no 12 para empezar).
3. Reducir features exogenas a las 10 de S5.
4. Entrenar 4 modelos:
   - `xgb_lags`: XGBoost con 10 feats + 5 lags, Huber/Tweedie loss. H=1, H=3, H=6 (uno por horizonte).
   - `tcn_direct`: TCN regresiva directa, mismo input, Huber + sample_weights.
   - `ar_xgb_hybrid`: AR(12) + XGB sobre residuos.
   - (opcional) `hurdle`: XGB clasificador + XGB regresor separados.
5. Evaluar TODOS con la bateria de metricas expandida (NSE por bucket + peak_lag + recall@50).

Coste estimado: 2-3 dias de trabajo para codigo + 1 jornada de entrenamiento en Colab. Todo antes de mayo 2026, dejando 4 meses para redaccion de TFM.
