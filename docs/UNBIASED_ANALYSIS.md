# UNBIASED_ANALYSIS.md

Revisión externa del dataset MC-CL-005. Documento producido sin acceso al código del proyecto, a sus iteraciones previas, ni a sus archivos de estado. La única fuente de evidencia citable es `docs/DATASET_STATS.md` (en adelante, **DS**), cuyas secciones se referencian explícitamente.

Autor: consultor externo. Fecha del análisis: 2026-04-28.

---

## 1. Caracterización objetiva del dataset

**Naturaleza y resolución.** Serie temporal univariante con covariables exógenas. Un único punto de medición (estación MC-CL-005) muestreado cada 5 minutos. La cobertura total es de 3.836 días (≈10,5 años, del 2015-08-01 al 2026-01-31), con **1.101.964 registros y cero valores faltantes** (DS §1). Esto es una densidad muestral muy alta: ≈105.120 pasos por año, ≈1,1 M en total.

**Dimensión del esquema.** 26 columnas: 1 timestamp, 22 features y 2 columnas adicionales (`stormflow_mgd` y `is_event`). DS no documenta la definición precisa de `is_event` (ver §7). Las 22 features se agrupan, por su nomenclatura y por las correlaciones reportadas (DS §4), en cinco bloques:

- **Lluvia instantánea y acumulada** (8 features): `rain_in`, `rain_sum_{10,15,30,60,120,180,360}m`. La lluvia instantánea es cero el **96,03 %** del tiempo (DS §7), y la mediana de la lluvia no-cero es 0,006 in/5min (DS §7).
- **Lluvia máxima por ventana** (3 features): `rain_max_{10,30,60}m`.
- **Humedad antecedente / memoria** (2 features): `api_dynamic` (índice antecedente), `minutes_since_last_rain`.
- **Derivadas de la propia serie de caudal** (2 features): `delta_flow_5m`, `delta_flow_15m`. Esta nomenclatura presupone una señal de "flow" subyacente cuya definición no aparece en DS (ver §7).
- **Calendario y temperatura** (5 features): `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `temp_daily_f`.
- **Derivadas de lluvia** (2 features): `delta_rain_10m`, `delta_rain_30m`.

**Propiedades estadísticas de `stormflow_mgd`.** Distribución con asimetría extrema: media 0,4901 MGD, desviación 3,0622 MGD, **skewness 19,81 y kurtosis 606,89** (DS §2). El **89,80 %** del tiempo el sistema está en régimen base (`stormflow_mgd` < 0,5 MGD). El cuantil p99 global es 9,92 MGD y el máximo histórico 225,33 MGD: hay tres órdenes de magnitud entre cola y cuerpo. La distribución es por tanto cuasi-degenerada en torno a cero con una cola muy pesada que domina la varianza y, por construcción, cualquier métrica L2.

**Régimen de eventos extremos.** Definidos por DS §3 como `stormflow_mgd ≥ 50 MGD`. Hay 833 muestras extremas en todo el dataset (≈0,076 % del total). Del total, 44 (5,3 %) ocurren con `rain_sum_60m` < 0,01 in, lo que es físicamente sospechoso (escorrentía retardada, deshielo, anomalía de sensor, intervención operativa).

**Estructura temporal.** La autocorrelación de `stormflow_mgd` (DS §6) es 0,909 a 5 min, 0,556 a 30 min, 0,275 a 120 min y cae a 0,043 a 1.440 min (24 h). El sistema es predominantemente "memoria corta": la información útil desaparece más allá del orden de 1-2 horas. La fracción de pasos con lluvia (3,97 %) y la duración mediana de las rachas de lluvia (10 min, máximo 11,1 h, 4.514 rachas; DS §7) confirman que la señal forzante es intermitente y burst-like.

**Estructura entre features (DS §5).** Hay **36 pares** con |Pearson| ≥ 0,7. Los más extremos: `rain_sum_10m`–`rain_max_10m` 0,985, `rain_sum_10m`–`rain_sum_15m` 0,970, `api_dynamic` correlaciona con 9 features distintas (con coeficientes 0,74-0,94). Existe además una correlación inversa fuerte entre `temp_daily_f` y `month_cos` (-0,870), que es trivial: ambas codifican estación del año. La dimensión efectiva del espacio de features es muy inferior a 22; estimación gruesa por inspección de la matriz: ≈8-10 ejes independientes.

**Splits.** El particionado es cronológico 70/15/15: train 2.689 días, val 573 días, test 573 días (DS §1). Val y test cubren **menos de un ciclo anual completo cada uno**, lo que introduce sesgo estacional sistemático en cualquier métrica que se calcule sobre ellos. Además, los splits están **desbalanceados en eventos extremos**: 677 / 97 / 59 muestras en train/val/test, con val conteniendo el máximo absoluto del dataset (225,3 MGD) por encima del de train (199,4 MGD) (DS §3).

---

## 2. Catálogo de problemas abordables

### P1. Forecasting puntual a corto horizonte (H = 1 a 3 pasos)
- **Familia:** Regresión.
- **Formulación:** dado un contexto de longitud L, predecir `stormflow_mgd(t+h)` con h ∈ {1,2,3} pasos (5-15 min). Métricas: RMSE, MAE, NSE, **siempre comparados contra persistencia naive**. Modelos candidatos: regresión lineal regularizada, gradient boosting tabular, modelos autorregresivos con exógenos (ARX), redes recurrentes/temporales.
- **Justificación:** ACF(5 min)=0,909 y ACF(15 min)=0,7157 (DS §6). El predictor naive ya alcanza NSE=0,811 a H=1 y 0,409 a H=3 (DS §8). La señal es predecible a estos horizontes.
- **Dificultad:** baja en absoluto, **alta en términos relativos**: superar al naive de forma estadísticamente significativa requiere extraer información que no esté ya contenida en `y(t)`. La cota inferior teórica de un AR(1) (DS §6) ya está casi saturada por el propio naive.
- **Valor operativo potencial:** prácticamente nulo a 5 min (ya cubierto por persistencia); marginal a 15 min como soporte a decisiones de operación inmediata.
- **Datos faltantes que mejorarían el problema:** lluvia futura (nowcast/QPF a 5-15 min) o estado del sistema aguas arriba.

### P2. Forecasting probabilístico / cuantílico
- **Familia:** Modelado de incertidumbre / regresión cuantil.
- **Formulación:** estimar cuantiles condicionales `Q_τ(stormflow_mgd(t+h) | x(t))` para τ ∈ {0,5; 0,9; 0,95; 0,99}. Métricas: pinball loss, calibración (PIT, reliability diagrams), CRPS. Modelos: regresión cuantil (lineal o gradient boosting), redes con quantile loss, ensembles.
- **Justificación:** la distribución condicional de `stormflow_mgd` es claramente heteroscedástica. Spearman feature-target en régimen evento (≥0,5 MGD) es muy distinto al de baseflow (DS §4): por ejemplo `api_dynamic` pasa de 0,160 (baseflow) a 0,706 (evento). La varianza del error condicional cambia con el estado, y un único punto puntual no captura el riesgo de cola.
- **Dificultad:** media. Modelos cuantílicos son estándar y robustos.
- **Valor operativo potencial:** alto si el caso de uso es gestión de riesgo: un intervalo predictivo calibrado al 95 % vale más operativamente que un punto sin incertidumbre.
- **Datos faltantes:** los mismos que P1 (cuanto más informativos los exógenos, más útil el cuantil alto).

### P3. Sistema de alerta de evento de gran magnitud (clasificación con horizonte)
- **Familia:** Sistema de alerta (clasificación binaria con coste asimétrico).
- **Formulación:** dado el estado actual `x(t)`, predecir si `max_{h∈[1,H]} stormflow_mgd(t+h) ≥ θ` para distintos umbrales θ (p.ej. p95=1,3 MGD, p99=9,9 MGD, "extremo"=50 MGD; ver DS §2 y §3). Salida binaria. Métricas adecuadas a fuerte desbalanceo: precision-recall AUC, F-β con β>1 (penaliza falsos negativos), POD/FAR, lead time mediano de la alerta correcta. Modelos: regresión logística con features bien diseñadas, gradient boosting con `scale_pos_weight`, calibración Platt/isotónica.
- **Justificación:** la asimetría del problema (89,80 % baseflow, 0,076 % muestras extremas; DS §2 y §3) favorece reformular como detección. El régimen evento exhibe correlaciones razonables con features físicas (Spearman 0,5-0,7 para múltiples acumuladas de lluvia y `api_dynamic` en DS §4). Hay 833 positivos absolutos en el dataset si se usa θ=50, y se pueden definir umbrales más bajos para incrementar el n positivo.
- **Dificultad:** media.
- **Valor operativo potencial:** alto. Una alarma binaria con buen recall a coste de cierto FAR es operacionalizable; un valor numérico ruidoso de pico no lo es.
- **Datos faltantes:** etiquetas de eventos verificados (no derivadas del propio caudal) y horizonte de QPF.

### P4. Modelado peak-over-threshold (POT) de los extremos
- **Familia:** Modelado de eventos raros (teoría del valor extremo).
- **Formulación:** sobre las excedencias `stormflow_mgd > u` (con u alto, p.ej. 50 MGD), ajustar una distribución generalizada de Pareto (GPD) y modelar la frecuencia de excedencias con un proceso de Poisson o Hawkes. Cuantificar nivel de retorno (return level) a 1, 5, 10 años. Variantes: GEV sobre máximos por bloque (mensual o por evento de tormenta), modelos no-estacionarios donde los parámetros de GPD dependen de covariables (`api_dynamic`, `rain_sum_*`).
- **Justificación:** kurtosis 606,89 y skewness 19,81 (DS §2) son síntomas claros de cola pesada que invita a un tratamiento extremista. n=833 excedencias (DS §3) es un tamaño razonable para ajustar GPD estacionaria; el componente no-estacionario es más dudoso pero explorable.
- **Dificultad:** media. Marco bien establecido (extRemes, evd, pyextremes).
- **Valor operativo potencial:** medio-alto para diseño/dimensionamiento, no para operación en tiempo real.
- **Datos faltantes:** información meta-estacional (cambios climáticos del periodo, modificaciones del sistema) que justifiquen no-estacionariedad.

### P5. Detección de anomalías de "evento sin lluvia"
- **Familia:** Detección de anomalías (semi-supervisada).
- **Formulación:** identificar muestras donde `stormflow_mgd` entra en régimen alto sin precursor pluvial razonable. DS §3 documenta 44 extremos con `rain_sum_60m` < 0,01 in. Modelo: Isolation Forest, autoencoder, o reglas físicas (residual contra modelo lluvia-caudal lineal o conceptual).
- **Justificación:** DS §3 explicita que estos 44 casos son candidatos a "errores de sensor, deshielo, escorrentía retardada".
- **Dificultad:** alta por tamaño muestral (44 instancias en todo el dataset).
- **Valor operativo potencial:** alto para garantía de calidad del dato, bajo como problema de modelado de pleno derecho.
- **Datos faltantes:** registros operativos de la red (compuertas, mantenimientos), datos de temperatura sub-diaria para descartar deshielo.

### P6. Clustering / tipología de eventos pluviales
- **Familia:** Clustering / segmentación.
- **Formulación:** sobre las **4.514 rachas de lluvia identificadas** (DS §7), construir vectores de características por evento (duración, intensidad pico, lluvia total, `api_dynamic` previo, mes, respuesta hidrológica observada) y aplicar k-means / DBSCAN / clustering jerárquico. Validación con silhouette y estabilidad sobre folds temporales.
- **Justificación:** 4.514 eventos es n suficiente para clustering robusto. La duración mediana 10 min vs máxima 11,1 h sugiere fuerte heterogeneidad inter-evento (DS §7).
- **Dificultad:** baja-media.
- **Valor operativo potencial:** medio. Una taxonomía de eventos guía el diseño de modelos especializados y de planes de respuesta operativa.
- **Datos faltantes:** clasificación meteorológica externa (tipo de frente, tipo de tormenta convectiva/estratiforme, datos de radar).

### P7. Reducción de dimensionalidad y estructura latente
- **Familia:** Reducción de dimensionalidad / análisis exploratorio puro.
- **Formulación:** PCA / sparse-PCA / factor analysis sobre el bloque de features de lluvia y antecedente para reducir las 22 features a su rango efectivo. Cuantificar varianza explicada y construir un conjunto reducido (≈8-10 features) para alimentar todos los demás problemas.
- **Justificación:** 36 pares con |Pearson| ≥ 0,7 (DS §5), incluido `rain_sum_10m`–`rain_max_10m` (0,985), efectivamente la misma señal. La multicolinealidad infla la varianza de coeficientes en modelos lineales y degrada interpretabilidad.
- **Dificultad:** baja.
- **Valor operativo potencial:** indirecto (mejora todos los demás problemas).
- **Datos faltantes:** ninguno necesario.

### P8. Análisis de duración de evento / tiempo hasta retorno a baseflow
- **Familia:** Modelado de eventos raros / análisis de supervivencia.
- **Formulación:** dado el inicio de un evento (cruce ascendente de un umbral), modelar P(duración > t) con Cox regression o modelos paramétricos (Weibull, log-normal). Covariables: `api_dynamic` al inicio, lluvia total acumulada durante el evento, mes, temperatura.
- **Justificación:** DS §7 reporta 4.514 rachas de lluvia con duración mediana 10 min y máxima 11,1 h: heterogeneidad fuerte y n alto. La distribución de duraciones de evento de caudal no se reporta directamente, pero es derivable.
- **Dificultad:** media.
- **Valor operativo potencial:** alto para planificación de operación durante un evento (cuándo se espera que termine).
- **Datos faltantes:** definición operativa estandarizada de "fin de evento" y posiblemente registros operativos de bombeos.

### P9. Forecasting puntual a horizonte largo (H ≥ 6, > 30 min)
- **Familia:** Regresión.
- **Formulación:** predecir `stormflow_mgd(t+h)` con h ≥ 6 (≥ 30 min). Mismo modelado que P1.
- **Justificación:** ACF a 30 min = 0,556 y a 60 min = 0,421 (DS §6); naive ya da NSE=0,081 a H=6 (DS §8). Sin información exógena futura, el contenido predictivo es marginal.
- **Dificultad:** alta-extrema sin variables exógenas adicionales.
- **Valor operativo potencial:** alto si fuera viable (mayor lead time = mejor margen operativo).
- **Datos faltantes:** crítico — pronóstico de lluvia (QPF/nowcast).

### P10. Análisis de sensibilidad / atribución por régimen
- **Familia:** Análisis causal o de atribución (en sentido suave: importancia condicional, no causalidad estricta).
- **Formulación:** con un modelo entrenado, calcular SHAP / permutation importance condicionados al régimen (baseflow vs evento). Cuantificar qué features dominan cada régimen.
- **Justificación:** DS §4 muestra que las correlaciones cambian de forma drástica entre regímenes (`api_dynamic`: 0,160 baseflow vs 0,706 evento; `delta_flow_5m`: 0,171 vs −0,120). Hay estructura condicional.
- **Dificultad:** baja-media (es post-hoc sobre cualquier modelo).
- **Valor operativo potencial:** alto como apoyo a la interpretación y al diseño de features.
- **Datos faltantes:** ninguno necesario.

---

## 3. Eliminación por viabilidad

**P9 (forecasting H ≥ 6) — descartado como problema principal.** Evidencia: ACF cae a 0,275 a 120 min y 0,043 a 1.440 min (DS §6), y el naive ya da NSE=0,081 a H=6 (DS §8). DS §9 (hallazgo 5) lo califica explícitamente como "límite físico". Con las features actuales, este problema no es resoluble con garantía. Es viable únicamente si se incorpora un nowcast de lluvia (no presente en el dataset).

**P5 (detección de anomalías sin lluvia) — descartado como problema principal por tamaño muestral.** 44 instancias en todo el dataset (DS §3); con un split 70/15/15 quedan ≈30/7/7. Imposible validar con potencia estadística suficiente. Útil como auditoría de calidad de datos, no como problema de ML formal.

**P10 (análisis de atribución) — viable pero no autónomo.** Es un complemento a cualquier otro problema, no un problema de pleno derecho que se pueda evaluar al final con una métrica binaria. Lo retiro del top como problema independiente; debe acompañar a P1/P2/P3.

**P8 (duración de evento) — supervivencia: viable pero con caveat.** DS no reporta directamente la distribución de duraciones de evento de caudal (solo de lluvia, DS §7). Antes de comprometerse hay que construirla. No descartado, pero degradado en confianza.

**P4 (POT/GEV) — viable con caveat.** n=833 excedencias es razonable para ajuste estacionario. La asimetría entre splits (DS §3) es menos relevante aquí porque POT no usa la convención train/val/test del ML supervisado, sino el conjunto completo de excedencias para inferencia. Pero el supuesto de estacionariedad (clima/sistema invariantes en 10,5 años) es cuestionable y debe testearse.

**P1 (forecasting H=1-3) — viable, con un caveat de honestidad.** El **valor real añadido** sobre el naive es de +0,050 NSE a H=1 y +0,062 a H=3 (DS §8). El problema es perfectamente abordable, pero el techo de mejora es bajo y debe reportarse siempre frente al naive.

**P2 (cuantílico) — viable.** No descartado. La heteroscedasticidad condicional (DS §4) lo justifica.

**P3 (alerta) — viable.** No descartado. Datos suficientes, métricas robustas a desbalanceo.

**P6 (clustering) — viable.** 4.514 rachas (DS §7), n más que suficiente. No descartado.

**P7 (reducción dimensional) — trivialmente viable.** No descartado, pero su valor es complementario, no de pleno derecho.

**Sobreviven al filtro:** P1, P2, P3, P4, P6, P7, P8.

---

## 4. Top 3 recomendaciones

### Top-1: **P3 — Sistema de alerta de evento de gran magnitud**

- **Por qué entra:** la geometría del problema (89,80 % baseflow, kurtosis 606,89, 833 extremos sobre 1,1 M; DS §2 y §3) está dominada por una distribución cuasi-degenerada con cola pesada. Reformular como clasificación con umbral elimina el problema de optimizar una métrica L2 dominada por el cuerpo. Las métricas naturales (PR-AUC, F-β) son insensibles al desbalanceo extremo cuando se comparan modelos.
- **Primer experimento concreto:** definir tres umbrales (θ₁=p90 train=0,589 MGD, θ₂=p95 train=1,442 MGD, θ₃=p99 train=10,512 MGD; DS §2). Construir target binario `Y(t) = 1{ max_{h∈[1,12]} stormflow_mgd(t+h) ≥ θ }` (lead time hasta 1 hora). Entrenar regresión logística con regularización L2 sobre las 22 features (sin selección previa) y comparar con gradient boosting. Reportar PR-AUC y matriz de confusión a un punto operativo definido por F2 máximo en val.
- **Métrica primaria:** **PR-AUC** (insensible a desbalanceo). Métrica secundaria: lead-time mediano de alertas verdaderas positivas. Calibración con reliability diagram.
- **Baseline trivial a batir:** clasificador que usa `Y_naive(t) = 1{stormflow_mgd(t) ≥ θ}` (persistencia binaria). Calcular PR-AUC del naive y exigir mejora estadísticamente significativa (bootstrap sobre test).

### Top-2: **P2 — Forecasting probabilístico / cuantílico (H=1-3)**

- **Por qué entra:** la heteroscedasticidad (DS §4: Spearman feature-target cambia drásticamente entre regímenes) y la cola pesada hacen que un punto sea engañoso. Un cuantil 0,9 ó 0,95 condicional informa de riesgo de manera honesta. Es además la base correcta para construir P3 desde un modelo de regresión (umbralizar cuantiles).
- **Primer experimento concreto:** quantile gradient boosting (LightGBM `objective='quantile'`) entrenado independientemente para τ ∈ {0,5; 0,9; 0,95}, predicción a H=1 y H=3.
- **Métrica primaria:** pinball loss en test, calibración por reliability (PIT histogram). Comparar con cuantil naive (cuantil incondicional empírico de la ventana reciente).
- **Baseline trivial a batir:** predicción cuantil constante = cuantil empírico de los últimos 24 h de la propia serie (ventana móvil). Pinball loss mejor con margen no trivial sobre val.

### Top-3: **P4 — Modelado POT/GEV de extremos**

- **Por qué entra:** la kurtosis 606,89 (DS §2) y el fenómeno de cola que dispara hasta 225,3 MGD pide un tratamiento desde teoría del valor extremo. n=833 excedencias por encima de 50 MGD (DS §3) es razonable para ajuste estacionario con buena estimación de parámetros (típicamente se considera adecuado a partir de ≈100). Aporta cuantificación de niveles de retorno que P1-P3 no dan.
- **Primer experimento concreto:** ajuste de GPD a las 833 excedencias por encima de u=50 MGD usando MLE, con diagnóstico de bondad por mean-residual-life plot y QQ-plot de los residuos GPD. Calcular niveles de retorno a 1 y 5 años con intervalos por bootstrap paramétrico.
- **Métrica primaria:** Anderson-Darling sobre las excedencias estandarizadas; cobertura de los intervalos de retorno por block-bootstrap con bloques anuales.
- **Baseline trivial a batir:** ajuste empírico (cuantiles de Hazen) sin distribución paramétrica. GPD debe extrapolar mejor más allá del rango observado.

---

## 5. Recomendación final única

**P3 — Sistema de alerta de evento de gran magnitud, formulado como clasificación binaria con horizonte H ∈ [1, 12] pasos.**

### Por qué frente a P2 y P4

- **Frente a P2 (cuantílico):** P2 es técnicamente más rico, pero su métrica (pinball loss) es difícil de defender como "valor operativo" sin un caso de uso concreto. P3 da una salida binaria que cualquier comité técnico evalúa con cifras inteligibles (precision, recall, falsos positivos por mes). Además, P3 puede construirse encima de P2 (umbralizar cuantil), por lo que **P3 absorbe parte del valor de P2** sin exigir compromiso previo a una formulación cuantílica.
- **Frente a P4 (POT):** P4 produce niveles de retorno, no operación en tiempo real. Su utilidad es de diseño/dimensionamiento, no de soporte a decisión continua. Además, P4 supone estacionariedad de la cola en 10,5 años que es cuestionable y no está testeada en DS.

### Familia de modelos a abordar primero, con razón estadística

**Gradient boosting tabular (LightGBM/XGBoost) con `scale_pos_weight` y calibración isotónica posterior.** Razones, no preferencia:

1. La señal en régimen evento es claramente **no lineal y heterocedástica** (Spearman ≠ Pearson en DS §4 — p.ej. `minutes_since_last_rain` tiene Pearson global −0,265 pero Spearman régimen evento −0,639). Modelos basados en árboles capturan no-linealidades sin requerir feature engineering manual.
2. La **multicolinealidad masiva** (36 pares con |Pearson| ≥ 0,7; DS §5) penaliza modelos lineales sin regularización fuerte y enturbia la interpretación. Los árboles son insensibles a esto.
3. Tamaño muestral de 771 K en train (DS §1) admite gradient boosting moderno con hiperparámetros estándar sin riesgo de underfitting.
4. La calibración por isotónica es estándar tras LightGBM con `scale_pos_weight`, y permite operar el clasificador en un punto del espacio precision-recall elegido por costes operativos.

Solo después de cerrar este baseline se justifica explorar arquitecturas más complejas (TCN, Transformer, modelos secuenciales). El comité no aceptará "necesito una red neuronal" sin haber agotado el baseline tabular.

### Riesgo principal y cómo se detectaría temprano

**Riesgo principal: data leakage entre la construcción del target binario y las features.** Si `is_event` (DS §1, no definido) está derivada de un umbral aplicado a `stormflow_mgd(t)` y se incluye como feature, el problema es trivial. Hay que **excluir `is_event` y `delta_flow_*` del conjunto de features** hasta entender exactamente cómo se construyen, porque ambos podrían codificar información que toca el target del horizonte futuro de forma indirecta.

**Detección temprana:** entrenar el modelo dos veces, con y sin el bloque `delta_flow_*`/`is_event`. Si la diferencia en PR-AUC val es enorme (>0,1), hay leakage y la versión con esas features no debe reportarse. Adicionalmente, sanity check: PR-AUC > 0,99 en val es sospechoso por sí solo.

### Techo realista de rendimiento

DS no contiene la PR-AUC de un baseline naive de clasificación, por lo que el techo no se puede acotar numéricamente con la evidencia disponible. Pero puedo acotarlo cualitativamente con tres argumentos basados en DS:

1. **Cota inferior:** la persistencia naive ya da NSE=0,811 a H=1 (DS §8). Traducido al problema binario, esto implica que la persistencia binaria también será un baseline muy fuerte para horizontes cortos. Mejorar sobre ella requerirá leadtime.
2. **Cota superior:** la ACF cae a 0,556 a 30 min y 0,421 a 60 min (DS §6). Predecir a H=12 (60 min) la categoría binaria de un sistema con ese decaimiento de información tendrá un techo claramente menor que a H=1.
3. **Cota por dato faltante:** 5,3 % de los extremos no tienen lluvia precursora dentro de la ventana de 60 min (DS §3). Estos casos son **físicamente impredecibles con las features disponibles**, lo que pone un techo duro sobre el recall del clasificador (recall_máx ≤ 1 − 0,053 = 0,947 si todos esos casos se pierden).

Mi expectativa razonada para H ∈ [6, 12]: PR-AUC en torno a 0,6-0,8 con θ alto (p99) y modelo bien construido. Para H=1 con θ bajo (p90) la PR-AUC será mucho mayor pero el problema es trivial por persistencia.

---

## 6. Datos que mejorarían el análisis

1. **Pronósticos de lluvia (QPF / nowcast) a 5-60 min.** Es el único input que rompería el techo del horizonte. Hace viable P9 (forecasting H=6+) y mejora drásticamente P3 a horizontes largos. DS §9 lo apunta como hallazgo 5.
2. **Series de estaciones vecinas (caudal, lluvia, niveles).** Permitirían correlación espacial y detección de anomalías "sin lluvia" por contraste regional. Reduciría la incertidumbre de los 44 casos extremos sin lluvia (DS §3).
3. **Registros operativos de la red.** Estado de compuertas, bombeos, derivaciones, mantenimientos. Sin esto, las anomalías que parecen "errores de sensor" pueden ser intervenciones operativas legítimas. Necesario para P5 con rigor.
4. **Definición operacional documentada de `is_event` y de la señal "flow" (la que `delta_flow_*` deriva).** No están en DS y son críticas para evitar leakage.
5. **Etiquetas de eventos verificados.** Catálogo de eventos clasificados manualmente (CSO confirmado / no confirmado, severidad). Permitiría supervisión limpia de P3 sin depender únicamente de umbrales aplicados a la propia serie.
6. **Granularidad sub-diaria de temperatura.** `temp_daily_f` es diaria. Para detectar deshielo o heladas que disparan respuesta sin lluvia se necesita resolución horaria.
7. **Información de uso de suelo / impermeabilidad de la cuenca.** Si la cuenca cambió en 10,5 años, los supuestos de estacionariedad para P4 son incorrectos.
8. **Datos de radar meteorológico.** Mejoran QPF y permiten clasificación de eventos pluviales (P6) por tipo (convectivo / estratiforme).

---

## 7. Comentarios metodológicos

1. **Splits desbalanceados en eventos extremos y con duración inferior a un ciclo anual.** DS §1 y §3 lo señalan: train/val/test contienen 677/97/59 muestras extremas (asimetría 8:1:1 frente a un 70:15:15 nominal), y val contiene el máximo absoluto del dataset (225,3 MGD) por encima del de train (199,4 MGD). Cualquier evaluación honesta debe usar **bootstrap o blocked-CV con embargo**, no un único punto val/test. Reportar intervalos de confianza, no una métrica puntual.

2. **`is_event` no está definida en DS.** Es un riesgo de leakage si está derivada de `stormflow_mgd` con un umbral. Hay que excluirla del feature set hasta verificar su definición (ver §5).

3. **Las features `delta_flow_5m` y `delta_flow_15m` referencian una señal "flow" que no se documenta en DS** salvo de forma indirecta (en DS §4 aparecen sus correlaciones pero no su definición). Si "flow" ≠ `stormflow_mgd`, hay una segunda serie de caudal sin caracterizar. Si "flow" = `stormflow_mgd`, entonces `delta_flow_5m` es leakage trivial. Esto debe aclararse antes de cualquier modelado.

4. **Multicolinealidad estructural masiva (DS §5).** Las 22 features tienen rango efectivo ≈8-10. Esto sesga la interpretación de feature importance hacia features arbitrariamente seleccionadas dentro de cada cluster correlacionado. Cualquier afirmación tipo "la feature X es la más importante" requiere PI grupal, no individual.

5. **Asimetría temperatura ↔ calendario.** `temp_daily_f` ↔ `month_cos` Pearson −0,870 (DS §5). Una de las dos es redundante: si el objetivo del modelo es producción, mantener `temp_daily_f` (físicamente interpretable); si es interpretabilidad lineal, mantener `month_cos`. No mantener ambas en un modelo lineal.

6. **El test contiene un régimen menos extremo que train (DS §3).** Test máx 135,2 MGD < train máx 199,4 MGD < val máx 225,3 MGD. Cualquier métrica de cola en test va a ser **optimista** respecto al riesgo real del sistema. Esto debe explicitarse en cualquier reporte.

7. **El supuesto de estacionariedad climática en 10,5 años es no testeado en DS.** El periodo 2015-2026 cubre cambios climáticos relevantes a escala regional. Antes de ajustar P4 (POT) hay que testear estacionariedad de la cola (Mann-Kendall sobre máximos anuales, test de Pettitt sobre breakpoints).

8. **DS reporta correlaciones de Pearson entre features y target globales, sin separar por régimen, en su tabla principal.** La tabla de DS §4 sí separa Spearman por régimen, lo cual es la decisión correcta: en datos con cola pesada, Spearman es la métrica de asociación adecuada. Pearson global sobre `stormflow_mgd` está dominado por un puñado de extremos y es poco informativo.

9. **DS §8 mezcla dos conceptos en la tabla de comparación naive vs modelo.** Mezcla la propiedad estadística del dato (NSE del naive, función solo del dato y la ACF) con la calidad del modelo entrenado (NSE del modelo, función del modelo). El primer concepto es el que importa para esta revisión y es el que he usado para acotar techos teóricos (DS §6). El segundo no lo discuto en este informe por las restricciones metodológicas.

10. **La definición de "evento extremo" como `stormflow_mgd ≥ 50 MGD` (DS §3) es ad-hoc.** Está cerca del p99,9 global (43,5 MGD; DS §2) pero no idéntica. Cualquier conclusión sobre extremos debe testearse a varios umbrales (p99, p99,5, p99,9, 50 MGD) y reportar la sensibilidad.

---

*Fin del informe. Las restricciones metodológicas de la revisión externa han sido respetadas: solo se ha leído `docs/DATASET_STATS.md`, no se han abierto archivos del proyecto referenciados en la lista prohibida, y todas las afirmaciones cuantitativas citan la sección concreta de DS de origen.*
