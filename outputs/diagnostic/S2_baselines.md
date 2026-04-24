# S2 - Baselines rigurosos

Bateria de baselines clasicos / ML tabular entrenados sobre el mismo split cronologico 70/15/15 y los mismos indices de test que la TCN (`seq_length=72`, `horizon h`). Objetivo: contextualizar la ganancia real del TwoStageTCN v1 y diagnosticar el peso del atajo `delta_flow`.

## Metodologia

- **Split**: train `iloc[:771374]` (hasta 2022-12-11), val `[771374:936669]`, test `[936669:]` (hasta 2026-01-31).
- **Ventana de evaluacion**: indices con ventana previa de 72 pasos completa y `y(t+h)` disponible (mismos indices que consumiria la TCN).
- **Target**: `stormflow_mgd(t+h)`.
- **Sin tuning**: hiperparametros fijos definidos en `s2_baselines.py`.

## Dimensiones por horizonte

| h | n_train | n_val | n_test | primer ts test | ultimo ts test |
|---|--------:|------:|-------:|----------------|----------------|
| 1 | 771,302 | 165,223 | 165,222 | 2024-07-07 07:25:00 | 2026-01-31 23:50:00 |
| 3 | 771,302 | 165,223 | 165,220 | 2024-07-07 07:25:00 | 2026-01-31 23:40:00 |
| 6 | 771,302 | 165,223 | 165,217 | 2024-07-07 07:25:00 | 2026-01-31 23:25:00 |

## Horizonte h=1

| Baseline | NSE | RMSE | MAE | ErrPico % | Peak pred | N features | Notas |
|---|---:|---:|---:|---:|---:|---:|---|
| **TCN v1 sinSF** (ref) | 0.8614 | 0.894 | 0.293 | -21.0 | 106.5 | 22 | n_test=165,223 |
| TCN v1 conSF (ref) | 0.8542 | 0.916 | 0.242 | +58.8 | 214.0 | 22+sf | - |
| Naive persistencia y(t) | 0.8107 | 1.046 | 0.101 | +0.0 | 135.2 | - |  |
| AR(1) analitico (rho + const) | 0.8196 | 1.021 | 0.121 | -9.1 | 122.9 | - | rho=0.9087 |
| AR(1) analitico (sin const) | 0.8195 | 1.022 | 0.097 | -9.1 | 122.8 | - |  |
| AR(5) lineal | 0.8266 | 1.001 | 0.120 | -3.4 | 130.6 | - |  |
| AR(12) lineal | 0.8273 | 0.999 | 0.115 | -6.1 | 126.9 | - |  |
| Lineal fisico (rain_60m+API) | 0.6186 | 1.485 | 0.315 | -37.4 | 84.6 | - |  |
| XGBoost 20 feats (sin delta_flow) | 0.6619 | 1.398 | 0.276 | -32.4 | 91.4 | 20 | t=8s |
| RandomForest 20 feats | 0.7021 | 1.312 | 0.268 | -35.1 | 87.7 | 20 | sub=300,000 |
| XGBoost 22 feats (con delta_flow) | 0.7898 | 1.102 | 0.202 | -7.3 | 125.3 | 22 | t=10s |

## Horizonte h=3

| Baseline | NSE | RMSE | MAE | ErrPico % | Peak pred | N features | Notas |
|---|---:|---:|---:|---:|---:|---:|---|
| **TCN v1 sinSF** (ref) | 0.4714 | 1.745 | 0.379 | -52.1 | 64.6 | 22 | n_test=165,221 |
| TCN v1 conSF (ref) | 0.4878 | 1.718 | 0.354 | -48.6 | 69.3 | 22+sf | - |
| Naive persistencia y(t) | 0.4094 | 1.848 | 0.197 | +0.0 | 135.2 | - |  |
| AR(1) analitico (rho + const) | 0.4962 | 1.707 | 0.280 | -28.3 | 96.9 | - | rho=0.7157 |
| AR(1) analitico (sin const) | 0.4944 | 1.710 | 0.185 | -28.4 | 96.7 | - |  |
| AR(5) lineal | 0.5012 | 1.698 | 0.268 | -28.2 | 97.1 | - |  |
| AR(12) lineal | 0.5085 | 1.686 | 0.255 | -33.1 | 90.4 | - |  |
| Lineal fisico (rain_60m+API) | 0.5055 | 1.691 | 0.349 | -43.5 | 76.3 | - |  |
| XGBoost 20 feats (sin delta_flow) | 0.5719 | 1.573 | 0.293 | -43.4 | 76.5 | 20 | t=9s |
| RandomForest 20 feats | 0.5867 | 1.546 | 0.293 | -40.4 | 80.6 | 20 | sub=300,000 |
| XGBoost 22 feats (con delta_flow) | 0.6558 | 1.411 | 0.243 | -25.2 | 101.1 | 22 | t=9s |

## Horizonte h=6

| Baseline | NSE | RMSE | MAE | ErrPico % | Peak pred | N features | Notas |
|---|---:|---:|---:|---:|---:|---:|---|
| **TCN v1 sinSF** (ref) | -1.2121 | 3.570 | 1.211 | -78.2 | 29.4 | 22 | n_test=165,218 |
| TCN v1 conSF (ref) | 0.2546 | 2.072 | 0.493 | -88.2 | 15.9 | 22+sf | - |
| Naive persistencia y(t) | 0.0808 | 2.305 | 0.272 | +0.0 | 135.2 | - |  |
| AR(1) analitico (rho + const) | 0.2911 | 2.024 | 0.404 | -44.2 | 75.4 | - | rho=0.5559 |
| AR(1) analitico (sin const) | 0.2868 | 2.031 | 0.249 | -44.4 | 75.1 | - |  |
| AR(5) lineal | 0.3053 | 2.004 | 0.383 | -47.8 | 70.5 | - |  |
| AR(12) lineal | 0.3170 | 1.987 | 0.367 | -53.6 | 62.8 | - |  |
| Lineal fisico (rain_60m+API) | 0.3064 | 2.002 | 0.426 | -49.5 | 68.2 | - |  |
| XGBoost 20 feats (sin delta_flow) | 0.3355 | 1.960 | 0.371 | -45.8 | 73.3 | 20 | t=8s |
| RandomForest 20 feats | 0.3096 | 1.998 | 0.386 | -50.2 | 67.3 | 20 | sub=300,000 |
| XGBoost 22 feats (con delta_flow) | 0.3819 | 1.890 | 0.330 | -52.1 | 64.7 | 22 | t=9s |

## Bias por bucket (H=1)

`bias = mean(y_pred - y_true)` dentro de cada rango de `y_true` (MGD). Negativo = subestima, positivo = sobrestima.

| Baseline | Base | Leve | Moderado | Alto | Extremo |
|---|---:|---:|---:|---:|---:|
| **TCN v1 sinSF** | +0.206 | +0.554 | +1.465 | +0.050 | -12.706 |
| Naive | +0.001 | +0.058 | +0.303 | -1.809 | -17.696 |
| AR(1) analitico | +0.044 | -0.038 | -0.691 | -4.623 | -22.375 |
| AR(5) | +0.044 | -0.034 | -0.745 | -4.529 | -20.864 |
| AR(12) | +0.036 | +0.024 | -0.652 | -4.646 | -21.256 |
| Fisico lineal | +0.146 | +0.521 | -0.225 | -8.800 | -34.746 |
| XGB-20 | +0.111 | +0.496 | -0.820 | -8.385 | -37.023 |
| RF-20 | +0.106 | +0.545 | -0.483 | -6.537 | -31.314 |
| XGB-22 | +0.056 | +0.160 | +0.193 | -1.856 | -21.779 |

## NSE por bucket (H=1)

NSE local dentro del bucket. NSE<0 indica que el modelo es peor que predecir la media del bucket.

| Baseline | Base | Leve | Moderado | Alto | Extremo |
|---|---:|---:|---:|---:|---:|
| **TCN v1 sinSF** | -14.513 | -0.256 | +0.191 | +0.021 | -0.986 |
| Naive | +0.804 | +0.793 | +0.355 | -3.224 | -2.853 |
| AR(1) | +0.570 | +0.833 | +0.443 | -2.919 | -2.889 |
| AR(5) | +0.552 | +0.836 | +0.493 | -2.690 | -2.857 |
| AR(12) | +0.622 | +0.804 | +0.502 | -2.652 | -2.860 |
| Fisico | -101.695 | -2.214 | +0.024 | -3.407 | -3.073 |
| XGB-20 | -21.562 | -3.275 | -0.289 | -3.548 | -3.839 |
| RF-20 | -18.953 | -3.259 | -0.302 | -2.628 | -2.709 |
| XGB-22 | -3.548 | -0.377 | -0.095 | -3.108 | -1.660 |

## Veredicto

Respuestas directas a las preguntas clave del diagnostico. Todos los numeros se refieren al test alineado (ventana de 72 pasos), sin tuning ni early stopping, y al target `stormflow_mgd(t+h)`.

### 1. XGBoost-20 vs TCN v1 (H=1)
- NSE XGB-20 (solo features exogenas, sin lags del target) = **0.6619**
- NSE TCN v1 sinSF (22 features, incluye `delta_flow` que SI deriva del target) = **0.8614**
- Diferencia TCN - XGB-20 = **+0.1995** NSE.

**Pero la comparacion justa NO es esta.** XGB-20 trabaja con cero informacion del propio caudal (ni siquiera `y(t)`), mientras que la TCN ve la ventana completa de 72 pasos. Comparar XGB-20 con TCN v1 sinSF es comparar un modelo "ciego al caudal" con uno que tiene memoria fluvial. El valor real del TCN frente a un GBM con la misma informacion temporal hay que medirlo con XGB-22 (que incluye `delta_flow_5m/15m`, los unicos proxies del target que la version "sinSF" se permite).

Resultado clave: **XGB-22 (NSE=0.7898)** queda **0.072 NSE por debajo** del TCN v1 sinSF (0.8614). Es decir, el TCN aporta valor arquitectonico real sobre un GBM con las mismas 22 features, pero el grueso de la calidad H=1 de la TCN ya es alcanzable con un GBM de 10 segundos de entrenamiento.

### 2. Peso real del atajo `delta_flow`
- NSE XGB-22 (con `delta_flow_*`) = **0.7898**
- NSE XGB-20 (sin `delta_flow_*`) = **0.6619**
- Contribucion del atajo en XGBoost = **+0.1279** NSE (de 0.66 a 0.79).
- NSE TCN v1 sinSF - NSE XGB-22 = **+0.072** (TCN sigue ganando cuando ambos ven `delta_flow`).

**Confirmado en un modelo no-TCN**: las dos features `delta_flow_5m` y `delta_flow_15m` aportan +0.13 NSE tambien en XGBoost. Esto valida cuantitativamente el hallazgo de iter16 (cuando se quitaron, el TCN cayo a niveles de naive). `delta_flow` es un proxy *muy* informativo del proximo paso: dado que predice `y(t+1)` con `seq[t-71..t]`, conocer la pendiente reciente del caudal es casi un proxy de `y(t+1)`.

Una nota incomoda: si se quita `delta_flow` *tambien* del TCN, lo previsible es que caiga al rango XGB-20 (NSE ~0.66). Y XGB-20 ya empata o pierde frente a baselines AR clasicos (ver punto 3). Es decir, el TCN actual *necesita* `delta_flow` para mantener la ventaja sobre AR(12).

### 3. XGBoost-20 vs baselines triviales (autorregresivos)
- XGB-20 - naive (y(t)) = **-0.149** NSE
- XGB-20 - AR(12)       = **-0.165** NSE

XGB-20 (sin features del target) **es claramente peor** que la persistencia. Esto NO significa que las features de lluvia no aporten nada en valor absoluto (el predictor fisico lineal alcanza 0.62), sino que **a H=1 la senal autorregresiva del caudal domina sobre la senal exogena**: con solo `y(t)` se obtiene NSE=0.811, mientras que con 20 features de lluvia/API y *cero* informacion del target solo se llega a 0.66. Para que un GBM gane a la persistencia hay que darle acceso al caudal reciente — exactamente lo que hace XGB-22 (NSE=0.79, ya por encima del naive).

Lo verdaderamente revelador: **AR(12) sobre solo 12 lags de `y` (NSE=0.827) supera a XGB-20 con 20 features de lluvia y supera tambien a XGB-22 (NSE=0.79)**. Es decir: a H=1, una regresion lineal con 12 valores pasados del caudal es mejor que XGBoost con 22 features (incluyendo lluvia y delta_flow). Esto sugiere que XGBoost esta perdiendo parte de la senal autorregresiva al no tener acceso explicito a los lags del target — los `delta_flow` son derivadas, no niveles, y no bastan.

### 4. Predictor fisico (2 features lineales) (H=1)
- NSE lineal(`rain_sum_60m` + `api_dynamic`) = **0.6186**

Resultado contundente: una **regresion lineal con dos features fisicas** alcanza NSE=0.62, casi tanto como XGBoost con 20 features (0.66). Confirma que (a) la senal lluvia->stormflow es real y aprendible incluso con un modelo trivial, (b) la mayor parte del valor predictivo a H=1 que aportan las 20 features no autorregresivas se concentra en `rain_sum_60m` + `api_dynamic`, y (c) XGBoost esta usando muy poca de la informacion adicional de las otras 18 features de lluvia.

### 5. Horizonte operativo
- Mejor NSE en H=1: **0.8614** (TCN v1 sinSF) ; mejor baseline puro: **0.8273** (AR(12))
- Mejor NSE en H=3: **0.6558** (XGB-22)         ; mejor baseline sin lags-de-target: **0.5719** (XGB-20)
- Mejor NSE en H=6: **0.3819** (XGB-22)         ; mejor sin atajo: **0.3355** (XGB-20)

Interpretacion: el techo de los baselines cae rapidamente con el horizonte. A **H=1** el problema es relativamente facil (varios baselines >0.8) pero *poco util operativamente* (5 minutos de antelacion). A **H=3** (15 min) hay degradacion fuerte: la mejor opcion sin TCN es XGB-22 con NSE=0.66, pero los TCN actuales (0.47-0.49) **estan por debajo de los baselines tabulares con las mismas features**. A **H=6** (30 min, el horizonte realmente util para CSO) el techo de baselines es 0.38, y el TCN v1 sinSF actual (NSE=-1.21) no llega ni a la persistencia.

**Conclusion operativa**: si se quiere mantener H=1, AR(12) ya da NSE=0.83 sin ML. Si se quiere ir a H=3, hay que pelear contra un techo de XGBoost-22 ~0.66, y la TCN actual lo pierde claramente. Para H=6, el TCN sinSF colapsa completamente (NSE negativo) — solo el TCN conSF (NSE=0.25) salva los muebles, pero su ErrPico=-88% lo hace inutilizable como alerta. **El proximo paso obligado es comparar el TCN entrenado a H=3 contra XGB-22 a H=3 con identico split**, porque hoy XGB-22 H=3 (0.66) > TCN v1 H=3 (0.47).


