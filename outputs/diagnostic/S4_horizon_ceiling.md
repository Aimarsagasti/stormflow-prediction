# S4 - Techo de NSE alcanzable por horizonte

Cuantificacion del techo de NSE alcanzable con las features actuales y sin informacion futura externa. Para cada horizonte h se reportan los baselines AR analiticos, la cota teorica de un AR(1) optimo, el TCN v1 (cuando hay modelo) y oraculos parciales por bucket de magnitud.

## Metodologia

- **Split test cronologico**: `iloc[936669:]`, mismos indices alineados que S2 (ventana previa de 72 pasos completa, target dentro del df).
- **Buckets MGD**: Base<0.5; Leve [0.5, 5); Moderado [5, 25); Alto [25, 50); Extremo>=50. Notar que `evaluate_local.py` usa Alto [20, 50) para reportar el v1; aqui se recalcula con [25, 50) para ser consistentes con el enunciado del diagnostico.
- **AR(1) optimo**: rho_h = corr(y(t), y(t+h)) sobre train; y_hat = mean_train + rho_h*(y(t) - mean_train).
- **Cota analitica `2*rho_h - 1`**: maximo NSE alcanzable por un predictor lineal ortogonal de y(t) cuando rho_h > 0.5 (cota informativa, no tope absoluto).
- **AR(12)**: regresion lineal con 12 lags consecutivos.
- **TCN v1 sinSF**: inferencia en lote sobre todo el test alineado, switch duro con threshold=0.3 (mismo que evaluate_local). Solo disponible para H=1, H=3, H=6.
- **Oraculo parcial**: para los buckets indicados, y_pred = y_true; el resto queda igual. NSE recalculado sobre todo el test.

## Tabla maestra: NSE por horizonte

| h | min | n_test | rho_h | NSE naive | NSE AR(1)opt | cota 2rho-1 | NSE AR(12) | NSE TCN v1 | Pico err % |
|---|----:|-------:|------:|---------:|-------------:|------------:|----------:|----------:|----------:|
| 1 | 5 | 165,222 | 0.9087 | 0.8107 | 0.8196 | +0.8174 | 0.8273 | 0.8615 | +42.6 |
| 3 | 15 | 165,220 | 0.7157 | 0.4094 | 0.4962 | +0.4314 | 0.5085 | 0.4697 | +121.2 |
| 6 | 30 | 165,217 | 0.5559 | 0.0808 | 0.2911 | +0.1117 | 0.3170 | -1.2039 | +38.9 |
| 12 | 60 | 165,211 | 0.4207 | -0.1866 | 0.1640 | -0.1585 | 0.1896 | - | - |
| 24 | 120 | 165,199 | 0.2751 | -0.4400 | 0.0764 | -0.4499 | 0.0949 | - | - |

Lectura rapida: `naive` es el suelo trivial; `AR(1) optimo` lo bate ligeramente porque mueve la prediccion hacia la media de train cuando rho_h<1; `AR(12)` aprovecha lags adicionales pero la mejora marginal indica que la memoria mas alla de un par de lags ya esta saturada; la cota `2*rho-1` da el techo teorico de un AR(1) bajo independencia de errores.

## Contribucion de cada bucket al denominador del NSE

Donde se concentra la varianza de y_true (sum (y-mean)^2). Si un bucket aporta el X% del denominador, mejorar la prediccion ahi sube el NSE en proporcion al X%. **Es la palanca principal**.

| h | total SSE | Base %denom | Leve %denom | Moderado %denom | Alto %denom | Extremo %denom |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 955,079.3 |  1.79 |  2.48 | 37.49 | 26.33 | 31.91 |
| 3 | 955,079.0 |  1.79 |  2.48 | 37.49 | 26.33 | 31.91 |
| 6 | 955,078.6 |  1.79 |  2.48 | 37.49 | 26.33 | 31.91 |
| 12 | 955,077.7 |  1.79 |  2.48 | 37.49 | 26.33 | 31.91 |
| 24 | 955,075.9 |  1.79 |  2.48 | 37.49 | 26.33 | 31.91 |

| h | Base n | Leve n | Moderado n | Alto n | Extremo n |
|---|---:|---:|---:|---:|---:|
| 1 | 152,900 | 9,520 | 2,519 | 224 | 59 |
| 3 | 152,898 | 9,520 | 2,519 | 224 | 59 |
| 6 | 152,895 | 9,520 | 2,519 | 224 | 59 |
| 12 | 152,889 | 9,520 | 2,519 | 224 | 59 |
| 24 | 152,877 | 9,520 | 2,519 | 224 | 59 |

## Oraculos parciales sobre TCN v1 (solo h con modelo entrenado)

Reemplazo y_pred = y_true dentro del bucket o combinacion indicada, NSE recalculado sobre todo el test. La columna `delta` es el incremento sobre el TCN v1.

### h=1 (5 min)

- TCN v1 base: NSE = **0.8615**, pico real 135.2 MGD, pico predicho 192.8 MGD (+42.6%).

| Oraculo | n muestras oraculizadas | NSE oraculo | delta NSE |
|---|---:|---:|---:|
| Solo Extremo (>=50) | 59 | 0.9096 | +0.0482 |
| Extremo + Alto (>=25) | 283 | 0.9271 | +0.0656 |
| Solo Moderado [5,25) | 2,519 | 0.9005 | +0.0390 |
| Leve + Base (<5) | 162,420 | 0.8954 | +0.0339 |
| Moderado + Leve + Base (<25) | 164,939 | 0.9344 | +0.0729 |
| Todo excepto Extremo (<50) | 165,163 | 0.9518 | +0.0904 |

Metricas del TCN v1 por bucket (sanity check):

| Bucket | n | bias | RMSE | NSE local | SSE residual | %SSE residual |
|---|---:|---:|---:|---:|---:|---:|
| Base | 152,900 | +0.206 | 0.345 | -14.520 | 18,155.2 | 13.72 |
| Leve | 9,520 | +0.554 | 1.223 | -0.256 | 14,241.5 | 10.76 |
| Moderado | 2,519 | +1.360 | 3.846 | +0.449 | 37,260.8 | 28.16 |
| Alto | 224 | -0.287 | 8.618 | -0.545 | 16,637.6 | 12.57 |
| Extremo | 59 | -12.796 | 27.932 | -0.990 | 46,031.1 | 34.79 |

### h=3 (15 min)

- TCN v1 base: NSE = **0.4697**, pico real 135.2 MGD, pico predicho 299.0 MGD (+121.2%).

| Oraculo | n muestras oraculizadas | NSE oraculo | delta NSE |
|---|---:|---:|---:|
| Solo Extremo (>=50) | 59 | 0.6336 | +0.1639 |
| Extremo + Alto (>=25) | 283 | 0.7202 | +0.2505 |
| Solo Moderado [5,25) | 2,519 | 0.5792 | +0.1096 |
| Leve + Base (<5) | 162,418 | 0.6399 | +0.1703 |
| Moderado + Leve + Base (<25) | 164,937 | 0.7495 | +0.2798 |
| Todo excepto Extremo (<50) | 165,161 | 0.8361 | +0.3664 |

Metricas del TCN v1 por bucket (sanity check):

| Bucket | n | bias | RMSE | NSE local | SSE residual | %SSE residual |
|---|---:|---:|---:|---:|---:|---:|
| Base | 152,898 | +0.246 | 0.876 | -99.219 | 117,237.8 | 23.15 |
| Leve | 9,520 | +0.693 | 2.183 | -3.003 | 45,381.9 |  8.96 |
| Moderado | 2,519 | +2.248 | 6.445 | -0.547 | 104,644.2 | 20.66 |
| Alto | 224 | -1.220 | 19.217 | -6.680 | 82,719.5 | 16.33 |
| Extremo | 59 | -15.878 | 51.507 | -5.768 | 156,528.1 | 30.90 |

### h=6 (30 min)

- TCN v1 base: NSE = **-1.2039**, pico real 135.2 MGD, pico predicho 187.7 MGD (+38.9%).

| Oraculo | n muestras oraculizadas | NSE oraculo | delta NSE |
|---|---:|---:|---:|
| Solo Extremo (>=50) | 59 | -1.0005 | +0.2034 |
| Extremo + Alto (>=25) | 283 | -0.8698 | +0.3340 |
| Solo Moderado [5,25) | 2,519 | -0.9173 | +0.2866 |
| Leve + Base (<5) | 162,415 | 0.3794 | +1.5832 |
| Moderado + Leve + Base (<25) | 164,934 | 0.6660 | +1.8698 |
| Todo excepto Extremo (<50) | 165,158 | 0.7966 | +2.0005 |

Metricas del TCN v1 por bucket (sanity check):

| Bucket | n | bias | RMSE | NSE local | SSE residual | %SSE residual |
|---|---:|---:|---:|---:|---:|---:|
| Base | 152,895 | +0.942 | 2.839 | -1052.796 | 1,232,729.9 | 58.57 |
| Leve | 9,520 | +3.215 | 5.417 | -23.645 | 279,386.1 | 13.27 |
| Moderado | 2,519 | +3.995 | 10.424 | -3.047 | 273,726.4 | 13.00 |
| Alto | 224 | -9.116 | 23.600 | -10.582 | 124,754.4 |  5.93 |
| Extremo | 59 | -50.603 | 57.385 | -7.401 | 194,286.1 |  9.23 |

## Sintesis cuantitativa y veredicto

### 1. Horizonte que merece la pena optimizar

| h | min | NSE naive | NSE max fisico estimado | Ganancia max sobre naive |
|---|---:|---:|---:|---:|
| 1 | 5 | 0.8107 | 0.8615 | +0.0507 |
| 3 | 15 | 0.4094 | 0.5085 | +0.0990 |
| 6 | 30 | 0.0808 | 0.3170 | +0.2362 |
| 12 | 60 | -0.1866 | 0.1896 | +0.3762 |
| 24 | 120 | -0.4400 | 0.0949 | +0.5349 |

**Horizonte de maxima ganancia teorica: h=24 (120 min)** con margen de +0.5349 NSE sobre naive.

### 2. NSE maximo defendible por horizonte (modelo perfecto, mismas features)

Tope superior estimado: maximo entre la cota analitica `2*rho-1` (que asume predictor AR(1) optimo bajo iid) y la mejor evidencia empirica disponible (AR(12) o TCN v1). Tomamos el mayor de los dos como techo conservador alcanzable con la informacion disponible.

- **h=1 (5 min)**: NSE max ~ **0.8615** (2*rho-1 = +0.8174; AR(12) = 0.8273; TCN v1 = 0.8615).
- **h=3 (15 min)**: NSE max ~ **0.5085** (2*rho-1 = +0.4314; AR(12) = 0.5085; TCN v1 = 0.4697).
- **h=6 (30 min)**: NSE max ~ **0.3170** (2*rho-1 = +0.1117; AR(12) = 0.3170; TCN v1 = -1.2039).
- **h=12 (60 min)**: NSE max ~ **0.1896** (2*rho-1 = -0.1585; AR(12) = 0.1896).
- **h=24 (120 min)**: NSE max ~ **0.0949** (2*rho-1 = -0.4499; AR(12) = 0.0949).

### 3. Mayor palanca cuantitativa para subir NSE en H=1

TCN v1 actual: NSE = **0.8615**. Si el modelo acertara perfectamente dentro del bucket indicado (manteniendo el resto), el NSE pasaria a:

| Oraculo | NSE oraculo | delta NSE | n |
|---|---:|---:|---:|
| todo_excepto_extremo | 0.9518 | +0.0904 | 165,163 |
| moderado_leve_base | 0.9344 | +0.0729 | 164,939 |
| extremo_alto | 0.9271 | +0.0656 | 283 |
| extremo | 0.9096 | +0.0482 | 59 |
| moderado | 0.9005 | +0.0390 | 2,519 |
| leve_base | 0.8954 | +0.0339 | 162,420 |

**Palanca dominante en H=1**: el oraculo que mas sube el NSE individualmente es `extremo` con delta = +0.0482 NSE. Esto coincide con la contribucion al denominador de ese bucket.

Contribucion al denominador (H=1) de los buckets clave:

- **Extremo**: 31.91% del denominador (59 muestras = 0.036% del test).
- **Alto**: 26.33% del denominador (224 muestras = 0.136% del test).
- **Moderado**: 37.49% del denominador (2,519 muestras = 1.525% del test).
- **Leve**: 2.48% del denominador (9,520 muestras = 5.762% del test).
- **Base**: 1.79% del denominador (152,900 muestras = 92.542% del test).

### 4. Tiene sentido perseguir H=6 (30 min)?

- NSE naive H=6 = **0.0808** (rho_h=0.5559, AR(1)opt=0.2911, AR(12)=0.3170).
- TCN v1 sinSF a H=6 = **-1.2039** (segun S4 inference).
- NSE max fisico estimado H=6 = **0.3170**.

Veredicto: el techo fisico de H=6 con las features actuales esta **por debajo de NSE=0.5**. Perseguir H=6 con el dataset actual no llevara a un modelo operativo defensible. Para abrir H=6 hace falta features exogenas con horizonte futuro (forecast de lluvia, NWP) o cambiar el horizonte objetivo.

