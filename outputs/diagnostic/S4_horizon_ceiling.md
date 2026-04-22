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
| 1 | 5 | 165,222 | 0.9087 | 0.8107 | 0.8196 | +0.8174 | 0.8273 | 0.7388 | +42.6 |
| 3 | 15 | 165,220 | 0.7157 | 0.4094 | 0.4962 | +0.4314 | 0.5085 | 0.2930 | +121.2 |
| 6 | 30 | 165,217 | 0.5559 | 0.0808 | 0.2911 | +0.1117 | 0.3170 | -1.2988 | +38.9 |
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

- TCN v1 base: NSE = **0.7388**, pico real 135.2 MGD, pico predicho 192.8 MGD (+42.6%).

| Oraculo | n muestras oraculizadas | NSE oraculo | delta NSE |
|---|---:|---:|---:|
| Solo Extremo (>=50) | 59 | 0.8425 | +0.1038 |
| Extremo + Alto (>=25) | 283 | 0.8912 | +0.1525 |
| Solo Moderado [5,25) | 2,519 | 0.8171 | +0.0784 |
| Leve + Base (<5) | 162,420 | 0.7692 | +0.0304 |
| Moderado + Leve + Base (<25) | 164,939 | 0.8475 | +0.1088 |
| Todo excepto Extremo (<50) | 165,163 | 0.8962 | +0.1575 |

Metricas del TCN v1 por bucket (sanity check):

| Bucket | n | bias | RMSE | NSE local | SSE residual | %SSE residual |
|---|---:|---:|---:|---:|---:|---:|
| Base | 152,900 | +0.205 | 0.313 | -11.768 | 14,936.4 |  5.99 |
| Leve | 9,520 | +0.597 | 1.217 | -0.245 | 14,110.7 |  5.66 |
| Moderado | 2,519 | +1.712 | 5.451 | -0.107 | 74,839.5 | 30.00 |
| Alto | 224 | -1.935 | 14.408 | -3.317 | 46,502.7 | 18.64 |
| Extremo | 59 | -24.964 | 40.984 | -3.285 | 99,101.3 | 39.72 |

### h=3 (15 min)

- TCN v1 base: NSE = **0.2930**, pico real 135.2 MGD, pico predicho 299.0 MGD (+121.2%).

| Oraculo | n muestras oraculizadas | NSE oraculo | delta NSE |
|---|---:|---:|---:|
| Solo Extremo (>=50) | 59 | 0.5232 | +0.2302 |
| Extremo + Alto (>=25) | 283 | 0.6464 | +0.3533 |
| Solo Moderado [5,25) | 2,519 | 0.4839 | +0.1909 |
| Leve + Base (<5) | 162,418 | 0.4558 | +0.1628 |
| Moderado + Leve + Base (<25) | 164,937 | 0.6467 | +0.3536 |
| Todo excepto Extremo (<50) | 165,161 | 0.7698 | +0.4768 |

Metricas del TCN v1 por bucket (sanity check):

| Bucket | n | bias | RMSE | NSE local | SSE residual | %SSE residual |
|---|---:|---:|---:|---:|---:|---:|
| Base | 152,898 | +0.240 | 0.806 | -83.989 | 99,420.9 | 14.72 |
| Leve | 9,520 | +0.801 | 2.426 | -3.942 | 56,023.3 |  8.30 |
| Moderado | 2,519 | +2.780 | 8.507 | -1.695 | 182,286.4 | 27.00 |
| Alto | 224 | -3.664 | 22.917 | -9.922 | 117,646.6 | 17.42 |
| Extremo | 59 | -32.406 | 61.040 | -8.506 | 219,824.7 | 32.56 |

### h=6 (30 min)

- TCN v1 base: NSE = **-1.2988**, pico real 135.2 MGD, pico predicho 187.7 MGD (+38.9%).

| Oraculo | n muestras oraculizadas | NSE oraculo | delta NSE |
|---|---:|---:|---:|
| Solo Extremo (>=50) | 59 | -1.0864 | +0.2124 |
| Extremo + Alto (>=25) | 283 | -0.9388 | +0.3600 |
| Solo Moderado [5,25) | 2,519 | -0.9784 | +0.3204 |
| Leve + Base (<5) | 162,415 | 0.3196 | +1.6184 |
| Moderado + Leve + Base (<25) | 164,934 | 0.6400 | +1.9388 |
| Todo excepto Extremo (<50) | 165,158 | 0.7876 | +2.0864 |

Metricas del TCN v1 por bucket (sanity check):

| Bucket | n | bias | RMSE | NSE local | SSE residual | %SSE residual |
|---|---:|---:|---:|---:|---:|---:|
| Base | 152,895 | +0.942 | 2.837 | -1050.902 | 1,230,514.6 | 56.05 |
| Leve | 9,520 | +3.384 | 5.754 | -26.805 | 315,205.3 | 14.36 |
| Moderado | 2,519 | +3.600 | 11.022 | -3.525 | 306,019.1 | 13.94 |
| Alto | 224 | -11.645 | 25.086 | -12.087 | 140,966.3 |  6.42 |
| Extremo | 59 | -53.158 | 58.631 | -7.770 | 202,816.8 |  9.24 |

## Sintesis cuantitativa y veredicto

### 1. Horizonte que merece la pena optimizar

| h | min | NSE naive | NSE max fisico estimado | Ganancia max sobre naive |
|---|---:|---:|---:|---:|
| 1 | 5 | 0.8107 | 0.8273 | +0.0165 |
| 3 | 15 | 0.4094 | 0.5085 | +0.0990 |
| 6 | 30 | 0.0808 | 0.3170 | +0.2362 |
| 12 | 60 | -0.1866 | 0.1896 | +0.3762 |
| 24 | 120 | -0.4400 | 0.0949 | +0.5349 |

**Horizonte de maxima ganancia teorica: h=24 (120 min)** con margen de +0.5349 NSE sobre naive.

### 2. NSE maximo defendible por horizonte (modelo perfecto, mismas features)

Tope superior estimado: maximo entre la cota analitica `2*rho-1` (que asume predictor AR(1) optimo bajo iid) y la mejor evidencia empirica disponible (AR(12) o TCN v1). Tomamos el mayor de los dos como techo conservador alcanzable con la informacion disponible.

- **h=1 (5 min)**: NSE max ~ **0.8273** (2*rho-1 = +0.8174; AR(12) = 0.8273; TCN v1 = 0.7388).
- **h=3 (15 min)**: NSE max ~ **0.5085** (2*rho-1 = +0.4314; AR(12) = 0.5085; TCN v1 = 0.2930).
- **h=6 (30 min)**: NSE max ~ **0.3170** (2*rho-1 = +0.1117; AR(12) = 0.3170; TCN v1 = -1.2988).
- **h=12 (60 min)**: NSE max ~ **0.1896** (2*rho-1 = -0.1585; AR(12) = 0.1896).
- **h=24 (120 min)**: NSE max ~ **0.0949** (2*rho-1 = -0.4499; AR(12) = 0.0949).

### 3. Mayor palanca cuantitativa para subir NSE en H=1

TCN v1 actual: NSE = **0.7388**. Si el modelo acertara perfectamente dentro del bucket indicado (manteniendo el resto), el NSE pasaria a:

| Oraculo | NSE oraculo | delta NSE | n |
|---|---:|---:|---:|
| todo_excepto_extremo | 0.8962 | +0.1575 | 165,163 |
| extremo_alto | 0.8912 | +0.1525 | 283 |
| moderado_leve_base | 0.8475 | +0.1088 | 164,939 |
| extremo | 0.8425 | +0.1038 | 59 |
| moderado | 0.8171 | +0.0784 | 2,519 |
| leve_base | 0.7692 | +0.0304 | 162,420 |

**Palanca dominante en H=1**: el oraculo que mas sube el NSE individualmente es `extremo` con delta = +0.1038 NSE. Esto coincide con la contribucion al denominador de ese bucket.

Contribucion al denominador (H=1) de los buckets clave:

- **Extremo**: 31.91% del denominador (59 muestras = 0.036% del test).
- **Alto**: 26.33% del denominador (224 muestras = 0.136% del test).
- **Moderado**: 37.49% del denominador (2,519 muestras = 1.525% del test).
- **Leve**: 2.48% del denominador (9,520 muestras = 5.762% del test).
- **Base**: 1.79% del denominador (152,900 muestras = 92.542% del test).

### 4. Tiene sentido perseguir H=6 (30 min)?

- NSE naive H=6 = **0.0808** (rho_h=0.5559, AR(1)opt=0.2911, AR(12)=0.3170).
- TCN v1 sinSF a H=6 = **-1.2988** (segun S4 inference).
- NSE max fisico estimado H=6 = **0.3170**.

Veredicto: el techo fisico de H=6 con las features actuales esta **por debajo de NSE=0.5**. Perseguir H=6 con el dataset actual no llevara a un modelo operativo defensible. Para abrir H=6 hace falta features exogenas con horizonte futuro (forecast de lluvia, NWP) o cambiar el horizonte objetivo.

