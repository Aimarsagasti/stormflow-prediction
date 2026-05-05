# Iter19 - TCN limpia (Bai 2018) vs xgb_lag6_feat10

## Resumen ejecutivo

Configuracion ganadora: **A4** (L=72, C=64, log1p_target=False). NSE TCN test H=1 = **0.8983**. NSE XGB iter17 referencia = 0.8630. Delta = **+0.0353** (umbral cierre = +0.020). VEREDICTO: **TCN_WINS**.

## Tabla de ablacion en val

| Run | L | C | log1p | NSE_val | RMSE_val | err_pico_val (%) | recall@50_val | best_epoch | train_s |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| A0 | 72 | 32 | sí | 0.8436 | 1.132 | +1.5 | 0.741 | 15 | 995 |
| A1 | 72 | 32 | no | 0.8617 | 1.065 | -24.9 | 0.741 | 12 | 863 |
| A2 | 144 | 32 | sí | 0.8146 | 1.233 | +6.7 | 0.815 | 7 | 717 |
| A3 | 72 | 64 | sí | 0.8497 | 1.110 | -23.3 | 0.630 | 25 | 1398 |
| A4 | 72 | 64 | no | 0.8640 | 1.056 | -25.3 | 0.741 | 7 | 683 |

## Justificacion de la config ganadora

- **log1p_target = False** — A0 (log1p=True) NSE_val=0.8436 vs A1 (log1p=False) NSE_val=0.8617. Delta=-0.0181. Eje resuelto a favor de log1p_target=False.
- **L = 72** — A0 (L=72) NSE_val=0.8436 vs A2 (L=144) NSE_val=0.8146. Delta=+0.0290. Eje resuelto a favor de L=72.
- **C = 64** — A0 (C=32) NSE_val=0.8436 vs A3 (C=64) NSE_val=0.8497. Delta=-0.0061. Eje resuelto a favor de C=64.

La combinacion ganadora (False, L=72, C=64) no coincide con A0..A3 ya entrenadas, por lo que se entreno A4 con esta config para validar empiricamente la seleccion por independencia de factores.

## Resultados del modelo final en test

| Metrica | Valor |
|---|---:|
| n_test | 165222 |
| NSE | 0.8983 |
| RMSE | 0.767 |
| MAE | 0.088 |
| peak_real (MGD) | 135.15 |
| peak_pred (MGD) | 141.45 |
| peak_err_pct | +4.7 |
| recall@25 | 0.895 |
| recall@50 | 0.783 |
| n_params | 89,985 |
| best_epoch | 18 |
| training_time_s | 1117 |

### NSE por bucket (test)

| Bucket | n | NSE | RMSE | bias (MGD) | peak_err_pct |
|---|---:|---:|---:|---:|---:|
| Base | 152902 | +0.085 | 0.084 | +0.002 | +1371.2 |
| Leve | 9518 | +0.765 | 0.529 | +0.136 | +150.6 |
| Moderado | 2303 | +0.527 | 2.780 | +0.350 | +135.8 |
| Alto | 440 | +0.005 | 7.392 | -1.045 | +20.7 |
| Extremo | 59 | -1.231 | 29.573 | -13.739 | +4.7 |

## Comparacion directa contra xgb_lag6_feat10

Ambos modelos evaluados sobre los mismos timestamps de test (L_winner=72, offset XGB=0 pasos).

| Modelo | NSE | err_pico (%) | bias_base (MGD) | recall@50 | NSE_extremo |
|---|---:|---:|---:|---:|---:|
| xgb_lag6_feat10 (inline) | 0.8631 | -12.4 | +0.028 | 0.609 | -1.808 |
| TCN (A4) | 0.8983 | +4.7 | +0.002 | 0.783 | -1.231 |
| **delta (TCN - XGB inline)** | **+0.0352** | +17.0 | -0.026 | +0.174 | +0.577 |

NSE de referencia oficial iter17 (xgb_lag6_feat10, n=165222): **0.8630**.
Delta usado para el veredicto: TCN(0.8983) - XGB_iter17(0.8630) = **+0.0353**.
Umbral de cierre del DIAGNOSTIC_REPORT §7.6: +0.020.

## Veredicto

**TCN gana** por margen >= 0.020 NSE. La TCN limpia (Bai 2018) supera al regresor XGBoost por +0.0353 NSE en H=1. Pasa a ser el modelo principal candidato para el TFM. Pendiente replicar a H=3 en una sesion posterior antes de sustituir definitivamente a `xgb_lag6_feat10`.

## Lectura honesta para el TFM

Esta es la unica iteracion de deep learning post-diagnostico, intencionalmente limitada (una seed, ablacion de 4 corridas, Huber loss simple) para evitar sobre-ingenieria que enmascare el resultado real. El umbral de cierre +0.020 NSE es arbitrario pero defendible: del orden del ruido entre configuraciones razonables, y muy por debajo del techo fisico H=1 ~0.86 derivado del analisis S4. 
El resultado favorece deep learning, pero la diferencia debe interpretarse a la luz de la complejidad anadida (GPU, dependencia PyTorch, mayor coste de inferencia) frente al beneficio operativo. Para uso real del MSD el coste/beneficio de mantener una TCN en produccion frente a un GBM debe documentarse explicitamente.

## Limitaciones reconocidas

- Una sola seed (=42). No se reportan barras de error sobre la NSE final.
- Ablacion minima (4 corridas) con seleccion por independencia de factores. Posibles interacciones entre log1p y L, o entre C y dropout, no se exploran.
- Loss Huber simple en espacio normalizado, sin componente de magnitud ni peak penalty. Variantes con loss asimetrica podrian mejorar el bucket Extremo, pero quedan fuera de scope por la spec.
- Cuando L>72 los origins de val/test se recortan en 0 pasos para no cruzar frontera de splits. Esto cambia ligeramente n_test respecto a iter17 (n=165222 con L=72).
- No se ha replicado a H=3 en esta iteracion. Si TCN gana, queda pendiente esa replica antes de cambiar el modelo principal en STATE.md y EXPERIMENTS.md.
