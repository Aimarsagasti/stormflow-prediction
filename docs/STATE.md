# STATE.md - Estado actual del proyecto

**Ultima actualizacion:** 2026-04-29 (iter17 e iter18 cerrados, iter19 pendiente).
**Mantenido por:** Aimar (actualizar al final de cada sesion significativa).

---

## Sistema actual (post-diagnostico)

Tras el diagnostico de abril el sistema antiguo (TwoStageTCN v1) fue
descartado por dos razones confirmadas: el atajo autoregresivo
`delta_flow_5m/15m` y bugs estructurales en `TwoStageLoss`. El
`DIAGNOSTIC_REPORT` recomendo construir un sistema de dos componentes
basado en XGBoost.

**Componentes operativos (cerrados):**

| Componente | Estado | Modelo |
|---|---|---|
| A - Regresor stormflow H=1, H=3 | Cerrado, en `main` | `xgb_lag6_feat10` |
| B - Clasificador alerta H=6, H=12 | Cerrado, rama pendiente merge | 4 variantes XGB binario |
| C - Comparacion deep learning | Pendiente (iter19) | TCN limpio Bai 2018 |

---

## Componente A: regresor `xgb_lag6_feat10` (iter17)

**Rama:** `iter17-xgboost-lags` (mergeada a `main`).
**Codigo:** `src/models/xgboost_baseline.py`, `src/evaluation/metrics_panel.py`,
`notebooks/iter17_xgboost_lags.py`.

### Configuracion

- 6 lags del target + 10 features reducidas de S5.
- Hiperparametros del DIAGNOSTIC_REPORT §7.2 sin tunear:
  `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
  tree_method=hist, early_stopping_rounds=20, random_state=42`.
- Trabaja en MGD reales, sin pipeline de normalizacion del v1.

### Resultados (test, n=165.222 a H=1)

| Metrica | H=1 | H=3 |
|---|---:|---:|
| NSE | 0.8630 | 0.6871 |
| RMSE | - | - |
| Err pico (%) | -5.9 | -13.1 |
| Bias Base (MGD) | +0.026 | - |
| recall@50 | 0.652 | 0.304 |
| NSE Extremo | -1.727 | - |

### Decision sobre lag6 vs lag12

El reporte proponia lag=12 como punto de partida. La ablacion mostro que
lag=6 produce mejor recall@50 (0.652 vs 0.565) y mejor error pico (-5.9%
vs -12.9%) con NSE casi identico (-0.0035). Como recall@50 es la metrica
operativa principal, se eligio lag=6.

### Atribucion de la mejora (H=1 NSE)

- `feat10_only` = 0.7157 -> los lags aportan +0.1473 NSE.
- `lag12_only` = 0.7778 -> las features aportan +0.0853 NSE.
- Ambas palancas son necesarias.

### Cumplimiento de criterios §7.4

Los 4 se cumplen: NSE H=1 >= 0.85, NSE H=3 >= 0.66, |err pico H=1| < 21%,
bias Base <= +0.05.

---

## Componente B: clasificador binario (iter18)

**Rama:** `iter18-xgboost-classifier` (pendiente de merge a `main`).
**Codigo:** `src/models/xgboost_classifier.py`,
`src/evaluation/classification_panel.py`,
`notebooks/iter18_xgboost_classifier.py`.

### Configuracion

- Target operacional: `max(stormflow[t+1..t+h]) >= U`.
  Esta formulacion (ventana completa, no instante puntual) la decidio el
  usuario tras que Codex la senalara como ambiguedad respecto a §7.5.
- Mismas features que el regresor: 6 lags + 10 features de S5.
- `objective='binary:logistic'`, `eval_metric='aucpr'`,
  `scale_pos_weight = neg_train / pos_train`, resto igual al regresor.
- Umbral operativo: el mayor que cumple `recall >= 0.85` en val.

### Resultados (test)

| Variante | Prevalencia | AUC-PR | ROC-AUC | P@op | R@op | Lead time mediano |
|---|---:|---:|---:|---:|---:|---:|
| h6_u25 | 0.41% | 0.5842 | 0.9806 | 0.102 | 0.884 | 25 min |
| h6_u50 | 0.12% | 0.2488 | 0.9678 | 0.007 | 0.949 | 30 min |
| h12_u25 | 0.66% | 0.4180 | 0.9397 | 0.042 | 0.830 | 60 min |
| h12_u50 | 0.21% | 0.2050 | 0.9187 | 0.008 | 0.901 | 60 min |

### Lectura honesta

- ROC-AUC alto (0.92-0.98): el modelo si distingue eventos de no-eventos.
- Lead times utiles: 25-60 min antes del rebasamiento real.
- Precision baja (0.7-10%): consecuencia directa de la prevalencia minuscula
  y del umbral operativo bajo necesario para mantener recall >= 0.85.
- En h6_u50 y h12_u50 el numero absoluto de FP es alto (26.000-41.000 en
  test). El reporte ya advertia en §8.3 que esto no invalida el TFM.
- Recomendacion para el documento academico: reportar curva PR completa,
  no un unico umbral operativo.

---

## Componente C: comparacion deep learning (iter19, pendiente)

Implementa el paso 7.6 del DIAGNOSTIC_REPORT. Es la ultima iteracion de
modelado del proyecto.

### Especificacion

- TCN estandar (Bai 2018), sin two-stage, sin switch duro, sin loss
  compuesta.
- Mismos inputs que el regresor primario: 6 lags + 10 features.
- Mismo split temporal que iter17 e iter18.
- Loss Huber simple.
- Solo H=1 inicialmente. Si gana, se replica a H=3.
- Entrenamiento en Colab Pro con T4 GPU.

### Criterio de cierre

- `NSE_TCN - NSE_XGB >= 0.02` -> TCN entra como modelo principal en TFM.
- `NSE_TCN - NSE_XGB <  0.02` -> deep learning se cierra para el TFM,
  XGBoost queda definitivo, y la seccion comparativa documenta esto como
  hallazgo academicamente valido.

### Decisiones pendientes (a tomar al inicio de iter19)

- Tamano de la TCN. La v1 tenia 104K params, claramente oversize. Para 6
  lags + 10 features Bai 2018 sugiere 3 bloques con dilations [1,2,4],
  32-64 canales, kernel_size=3, dropout=0.1 como punto de partida.
- Sequence length: 6 (igual que el regresor) o 72 (como v1).
- Normalizacion: NO usar el pipeline viejo (`src/pipeline/normalize.py`
  tiene bug latente BUG2 documentado en S1). Usar StandardScaler externo.

### Tiempo estimado

3-5 dias.

---

## Componentes deprecated (NO usar en iteraciones nuevas)

- `src/models/tcn.py` (TwoStageTCN v1).
- `src/models/loss.py` (CompositeLoss y TwoStageLoss).
- `src/pipeline/normalize.py` (bug latente documentado).
- `evaluate_local.py` (referencia historica solo).

Quedan en el repo por trazabilidad academica (tabla comparativa del TFM:
"modelo aparente -> diagnostico -> modelo real"), pero no se ejecutan ni se
modifican.

---

## Calendario hasta entrega

| Fase | Periodo | Entregable |
|---|---|---|
| iter17 | abril | Regresor cerrado |
| iter18 | abril | Clasificador cerrado |
| iter19 | mayo (3-5 dias) | TCN limpio + decision deep learning |
| Escritura TFM | junio-julio | Secciones 5, 6, 7, 8 |
| Revision y figuras | agosto | TFM completo |
| Entrega | septiembre 2026 | TFM final |

**Regla acordada:** no empezar iter20 ni iteraciones adicionales tras
iter19. Tras iter19, escritura.

---

## Bloqueos actuales

Ninguno. iter19 puede empezar cuando el usuario quiera.

---

## Referencias rapidas

- DIAGNOSTIC_REPORT: `outputs/diagnostic/DIAGNOSTIC_REPORT.md`.
- Resultados iter17: `outputs/diagnostic/iter17_comparison.md`.
- Resultados iter18: `outputs/diagnostic/iter18_comparison.md`.
- Bugs latentes: `outputs/diagnostic/S1_pipeline_audit.md`.
- Techo fisico: `outputs/diagnostic/S4_horizon_ceiling.md`.
- Justificacion features: `outputs/diagnostic/S5_feature_analysis.md`.
- Handoff de la transicion entre conversaciones: `HANDOFF_2026-04-29.md`.
