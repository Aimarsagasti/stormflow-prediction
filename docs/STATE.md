# STATE.md - Estado actual del proyecto

**Ultima actualizacion:** 2026-05-05 (iter19/19b/19c cerrados, modelado completo).
**Mantenido por:** Aimar (actualizar al final de cada sesion significativa).

---

## Sistema actual (post-diagnostico)

Tras el diagnostico de abril el sistema antiguo (TwoStageTCN v1) fue
descartado por dos razones confirmadas: el atajo autoregresivo
`delta_flow_5m/15m` y bugs estructurales en `TwoStageLoss`. El
`DIAGNOSTIC_REPORT` recomendo construir un sistema de dos componentes
basado en XGBoost. iter19 (TCN limpia Bai 2018) cerro la pregunta del
deep learning con resultado XGB_WINS en H=3, lo que mantiene a XGBoost
como modelo principal definitivo del TFM.

**Componentes operativos (todos cerrados):**

| Componente | Estado | Modelo |
|---|---|---|
| A - Regresor stormflow H=1, H=3 | Cerrado, en `main` | `xgb_lag6_feat10` |
| B - Clasificador alerta H=6, H=12 | Cerrado, mergeado a `main` | 4 variantes XGB binario |
| C - Comparacion deep learning | Cerrado, mergeado a `main` | TCN limpia A0 (descartada como modelo principal) |

**Fase actual del proyecto:** modelado completo, escritura del TFM en curso.

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

**Rama:** `iter18-xgboost-classifier` (mergeada a `main`).
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

## Componente C: comparacion deep learning (iter19/19b/19c)

**Rama:** `iter19-tcn-comparison` (mergeada a `main` en mayo 2026).
**Codigo:** `src/models/tcn_clean.py`, `src/pipeline/normalize_v2.py`,
`notebooks/iter19_tcn_clean.py`, `notebooks/iter19b_a0_test.py`,
`notebooks/iter19c_a0_h3.py`.

Implementa el paso 7.6 del DIAGNOSTIC_REPORT. Tres sub-iteraciones:
ablacion en H=1 (iter19), re-evaluacion de A0 con log1p en test (iter19b),
y replica a H=3 con A0 (iter19c).

### Configuracion final (TCN A0 limpia Bai 2018)

- Arquitectura: 4 bloques residuales con dilations `[1,2,4,8]`,
  `kernel_size=3`, `hidden_channels=32`, `dropout=0.1`.
- Receptive field = 61 pasos sobre ventana L=72.
- Inputs: 11 canales (1 target historico + 10 features de S5),
  shape `(B, T=72, F=11)`.
- Normalizacion: z-score sobre train, `log1p` sobre target.
- Loss: `HuberLoss(delta=1.0)` en espacio normalizado.
- Optimizer: AdamW lr=1e-3, wd=1e-4, batch=256, max_epochs=50, patience=10.
- Mismo split temporal e indices alineados que iter17 (n_test=165.222).
- 23.489 parametros.

### Mini-ablacion en val (iter19, 4 corridas + A4 extra)

| Run | L | C | log1p | NSE_val | err_pico_val | Notas |
|---|---:|---:|:---:|---:|---:|---|
| A0 | 72 | 32 | si | 0.8436 | +1.5% | baseline equilibrada |
| A1 | 72 | 32 | no | 0.8617 | -24.9% | gana en NSE val pero infraestima picos |
| A2 | 144 | 32 | si | 0.8146 | +6.7% | empeora con ventana doble |
| A3 | 72 | 64 | si | 0.8497 | -23.3% | empeora con C=64 + log1p |
| A4 | 72 | 64 | no | 0.8640 | -25.3% | combinacion ganadora por independencia |

La regla "max NSE_val por independencia de factores" eligio A4 inicialmente.

### Comparacion completa H=1 y H=3 en test

| Metrica | TCN A0 H=1 | XGB H=1 | TCN A0 H=3 | XGB H=3 |
|---|---:|---:|---:|---:|
| NSE global | 0.8890 | 0.8631 | 0.6096 | 0.6889 |
| NSE Base | +0.732 | similar | +0.711 | -4.37 |
| NSE Extremo | -1.571 | -1.808 | -8.064 | -5.553 |
| Err pico (%) | +6.6 | -13.5 | -53.2 | -13.5 |
| recall@50 | 0.741 | 0.609 | 0.087 | 0.261 |

### Veredicto y decision final

- **H=1:** TCN A0 supera marginalmente a XGB (`delta_NSE = +0.0259`,
  por encima del umbral +0.020 del DIAGNOSTIC_REPORT §7.6).
- **H=3:** TCN A0 PIERDE contra XGB (`delta_NSE = -0.0775`, MUY por
  debajo del umbral). La TCN colapsa a horizonte mayor: best_epoch=3,
  recall@50 cae a 0.087, error pico -53.2%.

**Decision final del TFM:** **XGBoost queda como modelo principal unico
en H=1 y H=3.** Razones:

1. La ventaja de TCN en H=1 es marginal (+0.026 NSE) y no se replica a H=3.
2. La TCN en H=3 es operativamente inservible (recall 8.7% vs 26.1% XGB).
3. Mantener un modelo unico (XGB) preserva coherencia narrativa,
   simplicidad operativa y menor coste de inferencia.
4. La narrativa del TFM se cierra como hallazgo academicamente fuerte:
   con los inputs disponibles (1 estacion pluviometrica, sin pronostico,
   1 cuenca), el GBM con lags explicitos absorbe la senal predecible
   disponible. El deep learning aportaria valor solo enriqueciendo la
   base de datos con: (a) datos espaciales de lluvia (radar NEXRAD),
   (b) integracion con pronostico meteorologico (HRRR/RAP del NWS), o
   (c) entrenamiento multi-estacion con datos fisicos de cuenca.

### Hallazgos tecnicos colaterales (para discusion del TFM)

- log1p en target es esencial tambien en la TCN limpia (confirma
  leccion 3 historica). Sin log1p, NSE_Base colapsa de 0.732 a 0.085.
- Las dos arquitecturas tienen sesgos opuestos: TCN aprende la "media"
  del proceso (flujo base), XGB aprende la "varianza" (eventos).
  Ninguna es estrictamente superior a la otra; coexisten.
- La ventaja de TCN en H=1 viene parcialmente de un solo evento extremo
  bien clavado (28 julio 2024). Resultado con varianza alta entre seeds.

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
| iter19/19b/19c | mayo | TCN limpia + decision deep learning cerrada |
| Escritura TFM | mayo-julio | Secciones 5, 6, 7, 8, 9 |
| Revision y figuras | agosto | TFM completo |
| Entrega | septiembre 2026 | TFM final |

**Regla acordada:** modelado cerrado tras iter19c. No habra mas iteraciones
de modelo. A partir de aqui, escritura.

---

## Bloqueos actuales

Ninguno. Modelado completo. Siguiente fase: TFM escrito.

---

## Referencias rapidas

- DIAGNOSTIC_REPORT: `outputs/diagnostic/DIAGNOSTIC_REPORT.md`.
- Resultados iter17: `outputs/diagnostic/iter17_comparison.md`.
- Resultados iter18: `outputs/diagnostic/iter18_comparison.md`.
- Resultados iter19 (H=1, ablacion): `outputs/diagnostic/iter19_comparison.md`.
- Resultados iter19b (A0 vs A4 en H=1): `outputs/diagnostic/iter19b_a0_vs_a4.md`.
- Resultados iter19c (A0 H=3 vs XGB): `outputs/diagnostic/iter19c_a0_h3_comparison.md`.
- Bugs latentes: `outputs/diagnostic/S1_pipeline_audit.md`.
- Techo fisico: `outputs/diagnostic/S4_horizon_ceiling.md`.
- Justificacion features: `outputs/diagnostic/S5_feature_analysis.md`.
- Handoff de transicion entre conversaciones: `HANDOFF_2026-04-29.md`.