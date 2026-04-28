# Iter17 - Comparativa XGB+lags vs referencias

Tabla unica con filas de baselines, TCN v1 y las variantes XGB+lags entrenadas en iter17. Columnas H=1 derivadas del panel multi-bucket (`src/evaluation/metrics_panel.py`) sobre el mismo test alineado que S2 (n=165,222). H=3 incluye solo NSE global (el panel completo esta en `iter17_xgb_results.json` si se necesita detalle).

| Modelo | NSE H=1 | NSE H=3 | Err pico H=1 (%) | Bias Base H=1 (MGD) | NSE Extremo H=1 | recall@50 H=1 |
|---|---:|---:|---:|---:|---:|---:|
| naive (S2) | 0.8107 | 0.4094 | +0.0 | +0.001 | -2.853 | n/a |
| AR(12) (S2) | 0.8273 | 0.5085 | -6.1 | +0.036 | -2.860 | n/a |
| XGB-20 feats (S2) | 0.6619 | 0.5719 | -32.4 | +0.111 | -3.839 | n/a |
| XGB-22 con delta_flow (S2) | 0.7898 | 0.6558 | -7.3 | +0.056 | -1.660 | n/a |
| TCN v1 sinSF (§7.1) | 0.8615 | 0.4697 | +42.6 | n/a | n/a | n/a |
| xgb_lag6_feat10 [PRIMARIO] (iter17) | 0.8630 | 0.6871 | -5.9 | +0.026 | -1.727 | 0.652 |
| xgb_lag12_feat10 (iter17) | 0.8665 | 0.6955 | -12.9 | +0.026 | -1.798 | 0.565 |
| xgb_feat10_only (iter17) | 0.7157 | 0.6012 | -38.8 | +0.118 | -3.133 | 0.304 |
| xgb_lag12_only (iter17) | 0.7778 | 0.5285 | -54.1 | +0.030 | -3.568 | 0.435 |
| xgb_lag24_feat10 (iter17) | 0.8634 | n/a | -26.4 | +0.025 | -1.806 | 0.565 |

## Criterios de exito (§7.4 DIAGNOSTIC_REPORT)

- NSE H=1 >= 0.85: **0.8630** -> OK
- NSE H=3 >= 0.66: **0.6871** -> OK
- |Err pico H=1| < 21%: **-5.9%** (abs=5.9) -> OK
- Bias Base H=1 <= +0.05 MGD: **+0.026** -> OK

## Seleccion del modelo primario y atribucion de la mejora (H=1, NSE)

Modelo primario: **lag6+feat10** (NSE=0.8630). El reporte §7.2 proponia lag=12 como punto de partida; la ablacion mostro que lag=6 produce recall@50=0.652 frente a 0.565 de lag=12, con diferencia de NSE de solo -0.0034. Dado que recall@50 es la metrica operativa principal del MSD (alertar CSOs), se elige lag=6.

Atribucion de la mejora sobre el primario (lag6+feat10):
- feat10 only = 0.7157  ->  lags aportan +0.1473 NSE
- lag12 only  = 0.7778  ->  features aportan +0.0853 NSE
- lag12+feat10 = 0.8665  (referencia del reporte, NSE similar)
- lag24+feat10 = 0.8634  (ablacion: lags largos empeoran pico)

## Figuras asociadas

- `outputs/figures/iter17/hydrograph_extreme_event_H1.png`
- `outputs/figures/iter17/scatter_real_vs_pred_H1.png`
- `outputs/figures/iter17/peak_error_by_bucket_H1.png`

## Nota sobre definicion de buckets

Las filas de baselines (naive, AR(12), XGB-20, XGB-22) provienen de `outputs/diagnostic/S2_baselines.json` y usan la definicion de buckets de `scripts/diagnostic/s2_baselines.py`: Moderado=[5, 25) MGD, Alto=[25, 50) MGD. Las filas de iter17 usan `src/evaluation/metrics_panel.py` con la definicion de `evaluate_local.py`: Moderado=[5, 20) MGD, Alto=[20, 50) MGD. **Las columnas Bias Base (<0.5 MGD) y NSE Extremo (>=50 MGD) NO se ven afectadas por esta diferencia** y son directamente comparables entre filas. Las columnas NSE Moderado y NSE Alto (no mostradas en esta tabla) si difieren en definicion entre fuentes y no deben compararse directamente.
