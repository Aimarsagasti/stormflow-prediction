# Iter17 - Comparativa XGB+lags vs referencias

Tabla unica con filas de baselines, TCN v1 y las variantes XGB+lags entrenadas en iter17. Columnas H=1 derivadas del panel multi-bucket (`src/evaluation/metrics_panel.py`) sobre el mismo test alineado que S2 (n=165,222). H=3 incluye solo NSE global (el panel completo esta en `iter17_xgb_results.json` si se necesita detalle).

| Modelo | NSE H=1 | NSE H=3 | Err pico H=1 (%) | Bias Base H=1 (MGD) | NSE Extremo H=1 | recall@50 H=1 |
|---|---:|---:|---:|---:|---:|---:|
| naive (S2) | 0.8107 | 0.4094 | +0.0 | +0.001 | -2.853 | n/a |
| AR(12) (S2) | 0.8273 | 0.5085 | -6.1 | +0.036 | -2.860 | n/a |
| XGB-20 feats (S2) | 0.6619 | 0.5719 | -32.4 | +0.111 | -3.839 | n/a |
| XGB-22 con delta_flow (S2) | 0.7898 | 0.6558 | -7.3 | +0.056 | -1.660 | n/a |
| TCN v1 sinSF (§7.1) | 0.8615 | 0.4697 | +42.6 | n/a | n/a | n/a |
| xgb_lag12_feat10 (iter17) | 0.8665 | 0.6955 | -12.9 | +0.026 | -1.798 | 0.565 |
| xgb_feat10_only (iter17) | 0.7157 | 0.6012 | -38.8 | +0.118 | -3.133 | 0.304 |
| xgb_lag12_only (iter17) | 0.7778 | 0.5285 | -54.1 | +0.030 | -3.568 | 0.435 |
| xgb_lag6_feat10 (iter17) | 0.8630 | n/a | -5.9 | +0.026 | -1.727 | 0.652 |
| xgb_lag24_feat10 (iter17) | 0.8634 | n/a | -26.4 | +0.025 | -1.806 | 0.565 |

## Criterios de exito (§7.4 DIAGNOSTIC_REPORT)

- NSE H=1 >= 0.85: **0.8665** -> OK
- NSE H=3 >= 0.66: **0.6955** -> OK
- |Err pico H=1| < 21%: **-12.9%** (abs=12.9) -> OK
- Bias Base H=1 <= +0.05 MGD: **+0.026** -> OK

## Atribucion de la mejora (H=1, NSE)

- lag12+feat10 = **0.8665**  (primario)
- feat10 only = 0.7157  -> los lags aportan +0.1508 NSE
- lag12 only = 0.7778  -> las features aportan +0.0887 NSE
- lag6+feat10 = 0.8630  |  lag24+feat10 = 0.8634  (ablation de longitud de lags)

## Figuras asociadas

- `outputs/figures/iter17/hydrograph_extreme_event_H1.png`
- `outputs/figures/iter17/scatter_real_vs_pred_H1.png`
- `outputs/figures/iter17/peak_error_by_bucket_H1.png`
