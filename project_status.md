# Project Status - 13 abril 2026

## Qué es esto
Modelo TCN (TwoStageTCN) que predice stormflow (MGD) para el MSD de Cincinnati. 6 modelos entrenados: 3 horizontes (H=1,3,6 = 5,15,30 min) x 2 variantes (sinSF = solo lluvia, conSF = con stormflow como feature).

## Resultados actuales (del JSON en outputs/data_analysis/local_eval_metrics.json)

| Modelo | NSE | RMSE | MAE | ErrPico% | BiasBase | BiasExt |
|--------|-----|------|-----|----------|----------|---------|
| H1_sinSF | 0.819 | 1.02 | 0.31 | +12.8% | +0.22 | -3.7 |
| H1_conSF | 0.853 | 0.92 | 0.24 | +59.4% | +0.18 | -8.8 |
| H3_sinSF | 0.536 | 1.64 | 0.39 | -48.2% | +0.26 | -25.8 |
| H3_conSF | 0.488 | 1.72 | 0.35 | -48.5% | +0.24 | -20.4 |
| H6_sinSF | -0.447 | 2.89 | 0.81 | -92.6% | +0.53 | -53.3 |
| H6_conSF | 0.255 | 2.08 | 0.49 | -88.2% | +0.28 | -57.5 |

## Datos clave
- Test set: ~165K muestras, periodo 2024-07 a 2026-01
- 59 eventos extremos (>50 MGD) en test, de los cuales ~15 NO tienen lluvia en la ventana de entrada
- 92% del dataset es flujo base (<0.5 MGD)
- El modelo funciona bien en eventos moderados (5-20 MGD), mal en extremos
- Los eventos sin lluvia son probablemente deshielo o escorrentía retardada, no fallos del modelo

## Arquitectura
- TwoStageTCN: clasificador binario + regresor con switch duro (prob >= 0.3 → usa regresor, si no → 0)
- 5 bloques residuales [32,64,64,64,32], dilations [1,2,4,8,16], GroupNorm
- seq_length=72 (6h), ~104K parámetros
- Loss: TwoStageLoss (0.3 BCE + 0.7 Huber asimétrico)

## Checkpoints
En `MC-CL-005/Pesos 13-04-2026/`:
- `modelo_H{1,3,6}_{sinSF,conSF}_weights.pt`
- `modelo_H{1,3,6}_{sinSF,conSF}_norm_params.json`
- `modelo_H{1,3,6}_{sinSF,conSF}_meta.json`

## Features
22 features (sinSF) o 23 (conSF = +stormflow_mgd):
rain_in, temp_daily_f, api_dynamic, rain_sum_{10,15,30,60,120,180,360}m, rain_max_{10,30,60}m, minutes_since_last_rain, delta_flow_{5,15}m, delta_rain_{10,30}m, hour_{sin,cos}, month_{sin,cos}

## Qué ya genera evaluate_local.py
- Métricas globales y por rangos → JSON
- Scatter predicho vs real (por modelo)
- Zoom a top 3 picos (por modelo)
- NSE por rango de magnitud (barplot comparativo)
- Curva de predicibilidad (NSE vs horizonte)
- Análisis extremos con/sin lluvia → JSON + scatter

## Qué FALTA (la tarea de este prompt)
Ver archivo prompt_codex.md
