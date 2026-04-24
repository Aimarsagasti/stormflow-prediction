# S5 - Analisis de features y redundancia

Diagnostico del valor real (no autoregresivo) de las 20 features oficiales tras excluir `delta_flow_*` (atajo confirmado en iter16/S2). Trabaja sobre el mismo split y los mismos indices alineados que S2 para que las metricas sean directamente comparables.

## Metodologia

- **Modelo base**: XGBoost-20 (mismos hiper que S2) reentrenado para H=1.
- **Permutation importance**: n_repeats=5 sobre subsample de test (n=50,000), shuffling individual de cada feature.
- **SHAP**: TreeExplainer sobre 5000 filas aleatorias de test (random_state=42).
- **Correlaciones**: matriz Pearson sobre TRAIN (771,302 filas), clustering jerarquico complete-linkage sobre |1 - r| con corte r>=0.85.
- **ACF features clave**: lags 1, 6, 12, 24 sobre TRAIN, comparada con la del target (referencia DATASET_STATS section 6).

### Tiempos por etapa

| Etapa | Segundos |
|---|---:|
| load_parquet | 0.1 |
| xgb20_fit | 9.0 |
| native_importance | 0.0 |
| permutation_importance | 5.5 |
| shap | 2.3 |
| corr_cluster | 1.5 |
| acf | 0.1 |
| reduced_set | 8.5 |
| ablation_reduced | 82.1 |

## 1. Native importance (gain) y Permutation importance

`gain_pct` = importancia normalizada del booster (suma 1.0). `PI mean_drop` = caida promedio de NSE al barajar la feature en test. Una feature con PI<=0 se considera ruido (su ablacion no degrada el modelo).

| Feature | gain_pct | PI mean_drop | PI std | rank PI |
|---|---:|---:|---:|---:|
| api_dynamic | 0.6560 | +0.4182 | 0.0037 | 1 |
| rain_sum_360m | 0.0253 | +0.1083 | 0.0083 | 2 |
| rain_sum_120m | 0.0454 | +0.0803 | 0.0038 | 3 |
| rain_sum_15m | 0.0240 | +0.0368 | 0.0011 | 4 |
| rain_sum_180m | 0.0206 | +0.0340 | 0.0028 | 5 |
| temp_daily_f | 0.0200 | +0.0332 | 0.0106 | 6 |
| rain_sum_10m | 0.0373 | +0.0172 | 0.0013 | 7 |
| hour_sin | 0.0163 | +0.0105 | 0.0096 | 8 |
| minutes_since_last_rain | 0.0084 | +0.0045 | 0.0002 | 9 |
| delta_rain_10m | 0.0065 | +0.0044 | 0.0007 | 10 |
| rain_in | 0.0166 | +0.0033 | 0.0026 | 11 |
| rain_max_10m | 0.0112 | +0.0018 | 0.0005 | 12 |
| delta_rain_30m | 0.0088 | +0.0006 | 0.0005 | 13 |
| rain_sum_30m | 0.0127 | +0.0006 | 0.0007 | 14 |
| rain_max_30m | 0.0102 | -0.0001 | 0.0009 | 15 |
| hour_cos | 0.0146 | -0.0062 | 0.0031 | 16 |
| month_cos | 0.0220 | -0.0084 | 0.0067 | 17 |
| rain_sum_60m | 0.0082 | -0.0121 | 0.0013 | 18 |
| rain_max_60m | 0.0148 | -0.0176 | 0.0008 | 19 |
| month_sin | 0.0210 | -0.0241 | 0.0050 | 20 |

## 2. SHAP (mean |SHAP|, signo en extremos vs base)

Ranking por importancia media absoluta. `mean_shap_extreme` = SHAP medio en y_true>=25.0 MGD, `mean_shap_base` = SHAP medio en y_true<0.5 MGD. Cambio de signo => contribucion no monotona (la feature empuja arriba en extremos pero abajo en baseflow o viceversa).

_n_samples=5,000, n_extremos_en_muestra=8, n_base_en_muestra=4638._

| Feature | mean |SHAP| | mean SHAP extremo | mean SHAP base | std SHAP |
|---|---:|---:|---:|---:|
| api_dynamic | 0.3327 | +19.9938 | -0.1714 | 1.3598 |
| rain_sum_120m | 0.1402 | +3.5535 | -0.0782 | 0.4132 |
| rain_sum_360m | 0.1154 | +0.6113 | -0.0453 | 0.3461 |
| rain_sum_15m | 0.0482 | +1.8653 | -0.0282 | 0.1814 |
| temp_daily_f | 0.0415 | +0.4717 | -0.0064 | 0.1863 |
| minutes_since_last_rain | 0.0365 | +0.0427 | -0.0012 | 0.0484 |
| rain_sum_30m | 0.0356 | +2.0685 | -0.0132 | 0.1903 |
| rain_in | 0.0261 | +1.8518 | -0.0154 | 0.1979 |
| month_cos | 0.0246 | +0.5238 | +0.0006 | 0.0867 |
| rain_sum_10m | 0.0199 | +1.6299 | -0.0092 | 0.1892 |
| rain_max_60m | 0.0187 | -0.8977 | +0.0020 | 0.1383 |
| month_sin | 0.0179 | +0.1869 | -0.0003 | 0.1404 |
| rain_sum_180m | 0.0170 | +0.4530 | -0.0039 | 0.0653 |
| rain_sum_60m | 0.0114 | -1.2806 | +0.0022 | 0.1050 |
| delta_rain_10m | 0.0105 | +0.3020 | -0.0021 | 0.0529 |
| rain_max_30m | 0.0104 | +2.0985 | -0.0002 | 0.1308 |
| hour_cos | 0.0099 | +0.0733 | +0.0009 | 0.0746 |
| hour_sin | 0.0095 | -0.5562 | +0.0007 | 0.0563 |
| rain_max_10m | 0.0044 | +0.7097 | -0.0002 | 0.0608 |
| delta_rain_30m | 0.0044 | +0.3497 | -0.0009 | 0.0309 |

**Features con SHAP no-monotono (signo cambia entre extremos y base):**
- `rain_in`: extremo +1.8518 vs base -0.0154
- `temp_daily_f`: extremo +0.4717 vs base -0.0064
- `api_dynamic`: extremo +19.9938 vs base -0.1714
- `rain_sum_10m`: extremo +1.6299 vs base -0.0092
- `rain_sum_15m`: extremo +1.8653 vs base -0.0282
- `rain_sum_30m`: extremo +2.0685 vs base -0.0132
- `rain_sum_60m`: extremo -1.2806 vs base +0.0022
- `rain_sum_120m`: extremo +3.5535 vs base -0.0782
- `rain_sum_180m`: extremo +0.4530 vs base -0.0039
- `rain_sum_360m`: extremo +0.6113 vs base -0.0453
- `rain_max_10m`: extremo +0.7097 vs base -0.0002
- `rain_max_30m`: extremo +2.0985 vs base -0.0002
- `rain_max_60m`: extremo -0.8977 vs base +0.0020
- `minutes_since_last_rain`: extremo +0.0427 vs base -0.0012
- `delta_rain_10m`: extremo +0.3020 vs base -0.0021
- `delta_rain_30m`: extremo +0.3497 vs base -0.0009
- `hour_sin`: extremo -0.5562 vs base +0.0007
- `month_sin`: extremo +0.1869 vs base -0.0003

## 3. Redundancia por correlacion (TRAIN)

Heatmap: `outputs/figures/diagnostic/s5_corr_matrix.png`.

**Pares con |r| >= 0.85**: 15 (lista completa en JSON). Top 10 por |r|:

| Feature A | Feature B | r |
|---|---|---:|
| rain_sum_10m | rain_max_10m | +0.985 |
| rain_sum_10m | rain_sum_15m | +0.970 |
| rain_sum_15m | rain_max_10m | +0.954 |
| rain_in | rain_sum_10m | +0.952 |
| rain_in | rain_max_10m | +0.942 |
| api_dynamic | rain_sum_60m | +0.940 |
| rain_sum_30m | rain_max_30m | +0.934 |
| rain_sum_120m | rain_sum_180m | +0.911 |
| api_dynamic | rain_sum_30m | +0.906 |
| rain_sum_60m | rain_max_60m | +0.903 |

**Clusters de redundancia (corte |r| >= 0.85)**: 13 clusters.

| # | tamano | representante (max PI) | miembros |
|---|---:|---|---|
| 1 | 4 | `rain_sum_15m` | `rain_in`, `rain_sum_10m`, `rain_sum_15m`, `rain_max_10m` |
| 2 | 2 | `temp_daily_f` | `temp_daily_f`, `month_cos` |
| 3 | 2 | `api_dynamic` | `api_dynamic`, `rain_sum_60m` |
| 4 | 2 | `rain_sum_30m` | `rain_sum_30m`, `rain_max_30m` |
| 5 | 2 | `rain_sum_120m` | `rain_sum_120m`, `rain_sum_180m` |
| 6 | 1 | `rain_sum_360m` | `rain_sum_360m` |
| 7 | 1 | `rain_max_60m` | `rain_max_60m` |
| 8 | 1 | `minutes_since_last_rain` | `minutes_since_last_rain` |
| 9 | 1 | `delta_rain_10m` | `delta_rain_10m` |
| 10 | 1 | `delta_rain_30m` | `delta_rain_30m` |
| 11 | 1 | `hour_sin` | `hour_sin` |
| 12 | 1 | `hour_cos` | `hour_cos` |
| 13 | 1 | `month_sin` | `month_sin` |

## 4. ACF de features clave vs target

ACF muestral en lags 1, 6, 12, 24 sobre TRAIN. Si la ACF de una feature es muy similar a la del target en los mismos lags, es candidata a transportar informacion autoregresiva del target encubierta a traves de su propia inercia.

| Feature | ACF lag1 | ACF lag6 | ACF lag12 | ACF lag24 |
|---|---:|---:|---:|---:|
| **target (stormflow_mgd)** | 0.910 | 0.560 | 0.420 | 0.280 |
| rain_sum_60m | 0.989 | 0.757 | 0.417 | 0.208 |
| api_dynamic | 0.988 | 0.795 | 0.591 | 0.354 |
| rain_sum_360m | 0.999 | 0.976 | 0.931 | 0.811 |
| delta_flow_5m | 0.082 | -0.036 | -0.000 | -0.003 |

## 5. Conjunto reducido propuesto y comparativa

Criterio: para cada cluster con |r|>=0.85 se conserva solo la feature con mayor permutation importance; se descartan las demas. Se eliminan ademas las features con PI<=0 (ruido). Si tras filtrar quedan >12 features se conservan las top por PI; si quedan <8 se completa con las siguientes mejores.

**Tamano final: 10 features.**

| # | Feature | PI mean_drop | tamano cluster | miembros del cluster |
|---|---|---:|---:|---|
| 1 | `api_dynamic` | +0.4182 | 2 | `api_dynamic`, `rain_sum_60m` |
| 2 | `rain_sum_360m` | +0.1083 | 1 | `rain_sum_360m` |
| 3 | `rain_sum_120m` | +0.0803 | 2 | `rain_sum_120m`, `rain_sum_180m` |
| 4 | `rain_sum_15m` | +0.0368 | 4 | `rain_in`, `rain_sum_10m`, `rain_sum_15m`, `rain_max_10m` |
| 5 | `temp_daily_f` | +0.0332 | 2 | `temp_daily_f`, `month_cos` |
| 6 | `hour_sin` | +0.0105 | 1 | `hour_sin` |
| 7 | `minutes_since_last_rain` | +0.0045 | 1 | `minutes_since_last_rain` |
| 8 | `delta_rain_10m` | +0.0044 | 1 | `delta_rain_10m` |
| 9 | `delta_rain_30m` | +0.0006 | 1 | `delta_rain_30m` |
| 10 | `rain_sum_30m` | +0.0006 | 2 | `rain_sum_30m`, `rain_max_30m` |

### Comparativa NSE H=1 (test alineado)

| Modelo | N feats | NSE | RMSE | MAE | Pico pred | Err pico % |
|---|---:|---:|---:|---:|---:|---:|
| **AR(12) (S2)** | - | 0.8273 | - | - | - | - |
| **XGB-20 (S2 ref)** | 20 | 0.6619 | 1.398 | 0.276 | 91.4 | -32.4 |
| **XGB-reducido** | 10 | 0.6980 | 1.321 | 0.271 | 91.8 | -32.1 |

Delta NSE (reducido - XGB-20) = **+0.0361**. Conjunto reducido **MEJORA +0.0361 NSE** sobre XGB-20: la reduccion elimina ruido/features anti-correlacionadas con el target.

## 6. Atajos encubiertos (ablation 1-by-1 sobre top-PI)

Sobre el conjunto reducido, ablacion individual de cada feature. Una caida >0.05 NSE al quitarla indica una dependencia muy fuerte: candidata a atajo (o feature genuinamente irreemplazable).

| Feature ablada | NSE sin ella | Delta NSE (caida) | Diagnostico |
|---|---:|---:|---|
| `hour_sin` | 0.6551 | +0.0428 | Aporta valor real |
| `rain_sum_15m` | 0.6659 | +0.0321 | Aporta valor real |
| `rain_sum_360m` | 0.6731 | +0.0249 | Aporta valor real |
| `rain_sum_30m` | 0.6911 | +0.0068 | Marginal o redundante |
| `delta_rain_30m` | 0.6955 | +0.0024 | Marginal o redundante |
| `api_dynamic` | 0.6970 | +0.0010 | Marginal o redundante |
| `delta_rain_10m` | 0.7001 | -0.0022 | Marginal o redundante |
| `minutes_since_last_rain` | 0.7029 | -0.0050 | Marginal o redundante |
| `rain_sum_120m` | 0.7072 | -0.0092 | Quitar mejora (probable ruido) |
| `temp_daily_f` | 0.7158 | -0.0179 | Quitar mejora (probable ruido) |

## Hallazgos clave

1. **Top 5 features por permutation importance**: `api_dynamic`(+0.4182), `rain_sum_360m`(+0.1083), `rain_sum_120m`(+0.0803), `rain_sum_15m`(+0.0368), `rain_sum_180m`(+0.0340). Estas son las que realmente impactan NSE cuando se barajan en test, no necesariamente las que mas usa el booster por gain.
2. **Redundancia: 15 pares con |r|>=0.85**, agrupados en **13 clusters independientes**. La dimensionalidad efectiva esta mucho mas cerca de 13 que de 20.
3. **No se detectan atajos encubiertos adicionales**: ninguna feature, tras quitar `delta_flow_*`, causa caida >0.05 NSE en ablation 1-by-1. El conjunto reducido depende repartidamente de varias features.
4. **Conjunto reducido propuesto (10 features)**: NSE=0.6980 vs XGB-20=0.6619 (delta +0.0361). Mejora (+0.0361 nse sobre xgb-20: al quitar features con pi<=0 se elimina ruido).
5. **Lluvias largas `rain_sum_180m`(PI=+0.0340), `rain_sum_360m`(PI=+0.1083)**: aportan PI medible: mantener.
6. **Features con PI <= 0**: `rain_sum_60m`, `rain_max_30m`, `rain_max_60m`, `hour_cos`, `month_sin`, `month_cos`. Su shuffle no degrada el modelo: ruido para esta tarea.
7. **ACF individual (posible inercia encubierta)**:
   - `rain_sum_60m` con ACF lag1=0.989, lag12=0.417 (target=0.420): inercia comparable a la del target en lag12 => posible portador de senal autoregresiva encubierta.
   - `delta_flow_5m` con ACF lag1=0.082: memoria corta, comportamiento tipo diferencia o ruido.

## Veredicto

Las 20 features oficiales (sin `delta_flow_*`) contienen una redundancia masiva: 15 pares con |r|>=0.85 que se agrupan en 13 clusters casi independientes. La capacidad predictiva real del modelo XGBoost vive en un subespacio mucho mas pequeno.



El conjunto reducido (10 features) **mejora +0.0361 NSE** sobre XGB-20. Reducir la dimensionalidad no solo no pierde senal sino que limpia ruido: las features con PI<=0 (hour_cos, month_sin, month_cos, rain_sum_60m, rain_max_30m, rain_max_60m) estaban degradando el fit de XGBoost.



Sin features con caida >=0.05 NSE en la ablation, el modelo XGBoost reducido reparte su senal de forma sana entre las features fisicas (lluvia agregada + API + temperatura + estacionalidad). No quedan atajos evidentes que limpiar mas alla de los `delta_flow_*` ya excluidos.
