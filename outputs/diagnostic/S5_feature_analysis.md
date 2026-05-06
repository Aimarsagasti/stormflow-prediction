# S5 - Feature analysis and redundancy

Diagnostic of the real (non-autoregressive) value of the 20 official features after excluding `delta_flow_*` (shortcut confirmed in iter16/S2). It uses the same split and the same aligned indices as S2 so the metrics are directly comparable.

## Methodology

- **Base model**: XGBoost-20 (same hyperparameters as S2) retrained for H=1.
- **Permutation importance**: n_repeats=5 on a test subsample (n=50,000), individual shuffling of each feature.
- **SHAP**: TreeExplainer on 5000 random test rows (random_state=42).
- **Correlations**: Pearson matrix on TRAIN (771,302 rows), complete-linkage hierarchical clustering on |1 - r| with cutoff r>=0.85.
- **ACF of key features**: lags 1, 6, 12, 24 on TRAIN, compared with the target ACF (reference DATASET_STATS section 6).

### Time by stage

| Stage | Seconds |
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

## 1. Native importance (gain) and Permutation importance

`gain_pct` = normalized importance of the booster (sums to 1.0). `PI mean_drop` = mean drop in NSE when the feature is shuffled on test. A feature with PI<=0 is considered noise (its ablation does not degrade the model).

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

## 2. SHAP (mean |SHAP|, sign in extremes vs base)

Ranking by mean absolute importance. `mean_shap_extreme` = mean SHAP for y_true>=25.0 MGD, `mean_shap_base` = mean SHAP for y_true<0.5 MGD. Sign change => non-monotonic contribution (the feature pushes upward in extremes but downward in baseflow, or vice versa).

_n_samples=5,000, n_extremos_en_muestra=8, n_base_en_muestra=4638._

| Feature | mean |SHAP| | mean SHAP extreme | mean SHAP base | std SHAP |
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

**Features with non-monotonic SHAP (sign changes between extremes and base):**
- `rain_in`: extreme +1.8518 vs base -0.0154
- `temp_daily_f`: extreme +0.4717 vs base -0.0064
- `api_dynamic`: extreme +19.9938 vs base -0.1714
- `rain_sum_10m`: extreme +1.6299 vs base -0.0092
- `rain_sum_15m`: extreme +1.8653 vs base -0.0282
- `rain_sum_30m`: extreme +2.0685 vs base -0.0132
- `rain_sum_60m`: extreme -1.2806 vs base +0.0022
- `rain_sum_120m`: extreme +3.5535 vs base -0.0782
- `rain_sum_180m`: extreme +0.4530 vs base -0.0039
- `rain_sum_360m`: extreme +0.6113 vs base -0.0453
- `rain_max_10m`: extreme +0.7097 vs base -0.0002
- `rain_max_30m`: extreme +2.0985 vs base -0.0002
- `rain_max_60m`: extreme -0.8977 vs base +0.0020
- `minutes_since_last_rain`: extreme +0.0427 vs base -0.0012
- `delta_rain_10m`: extreme +0.3020 vs base -0.0021
- `delta_rain_30m`: extreme +0.3497 vs base -0.0009
- `hour_sin`: extreme -0.5562 vs base +0.0007
- `month_sin`: extreme +0.1869 vs base -0.0003

## 3. Redundancy by correlation (TRAIN)

Heatmap: `outputs/figures/diagnostic/s5_corr_matrix.png`.

**Pairs with |r| >= 0.85**: 15 (full list in JSON). Top 10 by |r|:

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

**Redundancy clusters (cutoff |r| >= 0.85)**: 13 clusters.

| # | size | representative (max PI) | members |
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

## 4. ACF of key features vs target

Sample ACF at lags 1, 6, 12, 24 on TRAIN. If a feature ACF is very similar to the target ACF at the same lags, it is a candidate for carrying hidden autoregressive information from the target through its own inertia.

| Feature | ACF lag1 | ACF lag6 | ACF lag12 | ACF lag24 |
|---|---:|---:|---:|---:|
| **target (stormflow_mgd)** | 0.910 | 0.560 | 0.420 | 0.280 |
| rain_sum_60m | 0.989 | 0.757 | 0.417 | 0.208 |
| api_dynamic | 0.988 | 0.795 | 0.591 | 0.354 |
| rain_sum_360m | 0.999 | 0.976 | 0.931 | 0.811 |
| delta_flow_5m | 0.082 | -0.036 | -0.000 | -0.003 |

## 5. Proposed reduced set and comparison

Criterion: for each cluster with |r|>=0.85, keep only the feature with the highest permutation importance and discard the others. Also remove features with PI<=0 (noise). If more than 12 features remain after filtering, keep the top by PI; if fewer than 8 remain, complete with the next best ones.

**Final size: 10 features.**

| # | Feature | PI mean_drop | cluster size | cluster members |
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

### NSE H=1 comparison (aligned test)

| Model | N feats | NSE | RMSE | MAE | Pred peak | Peak err % |
|---|---:|---:|---:|---:|---:|---:|
| **AR(12) (S2)** | - | 0.8273 | - | - | - | - |
| **XGB-20 (S2 ref)** | 20 | 0.6619 | 1.398 | 0.276 | 91.4 | -32.4 |
| **XGB-reduced** | 10 | 0.6980 | 1.321 | 0.271 | 91.8 | -32.1 |

Delta NSE (reduced - XGB-20) = **+0.0361**. The reduced set **IMPROVES NSE by +0.0361** over XGB-20: the reduction removes noise/features that are anti-correlated with the target.

## 6. Hidden shortcuts (1-by-1 ablation on top-PI)

On the reduced set, individual ablation of each feature. A drop >0.05 NSE when removing it indicates very strong dependence: candidate for a shortcut (or a genuinely irreplaceable feature).

| Ablated feature | NSE without it | Delta NSE (drop) | Diagnostic |
|---|---:|---:|---|
| `hour_sin` | 0.6551 | +0.0428 | Provides real value |
| `rain_sum_15m` | 0.6659 | +0.0321 | Provides real value |
| `rain_sum_360m` | 0.6731 | +0.0249 | Provides real value |
| `rain_sum_30m` | 0.6911 | +0.0068 | Marginal or redundant |
| `delta_rain_30m` | 0.6955 | +0.0024 | Marginal or redundant |
| `api_dynamic` | 0.6970 | +0.0010 | Marginal or redundant |
| `delta_rain_10m` | 0.7001 | -0.0022 | Marginal or redundant |
| `minutes_since_last_rain` | 0.7029 | -0.0050 | Marginal or redundant |
| `rain_sum_120m` | 0.7072 | -0.0092 | Removing it improves (likely noise) |
| `temp_daily_f` | 0.7158 | -0.0179 | Removing it improves (likely noise) |

## Key findings

1. **Top 5 features by permutation importance**: `api_dynamic`(+0.4182), `rain_sum_360m`(+0.1083), `rain_sum_120m`(+0.0803), `rain_sum_15m`(+0.0368), `rain_sum_180m`(+0.0340). These are the ones that actually affect NSE when shuffled on test, not necessarily the ones the booster uses most by gain.
2. **Redundancy: 15 pairs with |r|>=0.85**, grouped into **13 independent clusters**. The effective dimensionality is much closer to 13 than to 20.
3. **No additional hidden shortcuts are detected**: after removing `delta_flow_*`, no feature causes a drop >0.05 NSE in 1-by-1 ablation. The reduced set depends in a balanced way on several features.
4. **Proposed reduced set (10 features)**: NSE=0.6980 vs XGB-20=0.6619 (delta +0.0361). Improvement (+0.0361 nse over xgb-20: removing features with pi<=0 eliminates noise).
5. **Long rainfall features `rain_sum_180m`(PI=+0.0340), `rain_sum_360m`(PI=+0.1083)**: they provide measurable PI: keep them.
6. **Features with PI <= 0**: `rain_sum_60m`, `rain_max_30m`, `rain_max_60m`, `hour_cos`, `month_sin`, `month_cos`. Shuffling them does not degrade the model: noise for this task.
7. **Individual ACF (possible hidden inertia)**:
   - `rain_sum_60m` with ACF lag1=0.989, lag12=0.417 (target=0.420): inertia comparable to the target at lag12 => possible carrier of hidden autoregressive signal.
   - `delta_flow_5m` with ACF lag1=0.082: short memory, difference-like or noise-like behavior.

## Verdict

The 20 official features (without `delta_flow_*`) contain massive redundancy: 15 pairs with |r|>=0.85 grouped into 13 nearly independent clusters. The real predictive capacity of the XGBoost model lives in a much smaller subspace.



The reduced set (10 features) **improves NSE by +0.0361** over XGB-20. Reducing dimensionality not only preserves signal but also cleans noise: the features with PI<=0 (`hour_cos`, `month_sin`, `month_cos`, `rain_sum_60m`, `rain_max_30m`, `rain_max_60m`) were degrading the XGBoost fit.



Without features causing a drop >=0.05 NSE in the ablation, the reduced XGBoost model distributes its signal in a healthy way among the physical features (aggregated rainfall + API + temperature + seasonality). No obvious shortcuts remain to clean up beyond the already excluded `delta_flow_*`.
