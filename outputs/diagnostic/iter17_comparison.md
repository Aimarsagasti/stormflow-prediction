# Iter17 - XGB+lags comparison vs references

Single table with rows for baselines, TCN v1, and the XGB+lags variants trained in iter17. H=1 columns are derived from the multi-bucket panel (`src/evaluation/metrics_panel.py`) on the same aligned test as S2 (n=165,222). H=3 includes only global NSE (the full panel is in `iter17_xgb_results.json` if more detail is needed).

| Model | NSE H=1 | NSE H=3 | Peak err H=1 (%) | Base Bias H=1 (MGD) | Extreme NSE H=1 | recall@50 H=1 |
|---|---:|---:|---:|---:|---:|---:|
| naive (S2) | 0.8107 | 0.4094 | +0.0 | +0.001 | -2.853 | n/a |
| AR(12) (S2) | 0.8273 | 0.5085 | -6.1 | +0.036 | -2.860 | n/a |
| XGB-20 feats (S2) | 0.6619 | 0.5719 | -32.4 | +0.111 | -3.839 | n/a |
| XGB-22 with delta_flow (S2) | 0.7898 | 0.6558 | -7.3 | +0.056 | -1.660 | n/a |
| TCN v1 sinSF (§7.1) | 0.8615 | 0.4697 | +42.6 | n/a | n/a | n/a |
| xgb_lag6_feat10 [PRIMARY] (iter17) | 0.8630 | 0.6871 | -5.9 | +0.026 | -1.727 | 0.652 |
| xgb_lag12_feat10 (iter17) | 0.8665 | 0.6955 | -12.9 | +0.026 | -1.798 | 0.565 |
| xgb_feat10_only (iter17) | 0.7157 | 0.6012 | -38.8 | +0.118 | -3.133 | 0.304 |
| xgb_lag12_only (iter17) | 0.7778 | 0.5285 | -54.1 | +0.030 | -3.568 | 0.435 |
| xgb_lag24_feat10 (iter17) | 0.8634 | n/a | -26.4 | +0.025 | -1.806 | 0.565 |

## Success criteria (§7.4 DIAGNOSTIC_REPORT)

- NSE H=1 >= 0.85: **0.8630** -> OK
- NSE H=3 >= 0.66: **0.6871** -> OK
- |Peak err H=1| < 21%: **-5.9%** (abs=5.9) -> OK
- Base Bias H=1 <= +0.05 MGD: **+0.026** -> OK

## Selection of the primary model and attribution of the improvement (H=1, NSE)

Primary model: **lag6+feat10** (NSE=0.8630). Report §7.2 proposed lag=12 as the starting point; the ablation showed that lag=6 gives recall@50=0.652 versus 0.565 for lag=12, with an NSE difference of only -0.0034. Since recall@50 is MSD's main operational metric (alerting CSOs), lag=6 was chosen.

Attribution of the improvement over the primary model (lag6+feat10):
- feat10 only = 0.7157  ->  lags contribute +0.1473 NSE
- lag12 only  = 0.7778  ->  features contribute +0.0853 NSE
- lag12+feat10 = 0.8665  (report reference, similar NSE)
- lag24+feat10 = 0.8634  (ablation: long lags worsen the peak)

## Associated figures

- `outputs/figures/iter17/hydrograph_extreme_event_H1.png`
- `outputs/figures/iter17/scatter_real_vs_pred_H1.png`
- `outputs/figures/iter17/peak_error_by_bucket_H1.png`

## Note on bucket definition

The baseline rows (naive, AR(12), XGB-20, XGB-22) come from `outputs/diagnostic/S2_baselines.json` and use the bucket definition from `scripts/diagnostic/s2_baselines.py`: Moderate=[5, 25) MGD, High=[25, 50) MGD. The iter17 rows use `src/evaluation/metrics_panel.py` with the definition from `evaluate_local.py`: Moderate=[5, 20) MGD, High=[20, 50) MGD. **The Base Bias (<0.5 MGD) and Extreme NSE (>=50 MGD) columns are NOT affected by this difference** and are directly comparable across rows. The Moderate NSE and High NSE columns (not shown in this table) do differ in definition between sources and should not be compared directly.
