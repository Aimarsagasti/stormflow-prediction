# Iter19 - Clean TCN (Bai 2018) vs xgb_lag6_feat10

## Executive summary

Winning configuration: **A4** (L=72, C=64, log1p_target=False). TCN test NSE H=1 = **0.8983**. Reference iter17 XGB NSE = 0.8630. Delta = **+0.0353** (closing threshold = +0.020). VERDICT: **TCN_WINS**.

## Ablation table on val

| Run | L | C | log1p | NSE_val | RMSE_val | err_pico_val (%) | recall@50_val | best_epoch | train_s |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| A0 | 72 | 32 | yes | 0.8436 | 1.132 | +1.5 | 0.741 | 15 | 995 |
| A1 | 72 | 32 | no | 0.8617 | 1.065 | -24.9 | 0.741 | 12 | 863 |
| A2 | 144 | 32 | yes | 0.8146 | 1.233 | +6.7 | 0.815 | 7 | 717 |
| A3 | 72 | 64 | yes | 0.8497 | 1.110 | -23.3 | 0.630 | 25 | 1398 |
| A4 | 72 | 64 | no | 0.8640 | 1.056 | -25.3 | 0.741 | 7 | 683 |

## Justification of the winning config

- **log1p_target = False** - A0 (log1p=True) NSE_val=0.8436 vs A1 (log1p=False) NSE_val=0.8617. Delta=-0.0181. Axis resolved in favor of log1p_target=False.
- **L = 72** - A0 (L=72) NSE_val=0.8436 vs A2 (L=144) NSE_val=0.8146. Delta=+0.0290. Axis resolved in favor of L=72.
- **C = 64** - A0 (C=32) NSE_val=0.8436 vs A3 (C=64) NSE_val=0.8497. Delta=-0.0061. Axis resolved in favor of C=64.

The winning combination (False, L=72, C=64) does not match the already trained A0..A3, so A4 was trained with this config to empirically validate the factor-independence selection.

## Final model results on test

| Metric | Value |
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

### NSE by bucket (test)

| Bucket | n | NSE | RMSE | bias (MGD) | peak_err_pct |
|---|---:|---:|---:|---:|---:|
| Base | 152902 | +0.085 | 0.084 | +0.002 | +1371.2 |
| Light | 9518 | +0.765 | 0.529 | +0.136 | +150.6 |
| Moderate | 2303 | +0.527 | 2.780 | +0.350 | +135.8 |
| High | 440 | +0.005 | 7.392 | -1.045 | +20.7 |
| Extreme | 59 | -1.231 | 29.573 | -13.739 | +4.7 |

## Direct comparison against xgb_lag6_feat10

Both models evaluated on the same test timestamps (L_winner=72, XGB offset=0 steps).

| Model | NSE | err_pico (%) | bias_base (MGD) | recall@50 | NSE_extremo |
|---|---:|---:|---:|---:|---:|
| xgb_lag6_feat10 (inline) | 0.8631 | -12.4 | +0.028 | 0.609 | -1.808 |
| TCN (A4) | 0.8983 | +4.7 | +0.002 | 0.783 | -1.231 |
| **delta (TCN - XGB inline)** | **+0.0352** | +17.0 | -0.026 | +0.174 | +0.577 |

Official reference iter17 NSE (xgb_lag6_feat10, n=165222): **0.8630**.
Delta used for the verdict: TCN(0.8983) - XGB_iter17(0.8630) = **+0.0353**.
Closing threshold from DIAGNOSTIC_REPORT §7.6: +0.020.

## Verdict

**TCN wins** by a margin >= 0.020 NSE. The clean TCN (Bai 2018) beats the XGBoost regressor by +0.0353 NSE at H=1. It becomes the main candidate model for the TFM. Pending replication at H=3 in a later session before definitively replacing `xgb_lag6_feat10`.

## Honest reading for the TFM

This is the only post-diagnostic deep learning iteration, intentionally limited (one seed, 4-run ablation, simple Huber loss) to avoid over-engineering that masks the real result. The +0.020 NSE closing threshold is arbitrary but defensible: on the order of the noise between reasonable configurations, and well below the H=1 physical ceiling ~0.86 derived from S4 analysis.
The result favors deep learning, but the difference must be interpreted in light of the added complexity (GPU, PyTorch dependency, higher inference cost) versus the operational benefit. For real MSD use, the cost/benefit of maintaining a production TCN versus a GBM must be documented explicitly.

## Recognized limitations

- A single seed (=42). No error bars are reported for the final NSE.
- Minimal ablation (4 runs) with factor-independence selection. Possible interactions between log1p and L, or between C and dropout, are not explored.
- Simple Huber loss in normalized space, without a magnitude component or peak penalty. Variants with asymmetric loss could improve the Extreme bucket, but they are out of scope by spec.
- When L>72 the val/test origins are trimmed by 0 steps so as not to cross the split boundary. This slightly changes n_test relative to iter17 (n=165222 with L=72).
- It has not been replicated at H=3 in this iteration. If TCN wins, that replication remains pending before changing the main model in STATE.md and EXPERIMENTS.md.
