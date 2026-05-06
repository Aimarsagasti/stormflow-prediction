# Iter19c - TCN A0 (log1p=True) at H=3 vs XGB iter17 H=3

## Executive summary

TCN A0 (L=72, C=32, log1p=True) is replicated at horizon H=3 (15 minutes ahead) with the same config as in iter19b (H=1). TCN A0 H=3 test NSE = **0.6096**. Official XGB iter17 H=3 NSE = 0.6871. Inline XGB H=3 NSE (aligned, sanity) = 0.6889. Delta = **-0.0775** (closing threshold = +0.020). H=3 VERDICT: **XGB_WINS**.

Sanity check of the inline XGB baseline vs official iter17 figure: |delta| = 0.0018 <= 0.02 (OK).

## TCN A0 H=3 results on test

| Metric | Value |
|---|---:|
| n_test | 165220 |
| NSE | 0.6096 |
| RMSE | 1.502 |
| MAE | 0.140 |
| peak_real (MGD) | 135.15 |
| peak_pred (MGD) | 63.24 |
| peak_err_pct | -53.2 |
| recall@25 | 0.333 |
| recall@50 | 0.087 |
| n_params | 23,489 |
| best_epoch | 3 |
| training_time_s | 517 |

### NSE by bucket (TCN A0 H=3 test)

| Bucket | n | NSE | RMSE | bias (MGD) | peak_err_pct |
|---|---:|---:|---:|---:|---:|
| Base | 152898 | +0.711 | 0.047 | +0.004 | +332.9 |
| Light | 9520 | +0.627 | 0.667 | +0.083 | +94.4 |
| Moderate | 2303 | -0.130 | 4.297 | -1.446 | +137.0 |
| High | 440 | -3.805 | 16.246 | -11.406 | +27.1 |
| Extreme | 59 | -8.064 | 59.604 | -55.002 | -59.9 |

## Direct comparison against xgb_lag6_feat10 H=3

Both models evaluated on the same test timestamps (L=72, H=3, n_TCN=165220, n_XGB=165220).

| Model | NSE | RMSE | err_pico (%) | recall@50 | NSE_extremo |
|---|---:|---:|---:|---:|---:|
| xgb_lag6_feat10 (inline) | 0.6889 | 1.341 | -13.5 | 0.261 | -5.553 |
| TCN A0 H=3 | 0.6096 | 1.502 | -53.2 | 0.087 | -8.064 |
| **delta (TCN - XGB inline)** | **-0.0792** | +0.161 | -39.7 | -0.174 | -2.510 |

Official reference iter17 H=3 NSE (xgb_lag6_feat10): **0.6871**.
Delta used for the verdict: TCN(0.6096) - XGB_iter17(0.6871) = **-0.0775**.
Closing threshold from DIAGNOSTIC_REPORT A7.6: +0.020.

## Honest reading

**H=1 vs H=3 consistency.** At H=1 (iter19b) TCN A0 reached NSE_test = 0.8890, versus XGB iter17 H=1 = 0.8630 (delta = +0.0260). At H=3 we get delta = -0.0775.
The TCN advantage observed at H=1 does NOT replicate at H=3: the GBM catches up as the horizon lengthens. This is consistent with the TCN's fixed convolutional nature: the receptive field optimized for H=1 does not transfer directly to H=3, where the autoregressive signal matters less and other dependencies (historical rainfall, API) gain relative weight. The GBM with explicit lags absorbs the latter efficiently.

**Implication for the TFM**: A0 is confirmed as the main model only at H=1. At H=3 the GBM matches or beats the TCN. There are two honest paths: (a) report the TCN as the main model at H=1 and XGB for H=3, which fragments the solution but reflects the real result; (b) report XGB as the single model for H=1 and H=3 at the cost of the +0.026 NSE operational gain at H=1, gaining coherence and simplicity. The user decides; this run does not force the answer.

**Operational behavior in the Extreme bucket (>=50 MGD).** TCN NSE_extremo = -8.064, XGB = -5.553. TCN bias_extremo = -55.002 MGD, XGB = -43.705 MGD.
The GBM has a smaller absolute bias here in critical events at H=3, reversing the operational advantage that A0 had at H=1. Another point to consider for the final main-model decision.

## Verdict and recommendation

**XGB_WINS at H=3** (TCN does not exceed the +0.020 NSE threshold). Recommendation: update STATE.md and EXPERIMENTS.md indicating that replication at H=3 does NOT confirm the H=1 gain. Explicitly decide the TFM strategy: dual model (TCN H=1 + XGB H=3) or single XGB model. Document the result as a positive methodological finding: iter17 GBM absorbs the predictable signal at medium horizon where the TCN adds no additional value.

## Recognized limitations

- A single seed (=42), the same as iter19/iter19b. The TCN vs XGB difference at H=3 may be within stochastic initialization noise. For greater certainty it would need repetition with several seeds.
- Comparison only at H=3. If a complete narrative is wanted, H=6 and H=12 remain pending.
- Inline XGB retrained with the same seed/params as iter17, but the dataset may have changed marginally since April 2026; the sanity check validates that the delta vs the official figure is 0.0018 <= 0.02.
- Simple Huber loss, without a magnitude component or peak penalty, the same as iter19. An asymmetric loss could improve the Extreme bucket at H=3, but it is out of scope.
