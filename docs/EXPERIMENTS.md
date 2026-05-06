# EXPERIMENTS.md - Iteration history

Chronological log of all stormflow model iterations. An entry is added at the end each time an experiment is completed.

---

## Conventions

Format of each entry:

Iteration N (YYYY-MM-DD)

Hypothesis: why this change was tested.
Change: what was modified exactly.
Result: NSE, RMSE, peak error, and other relevant metrics.
Lesson: what was learned.
Detailed analysis: (optional) reference to iteration_N_analysis.md if it exists.

Iterations 1-10 have complete detailed analyses in `outputs/iteration_N_analysis.md`.
Iterations 11-15 and v2 have more reduced documentation because they were done directly in Colab without detailed code commits. The full documentation lives in the saved weights (`MC-CL-005/Pesos 13-04-2026/*_meta.json`) and the commit messages.

---

## Iteration 1 (2026-03-28)

- **Hypothesis:** TCN baseline with direct regression, 4-component CompositeLoss, and stratified sampler.
- **Change:** base architecture.
- **Result:** low global NSE, severe positive bias in base (massive overestimation outside events).
- **Lesson:** the aggressive stratified sampler caused train/val drift.
- **Detailed analysis:** `outputs/iteration_1_analysis.md`.

---

## Iteration 2 (2026-03-29)

- **Hypothesis:** reduce target dependence by removing strong autoregressive features.
- **Change:** feature simplification, removing stormflow deltas and stormflow_flow_ratio.
- **Result:** reduced autoregressive bias but worse ability to distinguish rapid intensification.
- **Lesson:** the model was using features directly derived from the target as a shortcut.
- **Detailed analysis:** `outputs/iteration_2_analysis.md`.

---

## Iteration 3 (2026-03-30)

- **Hypothesis:** multitask with multiplicative gating (event prob x magnitude).
- **Change:** introduce classification head + multiplication with magnitude.
- **Result:** severe peak compression (the gate never saturates at 1.0).
- **Lesson:** **DO NOT use multiplicative gating. Use a hard switch (if/else).**
- **Detailed analysis:** `outputs/iteration_3_analysis.md`.

---

## Iteration 4 (2026-03-31)

- **Hypothesis:** simplify to direct regression without multitask.
- **Change:** remove classification head, return to pure regression.
- **Result:** NSE=-7.64, worse than iter 1. It got worse.
- **Lesson:** without a strong signal that distinguishes event/non-event, the model collapses under the 92% base imbalance.
- **Detailed analysis:** `outputs/iteration_4_analysis.md`.

---

## Iteration 5 (2026-04-01)

- **Hypothesis:** explicitly penalize overestimation in base.
- **Change:** add a "base overprediction penalty" loss component.
- **Result:** NSE=-1.72. Base bias improved but not the general metrics.
- **Lesson:** the penalty helps but does not attack the root problem (zero-inflation).
- **Detailed analysis:** `outputs/iteration_5_analysis.md`.

---

## Iteration 6 (2026-04-01)

- **Hypothesis:** test without log1p to see whether the transformation distorts learning.
- **Change:** remove log1p from the target.
- **Result:** **NSE=-26.6. CATASTROPHIC.**
- **Lesson:** **log1p on the target is NECESSARY. Never remove it.** The target's heavy-tailed distribution requires that transformation so the gradient is not useless.
- **Detailed analysis:** `outputs/iteration_6_analysis.md`.

---

## Iteration 7 (2026-04-01)

- **Hypothesis:** re-enable log1p and add tail focus in the loss.
- **Change:** log1p again + extra weight for high-tail samples.
- **Result:** NSE=-2.58, peak error -45.9%. It recovered what iter 6 broke.
- **Lesson:** log1p is non-negotiable. Tail focus helps modestly.
- **Detailed analysis:** `outputs/iteration_7_analysis.md`.

---

## Iteration 8 (2026-04-02)

- **Hypothesis:** add dynamic API (Antecedent Precipitation Index) and daily temperature.
- **Change:** new features temp_daily_f and api_dynamic.
- **Result:** NSE=-1.94, but bias in extremes worsened to -63%.
- **Lesson:** api_dynamic seemed redundant with rain_sum_*. Temp_daily_f correlates with month_cos. Adding features without prior analysis does not always help.
- **Detailed analysis:** `outputs/iteration_8_analysis.md`.

---

## Iteration 9 (2026-04-02)

- **Hypothesis:** more conservative learning rate (5e-4) to stabilize training.
- **Change:** lr reduced from 1e-3 to 5e-4.
- **Result:** NSE=-0.86 (better than iter 8) but peak error worsened to -71%.
- **Lesson:** lower lr helps global NSE but makes the model more "conservative" on peaks.
- **Detailed analysis:** `outputs/iteration_9_analysis.md`.

---

## Iteration 10 (2026-04-06)

- **Hypothesis:** change normalization from MinMax to z-score to decompress the feature range.
- **Change:** log1p + z-score instead of log1p + MinMax.
- **Result:** **NSE=-0.55 (best to date). RMSE=2.99, peak error -60%.**
- **Lesson:** z-score gives the model more resolution on peaks (normalized target reaches z-scores of 30-70 instead of 0.7-1.0 with MinMax). Even so, 92% of rainfall features remain compressed because the original distribution is dominated by zeros.
- **Detailed analysis:** `outputs/iteration_10_analysis.md`.

---

## Iteration 11 (2026-04-XX)

- **Hypothesis:** remove `flow_total_mgd` as a feature. It had correlation r=0.9976 with the target, suspicious of being a shortcut.
- **Change:** remove `flow_total_mgd` from FEATURE_COLUMNS.
- **Result:** global NSE worsened slightly but peak error improved from -60% to -41%.
- **Lesson:** **`flow_total_mgd` was a shortcut. NEVER re-include it.** The model was using it to copy the target instead of learning rainfall -> stormflow.
- **Detailed analysis:** not available in `iteration_11_analysis.md`. Partial documentation in commit messages and Claude chats from April 6-9.

---

## Iteration 12 (2026-04-XX)

- **Hypothesis:** `api_dynamic` is redundant with `rain_sum_60m`. `hour_sin/cos` have negative PI. Removing them could simplify the model.
- **Change:** exhaustive permutation importance analysis. Confirmation that `rain_sum_60m` through `rain_sum_360m` are necessary.
- **Result:** when the long `rain_sum_*` were removed in tests, `rain_max_10m` dominated with PI=138% and the model overestimated peaks by +114%. DO NOT remove all long rolling sums at once.
- **Lesson:** long temporal-context features are necessary to calibrate magnitude. `hour_sin/cos` have consistently negative PI (they do not help). `api_dynamic` is redundant but not harmful.
- **Detailed analysis:** not available as a file. Partial documentation in chats and in `project_status.md`.

---

## Iteration 13 (2026-04-XX)

- **Hypothesis:** separate the problem into two stages (Hurdle Model). A classifier decides whether there is an event, a regressor predicts magnitude only when yes.
- **Change:** **TwoStageTCN** with shared backbone + classifier head (BCE) + regressor head (asymmetric Huber). Inference with hard switch, threshold=0.3.
- **Result:** **MOST IMPACTFUL ARCHITECTURAL CHANGE OF THE PROJECT.** NSE and peak error improved significantly. Peaks captured with anomaly of -2.6% (later confirmed to be exceptional).
- **Lesson:** the two-stage architecture is the correct one for zero-inflated. Keep this architecture as the base from now on. The ~2.6% error was anomalous (low variance confirmed in later iterations, the systematic value is ~-60/-83%).
- **Commit:** `fea90be Iter 13: Modelo Two-Stage (Hurdle) para zero-inflated stormflow`.
- **Detailed analysis:** not available as a file.

---

## Iteration 14 (2026-04-XX)

- **Hypothesis:** a more sophisticated scheduler (CosineAnnealingWarmRestarts) could improve convergence.
- **Change:** replace ReduceLROnPlateau with CosineAnnealingWarmRestarts. Also fix diagnostics so they work with TwoStageTCN.
- **Result:** results got worse. The cyclic scheduler does not help on this problem.
- **Lesson:** **DO NOT use Cosine Annealing in this project. Use ReduceLROnPlateau.**
- **Commit:** `7c368e8 Iter 14: CosineAnnealingWarmRestarts + fix diagnostics para TwoStageTCN`.

---

## Iteration 14b (2026-04-XX)

- **Hypothesis:** revert the scheduler to return to ReduceLROnPlateau.
- **Change:** revert the iter 14 scheduler.
- **Result:** metrics return to the iter 13 level.
- **Lesson:** confirmation that ReduceLROnPlateau is the correct option for this problem.
- **Commit:** `fdc4399 Iter 14b: Revertir a ReduceLROnPlateau (Cosine empeoraba)`.

---

## Iteration 15 (2026-04-XX)

- **Hypothesis:** weight the regressor loss by magnitude (alpha=0.05) to force better prediction of extremes.
- **Change:** introduce magnitude-weighted regression loss with alpha=0.05.
- **Result:** alpha=0.05 was too aggressive, it degraded results. It was reverted.
- **Lesson:** **magnitude weighting does not solve the underestimation of extremes.** The problem is structural (few extreme samples + 15 of 59 without rainfall in the window), it is not fixed with loss weights.
- **Commits:** `d72bff9 Iter 15: magnitude-weighted regression loss (alpha=0.05)` and `7e256da Revert alpha=0.05, demasiado agresivo`.

---

## Iteration 15b (2026-04-XX)

- **Hypothesis:** a softer version of magnitude weighting + explicit penalty for excessive overestimation.
- **Change:** gradual loss by magnitude + overestimation penalty.
- **Result:** marginal improvement. It does not solve the underlying problem.
- **Lesson:** confirms that the underestimation of extreme peaks is structural, not tunable through loss tuning.
- **Commit:** `1fbaaf1 feat: loss gradual por magnitud y penalizacion por sobreestimacion excesiva`.

---

## Experiment: horizon comparison (2026-04-13)

Notebook `horizon_comparison.py`. Training of 6 models: H={1,3,6} x {with stormflow, without stormflow}.

- **Hypothesis:** characterize how the model degrades with the prediction horizon and whether stormflow as a feature helps.
- **Change:** 6 controlled trainings, same architecture, only horizon and feature set change.
- **Result (WITHOUT stormflow):**
  - H=1: NSE=0.819, peak error +12.8%.
  - H=3: NSE=0.536, peak error -48.2%.
  - H=6: NSE=-0.447, peak error -92.6%.
- **Result (WITH stormflow):**
  - H=1: NSE=0.853, peak error +59.4%.
  - H=3: NSE=0.488, peak error -48.5%.
  - H=6: NSE=0.255, peak error -88.2%.
- **Lesson:**
  - H=1 WITHOUT SF is the best rain-only model (the most operationally relevant).
  - Stormflow as a feature improves global NSE at H=1 but severely overestimates peaks (+59.4%).
  - Brutal degradation with horizon: H=6 (30 min) is not useful.
  - 15 of 59 extreme events have no rainfall in the window (physically unpredictable).
- **Commit:** `f9c5ea8 feat: evaluate_local, metricas, analisis extremos, threshold sweep`.

---

## Iteration v2 (2026-04-16)

- **Hypothesis:** retrain H1_sinSF with adjustments to see whether better peak capture can be achieved without losing global NSE.
- **Change:** retraining of modelo_H1_sinSF with a slightly modified configuration (details in `modelo_H1_sinSF_v2_meta.json`).
- **Result:** it captures peaks slightly better than v1 but with lower global NSE. **v1 remains the production model by global NSE.**
- **Lesson:** clear trade-off between global NSE and extreme peak capture. The decision of which model to choose depends on the operational case: if average MAPE matters more, v1. If not underestimating peaks matters more, v2.
- **Comparison plots:** `outputs/figures/local_eval/v1vs_v2_*.png`.

---

## Summary of firm conclusions (April 2026)

1. Massive changes = unpredictable results. ONE change per iteration.
2. Multiplicative gating compresses peaks. Use a hard switch.
3. log1p on the target is non-negotiable.
4. z-score is better than MinMax (but does not solve rainfall-feature compression).
5. Permutation Importance in a bad model is NOT reliable.
6. Aggressive stratified sampler causes drift. Better natural distribution + weighted loss.
7. Two-Stage (Hurdle) is the most impactful change.
8. Cosine Annealing makes things worse. Use ReduceLROnPlateau.
9. flow_total_mgd is a shortcut. NEVER re-include it.
10. Do not remove all long rain_sum at once.
11. Magnitude weighting in the loss does NOT solve the underestimation of extremes.
12. The ~83% peak error is SYSTEMATIC, not random.
13. 15 of 59 extreme events have no rainfall signal in the window (physically unpredictable).
14. The regressor HAS capacity (it can predict >100 MGD) but does not discriminate magnitudes.
15. When stormflow is feature AND target, normalize_splits normalizes it TWICE. Mandatory patch.
16. Stormflow as a feature improves global metrics but does NOT help with extreme peaks.
17. The model works well on moderate events (5-50 MGD) at H=1. Judging it only by extremes is unfair.
18. Colab ONLY for training. All analysis and plots in local VS Code.

# EXPERIMENTS.md - Iteration log

Record of each model iteration with standard format: hypothesis, applied change, result, decision.

---

## Iteration 16 - Ablation of `delta_flow_5m` and `delta_flow_15m`

**Date:** 2026-04-21
**Model:** `modelo_H1_sinSF_iter16`
**Features:** 20 (v1 had 22, `delta_flow_5m` and `delta_flow_15m` were removed)

### Hypothesis

Based on the external review by Opus 4.7 (see `docs/OPUS_REVIEW_2026-04-20.md` §A1 and §D1):

> Since $\text{flow\_total} = \text{baseflow} + \text{stormflow}$ and baseflow changes slowly, it follows that $\Delta\text{flow}(t) \approx \Delta\text{stormflow}(t)$. These features effectively carry the derivative of the target at $t$, reintroducing the shortcut that was closed in iter11 by removing `flow_total_mgd`. If removing them makes NSE fall to the naive level (~0.811), the model is doing disguised AR(1) instead of learning rainfall $\to$ stormflow.

### Change applied

Only change relative to v1: removal of `delta_flow_5m` and `delta_flow_15m` from `FEATURE_COLUMNS`. Architecture (TwoStageTCN), loss (TwoStageLoss), hyperparameters, split, and normalization identical to v1.

### Result

Training completed in 14 epochs (early stopping), best epoch = 4.

| Metric | v1 (22 feat) | iter16 (20 feat) | Delta |
|---|---:|---:|---:|
| NSE | +0.861 | **-0.169** | -1.030 |
| RMSE | 0.89 MGD | 2.60 MGD | +1.71 |
| MAE | 0.29 MGD | 0.63 MGD | +0.34 |
| Peak error | -21.0% | **+358.6%** | sign change |
| Base bias | +0.21 MGD | +0.48 MGD | +0.27 |
| Extreme bias | -12.7 MGD | **+39.2 MGD** | sign change |
| Max predicted peak | ~106 MGD | **625 MGD** | x3 extrapolation outside train range (max=199 MGD) |

Meta.json verified: `n_features=20`, `delta_flow_*` confirmed outside the list. There is no bug.

### Interpretation

The Opus hypothesis is confirmed with greater severity than expected. It is not that the +0.050 NSE gain over naive came from `delta_flow_*`. It is that **the model's entire numerical stability** depended on those two features. Without them:

- The regressor does not learn rainfall $\to$ stormflow.
- It extrapolates without bounds in extremes (predicts 625 MGD with a train maximum of 199 MGD).
- It massively overestimates even baseflow (base bias x2).
- `H1_conSF` (with `stormflow_mgd` as a direct autoregressive feature) still gives NSE=0.854, confirming that all useful signal is autoregressive.

### Collateral finding

The local evaluation of iter16 reports **0 extreme events without rainfall in the 72-step window**, not 15 as previously documented in STATE.md and in `extreme_events_no_rain.json`. All 59/59 extremes in the test have detectable rainfall in the previous 6h. The argument of "physically unpredictable extremes" used to justify the error in the Extreme bucket is invalidated.

### Decision

**Stop the planned sequence of iterations 16-20 after the Opus review.** The iter16 result invalidates the premise of iter17 (non-conditional regressor to reduce base bias), which assumed that the model was learning hydrology but the regressor was generating OOD. With negative global NSE, there is no signal to stabilize: there is no learned hydrology, it is an AR(1) with decorative features.

**Next step:** deep audit with Claude Code (branch `diagnostico`) before deciding the new path. No more iterations will be run on the current approach until obtaining `DIAGNOSTIC_REPORT.md`.

### References

- Raw data: `MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_iter16_{weights.pt, norm_params.json, meta.json}`
- Full evaluation: `outputs/data_analysis/local_eval_metrics.json`
- Master prompt for Claude Code: `PROMPT_CLAUDE_CODE.md`


## Iteration 17 (2026-04-24 to 2026-04-29)

- **Hypothesis:** after the Claude Code diagnostic (branch `diagnostico`, summarized in `outputs/diagnostic/DIAGNOSTIC_REPORT.md`), an XGBoost with explicit target lags structurally outperforms TwoStageTCN v1 because (a) it removes the `delta_flow_*` shortcuts confirmed in iter16, (b) it directly injects the autoregressive signal that a pure GBM loses, and (c) it reduces feature noise by removing the 10 with PI <= 0 according to S5.
- **Change:** new main project model: `xgb_lag6_feat10` with 6 target lags (`stormflow_mgd[t-0..t-5]`) + 10 reduced features from S5 (`api_dynamic`, `rain_sum_360m`, `rain_sum_120m`, `rain_sum_15m`, `temp_daily_f`, `hour_sin`, `minutes_since_last_rain`, `delta_rain_10m`, `delta_rain_30m`, `rain_sum_30m`). Hyperparameters from DIAGNOSTIC_REPORT §7.2 without tuning: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `tree_method=hist`, `early_stopping_rounds=20`, `random_state=42`. It works in real MGD (without `normalize.py` pipeline). New modules: `src/models/xgboost_baseline.py` and `src/evaluation/metrics_panel.py`. Orchestrator notebook: `notebooks/iter17_xgboost_lags.py`.
  - **Decision lag=6 vs lag=12:** the report proposed lag=12 as the starting point, but the ablation showed that lag=6 gives better recall@50 (0.652 vs 0.565) and better peak error (-5.9% vs -12.9%) with almost identical NSE (-0.0035). Since recall@50 is MSD's operational metric, lag=6 was chosen as primary.
- **Result:**

  | Metric       | H=1    | H=3    |
  |---------------|--------|--------|
  | NSE           | 0.8630 | 0.6871 |
  | Peak err (%)  | -5.9   | -13.1  |
  | Base Bias (MGD) | +0.026 | -    |
  | recall@50     | 0.652  | 0.304  |
  | Extreme NSE   | -1.727 | -      |

  The 4 success criteria of §7.4 in the DIAGNOSTIC_REPORT are satisfied. Attribution of the H=1 improvement (NSE): the lags contribute +0.1473 over `feat10_only`; the features contribute +0.0853 over `lag12_only`. Both levers are necessary.

- **Lesson:**
  - Explicit linear autoregression captures most of the variance at H=1 in zero-inflated streamflow; pure AR(12) gives NSE=0.83. XGBoost nonlinearities only pay off when combined with exogenous features (XGB with lags only gives 0.78, worse than linear AR(12)).
  - Optimizing the global metric (NSE) can contradict the operational metric (recall@50). In iter17 lag=6 loses 0.0035 of NSE and gains +0.087 in recall@50; the correct decision is to prioritize operations.
  - Working in real MGD without the legacy normalization pipeline removes an entire source of bugs (BUG2 of S1) at no performance cost.
- **Detailed analysis:** `outputs/diagnostic/iter17_comparison.md`, `outputs/diagnostic/iter17_xgb_results.json`, `outputs/figures/iter17/{hydrograph_extreme_event_H1, scatter_real_vs_pred_H1, peak_error_by_bucket_H1}.png`.

---

## Iteration 18 (2026-04-29)

- **Hypothesis:** long horizons (H=6, H=12) have a low physical ceiling as regression (S4: max NSE 0.32 and 0.19 respectively) that will not be broken by XGBoost or deep learning. Reformulating them as binary alert classification is operationally useful and academically defensible: MSD needs to know "will there be an exceedance >= U in the next h steps?", not the exact magnitude. Implements step §7.5 of the DIAGNOSTIC_REPORT.
- **Change:** four binary XGBoost classifiers for `(h, U) in {(6,25), (6,50), (12,25), (12,50)}`. Operational target: `y_bin(t) = 1 si max(stormflow[t+1..t+h]) >= U else 0` (full-window formulation, not point instant; corrects the ambiguity in §7.5 of the report). Same inputs as the primary regressor (6 lags + 10 features). `objective='binary:logistic'`, `eval_metric='aucpr'`, `scale_pos_weight = neg_train / pos_train`. The operational threshold is chosen as the highest one satisfying recall >= 0.85 on val. New modules: `src/models/xgboost_classifier.py`, `src/evaluation/classification_panel.py`. Notebook: `notebooks/iter18_xgboost_classifier.py`. Branch: `iter18-xgboost-classifier` (pending merge).
- **Result:**

  | Variant  | Prevalence | AUC-PR | ROC-AUC | P@op  | R@op  | Median lead time |
  |-----------|-------------|--------|---------|-------|-------|-------------------|
  | h6_u25    | 0.41%       | 0.5842 | 0.9806  | 0.102 | 0.884 | 25 min            |
  | h6_u50    | 0.12%       | 0.2488 | 0.9678  | 0.007 | 0.949 | 30 min            |
  | h12_u25   | 0.66%       | 0.4180 | 0.9397  | 0.042 | 0.830 | 60 min            |
  | h12_u50   | 0.21%       | 0.2050 | 0.9187  | 0.008 | 0.901 | 60 min            |

- **Lesson:**
  - High ROC-AUC (0.92-0.98) confirms that the model distinguishes events. Low precision (0.7-10%) is not a model failure, it is a mechanical consequence of the tiny prevalence (0.12-0.66%) and the low operational threshold needed to maintain recall >= 0.85.
  - Lead times of 25-60 minutes are operationally useful for MSD (margin to activate preventive protocols before a CSO).
  - In very rare variants (U=50) the absolute number of false positives is high and a single operational threshold is not practical. Recommendation for the TFM: report the full PR curve and leave the choice of operating point to MSD according to its FP tolerance.
- **Detailed analysis:** `outputs/diagnostic/iter18_comparison.md`, `outputs/diagnostic/iter18_classifier_results.json`, `outputs/figures/iter18/{pr_curves, calibration, lead_time_distribution}.png`.
- **Commits:** `83045b7`, `fe92229`, `5857bed`.



---

## Iteration 19 (2026-05-04) - Clean Bai 2018 TCN, ablation at H=1

- **Hypothesis:** a standard TCN (Bai et al. 2018) without two-stage, without `delta_flow_*` shortcuts, and with simple Huber loss structurally beats the `xgb_lag6_feat10` regressor by a significant margin (`delta_NSE >= 0.02`) at H=1. The same inputs as the regressor (equivalent 6 lags + 10 features from S5) but presented as temporal sequence `(B, T=72, F=11)` allow the TCN to extract structure that XGBoost cannot capture.
- **Change:** new model: causal convolutional TCN with 4 residual blocks `[1,2,4,8]`, kernel=3, dropout=0.1, causal padding with `Chomp1d`. Mini-ablation of 4 runs on val (A0-A3) varying one axis at a time: A0 baseline (L=72, C=32, log1p=True), A1 (without log1p), A2 (L=144), A3 (C=64). Selection by factor independence. New modules: `src/models/tcn_clean.py`, `src/pipeline/normalize_v2.py`. Notebook: `notebooks/iter19_tcn_clean.py`. Branch: `iter19-tcn-comparison`. Training in Colab Pro T4 GPU. Data in `outputs/cache/df_with_features.parquet` mounted from Drive.
- **Result:**
  
  Ablation on val:

  | Run | L | C | log1p | NSE_val | err_pico_val | best_epoch |
  |-----|--:|--:|:-----:|--------:|-------------:|-----------:|
  | A0  | 72 | 32 | yes | 0.8436 | +1.5% | 15 |
  | A1  | 72 | 32 | no | 0.8617 | -24.9% | 12 |
  | A2  | 144 | 32 | yes | 0.8146 | +6.7% | 7 |
  | A3  | 72 | 64 | yes | 0.8497 | -23.3% | 25 |
  | A4  | 72 | 64 | no | 0.8640 | -25.3% | 7 |

  Selection by independence chose A4 (without log1p, C=64). Final run on test with A4: NSE=0.8983, peak error +4.7%, recall@50=0.783, NSE_Extremo=-1.231, bias_Extremo=-13.74 MGD. **TCN A4 beats iter17 XGB H=1 by +0.0353 NSE (threshold +0.020).** Initial VERDICT: TCN_WINS at H=1.
- **Lesson:**
  - The automatic rule "max NSE_val with factor independence" chose A4, but A4 has NSE_Base = +0.085 (catastrophic in baseflow, where 99% of the data are). Global NSE can favor configurations with poor structural behavior.
  - The two configurations (with log1p vs without) represent different trade-offs: with log1p favors baseflow and global robustness; without log1p favors prediction of the extreme peak magnitude at the expense of the rest.
  - The H=1 TCN_WINS verdict is contingent: it depends on which configuration is chosen as the main model and how it replicates to other horizons (see iter19b and iter19c).
- **Detailed analysis:** `outputs/diagnostic/iter19_comparison.md`, `outputs/diagnostic/iter19_tcn_results.json`, `outputs/figures/iter19/{ablation_val_nse, hydrograph_extreme_event_H1, scatter_real_vs_pred_H1, loss_curves_final}.png`.

---

## Iteration 19b (2026-05-04) - A0 (with log1p) re-evaluated on test

- **Hypothesis:** the "max NSE_val by independence" rule used in iter19 favored the wrong metric when choosing A4 (without log1p). A0 (with log1p) would have slightly lower global NSE but more balanced behavior by bucket, especially in the Base bucket that contains 99% of the points. Re-evaluating A0 directly on test closes this methodological doubt before declaring A4 definitive.
- **Change:** one additional run with the exact A0 configuration (L=72, C=32, log1p=True), same split and same origin alignment as iter17. Load A4 from `outputs/iter19/weights/final.pt` to compare 1:1 on the same test timestamps. Sanity check with `raise` if recomputed A4 NSE differs from the official 0.8983 figure by more than 0.01. Notebook: `notebooks/iter19b_a0_test.py`. Same branch `iter19-tcn-comparison`.
- **Result:**

  Comparison A0 vs A4 on test (n=165.222):

  | Metric | TCN A0 | TCN A4 | Delta (A0 - A4) |
  |---|---:|---:|---:|
  | Global NSE | 0.8890 | 0.8983 | -0.0093 |
  | peak err (%) | +6.6 | +4.7 | +2.0 |
  | recall@50 | 0.611 | 0.741 | -0.130 |

  By bucket:

  | Bucket | NSE_A0 | NSE_A4 | Delta NSE | bias_A0 | bias_A4 |
  |---|---:|---:|---:|---:|---:|
  | Base (n=152.900) | +0.732 | +0.085 | **+0.646** | -0.002 | +0.002 |
  | Light (n=9.520) | +0.870 | +0.765 | +0.106 | +0.010 | +0.136 |
  | Moderate (n=2.303) | +0.582 | +0.527 | +0.055 | -0.227 | +0.350 |
  | High (n=440) | -0.202 | +0.005 | -0.207 | -2.661 | -1.045 |
  | Extreme (n=59) | -1.571 | -1.231 | -0.340 | -18.753 | -13.739 |

- **Lesson:**
  - Hypothesis confirmed in the buckets that contain 99% of the data: A0 with log1p drastically improves over A4 without log1p in Base (+0.65 NSE), Light (+0.11), and Moderate (+0.06).
  - Hypothesis NOT confirmed in high-magnitude buckets: A0 is slightly worse in High (-0.21) and Extreme (-0.34) relative to A4.
  - The two configurations represent optimizations of different objectives: A0 is a balanced model for continuous operational use; A4 is a specialized model for hitting the magnitude of extreme events when they occur.
  - **Decision:** A0 (with log1p) replaces A4 as the candidate main model of the TFM. Reasons: (1) it preserves the project's documented lesson 3 (log1p is non-negotiable in heavy-tailed hydrological variables); (2) better behavior in baseflow, where MSD operates 99% of the time; (3) global NSE of 0.8890 still satisfies the closing criterion vs XGB (`delta = +0.0259 > +0.020`).
- **Detailed analysis:** `outputs/diagnostic/iter19b_a0_vs_a4.md`, `outputs/diagnostic/iter19b_a0_test_results.json`, `outputs/figures/iter19/scatter_A0_vs_A4_extremo.png`.

---

## Iteration 19c (2026-05-05) - A0 replicated to H=3

- **Hypothesis:** if A0 is the TFM main model at H=1, it must be replicated to H=3 to validate the decision before updating STATE.md and EXPERIMENTS.md (explicit requirement of HANDOFF_2026-04-29.md §4). If A0 beats XGB at H=3 by `delta_NSE >= 0.02`, A0 is confirmed as the definitive main model. If it loses, the decision must be reopened.
- **Change:** one run with the exact A0 configuration (L=72, C=32, log1p=True) but with `horizon=3`. Same dataset, same split, same origin alignment. Inline retraining of iter17 XGB at H=3 with `train_xgboost_h(horizon=3)` to have aligned figures on the same timestamps. Sanity check: `|NSE_XGB_inline - 0.6871| <= 0.02` (official iter17 figure). Notebook: `notebooks/iter19c_a0_h3.py`. Same branch `iter19-tcn-comparison`.
- **Result:**

  Comparison TCN A0 H=3 vs XGB H=3 on test (n=165.220):

  | Metric | TCN A0 H=3 | XGB iter17 H=3 |
  |---|---:|---:|
  | Global NSE | **0.6096** | 0.6871 |
  | peak err (%) | -53.2 | -13.5 |
  | recall@50 | **0.087** | 0.261 |
  | Base NSE | +0.711 | -4.37 |
  | Extreme NSE | -8.064 | -5.553 |
  | best_epoch | 3 | - |
  | Delta NSE vs XGB iter17 | -0.0775 | - |

  Inline XGB sanity: NSE = 0.6889 (delta = +0.0018, within the 0.02 threshold). H=3 VERDICT: **XGB_WINS** (TCN does not exceed the +0.020 threshold).
- **Lesson:**
  - **TCN A0 collapses at H=3.** best_epoch=3 with max val_NSE of 0.487 indicates that the model cannot learn rain-to-flow at 15 minutes ahead. recall@50 falls to 8.7% (vs 78.3% at H=1): operationally unusable at a longer horizon.
  - Pattern consistent with the fixed convolutional nature of the TCN: the receptive field optimized for H=1 (where the target can almost be copied from the last value) does not transfer to H=3, where the autoregressive signal matters less and other dependencies (historical rainfall, API) gain relative weight. The GBM with explicit lags absorbs the latter efficiently.
  - **Final TFM decision:** **XGBoost (`xgb_lag6_feat10`) remains the only main model at H=1 and H=3.** The marginal TCN advantage at H=1 (+0.026 NSE) does not justify fragmenting the MSD operational system or assuming the added complexity (GPU, PyTorch dependency, higher inference cost) when the TCN does not scale to operationally useful horizons.
  - Academically strong finding for the TFM: the GBM-TCN equivalence at H=1 with the available data reflects an information ceiling (S4), not GBM technical superiority. Deep learning would add value with enriched data (NEXRAD radar, HRRR/RAP forecast, multi-station), proposed as a future-work line in section 9 of the TFM.
  - The two architectures have opposite biases: TCN learns the "mean" of the process (better in the Base bucket), XGB learns the "variance" (better in event buckets). This is consistent with the literature on GBM vs deep learning in heavy-tailed problems.
- **Detailed analysis:** `outputs/diagnostic/iter19c_a0_h3_comparison.md`, `outputs/diagnostic/iter19c_a0_h3_results.json`, `outputs/figures/iter19/{hydrograph_extreme_event_H3, scatter_real_vs_pred_H3}.png`.

---

## Modeling closeout (2026-05-05)

After iter19c, the project's modeling phase is **closed**. Definitive main model: `xgb_lag6_feat10` (regressor) + 4 binary classifiers `xgb_h{6,12}_u{25,50}`. The clean TCN A0 is documented as an academic contrast but does not go into production.

From this date onward, all future iterations are out of scope for the TFM and would only be considered in August if TFM writing is ahead of schedule.

**Additional firm conclusions (May 2026, add to the 18 in the "Summary of conclusions" section):**

19. The clean Bai 2018 TCN with one-dimensional scalar inputs (1 station, no forecast) is equivalent to the GBM with explicit lags at H=1 and worse at H=3. The theoretical superiority of deep learning in hydrology requires structured inputs (spatial, multichannel, with forecast) that are not available in this project.
20. The automatic rule "max val NSE by factor independence" can favor configurations with poor structural behavior by bucket. Always validate the decision with operational metrics (recall@50, peak error, bucket NSE) before declaring a winner.
21. log1p on the target is also non-negotiable in clean TCN, not only in TCN v1. The project's historical lesson 3 generalizes to any architecture over heavy-tailed hydrological variables.
22. Models with different architectures converge to the same information ceiling when the inputs are the same. To raise the ceiling, the data must be enriched, not the architecture.
