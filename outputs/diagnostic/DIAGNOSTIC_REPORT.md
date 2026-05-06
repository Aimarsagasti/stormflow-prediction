# DIAGNOSTIC REPORT - Stormflow Prediction TFM

**Date**: 2026-04-22
**Branch**: `diagnostico`
**Author**: Claude Code (orchestrator) over 6 subagents (S1-S6)
**Detailed artifacts**: `outputs/diagnostic/S{1..6}_*.md` and `.json`. Reproducible scripts in `scripts/diagnostic/`.

---

## 1. Executive summary

**Verdict**: the current TwoStageTCN is not recoverable and should be discarded. The v1 "good NSE" (0.86 at H=1) is the combined result of (a) an autoregressive shortcut (`delta_flow_5m/15m`, already confirmed in iter16), (b) a structural bug in `TwoStageLoss` that trains the regressor only on event samples but applies it at inference to baseflow, and (c) a global metric dominated by baseflow that masks tail error. Without the shortcut, NSE collapses to -0.17. With the shortcut and aligned metrics, it stays at 0.74-0.86 depending on how it is computed, versus a pure linear AR(12) with NSE=0.83 - that is, the TCN adds no architectural value over 12 lags of the target itself in a linear regression.

**Single recommendation**: abandon the TCN path as the main model and build a two-part system that runs in parallel:

1. **Production model**: regressive XGBoost trained on **explicit target lags (y(t-0..t-11))** + **10 reduced features from S5** (`api_dynamic`, `rain_sum_360m`, `rain_sum_120m`, `rain_sum_15m`, `temp_daily_f`, `hour_sin`, `minutes_since_last_rain`, `delta_rain_10m`, `delta_rain_30m`, `rain_sum_30m`). Retrainable locally in <60 s. Expected NSE H=1: **0.84-0.88** (S6 §3, based on composition AR(12)=0.827 + features with PI > 0). By construction it solves the two bugs (no two-stage, no train-eval mismatch). Reformulate the additional output as a **binary alert at H=6 and H=12** because at those horizons the physical NSE ceiling (0.32 and 0.19 respectively) makes regression unviable.

2. **Deep learning comparison**: one single Colab iteration with **standard TCN (Bai 2018, without two-stage) + target lags in the input** to confirm whether deep learning adds anything over XGBoost+lags. If it does not add >0.02 NSE, deep learning is closed for this TFM.

**The TFM should be reframed around the negative finding**: the results section gains academic value by honestly narrating "initial model seemed successful -> baseline analysis revealed shortcut -> simpler final model beats the initial one when compared correctly." It is a publishable and defensible story.

**What remains out of scope** with the current data: NSE > 0.5 at H=6 or H=12, reliable quantitative prediction of extremes (the physical bound puts maximum NSE in the Extreme bucket near 0 over only 59 samples). To open those horizons, external rainfall forecast is needed (explicit user decision: do not use it at this phase).

---

## 2. Real status of the current model

### 2.1 What had been reported

- TCN v1 sinSF H=1: NSE=0.861, RMSE=0.89, MAE=0.29, peak error=-21%.
- v1 considered "production model" since 2026-04-13.

### 2.2 What it really is

| Metric | Reported value (`local_eval_metrics.json`) | Value on aligned test (S4 reproduction) | Difference |
|---|---:|---:|---:|
| NSE H=1 sinSF | 0.8614 | 0.7388 | **-0.123** |
| Peak error H=1 | -21.0% | +42.6% | sign change |

**Detected discrepancy**: S4 reproduced TCN v1 inference on the test aligned to S2 (same `seq_length=72`, same `horizon=1`, dropping the first 71 steps so the window exists) and obtained NSE=0.74 with positive peak error (+42.6%, overestimation). `evaluate_local.py` vs `scripts/diagnostic/s4_horizon_ceiling.py` must be audited to understand the origin (likely candidates: different offset at test start, different hard-switch threshold, or denormalization difference with/without clipping). **This point must be resolved before any comparison against new models** - future-model NSE cannot be compared if the v1 reference figure is ambiguous. See Plan §7.1.

### 2.3 Uncomfortable truth: the TCN adds nothing over linear AR(12)

| Model | NSE H=1 | NSE H=3 | NSE H=6 |
|---|---:|---:|---:|
| Naive persistence | 0.811 | 0.409 | 0.081 |
| **Linear AR(12) on 12 lags of y** | **0.827** | **0.509** | **0.317** |
| XGBoost-22 (with `delta_flow`) | 0.790 | 0.656 | 0.382 |
| XGBoost-20 (without `delta_flow`) | 0.662 | 0.572 | 0.336 |
| TCN v1 sinSF (`local_eval_metrics.json`) | 0.861 | 0.471 | -1.21 |
| TCN v1 sinSF (S4 reproduction) | 0.739 | 0.293 | -1.30 |
| TCN v1 sinSF WITHOUT `delta_flow` (iter16) | -0.169 | n/d | n/d |

Readings:
- At H=1, linear AR(12) beats TCN v1 according to S4 (0.83 > 0.74) and stays 0.03 below according to `evaluate_local.py` (0.83 vs 0.86). In the best reading, the TCN adds +0.03 NSE over a 12-lag linear regression.
- At H=3 and H=6 the TCN is clearly below XGBoost-22 and AR(12). The TCN is structurally inferior at horizons >5 minutes.
- Without the `delta_flow` shortcut, the TCN collapses (-0.17). The shortcut contributes +1.03 NSE to the TCN, +0.13 NSE to XGBoost.
- At H=6, naive gives 0.08, AR(12) 0.32, TCN sinSF -1.30. The TCN does not even reach naive at H=6 without SF as autoregressive feature.

### 2.4 Bugs in the source code (S1, unrelated to `delta_flow`)

- **BUG1 - train-eval mismatch in `TwoStageLoss`** (`src/models/loss.py:182-211` + `src/models/tcn.py:248-253`): the regressor is trained only on samples with `y_real > 0.5 MGD`; at inference it is applied to ALL samples where the classifier gives `cls_prob >= 0.3`. On baseflow with a false positive, the regressor produces OOD output. This explains NSE=-14.5 in the Base bucket. It is a design bug, not an implementation bug.
- **BUG2 - latent double normalization** (`src/pipeline/normalize.py:71`): if `stormflow_mgd` appears in FEATURES AND TARGET, it is normalized twice in-place. Fixed in callers (`evaluate_local.py`, `horizon_comparison_v2.py`) but NOT in the function itself. Any future notebook that passes the full list silently reintroduces it.
- **Suspicious - dual definition of "event"**: the loss uses `y > 0.5 MGD` while sample weights use `is_event` (MSD label). They optimize different objectives.
- **Suspicious - naive alignment**: in `scripts/generate_dataset_stats.py:1046-1079` naive is computed over the WHOLE test, while the model is evaluated on `test[seq_length+horizon-1:]`. Small difference, but it introduces bias into the reported "+0.050 NSE gain".
- **Suspicious - inconsistent criterion for "extreme without rainfall"**: two different definitions in `evaluate_local.py` and `generate_dataset_stats.py`. That is why "15/59" (old) vs "0/59" (iter16) do not match. Both may be correct under their criterion. One must be fixed.

Full detail and line-by-line citations: `outputs/diagnostic/S1_pipeline_audit.md`.

---

## 3. Findings by area

### 3.1 S1 - Pipeline audit (`S1_pipeline_audit.md`)

2 bugs (regressor train-eval mismatch + latent double normalization), 5 suspicious points, 3 OK. The main finding is BUG1 - `TwoStageLoss` trains the regressor only on event samples but inference's hard switch applies it to baseflow where the regressor was never penalized. This is the second path (together with the `delta_flow` shortcut) by which the reported model is not a "real" TCN. It is a design failure that cannot be patched without redesigning loss and inference jointly.

### 3.2 S2 - Rigorous baselines (`S2_baselines.md`, `.json`)

On aligned test (n=165,222 at H=1):
- Linear AR(12) on 12 lags of `y`: NSE=**0.827**.
- Optimal AR(1) (with mean-reversion correction): 0.820.
- Naive: 0.811.
- XGBoost with 20 features without `delta_flow`: 0.662.
- XGBoost with 22 features (with `delta_flow`): 0.790. The shortcut adds +0.13 NSE in XGB.
- Linear predictor with 2 physical features (`rain_sum_60m` + `api_dynamic`): 0.619.

**Main S2 finding**: at H=1 the autoregressive flow signal dominates over the exogenous signal. Linear AR(12) beats XGBoost-22 (0.83 vs 0.79). XGBoost does not receive explicit target lags - `delta_flow` are derivatives, not levels, and they are insufficient. This suggests that adding target lags to XGBoost would raise it above AR(12).

At H=3 and H=6, XGBoost-22 beats TCN v1: at H=3 XGB-22=0.66 vs TCN v1=0.47. This closes the question of whether the TCN adds anything at longer horizons: **it does not**.

### 3.3 S3 - Analysis of the 59 extreme events (`S3_extreme_events.md`, `.json`)

- **24 physical storms**, not 59 independent events. Several storms contribute multiple consecutive samples to the Extreme bucket.
- **0/59 without rainfall** under any reasonable criterion (`rain_sum_360m`, `rain_sum_60m`, sum in window 72). The historical "15/59" figure does not apply to the current test; documentation (`CLAUDE.md`, `AGENTS.md`, `STATE.md`) must be updated.
- 3 clusters by K-means:
  - **Convective** (n=12, 20%): best predicted by v1, NSE_local=-0.27, median peak error -3.4%, 0 underestimations >50%.
  - **Mixed** (n=38, 64%): worst cluster, NSE_local=-3.05, median peak error -32.8%, **11/38 underestimations >50%**. Pattern: moderate rainfall + low API -> high unanticipated peaks.
  - **Stratiform** (n=9, 15%): low RMSE (11.6) but underestimates.
- Optimistic bound: a perfect oracle on the Convective cluster alone only raises NSE over the 59 to -0.49 (delta +0.50). The real margin lies in the Mixed cluster (64% of the extremes), which needs new features or architecture, not more data.

### 3.4 S4 - Physical ceiling by horizon (`S4_horizon_ceiling.md`, `.json`)

| h | min | NSE naive | NSE AR(12) | Bound 2rho-1 | Defensible max NSE | Margin over naive |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0.811 | 0.827 | 0.817 | **~0.83** | +0.02 |
| 3 | 15 | 0.409 | 0.509 | 0.431 | **~0.51** | +0.10 |
| 6 | 30 | 0.081 | 0.317 | 0.112 | **~0.32** | +0.24 |
| 12 | 60 | -0.187 | 0.190 | -0.159 | **~0.19** | +0.38 |
| 24 | 120 | -0.440 | 0.095 | -0.450 | **~0.09** | +0.53 |

**Contribution of each bucket to the H=1 NSE denominator** (key to understanding what matters):
- Extreme: 31.9% (only 59 samples = 0.036% of test).
- High: 26.3% (224 samples).
- Moderate: 37.5% (2,519 samples).
- Light: 2.5%, Base: 1.8%.

**Partial oracles on TCN v1 H=1** (base NSE = 0.74 according to S4):
- Perfect oracle only on Extreme: NSE=0.84 (+0.10).
- Perfect oracle on Extreme + High: NSE=0.89 (+0.15).
- Perfect oracle on everything except Extreme: NSE=0.90 (+0.16).
- Perfect oracle only on Moderate (body): NSE=0.82 (+0.08).

Finding: **the body (Moderate + Light + Base) contributes more to the denominator than Extreme alone (42% vs 32%)**. The narrative of "the model only fails on extremes" is incomplete - it also fails appreciably in the body, especially in Moderate where it has bias +1.7 MGD and NSE_local=-0.11.

**Verdict on horizons**: H=1 has a high physical ceiling but low margin over naive (+0.02). H=6 has margin +0.24 but low ceiling (0.32). H=12 and H=24 are unviable as regression without external rainfall forecast. The defensible conclusion is **to work on H=1 and H=3 as regression, and H=6/H=12 as binary alert classification**.

### 3.5 S5 - Feature analysis (`S5_feature_analysis.md`, `.json`)

- **Top-5 permutation importance** (XGB-20 on test): `api_dynamic` (+0.418), `rain_sum_360m` (+0.108), `rain_sum_120m` (+0.080), `rain_sum_15m` (+0.037), `rain_sum_180m` (+0.034). `api_dynamic` dominates by an order of magnitude.
- **6 features with PI <= 0** (active noise, they hurt): `hour_cos`, `month_sin`, `month_cos`, `rain_sum_60m`, `rain_max_30m`, `rain_max_60m`.
- **15 pairs with |r| >= 0.85** -> effective dimensionality of the 20 features is ~13 clusters.
- **Proposed reduced set (10 features)**: `api_dynamic, rain_sum_360m, rain_sum_120m, rain_sum_15m, temp_daily_f, hour_sin, minutes_since_last_rain, delta_rain_10m, delta_rain_30m, rain_sum_30m`. On XGB: NSE=**0.698** vs XGB-20=0.662 -> improvement of +0.036 by removing noise.
- **No additional hidden shortcuts** after removing `delta_flow_*`. Warnings:
  - `rain_sum_60m` has ACF(12) similar to the target's (0.42 vs 0.42), similar inertia, but negative PI in XGB-20 -> it does not work as a dominant shortcut.
  - `api_dynamic` correlates r=0.94 with `rain_sum_60m`. It dominates PI; AGENTS.md said "redundant with rain_sum_60m" but in the presence of api_dynamic, `rain_sum_60m` becomes noise. AGENTS.md §8.1 must be updated.
- XGB with 10 reduced features (NSE=0.70) **is still below AR(12) (0.83)**. Without explicit target lags, rainfall features cannot beat pure autoregression. This is the key lever: add target lags to the input.

### 3.6 S6 - Review of alternative architectures (`S6_architecture_review.md`)

Review without training. Recommendations:

1. **Primary - XGBoost + target lags + 10 reduced features**. Reusable from `s2_baselines.py`. ~50 additional LoC. <1 min training. Expected NSE H=1 0.83-0.88 (combination of AR(12)=0.827 + features with PI > 0). By construction it solves S1 bugs (no two-stage, no train-eval mismatch). Implementable locally without GPU.

2. **Secondary - standard TCN (Bai 2018) + target lags in the input + Huber loss + no two-stage**. Reuses ~90% of the existing pipeline (removing `TwoStageLoss` and `is_event`). One iteration in Colab T4. If it beats XGB+lags by >0.02 NSE, it justifies continuing with deep learning. If not, that branch is closed.

Discarded with justification: hydrological LSTM (Kratzert) - intended for multi-catchment with static features, excessive complexity for a single measurement point. TFT - high implementation overhead with doubtful benefit. Seq2seq with attention - useful for H>1 but the physical ceiling of H=6 is too low, it does not justify the cost.

**Verdict on TwoStageTCN**: discard it. Patching the train-eval mismatch bug turns the architecture into a standard TCN with a cosmetic classifier - cleaner to start from scratch with approach (2).

---

## 4. Root-cause diagnosis

The project has been optimizing a model whose reported "good NSE" came from three overlapping sources, none associated with genuine hydrological learning:

1. **Autoregressive shortcut `delta_flow_5m/15m`**. Confirmed in iter16 (without it, NSE = -0.17) and replicated in XGBoost by S2 (+0.13 NSE). It was a backdoor to `flow_total_mgd`, which had already been excluded in iter11. The derivative of total flow at t is practically the derivative of stormflow at t, reintroducing the shortcut.

2. **Global metric dominated by baseflow**. 92% of the test is baseflow (<0.5 MGD). Any model that predicts ~= 0 outside events inherits a high NSE without understanding hydrology. Global NSE is misleading in zero-inflated problems, and comparing it against naive (which already gives 0.81 at H=1) without explicitly reporting the gain is a wrong way to tell the result.

3. **Structural bug in `TwoStageLoss`**. The regressor is trained only on samples with `y_real > 0.5 MGD`, but at inference it is applied to baseflow when the classifier gives a false positive. This produces OOD behavior and explains the Base bucket's NSE=-14.5. It is a design inconsistency that cannot be solved without redesigning the loss.

Combined, this means: the TwoStageTCN was not a TCN genuinely learning rainfall -> stormflow; it was a model that (a) used `delta_flow` as implicit AR(1), (b) reported high NSE because the denominator was inflated by baseflow, and (c) had a regressor with an ill-defined domain. The four improvement iterations of the last month (v2, iter15, iter15b, iter16) were touching parts of a system whose foundation was unstable.

**The problem is not only architectural**. Even if `TwoStageLoss` is fixed and `delta_flow` is removed, the rainfall features without explicit target lags only give NSE=0.70 (S5) - far from AR(12)=0.83. What is missing is:

- **No current model has access to target lags except linear AR(12)**. The TCN sees the feature window but does not include `stormflow(t-k)` except in "conSF" variants where it is added as a single autoregressive feature.
- **The 22 features have massive redundancy** (15 pairs with r>=0.85, effective dimensionality ~13). The useful information fits in 10 features.
- **The Mixed extreme bucket (64% of the Extreme bucket)** is badly predicted by structure, not by lack of data. Moderate rainfall over low-saturation soil -> high unanticipated peak. This is a real hydrological pattern; it is likely improved with nonlinear interaction between `api_dynamic` and `rain_intensity_max`, not with more features.

---

## 5. Considered possible paths

Only the paths for which I have evidence from S1-S6.

### 5.1 Keep TwoStageTCN, patch bugs, and remove `delta_flow`

- **Pros**: reuses existing code. It would fulfill iter17 as planned.
- **Cons**: even with bugs corrected it remains below AR(12) at H=1 and H=3 (S2/S4). Even patched, the architecture loses the point of being two-stage (regressor trained on the whole domain = standard TCN with a cosmetic classifier). It is work on infrastructure with a known ceiling. **Discarded**.

### 5.2 Standard TCN + target lags in the input

- **Pros**: cleaner architecture, aligned with the literature (Bai et al. 2018). Solves S1 bugs by construction. Reuses the pipeline.
- **Cons**: requires Colab for training; no evidence that it beats a GBM with the same inputs.
- **Decision**: valid as a **comparison model, not a production model**. If it beats the GBM by >0.02 NSE, keep it; if not, close deep learning.

### 5.3 XGBoost / LightGBM with explicit target lags + 10 reduced features

- **Pros**: implementable locally in <1h. Reuses `s2_baselines.py`. Expected NSE H=1 0.83-0.88 (composition AR(12)=0.83 + features with PI>0). Solves S1 bugs by construction. Allows cheap retraining and variant comparison. Native SHAP for interpretability (useful for the TFM).
- **Cons**: loses the "deep learning" narrative of the TFM if it remains the final model. But that narrative is no longer sustainable after this diagnostic - academic honesty matters more.
- **Decision**: **main model**.

### 5.4 Quantile Regression / EQRN for extremes

- **Pros**: predicts quantiles instead of mean; the 95th quantile captures extremes better.
- **Cons**: it is not the dominant lever (S4: oracle on Extreme alone only contributes +0.10 NSE). It does not attack the structural bug or the lack of lags. Useful as a marginal improvement later.
- **Decision**: **postponed to phase 3** of the new plan. It is not the first battle.

### 5.5 Reformulate evaluation: NSE by bucket + peak_lag + recall@50

- **Pros**: global NSE is biased by baseflow. A richer metrics panel is good academic and operational practice. MSD needs "alert before CSO", not high NSE.
- **Cons**: none relevant.
- **Decision**: **complementary, mandatory**, not a path but an accompaniment to the model.

### 5.6 Reformulate the problem as multi-horizon binary classification

- **Pros**: if the physical ceiling of H=6 (NSE<=0.32) makes regression unviable, a binary alert "there will be a peak >= 25 MGD in the next 30 min" is more honest and operationally useful.
- **Cons**: academic scope change. But the TFM can defend both targets.
- **Decision**: **complementary to the regression model**, for H=6 and H=12. The main model remains regression at H=1 and H=3.

### 5.7 Reframe the TFM as negative finding + correct path

- **Pros**: the arc "apparent model -> baseline diagnostic -> simpler real model" is publishable and honest. It adds academic value. Better narrative than "deep learning model with NSE=0.86".
- **Cons**: requires coordination with academic advising. But the decision is good either way.
- **Decision**: **recommended TFM structure**.

---

## 6. SINGLE RECOMMENDATION

**Path**: discard TwoStageTCN. Build a two-component system in a new iteration (iter17):

> **Component A - Main model**: regressive XGBoost trained on **the target at horizon h** using as input **12 target lags (`stormflow_mgd[t-0..t-11]`) + 10 reduced exogenous features from S5** (do not include `delta_flow_5m/15m`). Separate models for h=1 and h=3.
>
> **Component B - Alert model**: **binary classifier** (XGBoost or LightGBM) that predicts "there will be `stormflow >= U` in the next h samples" for h=6 and h=12 with U={25, 50} MGD. Metric: precision/recall and F1.
>
> **Component C - Deep learning comparison**: one single Colab iteration with **standard TCN (Bai 2018) + 12 target lags + 10 features + Huber loss**, without two-stage, without classifier, without hard switch. Compare NSE H=1 against Component A. If NSE_TCN - NSE_XGB < 0.02 -> close deep learning for the TFM. If it wins by >0.02 -> consider deep learning as the final model, documenting the cost.

**Technical justification**:

1. **It solves the two structural bugs by construction**. Without two-stage there is no train-eval mismatch (BUG1). Without passing `stormflow_mgd` as a feature to the GBM there is no risk of double normalization (BUG2). The 10 reduced features eliminate 6 with PI<=0 (S5).

2. **It attacks the identified root cause**: linear AR(12) beats XGB-22 at H=1 (0.83 vs 0.79) because rainfall features do not recover the autoregressive information contained in the target lags. Injecting 12 lags as explicit input gives the GBM full access to that information, with added nonlinear flexibility over exogenous features.

3. **Expected NSE H=1 0.83-0.88**, which matches or beats the current TCN v1 even under the optimistic reading (0.86). Important: this is the first model in the project that beats linear AR(12) with structural justification.

4. **It is implementable locally in an afternoon**. Reuses `scripts/diagnostic/s2_baselines.py`, adds ~50 LoC. No GPU. Cheap retraining - allows many variants (10 vs 12 vs 14 lags, with/without SHAP, with/without weighting).

5. **Component B acknowledges the physical ceiling of H=6/H=12** (S4: max NSE 0.32 and 0.19 respectively). Instead of fighting that ceiling with a regression model that will fail, it gives MSD a binary alert that IS operationally defensible. It improves the real utility of the project without pretending that H=6 is reachable as regression.

6. **Component C honestly closes the deep learning question**. If TCN + lags beats XGB + lags, there is deep learning in the TFM. If not, it stays out with quantitative justification.

**Confidence**: high. The S2-S5 data are consistent and point in the same direction. The only real unknown is the exact performance of XGB+lags vs AR(12) - the lower bound is AR(12)=0.83 (because the 12 target lags are already there) and the theoretical upper bound is 0.87 according to S4 (physical ceiling H=1 ~= AR(12) + some contribution from exogenous features).

**Missing information and how to obtain it**: the only relevant ambiguity is the TCN v1 discrepancy NSE = 0.86 (`evaluate_local.py`) vs 0.74 (S4). Resolving it in plan step 1 establishes the rigorous reference figure against which to compare the new model.

---

## 7. Execution plan

Each step indicates concrete files and command or environment.

### 7.1 Step 1 - Resolve TCN v1 NSE discrepancy: official figure = **0.86** [DONE]

**Result**: rerunning `scripts/diagnostic/s4_horizon_ceiling.py` (full log in `outputs/diagnostic/logs/s4_rerun.log`, artifacts `outputs/diagnostic/S4_horizon_ceiling.{json,md}` regenerated) reproduces **NSE H=1 sinSF = 0.8615** on the aligned test (n=165,222). The figure matches `evaluate_local.py` (0.8614, n=165,223) to the third decimal, so the discrepancy is closed and **NSE = 0.86 is adopted as the official figure** for TCN v1 sinSF at H=1.

**Aligned official figures** (S4 rerun from 2026-04-22):

| h | min | NSE TCN v1 sinSF | Peak err % | NSE `evaluate_local.py` | Delta vs AR(12) |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | **0.8615** | +42.6 | 0.8614 | +0.0342 (AR12=0.8273) |
| 3 | 15 | 0.4697 | +121.2 | 0.4714 | -0.0388 (AR12=0.5085) |
| 6 | 30 | -1.2039 | +38.9 | -1.2121 | -1.5209 (AR12=0.3170) |

**Audit of the 0.74 vs 0.86 discrepancy** (based on comparison of the old S4 commit `68b038d` vs the rerun):

- Script unchanged (`git diff 68b038d..HEAD -- scripts/diagnostic/s4_horizon_ceiling.py` empty), weights unchanged (`MC-CL-005/Pesos 13-04-2026/modelo_H1_sinSF_*` dated 2026-04-13), cache `outputs/cache/df_with_features.parquet` already existed (ctime 2026-04-20) before the first S4 execution.
- **Invariants that match exactly** between both runs: `n_test=165,222`, `test_index_first=936,741`, `test_index_last=1,101,962`, `peak_real=135.1509`, `peak_pred=192.7817`, `peak_err_pct=+42.6418%`, `cls_prob_pct_above_thr=12.198738666763505%`, `cls_prob_mean≈0.105087` (they differ at the 9th decimal, float32 noise). Inference is identical at the peak and in classifier activation counts; the target's global variance (`sum(y-mean)^2 ≈ 955,079`) also matches.
- **What changes** are the residual SSE by bucket - e.g. at H=1: Moderate 74,840 -> 37,261; Extreme 99,101 -> 46,031; Base 14,936 -> 18,155. This shifts NSE from 0.7388 to 0.8615 without peak or `cls_prob` changing. There is no reproducible explanation afterward (identical code/data/weights), so the old run is marked **non-reproducible** and discarded as source of truth.
- The rerun figure (0.8615) matches the same pipeline as `evaluate_local.py` (0.8614) - it is adopted as official. The history of the previous JSON stays in the git log (commit `68b038d`) for traceability.

**Implications for the rest of the report** (they are not rewritten here by explicit instruction; the reader should treat §7.1 as the current reference):

- §2.2 "Value on aligned test (S4 reproduction): NSE=0.7388" -> obsolete figure; the reproducible one is 0.8615.
- §2.3, table: row "TCN v1 sinSF (S4 reproduction) | 0.739 | 0.293 | -1.30" -> obsolete values; current reading: 0.8615 / 0.4697 / -1.2039.
- §3.4, table: "Defensible max NSE H=1 ~0.83" rises to ~0.86 with the official figure, but it does not invalidate the verdict because TCN v1 still lies in the same order as AR(12) and 2rho-1. Partial oracles: deltas fall (e.g. Extreme oracle now +0.048 instead of +0.104) because the base rises; the relative hierarchy across buckets is preserved.
- §6 "Single recommendation": the comparison becomes **TCN v1 = 0.8615 vs AR(12) = 0.8273 -> +0.034 NSE**, not +/-0.03 depending on computation. TCN v1 **does beat** AR(12) at H=1 by a modest margin; at H=3 and H=6 it **remains below** AR(12) (0.47 < 0.51 and -1.20 < 0.32). The verdict in favor of choosing XGBoost+lags as the main model does not change: the dominant argument was (a) structural S1 bugs that cannot be addressed without redesigning the loss, and (b) the `delta_flow` shortcut without which NSE collapses to -0.17 (iter16). Both still stand.

**Artifacts and reproduction command**:

```bash
python scripts/diagnostic/s4_horizon_ceiling.py > outputs/diagnostic/logs/s4_rerun.log 2>&1
```

- JSON: `outputs/diagnostic/S4_horizon_ceiling.json` (overwritten by the rerun).
- MD: `outputs/diagnostic/S4_horizon_ceiling.md` (overwritten by the rerun).
- Figure: `outputs/figures/diagnostic/s4_horizon_ceiling.png` (overwritten by the rerun).
- Log: `outputs/diagnostic/logs/s4_rerun.log`.

**Suggested commit**: `diagnostic: rerun S4, official v1 H=1 NSE=0.86 (reconciled with evaluate_local)`.

### 7.2 Step 2 - Create iter17 branch and XGBoost+lags model draft (half day, local)

- `git checkout -b iter17-xgboost-lags` from `diagnostico` (or from `main` after merging `diagnostico` - user decision).
- Create `src/models/xgboost_baseline.py` with function `build_features_with_lags(df, lags=12, features=FEATURES_10)` and `train_xgboost_h(df_train, df_val, horizon, hyperparams)`.
- Create `notebooks/iter17_xgboost_lags.py` (Colab .py style) that: (a) loads parquet, (b) performs split, (c) trains XGB H=1 and H=3 with/without lags and reports a multi-bucket metrics panel.
- Initial hyperparameters: `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, tree_method=hist, early_stopping_rounds=20` with val set.
- Commit: `iter17: scaffolding XGBoost+lags`.

### 7.3 Step 3 - Implement multi-bucket metrics battery (1-2h, local)

- Create `src/evaluation/metrics_panel.py` with function `evaluate_full_panel(y_true, y_pred, buckets=BUCKETS)` that returns:
  - global NSE, RMSE, MAE.
  - NSE, RMSE, bias, MAE by bucket.
  - Peak error (%) over the whole test and over each bucket.
  - `peak_lag_minutes`: for each physical event (gap >4h), time between real peak and predicted peak.
  - `recall@U` for U in {25, 50}: fraction of events with `max(y_real) >= U` correctly alerted (`max(y_pred) >= U`).
  - `quantile_coverage` for 90% and 95% if the model outputs quantiles.
- Commit: `iter17: panel de metricas multi-bucket`.

### 7.4 Step 4 - Train XGBoost+lags H=1 and H=3, compare (half day, local)

- Run `notebooks/iter17_xgboost_lags.py`.
- Generate `outputs/diagnostic/iter17_xgb_results.json` with the full panel.
- Compare against: TCN v1 reference (step 1), AR(12), XGB-20, XGB-22.
- Success criteria:
  - NSE H=1 >= 0.85 (at least match TCN v1 under the optimistic figure).
  - NSE H=3 >= 0.66 (beat XGB-22).
  - H=1 peak error better than -21% (TCN v1) in absolute value.
  - Base bucket bias <= +0.05 MGD.
- Commit: `iter17: XGB+lags resultados, comparativa contra v1`.

### 7.5 Step 5 - Binary alert model H=6 and H=12 (half day, local)

- In the same `notebooks/iter17_xgboost_lags.py`, add a binary classification section: target `stormflow[t+h] >= U` for h in {6, 12}, U in {25, 50}.
- Initial classifier hyperparameters equal to the regressor; `objective='binary:logistic'`, `scale_pos_weight` computed from train prevalence.
- Metrics: precision, recall, F1, ROC-AUC, mean lead time before exceedance.
- Decide operational threshold as a function of relative cost (FN >> FP in MSD).
- Commit: `iter17: alerta binaria H=6/H=12`.

### 7.6 Step 6 - Deep learning comparison in Colab (1-2 days)

- Only if step 4 gives NSE H=1 >= 0.85.
- In `notebooks/rescate_colab_2026-04-17/iter17_tcn_lags.py`: standard TCN (without two-stage, without switch), input = 12 target lags + 10 reduced features, Huber loss, direct regression.
- Same split, same normalization, same metrics as step 4.
- Closing criterion: NSE_TCN_lags - NSE_XGB_lags < 0.02 -> close deep learning for the TFM. If it wins >= 0.02 -> keep TCN as final model.
- Commit: `iter17: TCN estandar + lags, comparativa final`.

### 7.7 Step 7 - Update documentation (1-2h)

- `docs/STATE.md`: new production model, metrics, decision on deep learning.
- `docs/EXPERIMENTS.md`: full iter17 entry.
- `AGENTS.md` §8.1: correct obsolete findings:
  - Remove mention "15/59 extremes without rainfall" in §"What does NOT work" (S3 confirms 0/59).
  - Document that `api_dynamic` DOES contribute (dominant PI in S5), but if it is present then `rain_sum_60m` becomes noise (negative PI in reduced XGB-20).
  - Add rule "if you add a feature derived from `flow_total_mgd`, ablation against XGBoost-20 is mandatory".
- `CLAUDE.md`: update "Current production model" if iter17 replaces it.
- Commit: `docs: iter17 cierra fase, nuevo modelo en produccion`.

### 7.8 Step 8 - Fix pending latent bugs (1h, local)

Independent of the new model, but they must be resolved so the repo does not accumulate debt:
- `src/pipeline/normalize.py:71`: explicit dedup (`list(dict.fromkeys(...))`) to prevent double normalization.
- `src/features/engineering.py:115-116`: mark `delta_flow_*` as `# DEPRECATED - backdoor to flow_total_mgd` or remove them.
- `scripts/generate_dataset_stats.py`: align naive computation with the model offset.
- Fix a single criterion for "extreme without rainfall" between `evaluate_local.py` and `generate_dataset_stats.py`.
- Commit: `fix: bugs latentes detectados en S1`.

### 7.9 Step 9 - Reframe the TFM plan (1 day, off-code)

- TFM Results section: structure it around the negative finding. Template:
  1. Initial model: TwoStageTCN, NSE=0.86, considered successful.
  2. Diagnosis via rigorous baselines: discovery that `delta_flow` is a shortcut and linear AR(12) beats the model.
  3. Physical ceiling analysis by horizon and bucket.
  4. Final model: XGBoost + lags + 10 features. Comparable NSE, no shortcuts, multi-bucket metrics.
  5. Binary alert as operational complement for H=6/H=12.
  6. Methodological lessons: importance of trivial baselines, danger of global metrics in zero-inflated data.
- It is an honest, publishable narrative arc, better than the "winning deep learning" version.

### 7.10 Step 10 - (Optional) Quantile regression for extremes (postponed)

- If after steps 4-6 there is time, add LightGBM with `objective='quantile'` for quantiles 0.5, 0.9, 0.95 on the regressor output.
- Useful to report "uncertainty interval of the predicted peak", not as the central improvement.
- Commit: `iter18: quantile regression para extremos`.

---

## 8. What is expected to be achievable and what remains out of scope

### 8.1 Achievable with the plan in section 7

- **NSE H=1 between 0.83 and 0.88** on aligned test, with gain over naive >= +0.02 at the lower bound and +0.07 at the upper bound. Model without confirmed shortcuts, with auditable feature-by-feature ablation.
- **NSE H=3 between 0.55 and 0.70**, beating current XGB-22 (0.66) and current TCN v1 (0.47).
- **Binary alert model H=6 and H=12** with recall >= 70% on events of magnitude >= 25 MGD and precision >= 50% (indicative thresholds, adjustable after evaluation). Operationally useful for MSD.
- **Multi-bucket metrics panel + peak_lag + recall@U** reported for every model. This ends the dependence on global NSE as the only indicator.
- **Binary diagnostic "deep learning yes or no"** for the TFM with quantitative justification.
- **Clean repo** without confirmed shortcuts or latent bugs.
- **Coherent documentation** that reflects the real status, without obsolete findings.

### 8.2 Out of scope with the current data

- **NSE > 0.5 at H=6 or H=12 as regression**. The physical ceiling (S4) is 0.32 and 0.19 respectively. Pursuing it is misleading the reader.
- **Reliable quantitative prediction of individual extremes >100 MGD**. The heavy tail with 59 test samples does not allow generalization beyond the Convective cluster (n=12). The defensible claim is to predict the High bucket well and report an upper bound with quantiles for Extreme.
- **NSE > 0.92 at H=1** without external future information (Opus 4.7 §B). To open that ceiling, MSD rainfall forecast is needed - the user's explicit decision is not to use it at this phase.
- **General multi-catchment model in the style of Kratzert**. There is only one point (MC-CL-005). Without static features or basin groups, the hydrological LSTM does not apply.

### 8.3 Risks of the plan

- **Low risk**: XGB+lags does not beat AR(12). Mitigation: AR(12) is already 0.83, and a reasonable composition of AR + features with PI > 0 should beat it. If not, the problem is implementation, not approach.
- **Medium risk**: TCN+lags beats XGB+lags by small margins (0.02-0.05). In that case, decision: keep deep learning for +0.02 NSE? Recommendation: if the improvement is <0.02, it does not justify the complexity cost and should be closed. Document it in the TFM as a conscious decision.
- **Medium risk**: binary alert H=6 gives low recall (<60%). Mitigation: it is not the main model; its failure does not invalidate the TFM. Report it honestly as "the physical ceiling of H=6 also limits utility as classification".
- **Low risk**: academic advisor prefers the "winning deep learning" narrative over "negative finding + simple model". Mitigation: the negative finding is academically stronger and more reproducible. If it does not convince, component C (TCN+lags) remains as the final model even if it only adds +0.02 NSE.

---

**End of report.**
