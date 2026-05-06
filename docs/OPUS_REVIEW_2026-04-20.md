# External review by Opus 4.7 (2026-04-20)

Review requested via Claude Code after completing `docs/DATASET_STATS.md`. Full model response, archived as a reference for future iterations.

- **Model:** Claude Opus 4.7 via Claude Code
- **Files reviewed:** CLAUDE.md, AGENTS.md, docs/STATE.md, docs/EXPERIMENTS.md, docs/WORKFLOW.md, docs/DATASET_STATS.md, outputs/data_analysis/local_eval_metrics.json, outputs/data_analysis/dataset_stats.json, src/, notebooks/claude_train.py
- **Prompt used:** available in HANDOFF_2026-04-20.md

---

I have read the files. Moving on to the diagnostic.

  A. Methodological diagnostic: hidden errors

  A1. Implicit leakage via delta_flow_5m and delta_flow_15m [MODERATE]

  src/features/engineering.py:115-116:
  df_feat["delta_flow_5m"] = df_feat["flow_total_mgd"].diff(periods=steps_5m)
  df_feat["delta_flow_15m"] = df_feat["flow_total_mgd"].diff(periods=steps_15m)

  The rule in AGENTS.md §8.1 excludes flow_total_mgd as a feature because r=0.9976 with stormflow. But these two features ARE included (notebook claude_train.py:142). Since flow_total = baseflow
   + stormflow and baseflow varies slowly, delta_flow_5m[t] ≈ delta_stormflow[t]. The model receives, de facto, the derivative of the target at t.

  Quantitative evidence from DATASET_STATS §4: Pearson(delta_flow_15m, stormflow) = 0.376, Spearman in event regime = -0.078. The global correlation is moderate, but in the first step after the jump
  in the target (exactly where it matters for H=1) this feature carries almost all the autoregressive information. This is consistent with the model beating the naive baseline by only +0.050 NSE: the
  marginal gain may be coming from the derivative of flow_total, not from rainfall. It is not future leakage, but it is a backdoor to the same shortcut that was closed in iter 11. Mandatory ablation to
  determine how much NSE the rainfall branch really contributes.

  A2. Train/eval inconsistency in the Two-Stage classifier [HIGH]

  src/models/loss.py:182-188:
  y_true_real = self._denormalize_target(y_true)
  event_label = (y_true_real > self.event_threshold).float()   # threshold=0.5 MGD
  ...
  event_mask = (event_label == 1.0).squeeze(1)
  if event_mask.any():
      reg_pred = reg_value[event_mask]

  The regressor is trained only on samples with y_true_real > 0.5 MGD (~10% of batches). It never sees baseflow samples. But at inference (tcn.py:248-253) the hard switch does torch.where(cls_prob >=
   0.3, reg_value, 0). When the classifier produces a false positive on baseflow (with threshold 0.3, recall is high at the expense of precision), the regressor produces an OOD output because it never
  trained in that region. This conditional-training / unconditional-inference asymmetry contaminates the Base bucket (NSE=-14.5, bias +0.21 MGD): the error is not regressor noise, it is undefined
  behavior.

  In addition, the classification label in training (y_true_real > 0.5) ≠ the is_event label that the pipeline attaches to the DataLoader (sequences.py:51). _compute_sample_weights in sequences.py:100
   uses is_event to give a 1.75× boost, but TwoStageLoss ignores that is_event and redefines the event as y>0.5. If is_event (generated in clean.py/events.py from the MSD .dat files) does not match
   y>0.5 MGD, the weighting system and classifier training are optimizing different things.

  A3. Sample weights applied to quantiles of NORMALIZED y [MINOR but confusing]

  sequences.py:185 computes train_thresholds over df_train[target_col] when it is already log1p+z-score (claude_train.py:168 calls normalize_splits BEFORE create_dataloaders). The p95/p99/p999
  thresholds live in normalized scale. Then _compute_sample_weights compares normalized y against them. This is internally consistent, but the comment in loss.py:79 (p99_threshold=
  float(p99_threshold)) suggests that real MGD used to be passed before. That conversion branch (thresholds_are_normalized=False + norm_params) exists in CompositeLoss, not in TwoStageLoss, which is what
  actually trains. It is not a bug, but it is dead code that makes future bugs easier.

  A4. Reported NSE, denormalization, and clipping to 0 [OK]

  src/evaluation/metrics.py:91-94: denormalizes with denormalize_target (which reverses z-score and then expm1 with clip≥0), and then clips again. Correct. NSE is computed on real MGD. There is no
  bug here.

  A5. Discrepancy in naive NSE [MINOR]

  DATASET_STATS §8 says NSE_naive(H=1) = 0.811. STATE.md §"Findings" says 0.826. It is the same test set with the same definition. Check how the naive baseline is aligned (whether it includes or not the
  first seq_length+horizon steps that the model does not predict). If the naive baseline is computed on the full test set and the model only on t ≥ seq_length-1+horizon, the comparison is biased.

  A6. Early stopping optimizes a regime that is not the test regime [HIGH, already partly covered by F2]

  I will not repeat F2, but I will add an operational consequence: trainer.py:185 saves the best model by val_loss_epoch. The val loss is dominated by the 97 extremes in val including the 225 MGD. The
  best epoch is 6 (STATE.md), extraordinarily early. That is characteristic of early stopping that rewards "not being too wrong in the very high tail" before the model learns moderate peaks. With test
  limited to 135 MGD, that optimum is not applicable. Specific recommendation in C.

  ---
  B. Diagnostic of the physical limit

  Hard data from DATASET_STATS:

  - ACF(1)=0.909 → lower bound on NSE for an AR(1) predictor: 2ρ−1 = 0.818. The naive baseline gives 0.811 empirically. 0.82 is the trivial floor.
  - Total variance of y in test ≈ σ² = 9.37 MGD² (σ=3.06).
  - Samples per bucket: 152,904 Base + 9,518 Mild + 2,305 Moderate + 437 High + 59 Extreme = 165,223.
  - Contribution to the NSE denominator by bucket (Σ(y−ȳ)²):
    - Extreme: 59 × (69.5−0.49)² ≈ 280,000 MGD² (≈18% of the total denominator)
    - High: 437 × (~30−0.49)² ≈ 380,000 MGD² (≈24%)
    - Moderate+Mild: ≈500,000 MGD² (≈32%)
    - Base: ≈400,000 MGD² (≈26%)

  Current squared errors (model H1_sinSF):
  - Extreme: 59 × 27.8² ≈ 45,600
  - High: 437 × 7.3² ≈ 23,300
  - Moderate: 2305 × 3.6² ≈ 30,000
  - Mild+Base: ~110,000 + ~18,000 ≈ 128,000
  - Total numerator ≈ 227,000. Denominator ≈ 1,560,000. NSE ≈ 1 − 0.146 = 0.854 (consistent with 0.861).

  Reachable upper bound at H=1:

  A perfect oracle on extremes with rainfall (44/59) while keeping the rest → numerator drops ~34,000 (30k rainfall extremes → 0) → NSE rises to ~0.88.

  A perfect oracle on rainfall-related extremes+highs and ideal rainfall→stormflow on moderates → numerator drops to ~80,000 → NSE ~0.95.

  Verdict: the maximum defendable NSE at H=1 with the current features is in the 0.88–0.92 range. Above 0.92 would require external signals (future rainfall radar, soil saturation state,
  snowmelt). The current 0.861 leaves ~0.06 of real headroom, not "almost closed." But that headroom lives in extremes, not in global NSE. If the target metric were peak_err_pct or RMSE of the Extreme
  bucket, the upper bound is different: the physical limit from the 15 extremes without rainfall imposes bias_ext of at least ~-3.5 MGD with an optimal regressor on the remaining 44 (assuming those are
  predicted perfectly), equivalent to peak_err_pct ≈ −5%.

  ---
  C. Single recommendation for the first 3 months

  Ablation and redesign of the supervision target: train the regressor with NON-conditional supervision.

  What exactly

  Three coordinated changes that are really ONE methodological change:

  1. src/features/engineering.py:115-116: remove delta_flow_5m and delta_flow_15m. They are the backdoor to flow_total_mgd. Replace them (if a trend signal is needed) with autoregressively computed
  delta_stormflow_5m from the target — but as an optional feature, in a comparison variant. Priority: first remove them and see how much NSE drops. That drop is the real leak you had.
  2. src/models/loss.py:187-209: train the regressor on ALL samples, not only on event_mask. Huber over the entire batch, with the magnitude weights you already compute. The classifier stays the same.
  At inference, keep the hard switch. This removes the train/inference mismatch from A2.
  3. src/pipeline/split.py: move to full-year splits (8 train / 1 val / 1.5 test). Specifically replace the current indices (iloc[:train_end]) with cuts at 2024-01-01 and 2025-01-01 (or similar). Val
   with ≥1 full year stops being an anomalous segment containing the absolute maximum (F2).

  Why this and not something else

  - Three options on the table in STATE.md: (a) API+ET, (b) multilevel classification, (c) future rainfall. (a) is a marginal refinement of a redundant feature (DATASET_STATS §5: api_dynamic has
  |r|≥0.7 with 9 features); it does not address anything structural. (b) changes the target metric without fixing the underlying model; if the current model does not discriminate magnitudes, a 5-class
   classifier will not either, it will only hide the problem. (c) depends on a meeting that has not happened yet and on an external artifact from MSD; it is not actionable in the next iteration.
  - Recommendation 1+2+3 attacks the three levers with quantitative evidence in the repo: (1) leakage from A1, (2) inconsistency from A2, (3) asymmetry F2.
  - It partly contradicts a learned lesson: "ONE change per iteration" (AGENTS.md §7). Yes. But the three modifications are methodological, not tuning, and they are INDEPENDENT for attribution: run 4
  trainings (baseline, without δflow, with non-conditional reg, with full-year split, and one with all three together). The computational cost is low (best epoch=6, batch 256) and the decoupling is
  preserved.

  How to measure whether it worked

  - Primary threshold: model NSE on test − naive NSE on test ≥ +0.10 at H=1. Today it is +0.050 (DATASET_STATS §8). If it does not exceed +0.10, the model still does not justify itself against a
   one-line script.
  - Secondary threshold: peak_err_pct in the Extreme bucket (filtered to the 44 rainfall-related events, not the 59 mixed ones) ≤ −15%. Today it is −21% on the 59 mixed events.
  - Consistency threshold: Bias in the Base bucket ≤ +0.05 MGD (today +0.21). If the non-conditional regressor makes it worse, the A2 hypothesis is false.

  If there are two months left afterward

  4. Second thing: reduce features to ~10 via permutation importance on the repaired model (not on the current one; as AGENTS.md §8.2 says, PI on a bad model is not reliable). Candidates to
  remove from DATASET_STATS §5: api_dynamic (redundant with 9), rain_sum_15m (0.970 with rain_sum_10m), rain_max_10m (0.985 with rain_sum_10m), hour_sin/cos (negative PI already confirmed), perhaps
  temp_daily_f (−0.870 with month_cos). Goal: 10 features with no loss in NSE.
  5. Third thing: bootstrap confidence intervals over the 59 test extremes (F3). With 59 samples and 1000× resampling, a 95% CI for Extreme NSE would probably be on the order of [−3, +0.3]. That
   makes explicit that the bucket cannot support the weight of the conclusions currently being placed on it. For the TFM this is academic shielding, not operational improvement.

  ---
  D. Questions you have not asked me

  D1. Does the +0.05 NSE gain over the naive baseline come from delta_flow_5m/15m or from the rainfall features?

  Why it matters: if removing the two delta features causes NSE to fall to ≤0.82 (≈naive), the model is not learning hydrology, it is doing disguised AR(1). The entire TwoStageTCN + 22-feature
  apparatus would be theater to add +0.05 NSE that one line of Python (y[t+1] = y[t] + 0.5*(y[t]-y[t-1])) would already cover. Evidence leading me to raise it: A1 + DATASET_STATS §4 (global Pearson
   delta_flow_15m × storm = 0.376) + the near equality NSE_model ≈ NSE_naive.

  D2. What is the timing error (lag between the real peak and the predicted peak) separated from the magnitude error?

  Why it matters: peak_err_pct = (max(y_pred) − max(y_true))/max(y_true) does not distinguish between "the model predicts the same peak but 5 minutes late" and "the model predicts a smaller peak at
  the correct time." The operational value changes a lot: a correct peak with a 5 min delay still leaves time to react; a peak underestimated in magnitude does not activate an alert. Evidence leading me
  to raise it: STATE.md says "the model gets quite right WHEN the peak occurs at H=1" but no metric is reported for that. It is being claimed without measurement.

  D3. Do the 15 rainfall-free extremes in test share any temporal pattern (snowmelt, day of year, previous temperature) that the model should be able to capture with temp_daily_f and long memory?

  Why it matters: AGENTS.md and STATE.md classify them as "physically unpredictable with the current data" and use them as a defense argument. If they are concentrated in January–March with
  temp_daily_f < 40°F preceded by snow accumulation (inferable from past precipitation), then they are predictable and the current claim is convenient but false. Evidence leading me to raise it:
  DATASET_STATS §3 mentions 44 extremes with rain_sum_60m < 0.01, candidates for snowmelt, but the analysis `extreme_events_no_rain.json` has not been crossed with temperature in the document. The
  "unpredictable" claim is being taken for granted without the confirming analysis.
