# S1 - Pipeline audit

Audit of the source code (`diagnostico` branch, commit c8bfe84). Objective: identify bugs, leakages, or questionable decisions beyond the `delta_flow_*` shortcut already confirmed in iter16.

## Summary

- BUGS: 2
- SUSPICIOUS: 5
- OK: 3
- Main finding: **`TwoStageLoss` (src/models/loss.py:163-220) trains the regressor ONLY on samples with `y_true_real > 0.5 MGD`, but at inference (`TwoStageTCN.predict`, src/models/tcn.py:248-253) the hard switch applies it to ALL samples with `p_evento >= 0.3`. This produces undefined behavior on baseflow under false positives and explains the NSE=-14.5 in the Base bucket. In addition, the event definition in the loss (`y > 0.5 MGD`) does not match the one used by the DataLoader (`is_event` boolean), so the sample weights and classifier supervision are optimizing different objectives.** Together with the already known backdoor through `delta_flow_*`, this is the second mechanism by which the reported system is not a "real" TCN: it is an AR(1) + a poorly supervised classifier.

---

## 1. Derived features and potential shortcuts

**Verdict: BUG (known + aggravating factor)**

Evidence: `src/features/engineering.py:115-118`

```
df_feat["delta_flow_5m"] = df_feat["flow_total_mgd"].diff(periods=steps_5m).fillna(0.0)
df_feat["delta_flow_15m"] = df_feat["flow_total_mgd"].diff(periods=steps_15m).fillna(0.0)
df_feat["delta_rain_10m"] = df_feat["rain_in"].diff(periods=steps_10m).fillna(0.0)
df_feat["delta_rain_30m"] = df_feat["rain_in"].diff(periods=steps_30m).fillna(0.0)
```

Analysis:
- `delta_flow_5m` and `delta_flow_15m` are derived from `flow_total_mgd = baseflow + stormflow`, which has r=0.9976 with the target. This is the backdoor already confirmed in iter16 (NSE dropped from 0.861 -> -0.169 when removed). Spearman in event regime for `delta_flow_15m` = -0.078 (almost random), but global Pearson is 0.376; in the first step after a jump in the target, the derivative carries almost the entire AR signal. `DATASET_STATS.md §4` shows this explicitly.
- No other feature is derived from the target: `api_dynamic` is computed only from `rain_in` and `temp_daily_f` (engineering.py:70-86, API recurrence = rain(t) + K*API(t-1), with K modulated by temperature). OK.
- `delta_rain_10m`/`delta_rain_30m` are derived only from `rain_in`, not from the target. OK.
- **Additional candidate to monitor**: `flow_total_mgd` is present as a COLUMN in `df_feat` (engineering.py uses it for deltas and keeps it in `output_columns` through `feature_columns` + target, although it is explicitly NOT in the `feature_columns` list on lines 128-152). That is, it is available in `df_feat` but is not passed to the model through FEATURE_COLUMNS. Low risk as long as the notebooks and `evaluate_local.py` do not accidentally add it back. Recheck when reviewing new iterations.
- `minutes_since_last_rain` (engineering.py:40-48): causal, derived only from `rain_in`. OK.

This remains the main shortcut mechanism. Iter16 already removed it from FEATURE_COLUMNS in the notebook, but the columns are still created in `engineering.py` and can easily reappear.

---

## 2. Chronological split and crossing of train/val/test windows

**Verdict: BUG (MODERATE FEATURE LEAKAGE, NOT TARGET LEAKAGE)**

Evidence:
- `src/pipeline/split.py:44-46`: `df_train = df_sorted.iloc[:train_end]; df_val = df_sorted.iloc[train_end:val_end]; df_test = df_sorted.iloc[val_end:]`. Strict split by index (not by timestamp, but equivalent after `sort_values("timestamp")`).
- `src/pipeline/sequences.py:65-70`: windows are built WITHIN each split, using only `feature_matrix` and `target_array` from the corresponding `df_split`.

Analysis:
- Good: no window crosses boundaries between splits, because `_build_window_arrays` operates on a single `df_split` at a time. There is no direct target leakage from one split into another.
- But: the rolling accumulated features from `engineering.py` (`rain_sum_*`, `rain_max_*`, `api_dynamic`, `minutes_since_last_rain`) are computed in `create_features(df_clean)` BEFORE the split. Therefore, the first record of `df_val` has `rain_sum_360m` values that include rainfall from the last 6h of `df_train`, and similarly for test. This is NOT target leakage, but it is feature "contamination": the first ~72-360 steps of each split (depending on the longest window) contain information from the previous split.
- Practical consequence: minimal for the model (it is just physically reasonable temporal continuity, and API is truly accumulative), but it invalidates the idea that `df_val` and `df_test` are "independent experiments." If the TFM wants to report clean test metrics, it is easy to fix: compute features per split or discard the first N steps of val/test. Recommended to document it.
- `split_chronological` does not apply a gap/buffer between splits. If timestamps were duplicated for some reason (not the case here after `drop_duplicates`/`sort`), there would be risk; with the real data it is clean.

---

## 3. Normalization and possible double normalization

**Verdict: BUG (latent, not always active)**

Evidence: `src/pipeline/normalize.py:71`

```
all_norm_columns = list(feature_columns) + [target_col]
```

Analysis:
- Statistics (`mean`, `std`) are computed ONLY on `train_transformed` (normalize.py:75-82). OK: there is no leakage from val/test into stats.
- But if `target_col` is already inside `feature_columns` (conSF case, when `stormflow_mgd` is passed as autoregressive input), line 71 includes it TWICE in the list of columns to scale. `_zscore_scale` (normalize.py:42-46) iterates and applies `(x - mean) / std` on the same column in-place on the copy, so it gets normalized twice. This is exactly the bug described in `AGENTS.md §11`.
- The source code in `normalize.py` is NOT fixed. The correction exists ONLY in the callers:
  - `evaluate_local.py:368`: `features_for_norm = [f for f in features if f != TARGET_COL]` (OK).
  - `notebooks/rescate_colab_2026-04-17/horizon_comparison_v2.py:270, 610, 830`: filters before calling (OK).
  - `notebooks/rescate_colab_2026-04-17/claude_train.py:181, 1028`: does NOT filter, but current iter16 does not include `stormflow_mgd` in `FEATURE_COLUMNS`, so it does not trigger. If a conSF experiment is reintroduced from that notebook tomorrow, the bug returns.
- Recommendation: fix the bug inside `normalize_splits` itself (dedup with `dict.fromkeys(all_norm_columns)` or an explicit warning) and do not depend on the caller remembering to do it. This is the kind of bug the pipeline should forbid by construction.

Additional verification OK:
- `normalize_target_values` (normalize.py:105-114) and `denormalize_target` (117-130) apply log1p/expm1 conditionally according to `norm_params["log1p_columns"]`. Symmetric. OK.
- log1p is applied before computing mean/std (normalize.py:67), which is the correct order.

---

## 4. Definition of `is_event`

**Verdict: OK**

Evidence: `src/data/clean.py:9-23, 76`.

Analysis:
- `_build_event_mask` uses timestamps with `searchsorted` on `event_start`/`event_end` from the MSD event file (not computed from the target). It marks `is_event=True` if `t in [event_start, event_end)`.
- It does NOT look into the future of the TARGET: the time window comes from the external definition (MSD storm files). There is no leakage from `stormflow_mgd` into `is_event`.
- However, physically, `event_start`/`event_end` DO include the moment of the peak, and that moment IS the target the model is trying to predict. This means that `is_event[t+h]` can encode "at t+h we are already inside a storm," which is what the model is trying to predict. The Dataset (sequences.py:51, 70) uses `event_array[target_index]` as `event_target`, which is legitimate as a multitask supervision label, but `is_event` cannot be used as an INPUT feature without introducing leakage.
- Review: it is not used as a feature. In `engineering.py:154` it is included in `output_columns` as an auxiliary column, and in `sequences.py:51` it is extracted as `event_array` for the DataLoader (it does not enter `feature_matrix`). OK.

---

## 5. Window construction (temporal leakage)

**Verdict: OK**

Evidence: `src/pipeline/sequences.py:65-70`.

```
for end_index in range(seq_length - 1, max_end_index + 1):
    start_index = end_index - seq_length + 1
    target_index = end_index + horizon
    windows_x.append(feature_matrix[start_index : end_index + 1])   # [t-71, ..., t]
    windows_y.append(float(target_array[target_index]))              # y(t + horizon)
```

Analysis:
- Input window: `[end_index - seq_length + 1, end_index]` inclusive = 72 steps.
- Target: `feature_matrix` covers up to index `end_index`; the target is taken at `end_index + horizon` with `horizon >= 1`. The target NEVER enters the input window. Correct.
- The loop starts at `seq_length - 1` (full window available) and ends at `max_end_index = total_rows - horizon - 1`, guaranteeing a valid `target_index`.
- One comment: with `horizon=1`, `end_index+1` is the target, and the last input element (`end_index`) is `t`. The model has access to `features[t]` (including `delta_flow_5m[t]` and in conSF `stormflow_mgd[t]`). For conSF this is explicit AR(1), and `stormflow[t]` has autocorrelation 0.909 with `stormflow[t+1]`, which is where the reported high NSE values come from. It is not future leakage, but it does confirm why the naive baseline almost matches the model.

---

## 6. Two-stage loss

**Verdict: BUG (A2 CONFIRMED) + SUSPICIOUS (A3 dead code) + SUSPICIOUS (double event definition)**

Evidence: `src/models/loss.py:182-211`.

```
y_true_real = self._denormalize_target(y_true)
event_label = (y_true_real > self.event_threshold).float()  # threshold=0.5 MGD

event_mask = (event_label == 1.0).squeeze(1)
if event_mask.any():
    reg_pred = reg_value[event_mask]
    reg_true = y_true[event_mask]
    ...
    reg_loss = (huber_values * reg_factor * reg_weights).mean()
else:
    reg_loss = torch.zeros(...)
```

Analysis:

**A2 (BUG, HIGH)**: the regressor is trained EXCLUSIVELY on samples with `y_true_real > 0.5`. It never sees baseflow. But at inference (`src/models/tcn.py:248-253`):

```
return torch.where(cls_prob >= threshold, reg_value, zeros)
```

With `threshold=0.3` (trainer.py:247), any false positive from the classifier on a baseflow sample triggers the regressor on an OOD (out-of-distribution) input. The regressor has complete freedom in that region (it is never penalized there) and produces arbitrary predictions. This matches `local_eval_metrics.json`, which reports the Base bucket with NSE=-14.5 and bias +0.21 MGD: it is not noise, it is undefined behavior.

**Double event definition (SUSPICIOUS)**:
- In the loss (loss.py:183), `event_label = (y_true_real > 0.5 MGD)`.
- In the DataLoader (sequences.py:51, 70), `event_array = df_split[aux_col]` where `aux_col="is_event"` comes from the MSD file (clean.py:76).
- `_compute_sample_weights` (sequences.py:100) uses `event_array` (MSD `is_event`) to apply a +1.75x boost.
- But `TwoStageLoss` uses `y > 0.5 MGD` (loss.py:183). These are different supervision signals for the same classifier: sample weights reward being inside the MSD window, while BCE optimizes `y > 0.5 MGD`. If `is_event` and `y>0.5` do not coincide 100%, the classification gradients and the sample weights pull in different directions.

**A3 (SUSPICIOUS, dead code)**: `CompositeLoss` (loss.py:14-110) exists but is not used in the current training (`claude_train.py` imports `CompositeLoss, TwoStageLoss` but only instantiates `TwoStageLoss` on lines 382 and 1090). It is old code from previous iterations. Its branch `thresholds_are_normalized` + `norm_params` (loss.py:34-42) makes it easy to confuse the reader about whether thresholds live in real MGD or in normalized space. In `TwoStageLoss` those thresholds do NOT exist because the loss compares in normalized space and only denormalizes internally to define `event_label`. Remove `CompositeLoss` or mark it deprecated.

**Sample weights in normalized space (A3 from Opus, MINOR)**: `_compute_quantile_thresholds` (sequences.py:79-87) computes p95/p99/p999 over `df_train[target_col]` when it is already log1p + z-score (confirmed: `claude_train.py:180-181` calls `normalize_splits` BEFORE `create_dataloaders`). The thresholds live in normalized space; `_compute_sample_weights` (sequences.py:95-103) compares normalized `y_array` against them. Internally consistent, but the values printed to the console (`"Weight thresholds(train): {p95: X, p99: Y}"`, sequences.py:228) are NOT real MGD, which can be confusing when reading logs. Not a bug, but a future trap.

---

## 7. Double normalization of `stormflow_mgd`

**Verdict: latent BUG** (see also point 3).

Evidence:
- The bug LIVES in `src/pipeline/normalize.py:71` (not fixed).
- Fixed by WORKAROUND in callers:
  - `evaluate_local.py:368`: filters before calling.
  - `horizon_comparison_v2.py:270, 610, 830`: filters before calling.
- NOT fixed in `notebooks/rescate_colab_2026-04-17/claude_train.py:181, 1028`: passes the full FEATURE_COLUMNS. In current iter16 `stormflow_mgd` is not in `FEATURE_COLUMNS`, so it does not trigger, but the latent bug remains.

Evaluation: the official source code (`normalize.py`) does NOT contain the correction. Any future refactor or notebook that passes a list including `stormflow_mgd` silently reintroduces the bug. This is exactly the kind of bug the pipeline should prevent by construction. Fixing it in the function itself is trivial:

```python
all_norm_columns = list(dict.fromkeys(list(feature_columns) + [target_col]))
```

---

## 8. Alignment of the naive predictor

**Verdict: SUSPICIOUS (A5 from Opus confirmed)**

Evidence:
- `scripts/generate_dataset_stats.py:1046-1079`:

```
df_test = splits["test"]                 # full df_test WITHOUT discarding the first 72 rows
target_series = df_test[TARGET_COLUMN].to_numpy()
...
y_true = target_series[h:]               # [h, ..., N-1]
y_pred_naive = target_series[:-h]        # [0, ..., N-1-h]
nse_naive = _nse(y_true, y_pred_naive)
```

- It compares against `model_nse_sin_sf = {1: 0.861, ...}` from `outputs/data_analysis/local_eval_metrics.json`, computed by `evaluate_local.py` on the range `test[seq_length+horizon-1 : ]` (the model discards the first 72 steps of test, evaluate_local.py:755 `offset = SEQ_LENGTH + horizon - 1`).

Analysis:
- The naive NSE is computed on the ENTIRE test set (~165k samples), while the model NSE is computed on `test[72+h-1:]` (~164.8k samples, 72 fewer steps). The sample difference is very small, but the NSE denominator depends on the range and can change in the 3rd decimal.
- More importantly: the first 72 points of test fall right at the start, which is usually a calm period (test begins at an arbitrary chronological boundary). The difference is minimal, but the reported naive NSE (0.811 in `DATASET_STATS §8` vs 0.826 in `STATE.md`) suggests it is being computed in two different ways. It must be aligned.
- Trivial fix: also discard the first `seq_length+horizon-1` points of test in the naive calculation. Without that fix, the comparison "gain +0.050 NSE" is not rigorous.

---

## 9. Inference (hard switch) and classifier false positives

**Verdict: BUG (direct consequence of A2, same bug as point 6)**

Evidence: `src/models/tcn.py:248-253` + `src/training/trainer.py:247`.

```
def predict(self, x, threshold=0.5):
    ...
    return torch.where(cls_prob >= threshold, reg_value, zeros)
```

In `predict()` from trainer.py it is called with `threshold=0.3`:

```
y_pred = model.predict(x_batch, threshold=0.3)
```

Analysis:
- With `threshold=0.3`, recall increases at the expense of precision -> more false positives.
- Each false positive activates the regressor, which only saw samples with `y_real > 0.5 MGD` during training -> OOD prediction.
- Instead of the hard switch, a clean fix is to train the regressor on ALL samples (with magnitude weighting, which is already computed) and keep the switch only as a low-magnitude filter. That way the regressor has a valid definition on baseflow and does not blow up under false positives. This is fix C1-2 suggested by Opus 4.7.

---

## 10. Criterion for "extreme events without rainfall"

**Verdict: SUSPICIOUS (criterion changes between runs)**

Evidence:
- `evaluate_local.py:606-655`: counts "extremes with/without rainfall" by iterating over `y_real > 50` in the prediction vector (after the 72-step offset), and for each one checks whether `rain_sum_60m > 0` appears in the 72-step input window before `df_idx = OFFSET + idx` in `df_test` (`df_test` is ALREADY NORMALIZED because the caller passes the normalized df to the plots... verify). Criterion: "some sample in the previous 6h has `rain_sum_60m > 0`".
- `scripts/generate_dataset_stats.py:502-507`: counts extremes where `rain_sum_60m < 0.01` at the PEAK INSTANT (not a window). Different criterion.
- The comment in 507-509 admits they are not the same criterion: "It is not the exact definition (the real one uses the model input window) but gives the right order of magnitude."

Analysis:
- That is why the figure "15 of 59 extremes without rainfall" from older docs (peak instant) does not match "0 extremes without rainfall" from iter16 (72-step window). The criterion was relaxed: a real storm almost always has SOME rainfall in the previous 6h, even if the peak happens after rainfall has already stopped. Both numbers can be correct; they simply measure different things.
- There is an additional subtlety: `evaluate_local.py:640` checks `rain_sum_60m > 0`, which is a rolling sum. In the raw dataset, `rain_sum_60m` at minute t collects rainfall over [t-60min, t]; if there was light rainfall 4h earlier, `rain_sum_60m` has already been zero for 3h, but the 72-step window still contains the old pulse at the beginning. That makes the detection very permissive: a single step with `rain_sum_60m > 0` in a 6h window is enough to mark "with rainfall."
- Additional sub-bug: line 635 `window_df = df_test.iloc[win_start:win_end]` uses the caller's `df_test`. If that `df_test` is NORMALIZED (log1p+z), `rain_sum_60m > 0` changes its physical threshold (`log1p(0)` is 0 and z-score(0) may be negative, so `> 0` is a different comparison in that space). Check `main()` to see which `df_test` gets passed to the plot.

Conclusion: the figure "0 extremes without rainfall" in iter16 may be correct, but under a very lax criterion. For a serious TFM it is necessary to:
1. Fix a single criterion (I propose: `rain_sum_180m > 0.05 pulgadas` at the peak instant, or accumulated rainfall over the window >= 0.1).
2. Document it.
3. Recompute both statistics under the same criterion.

---

## Actionable recommendations (priority)

1. **[HIGH] Fix A2 in `src/models/loss.py:182-211`**: train the regressor on all samples (not only `event_mask`), keeping the magnitude weights. Keep the hard switch only as a low-magnitude filter at inference. This removes OOD behavior on baseflow under false positives and unifies supervision. Likely to drastically improve the Base bucket (NSE=-14.5 -> something reasonable).

2. **[HIGH] Unify the event definition**: decide whether the classifier should predict `is_event` (MSD window) or `y > 0.5 MGD` (stormflow threshold). Today `TwoStageLoss` (loss.py:183) uses one and the sample weights (sequences.py:100) use the other. A single consistent criterion.

3. **[MEDIUM] Fix the double normalization bug INSIDE `src/pipeline/normalize.py:71`**: dedup with `dict.fromkeys`. Do not depend on the caller filtering it out. Latent bug that returns in any future conSF experiment launched from the main notebook.

4. **[MEDIUM] Align naive and model on the same index range in `scripts/generate_dataset_stats.py:compute_section_8_naive_baseline`**: discard `seq_length + horizon - 1` points at the start of test before computing naive NSE. This explains the 0.811 vs 0.826 discrepancy.

5. **[MEDIUM] Remove `delta_flow_5m` and `delta_flow_15m` from `src/features/engineering.py:115-116`** (or at least mark them clearly as "deprecated - backdoor to flow_total"). They are already out of FEATURE_COLUMNS in iter16 but are still generated.

6. **[MEDIUM] Fix a single criterion for "extreme without rainfall"** between `evaluate_local.py:637-642` and `scripts/generate_dataset_stats.py:504-507`. Recompute both outputs under the new criterion. That resolves the 15/59 vs 0/59 discrepancy.

7. **[LOW] Remove `CompositeLoss` (src/models/loss.py:14-110)** or mark it obsolete to avoid confusion about normalized thresholds vs MGD.

8. **[LOW] Document that rolling features (`rain_sum_*`, `api_dynamic`, `minutes_since_last_rain`) are computed before the split and inherit train->val->test continuity**. It is not target leakage, but it is a cross-dependency worth documenting.

9. **[LOW] Print thresholds p95/p99/p999 in real MGD in addition to normalized space in `sequences.py:228`** to avoid confusion when reading logs.
