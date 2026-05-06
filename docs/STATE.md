# STATE.md - Current project status

**Last update:** 2026-05-05 (iter19/19b/19c closed, modeling complete).
**Maintained by:** Aimar (update at the end of each significant session).

---

## Current system (post-diagnostic)

After the April diagnostic, the old system (TwoStageTCN v1) was
discarded for two confirmed reasons: the autoregressive shortcut
`delta_flow_5m/15m` and structural bugs in `TwoStageLoss`. The
`DIAGNOSTIC_REPORT` recommended building a two-component system
based on XGBoost. iter19 (clean Bai 2018 TCN) closed the
deep learning question with result XGB_WINS at H=3, which keeps XGBoost
as the definitive main model of the TFM.

**Operational components (all closed):**

| Component | Status | Model |
|---|---|---|
| A - stormflow regressor H=1, H=3 | Closed, in `main` | `xgb_lag6_feat10` |
| B - alert classifier H=6, H=12 | Closed, merged to `main` | 4 binary XGB variants |
| C - deep learning comparison | Closed, merged to `main` | clean TCN A0 (discarded as main model) |

**Current project phase:** modeling complete, TFM writing in progress.

---

## Component A: `xgb_lag6_feat10` regressor (iter17)

**Branch:** `iter17-xgboost-lags` (merged to `main`).
**Code:** `src/models/xgboost_baseline.py`, `src/evaluation/metrics_panel.py`,
`notebooks/iter17_xgboost_lags.py`.

### Configuration

- 6 target lags + 10 reduced features from S5.
- Hyperparameters from the DIAGNOSTIC_REPORT §7.2 without tuning:
  `n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
  tree_method=hist, early_stopping_rounds=20, random_state=42`.
- Works in real MGD, without the v1 normalization pipeline.

### Results (test, n=165.222 at H=1)

| Metric | H=1 | H=3 |
|---|---:|---:|
| NSE | 0.8630 | 0.6871 |
| Peak err (%) | -5.9 | -13.1 |
| Base Bias (MGD) | +0.026 | - |
| recall@50 | 0.652 | 0.304 |
| Extreme NSE | -1.727 | - |

### Decision on lag6 vs lag12

The report proposed lag=12 as the starting point. The ablation showed that
lag=6 produces better recall@50 (0.652 vs 0.565) and better peak error (-5.9%
vs -12.9%) with nearly identical NSE (-0.0035). Since recall@50 is the main
operational metric, lag=6 was chosen.

### Attribution of the improvement (H=1 NSE)

- `feat10_only` = 0.7157 -> the lags contribute +0.1473 NSE.
- `lag12_only` = 0.7778 -> the features contribute +0.0853 NSE.
- Both levers are necessary.

### Compliance with criteria §7.4

All 4 are satisfied: NSE H=1 >= 0.85, NSE H=3 >= 0.66, |peak err H=1| < 21%,
Base bias <= +0.05.

---

## Component B: binary classifier (iter18)

**Branch:** `iter18-xgboost-classifier` (merged to `main`).
**Code:** `src/models/xgboost_classifier.py`,
`src/evaluation/classification_panel.py`,
`notebooks/iter18_xgboost_classifier.py`.

### Configuration

- Operational target: `max(stormflow[t+1..t+h]) >= U`.
  This formulation (full window, not point instant) was decided by the
  user after Codex flagged it as an ambiguity relative to §7.5.
- Same features as the regressor: 6 lags + 10 features from S5.
- `objective='binary:logistic'`, `eval_metric='aucpr'`,
  `scale_pos_weight = neg_train / pos_train`, the rest equal to the regressor.
- Operational threshold: the highest one that satisfies `recall >= 0.85` on val.

### Results (test)

| Variant | Prevalence | AUC-PR | ROC-AUC | P@op | R@op | Median lead time |
|---|---:|---:|---:|---:|---:|---:|
| h6_u25 | 0.41% | 0.5842 | 0.9806 | 0.102 | 0.884 | 25 min |
| h6_u50 | 0.12% | 0.2488 | 0.9678 | 0.007 | 0.949 | 30 min |
| h12_u25 | 0.66% | 0.4180 | 0.9397 | 0.042 | 0.830 | 60 min |
| h12_u50 | 0.21% | 0.2050 | 0.9187 | 0.008 | 0.901 | 60 min |

### Honest reading

- High ROC-AUC (0.92-0.98): the model does distinguish events from non-events.
- Useful lead times: 25-60 min before the actual exceedance.
- Low precision (0.7-10%): direct consequence of the tiny prevalence
  and the low operational threshold needed to maintain recall >= 0.85.
- In h6_u50 and h12_u50 the absolute number of FP is high (26.000-41.000 in
  test). The report had already warned in §8.3 that this does not invalidate the TFM.
- Recommendation for the academic document: report the full PR curve,
  not a single operational threshold.

---

## Component C: deep learning comparison (iter19/19b/19c)

**Branch:** `iter19-tcn-comparison` (merged to `main` in May 2026).
**Code:** `src/models/tcn_clean.py`, `src/pipeline/normalize_v2.py`,
`notebooks/iter19_tcn_clean.py`, `notebooks/iter19b_a0_test.py`,
`notebooks/iter19c_a0_h3.py`.

Implements step 7.6 of the DIAGNOSTIC_REPORT. Three sub-iterations:
ablation at H=1 (iter19), re-evaluation of A0 with log1p on test (iter19b),
and replication to H=3 with A0 (iter19c).

### Final configuration (clean TCN A0 Bai 2018)

- Architecture: 4 residual blocks with dilations `[1,2,4,8]`,
  `kernel_size=3`, `hidden_channels=32`, `dropout=0.1`.
- Receptive field = 61 steps over window L=72.
- Inputs: 11 channels (1 historical target + 10 features from S5),
  shape `(B, T=72, F=11)`.
- Normalization: z-score on train, `log1p` on target.
- Loss: `HuberLoss(delta=1.0)` in normalized space.
- Optimizer: AdamW lr=1e-3, wd=1e-4, batch=256, max_epochs=50, patience=10.
- Same temporal split and aligned indices as iter17 (n_test=165.222).
- 23.489 parameters.

### Mini-ablation on val (iter19, 4 runs + extra A4)

| Run | L | C | log1p | NSE_val | err_pico_val | Notes |
|---|---:|---:|:---:|---:|---:|---|
| A0 | 72 | 32 | yes | 0.8436 | +1.5% | balanced baseline |
| A1 | 72 | 32 | no | 0.8617 | -24.9% | wins on val NSE but underestimates peaks |
| A2 | 144 | 32 | yes | 0.8146 | +6.7% | worsens with double window |
| A3 | 72 | 64 | yes | 0.8497 | -23.3% | worsens with C=64 + log1p |
| A4 | 72 | 64 | no | 0.8640 | -25.3% | winning combination by independence |

The "max NSE_val by factor independence" rule initially selected A4.

### Full comparison H=1 and H=3 on test

| Metric | TCN A0 H=1 | XGB H=1 | TCN A0 H=3 | XGB H=3 |
|---|---:|---:|---:|---:|
| Global NSE | 0.8890 | 0.8631 | 0.6096 | 0.6889 |
| Base NSE | +0.732 | similar | +0.711 | -4.37 |
| Extreme NSE | -1.571 | -1.808 | -8.064 | -5.553 |
| Peak err (%) | +6.6 | -13.5 | -53.2 | -13.5 |
| recall@50 | 0.741 | 0.609 | 0.087 | 0.261 |

### Verdict and final decision

- **H=1:** TCN A0 marginally beats XGB (`delta_NSE = +0.0259`,
  above the +0.020 threshold of DIAGNOSTIC_REPORT §7.6).
- **H=3:** TCN A0 LOSES against XGB (`delta_NSE = -0.0775`, WELL
  below the threshold). The TCN collapses at a longer prediction horizon: best_epoch=3,
  recall@50 drops to 0.087, peak error -53.2%.

**Final TFM decision:** **XGBoost remains the only main model
for H=1 and H=3.** Reasons:

1. The TCN advantage at H=1 is marginal (+0.026 NSE) and does not replicate at H=3.
2. The TCN at H=3 is operationally unusable (recall 8.7% vs 26.1% XGB).
3. Keeping a single model (XGB) preserves narrative coherence,
   operational simplicity, and lower inference cost.
4. The TFM narrative closes as an academically strong finding:
   with the available inputs (1 rain gauge, no forecast,
   1 basin), GBM with explicit lags absorbs the available
   predictable signal. Deep learning would add value only by enriching the
   database with: (a) spatial rainfall data (NEXRAD radar),
   (b) integration with weather forecast (HRRR/RAP from NWS), or
   (c) multi-station training with physical basin data.

### Collateral technical findings (for TFM discussion)

- log1p on the target is also essential in the clean TCN (confirms
  historical lesson 3). Without log1p, NSE_Base collapses from 0.732 to 0.085.
- The two architectures have opposite biases: TCN learns the "mean"
  of the process (baseflow), XGB learns the "variance" (events).
  Neither is strictly superior to the other; they coexist.
- The TCN advantage at H=1 comes partly from a single extreme event
  nailed very well (July 28, 2024). Result with high variance across seeds.

---

## Deprecated components (DO NOT use in new iterations)

- `src/models/tcn.py` (TwoStageTCN v1).
- `src/models/loss.py` (CompositeLoss and TwoStageLoss).
- `src/pipeline/normalize.py` (documented latent bug).
- `evaluate_local.py` (historical reference only).

They remain in the repo for academic traceability (TFM comparison table:
"apparent model -> diagnostic -> real model"), but are neither executed nor
modified.

---

## Timeline until submission

| Phase | Period | Deliverable |
|---|---|---|
| iter17 | April | Regressor closed |
| iter18 | April | Classifier closed |
| iter19/19b/19c | May | Clean TCN + deep learning decision closed |
| TFM writing | May-July | Sections 5, 6, 7, 8, 9 |
| Review and figures | August | Full TFM |
| Submission | September 2026 | Final TFM |

**Agreed rule:** modeling closed after iter19c. There will be no more model
iterations. From here on, writing.

---

## Current blockers

None. Modeling complete. Next phase: written TFM.

---

## Quick references

- DIAGNOSTIC_REPORT: `outputs/diagnostic/DIAGNOSTIC_REPORT.md`.
- iter17 results: `outputs/diagnostic/iter17_comparison.md`.
- iter18 results: `outputs/diagnostic/iter18_comparison.md`.
- iter19 results (H=1, ablation): `outputs/diagnostic/iter19_comparison.md`.
- iter19b results (A0 vs A4 in H=1): `outputs/diagnostic/iter19b_a0_vs_a4.md`.
- iter19c results (A0 H=3 vs XGB): `outputs/diagnostic/iter19c_a0_h3_comparison.md`.
- Latent bugs: `outputs/diagnostic/S1_pipeline_audit.md`.
- Physical ceiling: `outputs/diagnostic/S4_horizon_ceiling.md`.
- Feature justification: `outputs/diagnostic/S5_feature_analysis.md`.
- Transition handoff between conversations: `HANDOFF_2026-04-29.md`.
