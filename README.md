# Stormflow Prediction — Cincinnati MSD (MC-CL-005)

## 1. Project overview

Short-horizon prediction of stormflow (storm-driven flow) at sewer monitoring station MC-CL-005, operated by the Metropolitan Sewer District of Greater Cincinnati (MSD). The objective is to support a transition from reactive operation to predictive operation in order to anticipate Combined Sewer Overflow (CSO) events. This repository accompanies a Master's Thesis (TFM) and contains the data pipeline, models, and evaluation code used to reach the reported results.

## 2. Results summary

All metrics computed on the same chronologically held-out test set (n = 165,222 at H=1; H = horizon in 5-minute steps). NSE = Nash–Sutcliffe Efficiency. recall@50 = recall on samples with stormflow ≥ 50 MGD. AUC-PR = area under the precision–recall curve at the operational threshold (recall ≥ 0.85 on validation).

### 2.1 Regressor — `xgb_lag6_feat10` (production model)

| Metric         | H = 1 (5 min) | H = 3 (15 min) |
|----------------|--------------:|---------------:|
| NSE            |        0.8630 |         0.6871 |
| Peak error (%) |          −5.9 |          −13.1 |
| Base bias (MGD)|        +0.026 |              — |
| recall@50      |         0.652 |          0.304 |
| Extreme NSE    |        −1.727 |              — |

### 2.2 Binary alert classifiers (4 variants of `xgb_h{h}_u{U}`)

| Variant | Prevalence | AUC-PR | ROC-AUC | P@op  | R@op  | Median lead time |
|---------|-----------:|-------:|--------:|------:|------:|-----------------:|
| h6_u25  |     0.41 % | 0.5842 |  0.9806 | 0.102 | 0.884 |           25 min |
| h6_u50  |     0.12 % | 0.2488 |  0.9678 | 0.007 | 0.949 |           30 min |
| h12_u25 |     0.66 % | 0.4180 |  0.9397 | 0.042 | 0.830 |           60 min |
| h12_u50 |     0.21 % | 0.2050 |  0.9187 | 0.008 | 0.901 |           60 min |

### 2.3 Deep-learning comparison — clean TCN (Bai 2018), reported as academic contrast

| Metric     | TCN A0 H=1 | XGB H=1 | TCN A0 H=3 | XGB H=3 |
|------------|-----------:|--------:|-----------:|--------:|
| Global NSE |     0.8890 |  0.8631 |     0.6096 |  0.6889 |
| Peak err % |       +6.6 |   −13.5 |      −53.2 |   −13.5 |
| recall@50  |      0.741 |   0.609 |      0.087 |   0.261 |

The TCN beats XGBoost at H=1 by +0.026 NSE (below the +0.020 threshold for operational relevance once seed variance is considered) and clearly loses at H=3. XGBoost is therefore retained as the single production regressor.

## 3. System components

The deployed system has three components, all closed and merged to `main`:

- **Component A — regressor `xgb_lag6_feat10`** for H=1 and H=3. Inputs: 6 explicit target lags (`stormflow_mgd[t−0..t−5]`) plus 10 reduced exogenous features selected by permutation importance and redundancy clustering on training data. Code: `src/models/xgboost_baseline.py`, `src/evaluation/metrics_panel.py`. Notebook: `notebooks/iter17_xgboost_lags.py`.
- **Component B — four binary classifiers** for `(h, U) ∈ {(6, 25), (6, 50), (12, 25), (12, 50)}` MGD, framed as "will the maximum stormflow over the next h steps reach U?". Same input set as the regressor; `objective=binary:logistic`, `eval_metric=aucpr`, `scale_pos_weight = neg/pos`. Operational threshold chosen as the highest one satisfying recall ≥ 0.85 on validation. Code: `src/models/xgboost_classifier.py`, `src/evaluation/classification_panel.py`. Notebook: `notebooks/iter18_xgboost_classifier.py`.
- **Component C — deep-learning comparison.** Standard causal TCN (Bai et al. 2018) with 4 residual blocks, dilations [1, 2, 4, 8], kernel = 3, hidden channels = 32, dropout = 0.1, log1p target, Huber loss. Same split and aligned indices as Component A. Code: `src/models/tcn_clean.py`, `src/pipeline/normalize_v2.py`. Notebooks: `notebooks/iter19_tcn_clean.py`, `notebooks/iter19b_a0_test.py`, `notebooks/iter19c_a0_h3.py`.

## 4. Repository structure

```
stormflow-prediction/
├── README.md                     This document.
├── CLAUDE.md / AGENTS.md         Project entry point and permanent context.
├── requirements.txt              Python dependencies.
├── configs/default.yaml          Data paths (resolved at runtime in Colab).
├── docs/
│   ├── STATE.md                  Current project status (production models, metrics).
│   ├── EXPERIMENTS.md            Full iteration history (iter1 to iter19c).
│   ├── DATASET_STATS.md          Statistical summary of the dataset.
│   └── WORKFLOW.md               How to run the project (Colab, Git, Drive).
├── src/
│   ├── data/                     Loading and cleaning of .tsf files.
│   ├── features/                 Feature engineering (rainfall aggregations, API, calendar).
│   ├── pipeline/                 Chronological split, normalization, sequence builder.
│   ├── models/                   xgboost_baseline.py, xgboost_classifier.py, tcn_clean.py.
│   ├── training/                 Training loops for the TCN.
│   └── evaluation/               metrics_panel.py, classification_panel.py, diagnostics.
├── notebooks/                    Colab notebooks exported as .py (one per iteration).
├── outputs/
│   ├── diagnostic/               Iteration reports (iter17, iter18, iter19, iter19b, iter19c) and the master DIAGNOSTIC_REPORT.md.
│   ├── data_analysis/            JSON metric files and panels.
│   └── figures/                  Plots produced by the notebooks.
└── MC-CL-005/                    Raw data (NOT distributed in the repository).
```

## 5. Quickstart — Google Colab

The training and evaluation notebooks are designed to run on Google Colab with a Drive-mounted dataset. The raw data is not redistributed.

1. Open Colab and create a new notebook (a T4 GPU runtime is sufficient; only Component C needs the GPU).
2. Clone the repository and install dependencies:

   ```python
   !git clone https://github.com/Aimarsagasti/stormflow-prediction.git
   %cd stormflow-prediction
   !pip install -r requirements.txt xgboost shap
   ```

3. Mount Google Drive and place the raw dataset at the path expected by `configs/default.yaml`:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   The MC-CL-005 station data (`rain_list.tsf`, `flow_list.tsf`, `lstStormTs.tsf`, `daily_temperatures_2006_2026.tsf`, and the `lstEventsGenerated Events 0.dat` event file) must be reachable through the paths declared in `configs/default.yaml`. Edit the YAML if your Drive layout differs.
4. Open one of the notebook scripts as a notebook (`File → Open notebook → Upload`) or run it as a script:

   ```python
   !python notebooks/iter17_xgboost_lags.py     # Component A: XGB regressor
   !python notebooks/iter18_xgboost_classifier.py  # Component B: 4 binary classifiers
   !python notebooks/iter19_tcn_clean.py        # Component C: clean TCN ablation
   ```

5. Outputs (metrics JSON files, comparison Markdown, figures) land under `outputs/diagnostic/` and `outputs/figures/`.

## 6. Data

Dataset from MSD station MC-CL-005:

- **Temporal coverage:** 2015-08-01 to 2026-01-31 (10.5 years; 1,101,964 records).
- **Resolution:** one record every 5 minutes.
- **Variables:** total flow, stormflow (target), rainfall at the gauge, daily temperature, MSD-labelled storm events.
- **Split:** strictly chronological 70 / 15 / 15. Train: 2015-08 to 2022-12. Validation: 2022-12 to 2024-07. Test: 2024-07 to 2026-01.
- **Engineered features (final set):** 6 target lags plus 10 exogenous features (`api_dynamic`, `rain_sum_360m`, `rain_sum_120m`, `rain_sum_15m`, `temp_daily_f`, `hour_sin`, `minutes_since_last_rain`, `delta_rain_10m`, `delta_rain_30m`, `rain_sum_30m`).

The raw files are MSD-licensed and are not redistributed in this repository. Researchers should contact MSD directly to request access.

## 7. Key findings

- **Explicit target lags dominate at H=1.** A linear AR(12) on the target alone reaches NSE = 0.83 on the aligned test, and feeding 6 target lags into XGBoost together with the 10 reduced features lifts it to 0.8630 with peak error −5.9 %. The exogenous-only XGBoost reaches only 0.7157.
- **Initial deep-learning model relied on shortcuts.** The early TwoStageTCN model reached an apparent NSE of 0.86 only because two features (`delta_flow_5m`, `delta_flow_15m`) leaked the target derivative; removing them collapsed NSE to −0.17. A structural train/eval mismatch in `TwoStageLoss` was also identified. Both findings are documented in `outputs/diagnostic/DIAGNOSTIC_REPORT.md` and led to abandoning that architecture.
- **GBM and a clean Bai 2018 TCN converge to the same information ceiling.** With one rain gauge, no rainfall forecast, and one basin, the clean TCN beats XGBoost at H=1 by only +0.026 NSE and loses at H=3 (−0.078 NSE). The ceiling is set by available inputs, not architecture.
- **Long horizons are better framed as binary alerts.** The physical NSE ceiling at H=6 and H=12 makes regression unreliable, but the binary classifiers achieve ROC-AUC of 0.92–0.98 with median lead times of 25–60 minutes, which is operationally actionable for CSO mitigation.
- **`log1p` on a heavy-tailed target is non-negotiable.** Removing it collapses NSE both in the legacy TCN (−26.6) and in the clean TCN baseline NSE (0.732 → 0.085). The lesson generalises across architectures.

## 8. Future work

- **Spatial rainfall (NEXRAD radar).** Replacing the single rain gauge with NEXRAD reflectivity products would supply spatially resolved precipitation and is the single most likely route to lifting the H=3 and H=6 ceilings.
- **Hybrid forecasting with HRRR / RAP from NWS.** Operational atmospheric forecasts at 3 km / 13 km would give the model access to future rainfall, which today's input set lacks entirely.
- **Multi-station training.** Pooling additional MSD stations with physical basin descriptors would let a shared backbone exploit cross-basin generalisation, which a single-station GBM cannot.
- **Probabilistic prediction (quantile regression).** Reporting calibrated quantiles, not point estimates, would let MSD trade off precision and recall directly at the operational threshold.
- **TCN + GBM stacking ensemble.** The TCN learns the baseflow regime well and the GBM captures event variance; a stacked ensemble could combine their complementary biases.

## 9. References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271. (Architecture used in Component C.)
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of KDD '16, 785–794. (Library used in Components A and B.)
- Nash, J. E., & Sutcliffe, J. V. (1970). *River flow forecasting through conceptual models part I — A discussion of principles.* Journal of Hydrology, 10(3), 282–290. (NSE metric used throughout.)
