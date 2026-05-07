# -*- coding: utf-8 -*-
"""iter18_xgboost_classifier.py

Iter18 notebook/script for training binary XGBoost alert classifiers
over longer horizons.

Objective:
- Keep the iter17 regressor as the main model for H=1/H=3.
- Add an operational complement to answer:
  "will there be stormflow >= U at any point in t+1..t+h?".

The formulation confirmed by the user is the full-window one:
`max(stormflow[t+1..t+h]) >= U`.
"""

# %% [markdown]
# # Iter18 - Binary XGBoost alert classifier
#
# - Parquet cache check and regeneration if missing.
# - Temporal split identical to iter17 / S2 for a clean comparison.
# - Four fixed variants:
#   - h=6, U=25
#   - h=6, U=50
#   - h=12, U=25
#   - h=12, U=50
# - Operational threshold chosen on validation to satisfy recall >= 0.85
#   when possible, without tuning on test.
# - Artifacts:
#   - `outputs/diagnostic/iter18_classifier_results.json`
#   - `outputs/diagnostic/iter18_comparison.md`
#   - `outputs/figures/iter18/*.png`

# %%
from __future__ import annotations

import json
import inspect
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import pandas as pd


# We force a non-interactive backend so the script behaves the same locally.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("C:/Dev/TFM")
if str(ROOT) not in sys.path:
    # We insert the repo root to import `src.*` from the notebook.
    sys.path.insert(0, str(ROOT))

PARQUET_PATH = ROOT / "outputs" / "cache" / "df_with_features.parquet"
GENERATE_STATS_SCRIPT = ROOT / "scripts" / "generate_dataset_stats.py"
OUT_DIR = ROOT / "outputs" / "diagnostic"
FIG_DIR = ROOT / "outputs" / "figures" / "iter18"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# These are exactly the four variants requested for iter18.
VARIANTS = [
    ("h6_u25", 6, 25.0),
    ("h6_u50", 6, 50.0),
    ("h12_u25", 12, 25.0),
    ("h12_u50", 12, 50.0),
]


# %%
# We reproduce the iter17 pattern: if the cache does not exist, regenerate it
# before any training so we do not depend on manual steps.
if not PARQUET_PATH.exists():
    print(f"[iter18] Cache not found at {PARQUET_PATH}")
    print(f"[iter18] Regenerating via {GENERATE_STATS_SCRIPT}...")
    result = subprocess.run(
        [sys.executable, str(GENERATE_STATS_SCRIPT)],
        cwd=str(ROOT),
        check=True,
    )
    print(f"[iter18] generate_dataset_stats.py return_code={result.returncode}")
    if not PARQUET_PATH.exists():
        raise RuntimeError(
            f"The cache still does not exist after regenerating it: {PARQUET_PATH}. "
            "Check `scripts/generate_dataset_stats.py`."
        )
else:
    print(f"[iter18] Cache OK: {PARQUET_PATH}")

# %%
from src.evaluation.classification_panel import evaluate_classification_panel
from src.models.xgboost_baseline import FEATURES_10, SEQ_LENGTH, get_split_indices
from src.models.xgboost_classifier import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGB_CLASSIFIER_PARAMS,
    select_operational_threshold,
    train_xgboost_classifier,
)

print(f"[iter18] SEQ_LENGTH={SEQ_LENGTH}  FEATURES_10={len(FEATURES_10)}")
print(f"[iter18] Initial XGB params: {DEFAULT_XGB_CLASSIFIER_PARAMS}")
print(f"[iter18] early_stopping_rounds={DEFAULT_EARLY_STOPPING_ROUNDS}")

# %%
load_start = time.time()
df = pd.read_parquet(PARQUET_PATH)
print(f"[iter18] Parquet loaded in {time.time() - load_start:.1f}s. shape={df.shape}")
print(f"[iter18] timestamp range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

# %%
# Sanity check of split sizes by horizon so the results remain
# comparable with iter17 and the diagnostic.
for _, horizon, _ in VARIANTS:
    split_idx = get_split_indices(df, horizon=horizon)
    print(
        f"[iter18] H={horizon}: n_train={len(split_idx['train']):,}  "
        f"n_val={len(split_idx['val']):,}  n_test={len(split_idx['test']):,}"
    )


# %%
def _sanitize_json(obj: Any) -> Any:
    """Convert arrays/NaN/NumPy types into JSON-serializable objects."""
    if isinstance(obj, dict):
        return {key: _sanitize_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json(value) for value in obj]
    if isinstance(obj, pd.Series):
        return [_sanitize_json(value) for value in obj.tolist()]
    if isinstance(obj, np.ndarray):
        return [_sanitize_json(value) for value in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    return obj


def _fmt(value: Any, spec: str) -> str:
    """Format values for markdown with `n/a` when not applicable."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "n/a"
    return f"{value:{spec}}"


def _success_comment(variant_name: str, precision: float, recall: float) -> str:
    """Evaluate the report's guideline criteria for each variant."""
    if variant_name == "h6_u25":
        return "OK" if recall >= 0.70 and precision >= 0.50 else "NO"
    if variant_name == "h6_u50":
        return "OK" if recall >= 0.60 and precision >= 0.30 else "NO"
    if variant_name == "h12_u25":
        return "OK" if recall >= 0.60 else "NO"
    if variant_name == "h12_u50":
        return "OK" if recall >= 0.50 else "NO"
    return "n/a"


# %%
# ---------------------------------------------------------------------------
# Training and evaluation of the four variants
# ---------------------------------------------------------------------------
variant_outputs: Dict[str, Dict[str, Any]] = {}
results_json: Dict[str, Any] = {}

for variant_name, horizon, threshold_mgd in VARIANTS:
    print(f"\n[iter18] === Training {variant_name} (H={horizon}, U={threshold_mgd:.1f}) ===")
    trained = train_xgboost_classifier(
        df=df,
        horizon=horizon,
        threshold_mgd=threshold_mgd,
        lags=6,
        features=FEATURES_10,
        xgb_params=DEFAULT_XGB_CLASSIFIER_PARAMS,
        early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        verbose=False,
    )

    # We choose the operational threshold using validation only to avoid contaminating test.
    operational_threshold = select_operational_threshold(
        y_true_bin=trained["y_true_bin_val"],
        y_prob=trained["y_prob_val"],
        min_recall=0.85,
    )

    # We also evaluate validation to keep traceability of the chosen threshold.
    val_panel = evaluate_classification_panel(
        y_true_bin=trained["y_true_bin_val"],
        y_prob=trained["y_prob_val"],
        threshold_default=0.5,
        threshold_operational=operational_threshold,
    )

    # On test we add the raw series + U so we can reconstruct real lead times.
    test_panel = evaluate_classification_panel(
        y_true_bin=trained["y_true_bin_test"],
        y_prob=trained["y_prob_test"],
        timestamps=trained["timestamps_test"],
        y_true_raw={
            "values": trained["y_true_raw_test"],
            "threshold_mgd": threshold_mgd,
        },
        threshold_default=0.5,
        threshold_operational=operational_threshold,
    )

    variant_outputs[variant_name] = {
        "trained": trained,
        "val_panel": val_panel,
        "test_panel": test_panel,
        "operational_threshold": operational_threshold,
    }

    default_test = test_panel["default_threshold_metrics"]
    operational_test = test_panel["operational_threshold_metrics"]
    threshold_free = test_panel["threshold_free_metrics"]
    print(
        f"[iter18] {variant_name}: prevalence_train={trained['prevalence_train']:.4f}  "
        f"AUC-PR={threshold_free['auc_pr']:.4f}  ROC-AUC={threshold_free['roc_auc']:.4f}  "
        f"recall@op={operational_test['recall']:.3f}  precision@op={operational_test['precision']:.3f}  "
        f"thr_op={operational_threshold:.4f}  best_iter={trained['best_iteration']}"
    )

    # We persist only serializable objects; the model stays out of the JSON.
    results_json[variant_name] = {
        "config": trained["config"],
        "fit_seconds": trained["fit_seconds"],
        "best_iteration": trained["best_iteration"],
        "prevalence_train": trained["prevalence_train"],
        "scale_pos_weight": trained["scale_pos_weight"],
        "n_train": int(len(trained["idx_train"])),
        "n_val": int(len(trained["idx_val"])),
        "n_test": int(len(trained["idx_test"])),
        "operational_threshold": float(operational_threshold),
        "validation_panel": val_panel,
        "test_panel": test_panel,
    }

# %%
# ---------------------------------------------------------------------------
# Saving full results to JSON
# ---------------------------------------------------------------------------
iter18_json_path = OUT_DIR / "iter18_classifier_results.json"
json_payload = {
    "meta": {
        "iter": 18,
        "branch": "iter18-xgboost-classifier",
        "problem_formulation": "max(stormflow[t+1..t+h]) >= U",
        "split": {
            "seq_length": SEQ_LENGTH,
            "train_end_idx": 771374,
            "val_end_idx": 936669,
        },
        "xgb_params": DEFAULT_XGB_CLASSIFIER_PARAMS,
        "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
        "features_10": FEATURES_10,
        "variants": [
            {"name": name, "horizon": horizon, "threshold_mgd": threshold}
            for name, horizon, threshold in VARIANTS
        ],
    },
    "results": results_json,
}
with open(iter18_json_path, "w", encoding="utf-8") as handle:
    json.dump(_sanitize_json(json_payload), handle, indent=2, ensure_ascii=False)
print(f"[iter18] Written: {iter18_json_path}")

# %%
# ---------------------------------------------------------------------------
# Figure 1: overlaid PR curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 6.0))
for variant_name, _, _ in VARIANTS:
    pr_points = variant_outputs[variant_name]["test_panel"]["pr_curve"]
    if not pr_points:
        continue
    recall_values = [point["recall"] for point in pr_points]
    precision_values = [point["precision"] for point in pr_points]
    auc_pr = variant_outputs[variant_name]["test_panel"]["threshold_free_metrics"]["auc_pr"]
    ax.plot(recall_values, precision_values, marker="o", linewidth=1.5, label=f"{variant_name} (AUC-PR={auc_pr:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Sampled Precision-Recall curves (iter18)")
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
pr_fig_path = FIG_DIR / "pr_curves.png"
fig.savefig(pr_fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter18] Figure: {pr_fig_path}")

# %%
# ---------------------------------------------------------------------------
# Figure 2: calibration in 4 panels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True, sharey=True)
for axis, (variant_name, _, _) in zip(axes.flat, VARIANTS):
    calibration_rows = variant_outputs[variant_name]["test_panel"]["calibration"]
    pred_mean = [row["pred_mean"] for row in calibration_rows]
    observed = [row["observed_freq"] for row in calibration_rows]
    counts = [row["n"] for row in calibration_rows]
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=0.8)
    axis.plot(pred_mean, observed, marker="o", color="#1f77b4", linewidth=1.5)
    for x_value, y_value, count in zip(pred_mean, observed, counts):
        axis.text(x_value, y_value, f"n={count}", fontsize=8, ha="left", va="bottom")
    axis.set_title(variant_name)
    axis.grid(alpha=0.3)

fig.supxlabel("Mean predicted probability")
fig.supylabel("Observed empirical frequency")
fig.suptitle("Calibration by deciles (iter18)", y=0.98)
fig.tight_layout()
calibration_fig_path = FIG_DIR / "calibration.png"
fig.savefig(calibration_fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter18] Figure: {calibration_fig_path}")

# %%
# ---------------------------------------------------------------------------
# Figure 3: lead time distribution by variant
# ---------------------------------------------------------------------------
boxplot_labels = []
boxplot_data = []
for variant_name, _, _ in VARIANTS:
    per_event_rows = variant_outputs[variant_name]["test_panel"]["lead_time"]["per_event"]
    lead_values = [
        row["lead_time_minutes"]
        for row in per_event_rows
        if row.get("lead_time_minutes") is not None
    ]
    if lead_values:
        boxplot_labels.append(variant_name)
        boxplot_data.append(lead_values)

fig, ax = plt.subplots(figsize=(9.0, 5.0))
if boxplot_data:
    # We choose the parameter name based on the available signature to avoid
    # warnings between older and newer Matplotlib versions.
    boxplot_signature = inspect.signature(ax.boxplot)
    if "tick_labels" in boxplot_signature.parameters:
        ax.boxplot(boxplot_data, tick_labels=boxplot_labels, vert=True)
    else:
        ax.boxplot(boxplot_data, labels=boxplot_labels, vert=True)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Lead time (min)")
    ax.set_title("Lead time distribution by variant (detected events)")
else:
    ax.text(0.5, 0.5, "No positive events with measurable lead time were detected.", ha="center", va="center")
    ax.set_axis_off()

fig.tight_layout()
lead_time_fig_path = FIG_DIR / "lead_time_distribution.png"
fig.savefig(lead_time_fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter18] Figure: {lead_time_fig_path}")

# %%
# ---------------------------------------------------------------------------
# Required markdown comparison
# ---------------------------------------------------------------------------
md_lines = []
md_lines.append("# Iter18 - Binary XGBoost classifier comparison\n")
md_lines.append(
    "Binary classifiers trained on the same temporal split as iter17, "
    "with operational target `max(stormflow[t+1..t+h]) >= U`. "
    "The operational threshold was chosen exclusively on validation to satisfy "
    "recall >= 0.85 when possible.\n"
)
md_lines.append(
    "| Variante | Prevalencia test | AUC-PR | ROC-AUC | Prec@0.5 | Rec@0.5 | F1@0.5 | "
    "Prec@op | Rec@op | F1@op | Thr op | Lead time mediano (min) |"
)
md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for variant_name, _, _ in VARIANTS:
    test_panel = variant_outputs[variant_name]["test_panel"]
    default_metrics = test_panel["default_threshold_metrics"]
    operational_metrics = test_panel["operational_threshold_metrics"]
    threshold_free = test_panel["threshold_free_metrics"]
    lead_time = test_panel["lead_time"]
    prevalence = test_panel["class_distribution"]["prevalence"]
    md_lines.append(
        f"| {variant_name} | "
        f"{_fmt(prevalence, '.4f')} | "
        f"{_fmt(threshold_free['auc_pr'], '.4f')} | "
        f"{_fmt(threshold_free['roc_auc'], '.4f')} | "
        f"{_fmt(default_metrics['precision'], '.3f')} | "
        f"{_fmt(default_metrics['recall'], '.3f')} | "
        f"{_fmt(default_metrics['f1'], '.3f')} | "
        f"{_fmt(operational_metrics['precision'], '.3f')} | "
        f"{_fmt(operational_metrics['recall'], '.3f')} | "
        f"{_fmt(operational_metrics['f1'], '.3f')} | "
        f"{_fmt(operational_metrics['threshold'], '.4f')} | "
        f"{_fmt(lead_time['median_minutes'], '.1f')} |"
    )

md_lines.append("")
md_lines.append("## Guideline success criteria (§8.1)\n")
for variant_name, _, _ in VARIANTS:
    operational_metrics = variant_outputs[variant_name]["test_panel"]["operational_threshold_metrics"]
    precision_value = operational_metrics["precision"]
    recall_value = operational_metrics["recall"]
    status = _success_comment(variant_name, precision_value, recall_value)
    md_lines.append(
        f"- {variant_name}: precision@op={_fmt(precision_value, '.3f')}  "
        f"recall@op={_fmt(recall_value, '.3f')} -> {status}"
    )

md_lines.append("")
md_lines.append("## Brief narrative\n")
for variant_name, horizon, threshold_mgd in VARIANTS:
    test_panel = variant_outputs[variant_name]["test_panel"]
    lead_time = test_panel["lead_time"]
    threshold_free = test_panel["threshold_free_metrics"]
    operational_metrics = test_panel["operational_threshold_metrics"]
    md_lines.append(
        f"- {variant_name} (H={horizon}, U={threshold_mgd:.0f}): "
        f"AUC-PR={_fmt(threshold_free['auc_pr'], '.4f')}, "
        f"precision@op={_fmt(operational_metrics['precision'], '.3f')}, "
        f"recall@op={_fmt(operational_metrics['recall'], '.3f')}, "
        f"median lead time={_fmt(lead_time['median_minutes'], '.1f')} min."
    )

md_lines.append("")
md_lines.append("## Associated figures\n")
md_lines.append("- `outputs/figures/iter18/pr_curves.png`")
md_lines.append("- `outputs/figures/iter18/calibration.png`")
md_lines.append("- `outputs/figures/iter18/lead_time_distribution.png`")

iter18_md_path = OUT_DIR / "iter18_comparison.md"
iter18_md_path.write_text("\n".join(md_lines), encoding="utf-8")
print(f"[iter18] Written: {iter18_md_path}")
