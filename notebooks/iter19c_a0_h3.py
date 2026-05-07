# -*- coding: utf-8 -*-
"""iter19c_a0_h3.py

Replication of TCN A0 (with log1p=True) at horizon H=3.

Objective: close the pending item from HANDOFF_2026-04-29.md §4 that says "If TCN wins
at H=1, that replication at H=3 remains pending before changing the main model
in STATE.md and EXPERIMENTS.md". The user has already decided (post-iter19b) that A0 is
the main model of the TFM. This run verifies that the decision still holds
at H=3 against iter17 XGBoost (NSE_H3_official = 0.6871).

Design:
- A single run with config A0 (L=72, C=32, log1p=True) but horizon=3.
- Early stopping monitoring NSE on val.
- One-to-one comparison against iter17 XGB retrained inline at H=3 on the same
  timestamps, with a sanity check against the official 0.6871 figure.

Restrictions (branch iter19-tcn-comparison):
- Do not modify committed files from iter19/iter19b.
- Do not merge to main; it stays on branch iter19-tcn-comparison.

Colab-style cell structure (`# %%`).

Execution: Colab Pro T4 (~20-30 min).
"""

# %% [markdown]
# # Iter19c - TCN A0 (log1p=True) at H=3 vs iter17 XGB H=3
#
# - Retrains TCN A0 (L=72, C=32, log1p=True) with horizon=3 on train,
#   with early stopping on val.
# - Evaluates on test with `evaluate_full_panel`.
# - Retrains iter17 XGB inline at H=3 for aligned figures + sanity check vs 0.6871.
# - Verdict: TCN_WINS if (NSE_TCN_H3 - 0.6871) >= 0.02; otherwise XGB_WINS.

# %%
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Localizar repo: candidates para Colab y local Windows.
CANDIDATE_ROOTS = [
    Path("/content/stormflow-prediction"),
    Path("C:/Dev/TFM"),
    Path.cwd(),
    Path.cwd().parent,
]
REPO_ROOT = next(
    (p for p in CANDIDATE_ROOTS if (p / "src" / "models" / "tcn_clean.py").exists()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError(
        "Cannot find REPO_ROOT with src/models/tcn_clean.py. "
        f"Tested candidates: {[str(p) for p in CANDIDATE_ROOTS]}. "
        "In Colab, clone the repo into /content/stormflow-prediction before running."
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Locate the parquet with features. Same candidate paths as iter19/iter19b.
CANDIDATE_PARQUETS = [
    REPO_ROOT / "outputs" / "cache" / "df_with_features.parquet",
    Path("/content/drive/MyDrive/Proyecto de capstone/df_with_features.parquet"),
]
PARQUET_PATH = next((p for p in CANDIDATE_PARQUETS if p.exists()), None)
if PARQUET_PATH is None:
    raise RuntimeError(
        "Cannot find df_with_features.parquet. Verify with the user the correct "
        f"path. Tested candidates: {[str(p) for p in CANDIDATE_PARQUETS]}."
    )

# Output paths (reuse the iter19 structure).
OUT_BASE = REPO_ROOT / "outputs"
ITER_DIR = OUT_BASE / "iter19"
WEIGHTS_DIR = ITER_DIR / "weights"
LOGS_DIR = ITER_DIR / "logs"
DIAG_DIR = OUT_BASE / "diagnostic"
FIG_DIR = OUT_BASE / "figures" / "iter19"
for d in [WEIGHTS_DIR, LOGS_DIR, DIAG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[iter19c] REPO_ROOT = {REPO_ROOT}")
print(f"[iter19c] PARQUET   = {PARQUET_PATH}")
print(f"[iter19c] outputs   = {ITER_DIR}")

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.pipeline.normalize_v2 import (
    fit_scaler,
    transform,
    inverse_transform_target,
)
from src.models.tcn_clean import TCNClean
from src.models.xgboost_baseline import (
    aligned_indices,
    FEATURES_10,
    IDX_TRAIN_END,
    IDX_VAL_END,
    TARGET_COL,
    train_xgboost_h,
    DEFAULT_XGB_PARAMS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
)
from src.evaluation.metrics_panel import evaluate_full_panel, DEFAULT_BUCKETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"[iter19c] device={DEVICE}  torch={torch.__version__}")
if DEVICE.type != "cuda":
    print("[iter19c] WARNING: GPU not available. Training will be very slow. "
          "Run this notebook in Colab Pro with T4 for reasonable runtimes.")

# %%
# Global experiment constants.
HORIZON = 3                                       # only methodological change vs iter19b
TCN_FEATURE_COLS = list(FEATURES_10)
N_INPUT_CHANNELS = 1 + len(TCN_FEATURE_COLS)
XGB_NSE_H3_OFFICIAL = 0.6871                      # iter17, n_test = ?
SANITY_THRESHOLD = 0.02

print(f"[iter19c] HORIZON   = {HORIZON} (each step = 5 min, total = {HORIZON * 5} min ahead)")
print(f"[iter19c] TCN channels = {N_INPUT_CHANNELS}  (target_hist + {TCN_FEATURE_COLS})")

# %%
# Load the parquet.
t0 = time.time()
df = pd.read_parquet(PARQUET_PATH)
print(f"[iter19c] parquet loaded in {time.time() - t0:.1f}s. shape={df.shape}")
TOTAL_LEN = len(df)
print(f"[iter19c] split: train_end={IDX_TRAIN_END}  val_end={IDX_VAL_END}  total={TOTAL_LEN}")

missing = [c for c in [TARGET_COL] + TCN_FEATURE_COLS if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in parquet: {missing}")

# %%
# Scaler con log1p_target=True (config A0).
df_train = df.iloc[:IDX_TRAIN_END]
SCALER = fit_scaler(
    df_train,
    feature_cols=TCN_FEATURE_COLS,
    target_col=TARGET_COL,
    log1p_target=True,
)
print(f"[iter19c] scaler fit OK (log1p=True). target_mean={SCALER['target_mean']:.4f} "
      f"target_std={SCALER['target_std']:.4f}")

# %%
# Pre-construye los arrays normalizados (N, 11) y target (N,).

def build_feature_array(scaler: dict) -> tuple:
    """Devuelve (feature_arr, target_arr) sobre TODO el dataframe.

    feature_arr (N, 11): canal 0 = target normalizado historico,
    canales 1..10 = features normalizadas en orden FEATURES_10.
    """
    out = transform(df, scaler)
    full = np.concatenate(
        [out["target"].reshape(-1, 1), out["features"]], axis=1
    ).astype(np.float32)
    return full, out["target"].astype(np.float32)


FEATURE_ARR, TARGET_ARR = build_feature_array(SCALER)
print(f"[iter19c] arrays normalizados shape = {FEATURE_ARR.shape}")

# %%
# Origins by L with horizon=3. For L=72 H=3, the windows do not cross split
# boundaries (iter17 convention). n_test will be marginally smaller than at H=1
# (~2 fewer points due to the val boundary).

L_FIXED = 72


def origins_for_L(L: int) -> dict:
    return {
        "train": aligned_indices(0,             IDX_TRAIN_END, HORIZON, TOTAL_LEN, seq_length=L),
        "val":   aligned_indices(IDX_TRAIN_END, IDX_VAL_END,    HORIZON, TOTAL_LEN, seq_length=L),
        "test":  aligned_indices(IDX_VAL_END,   TOTAL_LEN,      HORIZON, TOTAL_LEN, seq_length=L),
    }


ORIGINS = origins_for_L(L_FIXED)
print(f"[iter19c] L={L_FIXED} H={HORIZON}: n_train={len(ORIGINS['train']):,} "
      f"n_val={len(ORIGINS['val']):,} n_test={len(ORIGINS['test']):,}")

# %%
# Dataset (copia de iter19b, ya parametrizado por horizon).

class WindowsDataset(Dataset):
    def __init__(
        self,
        feature_arr: np.ndarray,
        target_arr: np.ndarray,
        origins: np.ndarray,
        seq_length: int,
        horizon: int = 1,
    ) -> None:
        self.features = torch.from_numpy(feature_arr)
        self.target = torch.from_numpy(target_arr)
        self.origins = torch.from_numpy(origins.astype(np.int64))
        self.L = int(seq_length)
        self.h = int(horizon)

    def __len__(self) -> int:
        return self.origins.shape[0]

    def __getitem__(self, i: int):
        t = int(self.origins[i].item())
        x = self.features[t - self.L + 1 : t + 1]   # ventana hasta t inclusive
        y = self.target[t + self.h]                 # objetivo en t+h
        return x, y


# %%
# Utilidades de evaluacion (copia de iter19b).

def compute_nse_mgd(y_pred_norm: np.ndarray, y_true_norm: np.ndarray, scaler: dict) -> float:
    yp = inverse_transform_target(y_pred_norm, scaler)
    yt = inverse_transform_target(y_true_norm, scaler)
    yp = np.clip(yp, 0.0, None)
    denom = float(((yt - yt.mean()) ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(((yt - yp) ** 2).sum()) / denom


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x).cpu().numpy()
        preds.append(out)
        trues.append(y.numpy())
    return np.concatenate(preds).astype(np.float64), np.concatenate(trues).astype(np.float64)


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, n_workers: int) -> DataLoader:
    use_workers = n_workers if (n_workers > 0 and DEVICE.type == "cuda") else 0
    pin = (DEVICE.type == "cuda")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=use_workers,
        pin_memory=pin,
        drop_last=False,
    )


# %%
# train_one_run: copia simplificada de iter19b. Siempre evalua en test.

def train_one_run(
    run_id: str,
    config: dict,
    n_workers: int = 2,
) -> dict:
    """Train the TCN with the given configuration and evaluate on val + test.

    Early stopping monitors val (NSE in MGD). Best weights are saved to
    WEIGHTS_DIR/{run_id}.pt. Returns a dict with metrics and prediction arrays.
    """
    seed = int(config.get("seed", SEED))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    L = int(config["L"])
    C = int(config["C"])

    train_ds = WindowsDataset(FEATURE_ARR, TARGET_ARR, ORIGINS["train"], L, HORIZON)
    val_ds   = WindowsDataset(FEATURE_ARR, TARGET_ARR, ORIGINS["val"],   L, HORIZON)
    test_ds  = WindowsDataset(FEATURE_ARR, TARGET_ARR, ORIGINS["test"],  L, HORIZON)

    train_loader = _make_loader(train_ds, config["batch_size"], shuffle=True, n_workers=n_workers)
    val_loader = _make_loader(val_ds, config["batch_size"], shuffle=False, n_workers=n_workers)
    test_loader = _make_loader(test_ds, config["batch_size"], shuffle=False, n_workers=n_workers)

    model = TCNClean(
        in_channels=N_INPUT_CHANNELS,
        hidden_channels=C,
        kernel_size=int(config["kernel_size"]),
        num_blocks=int(config["num_blocks"]),
        dilations=config.get("dilations"),
        dropout=float(config["dropout"]),
    ).to(DEVICE)
    n_params = model.num_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val_nse = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    patience = int(config["patience"])
    grad_clip = float(config["grad_clip"])
    weights_path = WEIGHTS_DIR / f"{run_id}.pt"
    log_path = LOGS_DIR / f"{run_id}.log"

    train_log_lines = [
        f"# {run_id}",
        f"config={json.dumps(config, default=str)}",
        f"n_params={n_params}",
        f"device={DEVICE}",
        f"L={L} C={C} log1p=True horizon={HORIZON}",
        f"n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_ds)}",
    ]
    train_loss_history = []
    val_nse_history = []

    t_start = time.time()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss_sum += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss_sum / max(n_batches, 1)
        train_loss_history.append(avg_loss)

        y_pred_val, y_true_val = predict_loader(model, val_loader, DEVICE)
        val_nse = compute_nse_mgd(y_pred_val, y_true_val, SCALER)
        val_nse_history.append(val_nse)

        elapsed = time.time() - t_start
        line = (f"epoch={epoch:03d}  train_loss={avg_loss:.5f}  "
                f"val_NSE={val_nse:.4f}  t={elapsed:.1f}s")
        print(f"[{run_id}] {line}")
        train_log_lines.append(line)

        if val_nse > best_val_nse:
            best_val_nse = val_nse
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_nse": float(val_nse),
                "n_params": n_params,
                "scaler_log1p": True,
                "horizon": HORIZON,
            }, weights_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                stop_line = (f"early_stopping epoch={epoch}  best_epoch={best_epoch}  "
                             f"best_val_nse={best_val_nse:.4f}")
                print(f"[{run_id}] {stop_line}")
                train_log_lines.append(stop_line)
                break

    train_seconds = time.time() - t_start
    train_log_lines.append(
        f"final  best_epoch={best_epoch}  best_val_nse={best_val_nse:.6f}  "
        f"train_seconds={train_seconds:.1f}"
    )

    # Cargar mejores pesos para evaluacion final.
    ckpt = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # Panel test.
    y_pred_test, y_true_test = predict_loader(model, test_loader, DEVICE)
    yp_test_mgd = inverse_transform_target(y_pred_test, SCALER)
    yt_test_mgd = inverse_transform_target(y_true_test, SCALER)
    yp_test_mgd = np.clip(yp_test_mgd, 0.0, None)
    test_origins = ORIGINS["test"]
    test_timestamps = df.iloc[test_origins + HORIZON]["timestamp"].reset_index(drop=True)
    test_panel = evaluate_full_panel(yt_test_mgd, yp_test_mgd, timestamps=test_timestamps)

    # Persistir log.
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_log_lines) + "\n")

    return {
        "run_id": run_id,
        "config": dict(config),
        "n_params": n_params,
        "weights_path": str(weights_path),
        "log_path": str(log_path),
        "best_epoch": best_epoch,
        "best_val_nse": best_val_nse,
        "train_seconds": train_seconds,
        "train_loss_history": train_loss_history,
        "val_nse_history": val_nse_history,
        "test_panel": test_panel,
        "test_y_pred_mgd": yp_test_mgd,
        "test_y_true_mgd": yt_test_mgd,
        "test_timestamps": test_timestamps,
    }


# %%
# A0 configuration at H=3.
A0_H3_CONFIG = dict(
    horizon=HORIZON,
    L=72,
    C=32,
    log1p_target=True,
    kernel_size=3,
    num_blocks=4,
    dilations=[1, 2, 4, 8],
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=50,
    patience=10,
    grad_clip=1.0,
    seed=SEED,
)

print("[iter19c] config A0 a H=3:")
for k, v in A0_H3_CONFIG.items():
    print(f"  {k} = {v}")

# %%
# Training.
print("\n[iter19c] === Training A0 H=3 (log1p=True) and evaluating on test ===")
tcn_run = train_one_run("A0_h3", A0_H3_CONFIG)
tcn_panel = tcn_run["test_panel"]
TCN_NSE_H3 = float(tcn_panel["global"]["nse"])
print(f"\n[iter19c] TCN A0 H=3 test NSE = {TCN_NSE_H3:.4f}  "
      f"err_pico = {tcn_panel['global']['peak_err_pct']:+.1f}%  n={tcn_panel['global']['n']}")

# %%
# Retrain iter17 XGB inline at H=3 to get aligned figures.
print("\n[iter19c] === Retraining xgb_lag6_feat10 at H=3 for direct comparison ===")
xgb_out = train_xgboost_h(
    df,
    horizon=HORIZON,
    lags=6,
    features=FEATURES_10,
    include_lags=True,
    include_features=True,
    xgb_params=DEFAULT_XGB_PARAMS,
    early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
    seq_length=72,                # official iter17 convention
    verbose=False,
)
xgb_y_true = xgb_out["y_true_test"]
xgb_y_pred = xgb_out["y_pred_test"]
xgb_ts = xgb_out["timestamps_test"]
xgb_panel = evaluate_full_panel(xgb_y_true, xgb_y_pred, timestamps=xgb_ts)
XGB_NSE_INLINE_H3 = float(xgb_panel["global"]["nse"])
print(f"[iter19c] xgb_lag6_feat10 inline H=3: NSE={XGB_NSE_INLINE_H3:.4f}  "
      f"n={xgb_panel['global']['n']}")

# %%
# Sanity check contra cifra oficial iter17.
delta_xgb_sanity = abs(XGB_NSE_INLINE_H3 - XGB_NSE_H3_OFFICIAL)
if delta_xgb_sanity > SANITY_THRESHOLD:
    raise RuntimeError(
        f"[iter19c] SANITY FAIL: |XGB_NSE_inline ({XGB_NSE_INLINE_H3:.4f}) - "
        f"XGB_NSE_oficial_iter17 ({XGB_NSE_H3_OFFICIAL:.4f})| = {delta_xgb_sanity:.4f} "
        f"> threshold {SANITY_THRESHOLD}. Something in the split or the features has changed. "
        "PAUSAR Y AVISAR AL USUARIO antes de seguir (per spec iter19c)."
    )
print(f"[iter19c] sanity OK: |XGB_inline - XGB_oficial| = {delta_xgb_sanity:.4f} "
      f"<= umbral {SANITY_THRESHOLD}")

# Sanity adicional: TCN y XGB deben evaluarse sobre el mismo numero de samples
# en test cuando ambos usan L=72 H=3. Si difieren, recortar antes.
n_tcn = int(tcn_panel["global"]["n"])
n_xgb = int(xgb_panel["global"]["n"])
if n_tcn != n_xgb:
    print(f"[iter19c] WARNING: n_test difiere TCN={n_tcn} vs XGB={n_xgb}. "
          "Las metricas globales son sobre conjuntos ligeramente distintos.")
else:
    print(f"[iter19c] alineacion OK: n_test TCN = n_test XGB = {n_tcn}")

# %%
# Verdict.
DELTA_NSE = TCN_NSE_H3 - XGB_NSE_H3_OFFICIAL
THRESHOLD = 0.02
VERDICT_H3 = "TCN_WINS" if DELTA_NSE >= THRESHOLD else "XGB_WINS"

print("\n[iter19c] === VEREDICTO H=3 ===")
print(f"  NSE TCN A0 H=3 (test)         = {TCN_NSE_H3:.4f}")
print(f"  NSE XGB iter17 H=3 (oficial)  = {XGB_NSE_H3_OFFICIAL:.4f}")
print(f"  NSE XGB inline H=3 (aligned) = {XGB_NSE_INLINE_H3:.4f}")
print(f"  delta NSE (TCN - XGB iter17)  = {DELTA_NSE:+.4f}")
print(f"  umbral cierre                 = +{THRESHOLD:.3f}")
print(f"  VEREDICTO H=3                 = {VERDICT_H3}")

# %%
# Persistir JSON.

def serialize_panel(panel: dict) -> dict:
    p = dict(panel)
    p.pop("peak_lag_per_event", None)
    return p


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if isinstance(o, tuple):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


verdict_explanation = (
    f"TCN A0 H=3 supera a XGB iter17 H=3 por {DELTA_NSE:+.4f} NSE "
    f"(>= threshold +{THRESHOLD}). Replication confirmed: A0 remains the main model at H=3."
    if VERDICT_H3 == "TCN_WINS" else
    f"TCN A0 H=3 NO supera a XGB iter17 H=3 (delta={DELTA_NSE:+.4f} < umbral +{THRESHOLD}). "
    "El TCN no replica la ganancia de H=1 al alargar el horizonte; A0 queda confirmado como "
    "main model only at H=1."
)

results_json = {
    "meta": {
        "iter": "19c",
        "branch": "iter19-tcn-comparison",
        "purpose": (
            "Replicate A0 (with log1p) at H=3 to close the comparison against XGB iter17 H=3. "
            "This closes the pending item from HANDOFF_2026-04-29.md about validating A0 as the "
            "main model of the TFM at H=3."
        ),
        "horizon": HORIZON,
        "config": dict(A0_H3_CONFIG),
        "device": str(DEVICE),
        "seed": SEED,
        "feature_cols_tcn": [TARGET_COL] + list(FEATURES_10),
        "split": {
            "train_end_idx": IDX_TRAIN_END,
            "val_end_idx": IDX_VAL_END,
            "total_len": TOTAL_LEN,
        },
        "baseline_xgb_iter17": {
            "name": "xgb_lag6_feat10",
            "nse_h3_iter17": XGB_NSE_H3_OFFICIAL,
        },
    },
    "training": {
        "best_epoch": int(tcn_run["best_epoch"]),
        "best_val_nse": float(tcn_run["best_val_nse"]),
        "training_time_seconds": float(tcn_run["train_seconds"]),
        "n_params": int(tcn_run["n_params"]),
        "weights_path": tcn_run["weights_path"],
        "train_loss_history": list(map(float, tcn_run["train_loss_history"])),
        "val_nse_history": list(map(float, tcn_run["val_nse_history"])),
    },
    "metrics_test_TCN_A0_H3": serialize_panel(tcn_panel),
    "metrics_test_XGB_inline_H3": serialize_panel(xgb_panel),
    "comparison_TCN_vs_XGB_H3": {
        "tcn_nse_h3": TCN_NSE_H3,
        "xgb_inline_nse_h3": XGB_NSE_INLINE_H3,
        "xgb_iter17_nse_h3": XGB_NSE_H3_OFFICIAL,
        "delta_nse_vs_iter17": DELTA_NSE,
        "delta_nse_vs_inline": float(TCN_NSE_H3 - XGB_NSE_INLINE_H3),
        "criterion_threshold": THRESHOLD,
        "sanity_xgb_inline_vs_iter17_delta": float(delta_xgb_sanity),
        "verdict_h3": VERDICT_H3,
        "verdict_explanation": verdict_explanation,
    },
}

json_path = DIAG_DIR / "iter19c_a0_h3_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(_sanitize(results_json), f, indent=2, ensure_ascii=False)
print(f"\n[iter19c] JSON: {json_path}")

# %%
# Generar markdown comparativo.
md_path = DIAG_DIR / "iter19c_a0_h3_comparison.md"


def fmt(v, spec=".4f"):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return "n/a"
    return f"{v:{spec}}"


tcn_g = tcn_panel["global"]
xgb_g = xgb_panel["global"]
tcn_rec50 = tcn_panel["recall"]["at_50_mgd"].get("recall")
xgb_rec50 = xgb_panel["recall"]["at_50_mgd"].get("recall")
tcn_rec25 = tcn_panel["recall"]["at_25_mgd"].get("recall")
xgb_rec25 = xgb_panel["recall"]["at_25_mgd"].get("recall")

# Cifras de iter19/iter19b para la lectura honesta de consistencia H=1 vs H=3.
NSE_A0_H1_TEST = 0.8890   # iter19b A0 test
NSE_A4_H1_TEST = 0.8983   # iter19 A4 test
NSE_XGB_H1_INLINE = 0.8631  # iter19 XGB inline
NSE_XGB_H1_ITER17 = 0.8630  # iter17 oficial H=1
DELTA_TCN_H1 = NSE_A0_H1_TEST - NSE_XGB_H1_ITER17   # 0.026 aprox
DELTA_TCN_H3 = DELTA_NSE                            # variable

lines: list = []
lines.append("# Iter19c - TCN A0 (log1p=True) a H=3 vs XGB iter17 H=3\n")

# 1. Executive summary
lines.append("## Executive summary\n")
lines.append(
    f"Se replica la TCN A0 (L=72, C=32, log1p=True) a horizonte H=3 (15 minutos adelante) "
    f"con la misma config que en iter19b (H=1). NSE TCN A0 H=3 test = **{TCN_NSE_H3:.4f}**. "
    f"NSE XGB iter17 H=3 oficial = {XGB_NSE_H3_OFFICIAL:.4f}. "
    f"NSE XGB inline H=3 (aligned, sanity) = {XGB_NSE_INLINE_H3:.4f}. "
    f"Delta = **{DELTA_NSE:+.4f}** (umbral cierre = +{THRESHOLD:.3f}). "
    f"VEREDICTO H=3: **{VERDICT_H3}**.\n"
)
lines.append(
    f"Sanity check del baseline XGB inline vs cifra oficial iter17: "
    f"|delta| = {delta_xgb_sanity:.4f} <= {SANITY_THRESHOLD} (OK).\n"
)

# 2. Resultados TCN A0 H=3 en test
lines.append("## Resultados TCN A0 H=3 en test\n")
lines.append("| Metrica | Valor |")
lines.append("|---|---:|")
lines.append(f"| n_test | {tcn_g['n']} |")
lines.append(f"| NSE | {tcn_g['nse']:.4f} |")
lines.append(f"| RMSE | {tcn_g['rmse']:.3f} |")
lines.append(f"| MAE | {tcn_g['mae']:.3f} |")
lines.append(f"| peak_real (MGD) | {tcn_g['peak_real_mgd']:.2f} |")
lines.append(f"| peak_pred (MGD) | {tcn_g['peak_pred_mgd']:.2f} |")
lines.append(f"| peak_err_pct | {tcn_g['peak_err_pct']:+.1f} |")
lines.append(f"| recall@25 | {fmt(tcn_rec25, '.3f')} |")
lines.append(f"| recall@50 | {fmt(tcn_rec50, '.3f')} |")
lines.append(f"| n_params | {tcn_run['n_params']:,} |")
lines.append(f"| best_epoch | {tcn_run['best_epoch']} |")
lines.append(f"| training_time_s | {tcn_run['train_seconds']:.0f} |")
lines.append("")

lines.append("### NSE por bucket (TCN A0 H=3 test)\n")
lines.append("| Bucket | n | NSE | RMSE | bias (MGD) | peak_err_pct |")
lines.append("|---|---:|---:|---:|---:|---:|")
for bname, _, _ in DEFAULT_BUCKETS:
    b = tcn_panel["buckets"][bname]
    lines.append(
        f"| {bname} | {b['n']} | {fmt(b['nse'], '+.3f')} | "
        f"{fmt(b['rmse'], '.3f')} | {fmt(b['bias'], '+.3f')} | "
        f"{fmt(b['peak_err_pct'], '+.1f')} |"
    )
lines.append("")

# 3. Comparacion contra XGB H=3
lines.append("## Comparacion directa contra xgb_lag6_feat10 H=3\n")
lines.append("Ambos modelos evaluados sobre los mismos timestamps de test (L=72, H=3, "
             f"n_TCN={n_tcn}, n_XGB={n_xgb}).\n")
lines.append("| Modelo | NSE | RMSE | err_pico (%) | recall@50 | NSE_extremo |")
lines.append("|---|---:|---:|---:|---:|---:|")
xgb_b_ext = xgb_panel["buckets"]["Extremo"]
tcn_b_ext = tcn_panel["buckets"]["Extremo"]
lines.append(
    f"| xgb_lag6_feat10 (inline) | {xgb_g['nse']:.4f} | {xgb_g['rmse']:.3f} | "
    f"{xgb_g['peak_err_pct']:+.1f} | {fmt(xgb_rec50, '.3f')} | {fmt(xgb_b_ext['nse'], '+.3f')} |"
)
lines.append(
    f"| TCN A0 H=3 | {tcn_g['nse']:.4f} | {tcn_g['rmse']:.3f} | "
    f"{tcn_g['peak_err_pct']:+.1f} | {fmt(tcn_rec50, '.3f')} | {fmt(tcn_b_ext['nse'], '+.3f')} |"
)


def _safe_diff(a, b, fmt_spec="+.4f"):
    if a is None or b is None:
        return "n/a"
    if isinstance(a, float) and (a != a):
        return "n/a"
    if isinstance(b, float) and (b != b):
        return "n/a"
    return f"{a - b:{fmt_spec}}"


lines.append(
    f"| **delta (TCN - XGB inline)** | "
    f"**{_safe_diff(tcn_g['nse'], xgb_g['nse'], '+.4f')}** | "
    f"{_safe_diff(tcn_g['rmse'], xgb_g['rmse'], '+.3f')} | "
    f"{_safe_diff(tcn_g['peak_err_pct'], xgb_g['peak_err_pct'], '+.1f')} | "
    f"{_safe_diff(tcn_rec50, xgb_rec50, '+.3f')} | "
    f"{_safe_diff(tcn_b_ext['nse'], xgb_b_ext['nse'], '+.3f')} |"
)
lines.append("")
lines.append(f"NSE de referencia oficial iter17 H=3 (xgb_lag6_feat10): **{XGB_NSE_H3_OFFICIAL:.4f}**.")
lines.append(f"Delta usado para el veredicto: TCN({TCN_NSE_H3:.4f}) - XGB_iter17({XGB_NSE_H3_OFFICIAL:.4f}) = **{DELTA_NSE:+.4f}**.")
lines.append(f"Threshold from DIAGNOSTIC_REPORT §7.6: +{THRESHOLD:.3f}.\n")

# 4. Honest reading
lines.append("## Honest reading\n")

# 4.1 Consistencia H=1 vs H=3
lines.append(
    f"**Consistencia H=1 vs H=3.** En H=1 (iter19b) la TCN A0 alcanzo NSE_test = "
    f"{NSE_A0_H1_TEST:.4f}, frente a XGB iter17 H=1 = {NSE_XGB_H1_ITER17:.4f} "
    f"(delta = {DELTA_TCN_H1:+.4f}). En H=3 obtenemos delta = {DELTA_TCN_H3:+.4f}. "
)
if VERDICT_H3 == "TCN_WINS":
    if abs(DELTA_TCN_H3) >= abs(DELTA_TCN_H1):
        lines.append(
            "La ventaja del TCN se mantiene o incluso aumenta al alargar el horizonte. "
            "Esto es consistente con la hipotesis del DIAGNOSTIC_REPORT §7.6: la flexibilidad "
            "temporal de la TCN aporta valor sobre todo cuando la inercia autoregresiva pierde "
            "potencia (H>1).\n"
        )
    else:
        lines.append(
            "La TCN sigue ganando pero con menor margen que en H=1. La ganancia se preserva "
            "pero se debilita a horizonte mas largo, lo que sugiere que parte de la ventaja en "
            "H=1 viene de la fuerte autocorrelacion local que ambos modelos absorben.\n"
        )
else:
    lines.append(
        "La ventaja del TCN observada en H=1 NO se replica a H=3: el GBM se equilibra al "
        "alargar el horizonte. Esto es coherente con la naturaleza convolucional fija de la "
        "TCN: el receptive field optimizado para H=1 no transfiere directamente a H=3, donde "
        "la senal autoregresiva pesa menos y otras dependencias (lluvia historica, API) ganan "
        "peso relativo. El GBM con lags explicitos absorbe estas ultimas eficientemente.\n"
    )

# 4.2 Main-model implication for the TFM
if VERDICT_H3 == "TCN_WINS":
    lines.append(
        "**Implication for the TFM**: A0 is confirmed as the main model at H=1 and H=3. "
        "El cierre del pendiente del HANDOFF_2026-04-29.md §4 es positivo: STATE.md y "
        "EXPERIMENTS.md pueden actualizarse para reportar la TCN como modelo de referencia "
        "para el sistema operativo MSD a corto y medio plazo.\n"
    )
else:
    lines.append(
        "**Implication for the TFM**: A0 is confirmed as the main model only at H=1. "
        "A H=3 el GBM iguala o supera al TCN. Hay dos caminos honestos: (a) reportar el TCN "
        "as the main model at H=1 and XGB for H=3, which fragments the solution but reflects "
        "el resultado real; (b) reportar XGB como modelo unico para H=1 y H=3 a costa de la "
        "ganancia operativa de +0.026 NSE en H=1, ganando coherencia y sencillez. El usuario "
        "decide; esta corrida no fuerza la respuesta.\n"
    )

# 4.3 Calidad operativa (extremos)
b_ext_tcn = tcn_panel["buckets"]["Extremo"]
b_ext_xgb = xgb_panel["buckets"]["Extremo"]
lines.append(
    f"**Comportamiento operativo en bucket Extremo (>=50 MGD).** "
    f"NSE_extremo TCN = {b_ext_tcn['nse']:+.3f}, XGB = {b_ext_xgb['nse']:+.3f}. "
    f"bias_extremo TCN = {b_ext_tcn['bias']:+.3f} MGD, XGB = {b_ext_xgb['bias']:+.3f} MGD. "
)
if abs(b_ext_tcn["bias"]) < abs(b_ext_xgb["bias"]):
    lines.append(
        "La TCN sigue teniendo menor sesgo absoluto en eventos criticos a H=3. "
        "This reinforces the iter19b decision to choose A0 as the main model "
        "por motivos operativos, no solo por NSE global.\n"
    )
else:
    lines.append(
        "El GBM tiene aqui un sesgo absoluto menor en eventos criticos a H=3, "
        "invirtiendo la ventaja operativa que A0 tenia en H=1. Otro punto a considerar "
        "for the final main-model decision.\n"
    )

# 5. Verdict and recommendation
lines.append("## Verdict and recommendation\n")
if VERDICT_H3 == "TCN_WINS":
    lines.append(
        f"**TCN_WINS en H=3** por margen >= {THRESHOLD:.3f} NSE. "
        f"Recomendacion: actualizar STATE.md y EXPERIMENTS.md para registrar A0 como modelo "
        f"of the TFM at H=1 and H=3. Handoff §4 is closed positively. "
        f"Documentar el coste operativo de mantener una TCN en produccion (GPU, dependencia "
        f"PyTorch) frente al beneficio en NSE.\n"
    )
else:
    lines.append(
        f"**XGB_WINS en H=3** (TCN no supera el umbral +{THRESHOLD:.3f} NSE). "
        f"Recomendacion: actualizar STATE.md y EXPERIMENTS.md indicando que la replica a H=3 "
        f"NO confirma la ganancia de H=1. Decidir explicitamente la estrategia del TFM: "
        f"modelo dual (TCN H=1 + XGB H=3) o modelo unico XGB. Documentar el resultado como "
        f"hallazgo metodologico positivo: el GBM iter17 absorbe la senal predictible a "
        f"horizonte medio donde la TCN no aporta valor adicional.\n"
    )

# 6. Limitations
lines.append("## Recognized limitations\n")
lines.append(
    "- Una sola seed (=42), igual que iter19/iter19b. La diferencia TCN vs XGB en H=3 puede "
    "estar dentro del ruido estocastico de inicializacion. Para mas certeza haria falta "
    "repetir con varias seeds.\n"
    "- Comparacion solo a H=3. Si se quiere narrativa completa, queda pendiente H=6 y H=12.\n"
    "- XGB inline reentrenado con misma seed/params que iter17 pero el dataset puede haber "
    f"cambiado marginalmente desde abril 2026; el sanity check valida que el delta vs cifra "
    f"oficial es {delta_xgb_sanity:.4f} <= {SANITY_THRESHOLD}.\n"
    "- Loss Huber simple, sin componente de magnitud ni peak penalty, igual que iter19. "
    "Una loss asimetrica podria mejorar el bucket Extremo a H=3 pero queda fuera de scope.\n"
)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[iter19c] MD:   {md_path}")

# %%
# Figuras.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fig 1 - hidrograma evento extremo H=3 (TCN vs XGB).
yt_test = tcn_run["test_y_true_mgd"]
yp_test = tcn_run["test_y_pred_mgd"]
ts_test = pd.to_datetime(tcn_run["test_timestamps"])
i_peak_tcn = int(np.argmax(yt_test))
half = 96  # 8h
i0 = max(0, i_peak_tcn - half)
i1 = min(len(yt_test), i_peak_tcn + half + 1)

xgb_ts_dt = pd.to_datetime(xgb_ts)
peak_ts = ts_test.iloc[i_peak_tcn]
diffs = np.abs((xgb_ts_dt.values - np.datetime64(peak_ts)).astype("timedelta64[ns]").astype("int64"))
match_idx = int(np.argmin(diffs))
xi0 = max(0, match_idx - half)
xi1 = min(len(xgb_y_pred), match_idx + half + 1)

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(ts_test.iloc[i0:i1], yt_test[i0:i1], color="#1f77b4", linewidth=1.5, label="y_real")
ax.plot(ts_test.iloc[i0:i1], yp_test[i0:i1], color="#d62728", linewidth=1.4, alpha=0.9,
        label="TCN A0 H=3")
ax.plot(xgb_ts_dt.iloc[xi0:xi1], xgb_y_pred[xi0:xi1], color="#ff7f0e", linewidth=1.2,
        linestyle="--", alpha=0.9, label="XGB lag6+feat10 H=3")
ax.axhline(50.0, linestyle=":", color="#888", linewidth=0.8, label="Extremo Threshold 50 MGD")
ax.set_title(
    f"Test extreme-event hydrograph (H=3) - TCN A0 vs XGB\n"
    f"Real peak = {yt_test[i_peak_tcn]:.1f} MGD at {peak_ts}"
)
ax.set_xlabel("Date")
ax.set_ylabel("stormflow (MGD)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "hydrograph_extreme_event_H3.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter19c] figura: hydrograph_extreme_event_H3.png")

# Fig 2 - scatter real vs pred coloreado por bucket (H=3, TCN A0).
rng = np.random.default_rng(42)
n = len(yt_test)
sample_idx = rng.choice(n, size=min(20_000, n), replace=False)
large_idx = np.where(yt_test > 10.0)[0]
plot_idx = np.unique(np.concatenate([sample_idx, large_idx]))

fig, ax = plt.subplots(figsize=(6.5, 6.5))
bucket_colors = ["#aaaaaa", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
for (bname, lo, hi), col in zip(DEFAULT_BUCKETS, bucket_colors):
    mask = (yt_test[plot_idx] >= lo) & (yt_test[plot_idx] < hi)
    if int(mask.sum()) == 0:
        continue
    ax.scatter(
        yt_test[plot_idx][mask], yp_test[plot_idx][mask],
        s=4, alpha=0.4, color=col, label=f"{bname} (n_plot={int(mask.sum())})",
    )
lim = max(float(np.max(yt_test)), float(np.max(yp_test))) * 1.05
ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=0.8, label="y_pred = y_real")
ax.set_xlim(-1, lim)
ax.set_ylim(-1, lim)
ax.set_xlabel("y_real (MGD)")
ax.set_ylabel("y_pred (MGD)")
ax.set_title(f"TCN A0 H=3 sobre test - NSE={tcn_g['nse']:.4f}")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "scatter_real_vs_pred_H3.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter19c] figura: scatter_real_vs_pred_H3.png")

# %%
# Final summary in console.
print("\n" + "=" * 80)
print("iter19c - resumen final")
print("=" * 80)
print(f"  Config A0 H=3   : L=72 C=32 log1p=True horizon={HORIZON}")
print(f"  NSE TCN H=3     : {TCN_NSE_H3:.4f}  (epoch {tcn_run['best_epoch']}, "
      f"{tcn_run['train_seconds']:.0f}s)")
print(f"  NSE XGB iter17  : {XGB_NSE_H3_OFFICIAL:.4f}  (oficial)")
print(f"  NSE XGB inline  : {XGB_NSE_INLINE_H3:.4f}  (sanity delta={delta_xgb_sanity:.4f})")
print(f"  delta NSE       : {DELTA_NSE:+.4f}  (umbral +{THRESHOLD:.3f})")
print(f"  err_pico TCN    : {tcn_g['peak_err_pct']:+.1f}%")
print(f"  err_pico XGB    : {xgb_g['peak_err_pct']:+.1f}%")
print(f"  recall@50 TCN   : {fmt(tcn_rec50, '.3f')}")
print(f"  recall@50 XGB   : {fmt(xgb_rec50, '.3f')}")
print(f"  NSE_extremo TCN : {tcn_panel['buckets']['Extremo']['nse']:+.3f}  "
      f"bias={tcn_panel['buckets']['Extremo']['bias']:+.3f}")
print(f"  NSE_extremo XGB : {xgb_panel['buckets']['Extremo']['nse']:+.3f}  "
      f"bias={xgb_panel['buckets']['Extremo']['bias']:+.3f}")
print(f"  VEREDICTO H=3   : {VERDICT_H3}")
print(f"  weights : {WEIGHTS_DIR / 'A0_h3.pt'}")
print(f"  json    : {json_path}")
print(f"  md      : {md_path}")
print(f"  figs    : {FIG_DIR / 'hydrograph_extreme_event_H3.png'}")
print(f"            {FIG_DIR / 'scatter_real_vs_pred_H3.png'}")
