# -*- coding: utf-8 -*-
"""iter19_tcn_clean.py

Iter19 orchestrator notebook (branch `iter19-tcn-comparison`).

Implements the standard TCN (Bai et al. 2018) described in `outputs/diagnostic/
DIAGNOSTIC_REPORT.md` §7.6. A single final run on test, preceded by
a mini-ablation on val (4 configurations) to set log1p, L, and C.

Full spec: branch `iter19-tcn-comparison`, original prompt file.

Design:
- No two-stage, no classifier, no hard switch.
- Simple Huber loss in normalized space.
- Inputs (B, T, F): 1 historical target channel + 10 S5 features = 11 channels.
- Window origins: `aligned_indices` from iter17 (with the experiment's seq_length=L)
  to guarantee that windows do not cross the split boundary.
  For L=72 the indices match `xgb_lag6_feat10` 1:1 (n_test=165222).

Colab-style cell structure (`# %%`).

Execution:
- Local (Windows / VS Code): `python notebooks/iter19_tcn_clean.py` or run
  celda a celda. Funciona en CPU pero es muy lento — usar Colab T4.
- Colab Pro with T4 GPU: clone the repo into /content/stormflow-prediction and
  set the cwd there. Verify with the user the parquet path
  (it may be in the repo local cache or in Drive).
"""

# %% [markdown]
# # Iter19 - Clean TCN (Bai 2018) vs xgb_lag6_feat10
#
# - Mini-ablation on val: A0 (baseline), A1 (without log1p), A2 (L=144), A3 (C=64).
# - Winning selection assuming factor independence; extra run A4
#   if the winning combination does not match any already trained A0..A3.
# - Final run on test with the winning config; comparison against
#   `xgb_lag6_feat10` (NSE H=1 = 0.8630, official iter17 figure).
# - Verdict: if `NSE_TCN - NSE_XGB >= 0.02` -> TCN wins; otherwise, deep
#   learning is closed for the TFM.

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

# Locate the parquet with features. Confirm with the user if none applies.
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

# Output paths.
OUT_BASE = REPO_ROOT / "outputs"
ITER_DIR = OUT_BASE / "iter19"
WEIGHTS_DIR = ITER_DIR / "weights"
LOGS_DIR = ITER_DIR / "logs"
DIAG_DIR = OUT_BASE / "diagnostic"
FIG_DIR = OUT_BASE / "figures" / "iter19"
for d in [WEIGHTS_DIR, LOGS_DIR, DIAG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[iter19] REPO_ROOT = {REPO_ROOT}")
print(f"[iter19] PARQUET   = {PARQUET_PATH}")
print(f"[iter19] outputs   = {ITER_DIR}")

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
print(f"[iter19] device={DEVICE}  torch={torch.__version__}")
if DEVICE.type != "cuda":
    print("[iter19] WARNING: GPU not available. Training will be very slow. "
          "Run this notebook in Colab Pro with T4 for reasonable runtimes.")

# %%
# Global constants of the experiment.
HORIZON = 1
TCN_FEATURE_COLS = list(FEATURES_10)              # 10 features (no incluye target)
N_INPUT_CHANNELS = 1 + len(TCN_FEATURE_COLS)      # 1 (target hist) + 10 features = 11
print(f"[iter19] TCN channels = {N_INPUT_CHANNELS}  (target_hist + {TCN_FEATURE_COLS})")

# %%
# Load the parquet.
t0 = time.time()
df = pd.read_parquet(PARQUET_PATH)
print(f"[iter19] parquet loaded in {time.time() - t0:.1f}s. shape={df.shape}")
TOTAL_LEN = len(df)
print(f"[iter19] split: train_end={IDX_TRAIN_END}  val_end={IDX_VAL_END}  total={TOTAL_LEN}")

missing = [c for c in [TARGET_COL] + TCN_FEATURE_COLS if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in parquet: {missing}")

# %%
# Scalers (uno con log1p_target=True, otro con False) — se reutilizan en cada run
# according to the ablation configuration.
df_train = df.iloc[:IDX_TRAIN_END]


def build_scaler(log1p_target: bool) -> dict:
    return fit_scaler(
        df_train,
        feature_cols=TCN_FEATURE_COLS,
        target_col=TARGET_COL,
        log1p_target=log1p_target,
    )


SCALERS = {
    True: build_scaler(log1p_target=True),
    False: build_scaler(log1p_target=False),
}
print("[iter19] scalers fit OK (log1p True/False).")
print(f"[iter19]   scaler[True]: target_mean={SCALERS[True]['target_mean']:.4f} "
      f"target_std={SCALERS[True]['target_std']:.4f}")
print(f"[iter19]   scaler[False]: target_mean={SCALERS[False]['target_mean']:.4f} "
      f"target_std={SCALERS[False]['target_std']:.4f}")

# %%
# Pre-construye los arrays normalizados (N, 11) y target (N,) para cada scaler.
# Memoria: 1.1M * 11 * float32 ~ 50MB por scaler -> 100MB total. Aceptable.

def build_feature_array(scaler: dict) -> tuple:
    """Devuelve (feature_arr, target_arr) sobre TODO el dataframe.

    feature_arr (N, 11): canal 0 = target normalizado historico,
    canales 1..10 = features normalizadas en orden FEATURES_10.
    target_arr (N,) = mismo que feature_arr[:, 0]; se mantiene en variable
    separada por claridad cuando se indexa por t+horizon.
    """
    out = transform(df, scaler)        # features (N, 10), target (N,)
    full = np.concatenate(
        [out["target"].reshape(-1, 1), out["features"]], axis=1
    ).astype(np.float32)
    return full, out["target"].astype(np.float32)


ARRAYS = {
    True: build_feature_array(SCALERS[True]),
    False: build_feature_array(SCALERS[False]),
}
print(f"[iter19] arrays normalizados shape = {ARRAYS[True][0].shape}")

# %%
# Origins por L. Reusa aligned_indices de iter17 con seq_length=L para que las
# windows do not cross split boundaries. For L=72 it matches 1:1 with the convention
# de `xgb_lag6_feat10` (n_test = 165,222 a H=1).

def origins_for_L(L: int) -> dict:
    return {
        "train": aligned_indices(0,             IDX_TRAIN_END, HORIZON, TOTAL_LEN, seq_length=L),
        "val":   aligned_indices(IDX_TRAIN_END, IDX_VAL_END,    HORIZON, TOTAL_LEN, seq_length=L),
        "test":  aligned_indices(IDX_VAL_END,   TOTAL_LEN,      HORIZON, TOTAL_LEN, seq_length=L),
    }


for L_try in [72, 144]:
    o = origins_for_L(L_try)
    print(f"[iter19] L={L_try:3d}: n_train={len(o['train']):,} "
          f"n_val={len(o['val']):,} n_test={len(o['test']):,}")

# %%
# Dataset: cada item es (window (L, 11), y_norm escalar).

class WindowsDataset(Dataset):
    def __init__(
        self,
        feature_arr: np.ndarray,
        target_arr: np.ndarray,
        origins: np.ndarray,
        seq_length: int,
        horizon: int = 1,
    ) -> None:
        self.features = torch.from_numpy(feature_arr)   # (N, F) float32
        self.target = torch.from_numpy(target_arr)      # (N,) float32
        self.origins = torch.from_numpy(origins.astype(np.int64))
        self.L = int(seq_length)
        self.h = int(horizon)

    def __len__(self) -> int:
        return self.origins.shape[0]

    def __getitem__(self, i: int):
        t = int(self.origins[i].item())
        x = self.features[t - self.L + 1 : t + 1]      # (L, F)
        y = self.target[t + self.h]                     # ()
        return x, y


# %%
# Utilidades de evaluacion.

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


# %%
# Train one run.

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


def train_one_run(
    run_id: str,
    config: dict,
    eval_split: str = "val",   # "val" in ablation; "test" in final run
    n_workers: int = 2,
) -> dict:
    """Train the TCN with the given configuration.

    Args:
        run_id: identifier (used as the weights and log filename).
        config: dict with all hyperparameters (see BASELINE).
        eval_split: "val" -> early stopping and panel on val.
                    "test" -> early stopping on val + panel also on test.
        n_workers: DataLoader workers.

    Returns:
        Dict with metrics, panels, and prediction arrays.
    """
    seed = int(config.get("seed", SEED))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    L = int(config["L"])
    C = int(config["C"])
    log1p = bool(config["log1p_target"])

    scaler = SCALERS[log1p]
    feature_arr, target_arr = ARRAYS[log1p]
    origins = origins_for_L(L)

    train_ds = WindowsDataset(feature_arr, target_arr, origins["train"], L, HORIZON)
    val_ds = WindowsDataset(feature_arr, target_arr, origins["val"], L, HORIZON)
    test_ds = (
        WindowsDataset(feature_arr, target_arr, origins["test"], L, HORIZON)
        if eval_split == "test" else None
    )

    train_loader = _make_loader(train_ds, config["batch_size"], shuffle=True, n_workers=n_workers)
    val_loader = _make_loader(val_ds, config["batch_size"], shuffle=False, n_workers=n_workers)
    test_loader = (
        _make_loader(test_ds, config["batch_size"], shuffle=False, n_workers=n_workers)
        if test_ds is not None else None
    )

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
        f"L={L} C={C} log1p={log1p}",
        f"n_train={len(train_ds)} n_val={len(val_ds)}"
        + (f" n_test={len(test_ds)}" if test_ds is not None else ""),
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
        val_nse = compute_nse_mgd(y_pred_val, y_true_val, scaler)
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
                "scaler_log1p": log1p,
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

    # Load best weights for final evaluation.
    ckpt = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # Val panel (always).
    y_pred_val, y_true_val = predict_loader(model, val_loader, DEVICE)
    yp_val_mgd = inverse_transform_target(y_pred_val, scaler)
    yt_val_mgd = inverse_transform_target(y_true_val, scaler)
    val_origins = origins["val"]
    val_timestamps = df.iloc[val_origins + HORIZON]["timestamp"].reset_index(drop=True)
    val_panel = evaluate_full_panel(yt_val_mgd, yp_val_mgd, timestamps=val_timestamps)

    # Test panel (only in the final run).
    test_panel = None
    test_y_pred_mgd = None
    test_y_true_mgd = None
    test_timestamps = None
    if test_loader is not None:
        y_pred_test, y_true_test = predict_loader(model, test_loader, DEVICE)
        test_y_pred_mgd = inverse_transform_target(y_pred_test, scaler)
        test_y_true_mgd = inverse_transform_target(y_true_test, scaler)
        test_origins = origins["test"]
        test_timestamps = df.iloc[test_origins + HORIZON]["timestamp"].reset_index(drop=True)
        test_panel = evaluate_full_panel(
            test_y_true_mgd, test_y_pred_mgd, timestamps=test_timestamps
        )

    # Persist log.
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
        "val_panel": val_panel,
        "val_y_pred_mgd": yp_val_mgd,
        "val_y_true_mgd": yt_val_mgd,
        "val_timestamps": val_timestamps,
        "test_panel": test_panel,
        "test_y_pred_mgd": test_y_pred_mgd,
        "test_y_true_mgd": test_y_true_mgd,
        "test_timestamps": test_timestamps,
        "scaler_log1p": log1p,
    }


# %%
# Baseline configuration + 4 ablation variants (spec §"Mini-ablation on VAL").
BASELINE = dict(
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

ABLATIONS = {
    "A0": dict(BASELINE),
    "A1": {**BASELINE, "log1p_target": False},
    "A2": {**BASELINE, "L": 144},
    "A3": {**BASELINE, "C": 64},
}

print("[iter19] configured ablation:")
for k, v in ABLATIONS.items():
    print(f"  {k}: L={v['L']} C={v['C']} log1p={v['log1p_target']}")

# %%
# Val ablation loop.
ablation_runs: dict = {}
for run_id, cfg in ABLATIONS.items():
    print(f"\n[iter19] === Running ablation {run_id} ===")
    print(f"[iter19]   {cfg}")
    ablation_runs[run_id] = train_one_run(run_id, cfg, eval_split="val")

# %%
# Summary table of the ablation.
print("\n[iter19] === Ablation summary (val) ===")
print(f"{'run':<4} {'L':>4} {'C':>4} {'log1p':>6} {'NSE':>8} {'RMSE':>7} "
      f"{'pico%':>7} {'rec@50':>7} {'epoch':>6} {'sec':>6}")
for rid, r in ablation_runs.items():
    p = r["val_panel"]
    g = p["global"]
    rec50 = p["recall"]["at_50_mgd"].get("recall")
    rec50_str = "n/a" if rec50 is None or rec50 != rec50 else f"{rec50:.3f}"
    print(f"{rid:<4} {r['config']['L']:>4} {r['config']['C']:>4} "
          f"{str(r['config']['log1p_target']):>6} "
          f"{g['nse']:>8.4f} {g['rmse']:>7.3f} {g['peak_err_pct']:>+7.1f} "
          f"{rec50_str:>7} {r['best_epoch']:>6} {r['train_seconds']:>6.0f}")

# %%
# Winner selection (factor independence).

def pick_winner(ablation_runs: dict) -> dict:
    """Choose the winning config assuming that log1p, L, and C are independent.

    For each axis, compare A0 against the variant that changes that axis and keep
    the value with the higher NSE_val.
    """
    nse = {rid: ablation_runs[rid]["val_panel"]["global"]["nse"] for rid in ablation_runs}
    a0 = ablation_runs["A0"]["config"]

    log1p_win = bool(a0["log1p_target"]) if nse["A0"] >= nse["A1"] else bool(ablation_runs["A1"]["config"]["log1p_target"])
    L_win = int(a0["L"]) if nse["A0"] >= nse["A2"] else int(ablation_runs["A2"]["config"]["L"])
    C_win = int(a0["C"]) if nse["A0"] >= nse["A3"] else int(ablation_runs["A3"]["config"]["C"])

    winner_cfg = {**BASELINE, "log1p_target": log1p_win, "L": L_win, "C": C_win}

    matched = None
    for rid, r in ablation_runs.items():
        c = r["config"]
        if (bool(c["log1p_target"]) == log1p_win
                and int(c["L"]) == L_win
                and int(c["C"]) == C_win):
            matched = rid
            break

    return {
        "config": winner_cfg,
        "matched": matched,
        "log1p_win": log1p_win,
        "L_win": L_win,
        "C_win": C_win,
        "nse_per_run": nse,
    }


winner = pick_winner(ablation_runs)
print(f"\n[iter19] winner: log1p={winner['log1p_win']} "
      f"L={winner['L_win']} C={winner['C_win']} matched={winner['matched']}")

if winner["matched"] is None:
    print(f"[iter19] The winning combination does not match A0..A3. Training A4...")
    ablation_runs["A4"] = train_one_run("A4", winner["config"], eval_split="val")
    p_a4 = ablation_runs["A4"]["val_panel"]
    print(f"[A4] NSE_val = {p_a4['global']['nse']:.4f}")
    winner_run_id = "A4"
    winner_cfg = winner["config"]
else:
    winner_run_id = winner["matched"]
    winner_cfg = ablation_runs[winner_run_id]["config"]
    print(f"[iter19] winner ya entrenado como {winner_run_id}")

# %%
# Final run on TEST with the winning config. Train only on train,
# monitor val with early stopping, evaluate on test (simple alternative
# from the spec, more comparable to iter17).
print(f"\n[iter19] === Final run on test with winning config ({winner_run_id}) ===")
final_run = train_one_run("final", winner_cfg, eval_split="test")
final_panel = final_run["test_panel"]
TCN_NSE_FINAL = float(final_panel["global"]["nse"])
print(f"\n[iter19] FINAL TEST NSE = {TCN_NSE_FINAL:.4f}")

# %%
# Re-entrenar xgb_lag6_feat10 inline para tener predicciones sobre los mismos
# indices y poder comparar 1:1 (mismo n_test cuando L=72; recortado cuando L=144).
print("\n[iter19] === Retraining xgb_lag6_feat10 for direct comparison ===")
xgb_out = train_xgboost_h(
    df,
    horizon=HORIZON,
    lags=6,
    features=FEATURES_10,
    include_lags=True,
    include_features=True,
    xgb_params=DEFAULT_XGB_PARAMS,
    early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
    seq_length=72,           # official iter17 convention
    verbose=False,
)
xgb_y_true_72 = xgb_out["y_true_test"]
xgb_y_pred_72 = xgb_out["y_pred_test"]
xgb_ts_72 = xgb_out["timestamps_test"]

# Si la TCN ganadora usa L>72, recorta los primeros (L-72) puntos del XGB para
# que ambos modelos se evaluen sobre los mismos timestamps.
L_winner = int(winner_cfg["L"])
offset_xgb = L_winner - 72
if offset_xgb > 0:
    xgb_y_true = xgb_y_true_72[offset_xgb:]
    xgb_y_pred = xgb_y_pred_72[offset_xgb:]
    xgb_ts = xgb_ts_72.iloc[offset_xgb:].reset_index(drop=True)
    print(f"[iter19] L_winner={L_winner} -> recortados {offset_xgb} puntos del XGB para alineacion")
else:
    xgb_y_true = xgb_y_true_72
    xgb_y_pred = xgb_y_pred_72
    xgb_ts = xgb_ts_72

xgb_panel = evaluate_full_panel(xgb_y_true, xgb_y_pred, timestamps=xgb_ts)
print(f"[iter19] xgb_lag6_feat10 (aligned): NSE={xgb_panel['global']['nse']:.4f}")

# %%
# Verdict.
XGB_NSE_REF_ITER17 = 0.8630   # cifra oficial de iter17 (n=165222)
XGB_NSE_INLINE = float(xgb_panel["global"]["nse"])
DELTA_NSE = TCN_NSE_FINAL - XGB_NSE_REF_ITER17
THRESHOLD = 0.02
VERDICT = "TCN_WINS" if DELTA_NSE >= THRESHOLD else "XGB_WINS"

print("\n[iter19] === VEREDICTO ===")
print(f"NSE TCN (test, H=1)            = {TCN_NSE_FINAL:.4f}")
print(f"NSE XGB iter17 (referencia)    = {XGB_NSE_REF_ITER17:.4f}")
print(f"NSE XGB inline (aligned)       = {XGB_NSE_INLINE:.4f}")
print(f"delta NSE (TCN - XGB iter17)   = {DELTA_NSE:+.4f}")
print(f"umbral de cierre               = +{THRESHOLD:.3f}")
print(f"VEREDICTO                      = {VERDICT}")

# %%
# Guardar JSON de resultados.

def serialize_panel(panel: dict) -> dict:
    p = dict(panel)
    p.pop("peak_lag_per_event", None)   # lista larga, fuera del JSON resumen
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


ablation_block = {}
for rid, r in ablation_runs.items():
    p = r["val_panel"]
    g = p["global"]
    ablation_block[rid] = {
        "config": dict(r["config"]),
        "metrics_val": {
            "NSE": g["nse"],
            "RMSE": g["rmse"],
            "MAE": g["mae"],
            "peak_err_pct": g["peak_err_pct"],
            "recall_at_50": p["recall"]["at_50_mgd"].get("recall"),
        },
        "best_epoch": r["best_epoch"],
        "train_seconds": r["train_seconds"],
        "n_params": r["n_params"],
    }

results_json = {
    "meta": {
        "iter": 19,
        "branch": "iter19-tcn-comparison",
        "horizon": HORIZON,
        "device": str(DEVICE),
        "seed": SEED,
        "feature_cols_tcn": [TARGET_COL] + list(FEATURES_10),
        "split": {
            "train_end_idx": IDX_TRAIN_END,
            "val_end_idx": IDX_VAL_END,
            "total_len": TOTAL_LEN,
        },
        "baseline_xgb": {
            "name": "xgb_lag6_feat10",
            "nse_h1_iter17": XGB_NSE_REF_ITER17,
            "nse_h1_inline_aligned": XGB_NSE_INLINE,
        },
    },
    "ablation_val": ablation_block,
    "winner": {
        "id": winner_run_id,
        "config": winner_cfg,
        "selection_rule": "max NSE_val asumiendo independencia de factores (log1p, L, C)",
    },
    "final_test": {
        "config": winner_cfg,
        "metrics_test": serialize_panel(final_panel),
        "best_epoch": final_run["best_epoch"],
        "training_time_seconds": final_run["train_seconds"],
        "n_params": final_run["n_params"],
        "weights_path": final_run["weights_path"],
    },
    "comparison": {
        "xgb_lag6_feat10_NSE_H1_iter17": XGB_NSE_REF_ITER17,
        "xgb_lag6_feat10_NSE_H1_inline": XGB_NSE_INLINE,
        "tcn_NSE_H1": TCN_NSE_FINAL,
        "delta_NSE": DELTA_NSE,
        "criterion_threshold": THRESHOLD,
        "verdict": VERDICT,
        "verdict_explanation": (
            "TCN supera a XGB por mas de 0.02 NSE en H=1." if VERDICT == "TCN_WINS"
            else "TCN no supera a XGB por mas de 0.02 NSE en H=1; deep learning se cierra para el TFM."
        ),
    },
}

json_path = DIAG_DIR / "iter19_tcn_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(_sanitize(results_json), f, indent=2, ensure_ascii=False)
print(f"\n[iter19] JSON: {json_path}")

# %%
# Generar iter19_comparison.md.
md_path = DIAG_DIR / "iter19_comparison.md"


def fmt(v, spec=".4f"):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return "n/a"
    return f"{v:{spec}}"


lines: list = []
lines.append("# Iter19 - TCN limpia (Bai 2018) vs xgb_lag6_feat10\n")

# 1. Executive summary
lines.append("## Executive summary\n")
lines.append(
    f"Winning configuration: **{winner_run_id}** "
    f"(L={winner_cfg['L']}, C={winner_cfg['C']}, log1p_target={winner_cfg['log1p_target']}). "
    f"NSE TCN test H=1 = **{TCN_NSE_FINAL:.4f}**. "
    f"NSE XGB iter17 referencia = {XGB_NSE_REF_ITER17:.4f}. "
    f"Delta = **{DELTA_NSE:+.4f}** (umbral cierre = +{THRESHOLD:.3f}). "
    f"VEREDICTO: **{VERDICT}**.\n"
)

# 2. Val ablation table
lines.append("## Table of val ablation\n")
lines.append("| Run | L | C | log1p | NSE_val | RMSE_val | err_pico_val (%) | recall@50_val | best_epoch | train_s |")
lines.append("|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|")
for rid, r in ablation_runs.items():
    p = r["val_panel"]
    g = p["global"]
    rec50 = p["recall"]["at_50_mgd"].get("recall")
    lines.append(
        f"| {rid} | {r['config']['L']} | {r['config']['C']} | "
        f"{'yes' if r['config']['log1p_target'] else 'no'} | "
        f"{fmt(g['nse'])} | {fmt(g['rmse'], '.3f')} | "
        f"{fmt(g['peak_err_pct'], '+.1f')} | "
        f"{fmt(rec50, '.3f')} | "
        f"{r['best_epoch']} | {r['train_seconds']:.0f} |"
    )
lines.append("")

# 3. Justification of the winning configuration
lines.append("## Justification of the winning configuration\n")
nse_a0 = ablation_runs["A0"]["val_panel"]["global"]["nse"]
nse_a1 = ablation_runs["A1"]["val_panel"]["global"]["nse"]
nse_a2 = ablation_runs["A2"]["val_panel"]["global"]["nse"]
nse_a3 = ablation_runs["A3"]["val_panel"]["global"]["nse"]
lines.append(
    f"- **log1p_target = {winner_cfg['log1p_target']}** — A0 (log1p=True) NSE_val={nse_a0:.4f} vs "
    f"A1 (log1p=False) NSE_val={nse_a1:.4f}. Delta={nse_a0 - nse_a1:+.4f}. "
    f"Eje resuelto a favor de log1p_target={winner_cfg['log1p_target']}.\n"
    f"- **L = {winner_cfg['L']}** — A0 (L=72) NSE_val={nse_a0:.4f} vs "
    f"A2 (L=144) NSE_val={nse_a2:.4f}. Delta={nse_a0 - nse_a2:+.4f}. "
    f"Eje resuelto a favor de L={winner_cfg['L']}.\n"
    f"- **C = {winner_cfg['C']}** — A0 (C=32) NSE_val={nse_a0:.4f} vs "
    f"A3 (C=64) NSE_val={nse_a3:.4f}. Delta={nse_a0 - nse_a3:+.4f}. "
    f"Eje resuelto a favor de C={winner_cfg['C']}.\n"
)
if winner["matched"] is None:
    lines.append(
        f"La combinacion ganadora ({winner_cfg['log1p_target']}, L={winner_cfg['L']}, C={winner_cfg['C']}) "
        f"no coincide con A0..A3 ya entrenadas, por lo que se entreno A4 con esta config para validar "
        f"empiricamente la seleccion por independencia de factores.\n"
    )
else:
    lines.append(
        f"The winning combination matches run {winner['matched']} already trained in the ablation.\n"
    )

# 4. Resultados del modelo final en test
lines.append("## Resultados del modelo final en test\n")
fp = final_panel
lines.append("| Metrica | Valor |")
lines.append("|---|---:|")
lines.append(f"| n_test | {fp['global']['n']} |")
lines.append(f"| NSE | {fp['global']['nse']:.4f} |")
lines.append(f"| RMSE | {fp['global']['rmse']:.3f} |")
lines.append(f"| MAE | {fp['global']['mae']:.3f} |")
lines.append(f"| peak_real (MGD) | {fp['global']['peak_real_mgd']:.2f} |")
lines.append(f"| peak_pred (MGD) | {fp['global']['peak_pred_mgd']:.2f} |")
lines.append(f"| peak_err_pct | {fp['global']['peak_err_pct']:+.1f} |")
lines.append(f"| recall@25 | {fmt(fp['recall']['at_25_mgd'].get('recall'), '.3f')} |")
lines.append(f"| recall@50 | {fmt(fp['recall']['at_50_mgd'].get('recall'), '.3f')} |")
lines.append(f"| n_params | {final_run['n_params']:,} |")
lines.append(f"| best_epoch | {final_run['best_epoch']} |")
lines.append(f"| training_time_s | {final_run['train_seconds']:.0f} |")
lines.append("")

lines.append("### NSE por bucket (test)\n")
lines.append("| Bucket | n | NSE | RMSE | bias (MGD) | peak_err_pct |")
lines.append("|---|---:|---:|---:|---:|---:|")
for bname in [b[0] for b in DEFAULT_BUCKETS]:
    b = fp["buckets"][bname]
    lines.append(
        f"| {bname} | {b['n']} | {fmt(b['nse'], '+.3f')} | "
        f"{fmt(b['rmse'], '.3f')} | {fmt(b['bias'], '+.3f')} | "
        f"{fmt(b['peak_err_pct'], '+.1f')} |"
    )
lines.append("")

# 5. Comparison against xgb_lag6_feat10
lines.append("## Direct comparison against xgb_lag6_feat10\n")
lines.append("Ambos modelos evaluados sobre los mismos timestamps de test "
             f"(L_winner={L_winner}, offset XGB={offset_xgb} pasos).\n")
xgb_g = xgb_panel["global"]
xgb_b_base = xgb_panel["buckets"]["Base"]
xgb_b_ext = xgb_panel["buckets"]["Extremo"]
xgb_rec50 = xgb_panel["recall"]["at_50_mgd"].get("recall")
tcn_b_base = fp["buckets"]["Base"]
tcn_b_ext = fp["buckets"]["Extremo"]
tcn_rec50 = fp["recall"]["at_50_mgd"].get("recall")

lines.append("| Modelo | NSE | err_pico (%) | bias_base (MGD) | recall@50 | NSE_extremo |")
lines.append("|---|---:|---:|---:|---:|---:|")
lines.append(
    f"| xgb_lag6_feat10 (inline) | {xgb_g['nse']:.4f} | {xgb_g['peak_err_pct']:+.1f} | "
    f"{xgb_b_base['bias']:+.3f} | {fmt(xgb_rec50, '.3f')} | {fmt(xgb_b_ext['nse'], '+.3f')} |"
)
lines.append(
    f"| TCN ({winner_run_id}) | {fp['global']['nse']:.4f} | {fp['global']['peak_err_pct']:+.1f} | "
    f"{tcn_b_base['bias']:+.3f} | {fmt(tcn_rec50, '.3f')} | {fmt(tcn_b_ext['nse'], '+.3f')} |"
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
    f"**{_safe_diff(fp['global']['nse'], xgb_g['nse'], '+.4f')}** | "
    f"{_safe_diff(fp['global']['peak_err_pct'], xgb_g['peak_err_pct'], '+.1f')} | "
    f"{_safe_diff(tcn_b_base['bias'], xgb_b_base['bias'], '+.3f')} | "
    f"{_safe_diff(tcn_rec50, xgb_rec50, '+.3f')} | "
    f"{_safe_diff(tcn_b_ext['nse'], xgb_b_ext['nse'], '+.3f')} |"
)
lines.append("")
lines.append(f"NSE de referencia oficial iter17 (xgb_lag6_feat10, n=165222): **{XGB_NSE_REF_ITER17:.4f}**.")
lines.append(f"Delta usado para el veredicto: TCN({TCN_NSE_FINAL:.4f}) - XGB_iter17({XGB_NSE_REF_ITER17:.4f}) = **{DELTA_NSE:+.4f}**.")
lines.append(f"Threshold from DIAGNOSTIC_REPORT §7.6: +{THRESHOLD:.3f}.\n")

# 6. Verdict
lines.append("## Verdict\n")
if VERDICT == "TCN_WINS":
    lines.append(
        f"**TCN gana** por margen >= {THRESHOLD:.3f} NSE. La TCN limpia (Bai 2018) supera al "
        f"XGBoost regressor by {DELTA_NSE:+.4f} NSE at H=1. It becomes the main candidate "
        f"model for the TFM. Replication at H=3 remains pending in a later session before "
        f"sustituir definitivamente a `xgb_lag6_feat10`.\n"
    )
else:
    lines.append(
        f"**XGBoost gana** (TCN no supera el umbral +{THRESHOLD:.3f} NSE). La diferencia es "
        f"{DELTA_NSE:+.4f} NSE en H=1, dentro del margen de no significancia operativa fijado "
        f"por la spec. Deep learning se cierra para el TFM y `xgb_lag6_feat10` queda como "
        f"definitive main model. This quantitatively validates the hypothesis from "
        f"DIAGNOSTIC_REPORT §6: un GBM con 6 lags + 10 features captura toda la senal "
        f"predictible disponible a H=1 sobre MC-CL-005, y la flexibilidad temporal extra de "
        f"la TCN (receptive field 61 pasos) no aporta valor sobre la senal autoregresiva "
        f"corta del sistema (lag optimo 10 min).\n"
    )

# 7. Honest reading for the TFM
lines.append("## Honest reading for the TFM\n")
lines.append(
    "Esta es la unica iteracion de deep learning post-diagnostico, intencionalmente limitada "
    "(one seed, 4-run ablation, simple Huber loss) to avoid over-engineering that "
    f"enmascare el resultado real. El umbral de cierre +{THRESHOLD:.3f} NSE es arbitrario pero "
    "defendible: del orden del ruido entre configuraciones razonables, y muy por debajo del "
    "techo fisico H=1 ~0.86 derivado del analisis S4. "
)
if VERDICT == "TCN_WINS":
    lines.append(
        "El resultado favorece deep learning, pero la diferencia debe interpretarse a la luz "
        "de la complejidad anadida (GPU, dependencia PyTorch, mayor coste de inferencia) "
        "frente al beneficio operativo. Para uso real del MSD el coste/beneficio de mantener "
        "una TCN en produccion frente a un GBM debe documentarse explicitamente.\n"
    )
else:
    lines.append(
        "El resultado, lejos de ser un fracaso, es academicamente fuerte: muestra que sobre "
        "una unica estacion (MC-CL-005) con ~770k muestras de train, un GBM con memoria "
        "explicita absorbe la senal autoregresiva sin necesidad de receptive fields "
        "convolucionales mas largos. Es el cierre cuantitativo de la pregunta planteada en "
        "el DIAGNOSTIC_REPORT §7.6 y refuerza la narrativa del TFM como hallazgo metodologico "
        "(modelo aparente -> diagnostico -> modelo real mas simple).\n"
    )

# 8. Limitations
lines.append("## Recognized limitations\n")
lines.append(
    "- Una sola seed (=42). No se reportan barras de error sobre la NSE final.\n"
    "- Ablacion minima (4 corridas) con seleccion por independencia de factores. "
    "Posibles interacciones entre log1p y L, o entre C y dropout, no se exploran.\n"
    "- Loss Huber simple en espacio normalizado, sin componente de magnitud ni peak penalty. "
    "Variantes con loss asimetrica podrian mejorar el bucket Extremo, pero quedan fuera de "
    "scope por la spec.\n"
    f"- When L>72, val/test origins are trimmed by {offset_xgb} steps to avoid crossing "
    "split boundaries. This changes n_test slightly relative to iter17 (n=165222 with L=72).\n"
    "- No se ha replicado a H=3 en esta iteracion. Si TCN gana, queda pendiente esa replica "
    "before changing the main model in STATE.md and EXPERIMENTS.md.\n"
)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[iter19] MD:   {md_path}")

# %%
# Figuras.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fig 1 - val ablation NSE bars.
rids = list(ablation_runs.keys())
nses = [ablation_runs[r]["val_panel"]["global"]["nse"] for r in rids]
fig, ax = plt.subplots(figsize=(7.0, 4.0))
colors = ["#2ca02c" if r == winner_run_id else "#1f77b4" for r in rids]
bars = ax.bar(rids, nses, color=colors, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, nses):
    ax.text(
        b.get_x() + b.get_width() / 2.0,
        b.get_height() + (max(nses) - min(nses)) * 0.02 if max(nses) != min(nses) else b.get_height() + 0.001,
        f"{v:.4f}",
        ha="center", va="bottom", fontsize=9,
    )
ax.set_ylabel("NSE val (MGD)")
ax.set_title("iter19 - val ablation: NSE by configuration (green=winner)")
ax.grid(alpha=0.3, axis="y")
if max(nses) - min(nses) > 0:
    ax.set_ylim(min(nses) - (max(nses) - min(nses)) * 0.4, max(nses) + (max(nses) - min(nses)) * 0.2)
fig.tight_layout()
fig.savefig(FIG_DIR / "ablation_val_nse.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter19] figura: ablation_val_nse.png")

# Fig 2 — hidrograma evento extremo (TCN vs XGB).
yt_test = final_run["test_y_true_mgd"]
yp_test = final_run["test_y_pred_mgd"]
ts_test = pd.to_datetime(final_run["test_timestamps"])
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
        label=f"TCN ({winner_run_id})")
ax.plot(xgb_ts_dt.iloc[xi0:xi1], xgb_y_pred[xi0:xi1], color="#ff7f0e", linewidth=1.2,
        linestyle="--", alpha=0.9, label="XGB lag6+feat10")
ax.axhline(50.0, linestyle=":", color="#888", linewidth=0.8, label="Extremo Threshold 50 MGD")
ax.set_title(
    f"Test extreme-event hydrograph (H=1) - TCN {winner_run_id} vs XGB\n"
    f"Real peak = {yt_test[i_peak_tcn]:.1f} MGD at {peak_ts}"
)
ax.set_xlabel("Date")
ax.set_ylabel("stormflow (MGD)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "hydrograph_extreme_event_H1.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter19] figura: hydrograph_extreme_event_H1.png")

# Fig 3 — scatter real vs pred coloreado por bucket.
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
ax.set_title(f"TCN final ({winner_run_id}) sobre test (H=1)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "scatter_real_vs_pred_H1.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter19] figura: scatter_real_vs_pred_H1.png")

# Fig 4 - train loss and val NSE curves of the final run.
epochs_x = list(range(1, len(final_run["train_loss_history"]) + 1))
fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
l1 = ax1.plot(epochs_x, final_run["train_loss_history"], color="#1f77b4", label="train Huber loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("train Huber loss", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2 = ax1.twinx()
l2 = ax2.plot(epochs_x, final_run["val_nse_history"], color="#d62728", label="val NSE (MGD)")
ax2.set_ylabel("val NSE (MGD)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.axvline(final_run["best_epoch"], color="black", linestyle=":", linewidth=0.8)
ax2.text(final_run["best_epoch"], max(final_run["val_nse_history"]),
         f" best_epoch={final_run['best_epoch']}", fontsize=9, va="top")
fig.suptitle(f"iter19 final run ({winner_run_id}) - training curves")
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "loss_curves_final.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter19] figura: loss_curves_final.png")

# %%
# Final summary in console.
print("\n" + "=" * 80)
print("iter19 - resumen final")
print("=" * 80)
print(f"  Config ganadora: {winner_run_id} (L={winner_cfg['L']} C={winner_cfg['C']} "
      f"log1p={winner_cfg['log1p_target']})")
print(f"  NSE TCN test H=1   : {TCN_NSE_FINAL:.4f}")
print(f"  NSE XGB iter17 ref : {XGB_NSE_REF_ITER17:.4f}")
print(f"  delta NSE          : {DELTA_NSE:+.4f}  (umbral +{THRESHOLD:.3f})")
print(f"  VEREDICTO          : {VERDICT}")
print(f"  weights : {WEIGHTS_DIR}")
print(f"  logs    : {LOGS_DIR}")
print(f"  json    : {json_path}")
print(f"  md      : {md_path}")
print(f"  figs    : {FIG_DIR}")
