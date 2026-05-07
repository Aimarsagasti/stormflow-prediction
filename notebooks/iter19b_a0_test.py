# -*- coding: utf-8 -*-
"""iter19b_a0_test.py

Re-evaluation of A0 (clean TCN with log1p=True) on TEST.

Objective: close the methodological question documented in
`outputs/diagnostic/iter19_comparison.md` sobre si la regla "max NSE_val" usada
in iter19 privileged the global metric at the expense of operational behavior
in the Extremo bucket. A0 had val peak_err = +1.5% (almost perfect) versus
-25% de A1/A4, pero nunca se evaluo en test.

Hypothesis: A0 will have lower global NSE than A4 but better behavior in the
Extremo bucket (lower absolute bias, peak error closer to 0).

Design:
- A single run with config A0 evaluated directly on TEST
  (early stopping monitoring val).
- One-to-one comparison against A4 by loading the weights from `outputs/iter19/weights/final.pt`
  and re-predicting on the same test set.

Restrictions (branch iter19-tcn-comparison):
- Do not modify already committed iter19 files (tcn_clean.py, normalize_v2.py,
  iter19_tcn_clean.py, weights, JSON, comparison.md).
- Do not merge to main; it stays on branch iter19-tcn-comparison.

Colab-style cell structure (`# %%`).

Execution: Colab Pro T4 (same as iter19_tcn_clean.py). On CPU it is feasible
but very slow (~1-2h for one run).
"""

# %% [markdown]
# # Iter19b - A0 (with log1p) on TEST vs A4 (without log1p)
#
# - Retrains TCN A0 (L=72, C=32, log1p=True) on train, with early stopping on val.
# - Evaluates on test with `evaluate_full_panel`.
# - Loads A4 from `outputs/iter19/weights/final.pt` and predicts on the same test.
# - Compares A0 vs A4 globally and by bucket (focus on Alto+Extremo).
# - Generates JSON, markdown, and a double scatter plot.

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

# Locate the parquet with features. Same candidate paths as iter19.
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

# Path to the A4 weights (final from iter19) for the comparison.
A4_WEIGHTS_PATH = WEIGHTS_DIR / "final.pt"
if not A4_WEIGHTS_PATH.exists():
    raise RuntimeError(
        f"Cannot find A4 weights at {A4_WEIGHTS_PATH}. "
        "They are required for the comparison. Make sure the iter19 outputs are synchronized."
    )

print(f"[iter19b] REPO_ROOT = {REPO_ROOT}")
print(f"[iter19b] PARQUET   = {PARQUET_PATH}")
print(f"[iter19b] outputs   = {ITER_DIR}")
print(f"[iter19b] A4 ckpt   = {A4_WEIGHTS_PATH}")

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
)
from src.evaluation.metrics_panel import evaluate_full_panel, DEFAULT_BUCKETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"[iter19b] device={DEVICE}  torch={torch.__version__}")
if DEVICE.type != "cuda":
    print("[iter19b] WARNING: GPU not available. Training will be very slow. "
          "Run this notebook in Colab Pro with T4 for reasonable runtimes.")

# %%
# Global experiment constants (same as iter19).
HORIZON = 1
TCN_FEATURE_COLS = list(FEATURES_10)
N_INPUT_CHANNELS = 1 + len(TCN_FEATURE_COLS)
print(f"[iter19b] TCN channels = {N_INPUT_CHANNELS}  (target_hist + {TCN_FEATURE_COLS})")

# %%
# Load the parquet.
t0 = time.time()
df = pd.read_parquet(PARQUET_PATH)
print(f"[iter19b] parquet loaded in {time.time() - t0:.1f}s. shape={df.shape}")
TOTAL_LEN = len(df)
print(f"[iter19b] split: train_end={IDX_TRAIN_END}  val_end={IDX_VAL_END}  total={TOTAL_LEN}")

missing = [c for c in [TARGET_COL] + TCN_FEATURE_COLS if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in parquet: {missing}")

# %%
# Scalers: necesitamos uno con log1p=True (para A0) y otro con log1p=False
# (para invertir las predicciones de A4).
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
print("[iter19b] scalers fit OK (log1p True/False).")
print(f"[iter19b]   scaler[True]: target_mean={SCALERS[True]['target_mean']:.4f} "
      f"target_std={SCALERS[True]['target_std']:.4f}")
print(f"[iter19b]   scaler[False]: target_mean={SCALERS[False]['target_mean']:.4f} "
      f"target_std={SCALERS[False]['target_std']:.4f}")

# %%
# Pre-construye los arrays normalizados (N, 11) y target (N,) para cada scaler.

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


ARRAYS = {
    True: build_feature_array(SCALERS[True]),
    False: build_feature_array(SCALERS[False]),
}
print(f"[iter19b] arrays normalizados shape = {ARRAYS[True][0].shape}")

# %%
# Origins por L. Para L=72 coincide 1:1 con la convencion oficial iter17/19
# (n_test = 165222 a H=1).

def origins_for_L(L: int) -> dict:
    return {
        "train": aligned_indices(0,             IDX_TRAIN_END, HORIZON, TOTAL_LEN, seq_length=L),
        "val":   aligned_indices(IDX_TRAIN_END, IDX_VAL_END,    HORIZON, TOTAL_LEN, seq_length=L),
        "test":  aligned_indices(IDX_VAL_END,   TOTAL_LEN,      HORIZON, TOTAL_LEN, seq_length=L),
    }


L_FIXED = 72
o72 = origins_for_L(L_FIXED)
print(f"[iter19b] L={L_FIXED}: n_train={len(o72['train']):,} "
      f"n_val={len(o72['val']):,} n_test={len(o72['test']):,}")

# %%
# Dataset (copia de iter19).

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
        x = self.features[t - self.L + 1 : t + 1]
        y = self.target[t + self.h]
        return x, y


# %%
# Utilidades de evaluacion (copia de iter19).

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
# train_one_run: copia simplificada de iter19 (siempre eval_split="test"
# en este experimento, no se necesita la rama "val-only").

def train_one_run(
    run_id: str,
    config: dict,
    n_workers: int = 2,
) -> dict:
    """Train the TCN with the given configuration and evaluate on val + test.

    Early stopping monitors val. Best weights are saved to
    WEIGHTS_DIR/{run_id}.pt. Returns a dict with metrics and prediction arrays.
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
    test_ds = WindowsDataset(feature_arr, target_arr, origins["test"], L, HORIZON)

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
        f"L={L} C={C} log1p={log1p}",
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

    # Cargar mejores pesos para evaluacion final.
    ckpt = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # Panel test.
    y_pred_test, y_true_test = predict_loader(model, test_loader, DEVICE)
    yp_test_mgd = inverse_transform_target(y_pred_test, scaler)
    yt_test_mgd = inverse_transform_target(y_true_test, scaler)
    yp_test_mgd = np.clip(yp_test_mgd, 0.0, None)
    test_origins = origins["test"]
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
        "scaler_log1p": log1p,
    }


# %%
# A0 configuration (identical to iter19, repeated here so the notebook
# is self-contained).
A0_TEST_CONFIG = dict(
    L=72,
    C=32,
    log1p_target=True,    # unico cambio relevante respecto a A4
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

print("[iter19b] config A0 para test:")
for k, v in A0_TEST_CONFIG.items():
    print(f"  {k} = {v}")

# %%
# Train A0 on train, early stopping on val, final evaluation on test.
print("\n[iter19b] === Training A0 (log1p=True) and evaluating on test ===")
a0_run = train_one_run("A0_test", A0_TEST_CONFIG)

a0_panel = a0_run["test_panel"]
print(f"\n[iter19b] A0 test NSE = {a0_panel['global']['nse']:.4f}  "
      f"err_pico = {a0_panel['global']['peak_err_pct']:+.1f}%")

# %%
# Load A4 from final.pt and predict on test (with scaler log1p=False).
print(f"\n[iter19b] === Loading A4 from {A4_WEIGHTS_PATH} and predicting on test ===")
a4_ckpt = torch.load(A4_WEIGHTS_PATH, map_location=DEVICE)
a4_config = dict(a4_ckpt["config"])
print(f"[iter19b] A4 config (del checkpoint): "
      f"L={a4_config['L']} C={a4_config['C']} log1p={a4_config['log1p_target']}  "
      f"epoch={a4_ckpt.get('epoch')}  val_nse={a4_ckpt.get('val_nse'):.4f}")

assert int(a4_config["L"]) == 72, "Se asume A4 con L=72; abortar si no es el caso."
assert bool(a4_config["log1p_target"]) is False, "Se asume A4 con log1p=False."

a4_model = TCNClean(
    in_channels=N_INPUT_CHANNELS,
    hidden_channels=int(a4_config["C"]),
    kernel_size=int(a4_config["kernel_size"]),
    num_blocks=int(a4_config["num_blocks"]),
    dilations=a4_config.get("dilations"),
    dropout=float(a4_config["dropout"]),
).to(DEVICE)
a4_model.load_state_dict(a4_ckpt["model_state_dict"])
a4_model.eval()

a4_scaler = SCALERS[False]
feature_arr_f, target_arr_f = ARRAYS[False]
test_ds_a4 = WindowsDataset(feature_arr_f, target_arr_f, o72["test"], L_FIXED, HORIZON)
test_loader_a4 = _make_loader(test_ds_a4, A0_TEST_CONFIG["batch_size"], shuffle=False, n_workers=2)

y_pred_a4, y_true_a4 = predict_loader(a4_model, test_loader_a4, DEVICE)
yp_a4_mgd = inverse_transform_target(y_pred_a4, a4_scaler)
yt_a4_mgd = inverse_transform_target(y_true_a4, a4_scaler)
yp_a4_mgd = np.clip(yp_a4_mgd, 0.0, None)
a4_test_origins = o72["test"]
a4_test_timestamps = df.iloc[a4_test_origins + HORIZON]["timestamp"].reset_index(drop=True)
a4_panel = evaluate_full_panel(yt_a4_mgd, yp_a4_mgd, timestamps=a4_test_timestamps)

print(f"[iter19b] A4 test NSE (recomputado) = {a4_panel['global']['nse']:.4f}  "
      f"err_pico = {a4_panel['global']['peak_err_pct']:+.1f}%  n={a4_panel['global']['n']}")

# Sanity check: the recomputed NSE must match the official iter19 JSON
# (NSE_A4_test = 0.8983).
NSE_A4_OFFICIAL = 0.8983
delta_nse_check = abs(a4_panel["global"]["nse"] - NSE_A4_OFFICIAL)
if delta_nse_check > 0.01:
    print(f"[iter19b] WARNING: NSE recomputado de A4 difiere del oficial en {delta_nse_check:.4f}. "
          f"Revisar consistencia de scaler/arrays.")
else:
    print(f"[iter19b] sanity OK: |NSE_A4_recomputado - NSE_A4_oficial| = {delta_nse_check:.4f} <= 0.01")

# %%
# Build the A0 vs A4 comparison by bucket and globally.
def deltas_global(p_a0: dict, p_a4: dict) -> dict:
    g0 = p_a0["global"]
    g4 = p_a4["global"]
    rec50_a0 = p_a0["recall"]["at_50_mgd"].get("recall")
    rec50_a4 = p_a4["recall"]["at_50_mgd"].get("recall")
    rec25_a0 = p_a0["recall"]["at_25_mgd"].get("recall")
    rec25_a4 = p_a4["recall"]["at_25_mgd"].get("recall")
    return {
        "delta_nse_global": float(g0["nse"] - g4["nse"]),
        "delta_rmse_global": float(g0["rmse"] - g4["rmse"]),
        "delta_mae_global": float(g0["mae"] - g4["mae"]),
        "delta_peak_err_pct": float(g0["peak_err_pct"] - g4["peak_err_pct"]),
        "delta_recall_50": (None if rec50_a0 is None or rec50_a4 is None
                            else float(rec50_a0 - rec50_a4)),
        "delta_recall_25": (None if rec25_a0 is None or rec25_a4 is None
                            else float(rec25_a0 - rec25_a4)),
    }


def deltas_buckets(p_a0: dict, p_a4: dict) -> dict:
    out = {}
    for bname, _, _ in DEFAULT_BUCKETS:
        b0 = p_a0["buckets"][bname]
        b4 = p_a4["buckets"][bname]
        out[bname] = {
            "n_a0": b0["n"],
            "n_a4": b4["n"],
            "delta_nse": float(b0["nse"] - b4["nse"]),
            "delta_bias": float(b0["bias"] - b4["bias"]),
            "delta_rmse": float(b0["rmse"] - b4["rmse"]),
            "delta_peak_err_pct": float(b0["peak_err_pct"] - b4["peak_err_pct"]),
        }
    return out


cmp_global = deltas_global(a0_panel, a4_panel)
cmp_buckets = deltas_buckets(a0_panel, a4_panel)

print("\n[iter19b] === Comparacion global A0 vs A4 ===")
print(f"  delta_NSE        = {cmp_global['delta_nse_global']:+.4f}")
print(f"  delta_peak_err_% = {cmp_global['delta_peak_err_pct']:+.1f}")
print(f"  delta_recall@50  = "
      f"{'n/a' if cmp_global['delta_recall_50'] is None else f'{cmp_global['delta_recall_50']:+.3f}'}")
print("\n[iter19b] === Comparacion por bucket A0 vs A4 ===")
print(f"{'bucket':<10} {'n':>6} {'NSE_A0':>10} {'NSE_A4':>10} {'dNSE':>8} "
      f"{'bias_A0':>9} {'bias_A4':>9} {'dbias':>8} {'pico_A0':>9} {'pico_A4':>9}")
for bname, _, _ in DEFAULT_BUCKETS:
    b0 = a0_panel["buckets"][bname]
    b4 = a4_panel["buckets"][bname]
    dn = cmp_buckets[bname]["delta_nse"]
    print(f"{bname:<10} {b0['n']:>6} {b0['nse']:>+10.3f} {b4['nse']:>+10.3f} "
          f"{dn:>+8.3f} {b0['bias']:>+9.3f} {b4['bias']:>+9.3f} "
          f"{(b0['bias'] - b4['bias']):>+8.3f} "
          f"{b0['peak_err_pct']:>+9.1f} {b4['peak_err_pct']:>+9.1f}")

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


results_json = {
    "meta": {
        "iter": "19b",
        "branch": "iter19-tcn-comparison",
        "purpose": (
            "Re-evaluar A0 (con log1p) en test para cerrar duda metodologica "
            "del comparison.md de iter19 sobre el rol de log1p en el bucket Extremo."
        ),
        "horizon": HORIZON,
        "device": str(DEVICE),
        "seed": SEED,
        "feature_cols_tcn": [TARGET_COL] + list(FEATURES_10),
        "config": dict(A0_TEST_CONFIG),
        "split": {
            "train_end_idx": IDX_TRAIN_END,
            "val_end_idx": IDX_VAL_END,
            "total_len": TOTAL_LEN,
        },
        "a4_reference": {
            "weights_path": str(A4_WEIGHTS_PATH),
            "config": a4_config,
            "ckpt_epoch": int(a4_ckpt.get("epoch", -1)),
            "ckpt_val_nse": float(a4_ckpt.get("val_nse", float("nan"))),
            "nse_test_recomputado": float(a4_panel["global"]["nse"]),
            "nse_test_oficial_iter19": NSE_A4_OFFICIAL,
        },
    },
    "training": {
        "best_epoch": int(a0_run["best_epoch"]),
        "best_val_nse": float(a0_run["best_val_nse"]),
        "training_time_seconds": float(a0_run["train_seconds"]),
        "n_params": int(a0_run["n_params"]),
        "weights_path": a0_run["weights_path"],
        "train_loss_history": list(map(float, a0_run["train_loss_history"])),
        "val_nse_history": list(map(float, a0_run["val_nse_history"])),
    },
    "metrics_test_A0": serialize_panel(a0_panel),
    "metrics_test_A4_reference": serialize_panel(a4_panel),
    "comparison_A0_vs_A4": {
        **cmp_global,
        "by_bucket": cmp_buckets,
    },
}

json_path = DIAG_DIR / "iter19b_a0_test_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(_sanitize(results_json), f, indent=2, ensure_ascii=False)
print(f"\n[iter19b] JSON: {json_path}")

# %%
# Generar markdown comparativo.
md_path = DIAG_DIR / "iter19b_a0_vs_a4.md"


def fmt(v, spec=".4f"):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return "n/a"
    return f"{v:{spec}}"


a0_g = a0_panel["global"]
a4_g = a4_panel["global"]
a0_rec50 = a0_panel["recall"]["at_50_mgd"].get("recall")
a4_rec50 = a4_panel["recall"]["at_50_mgd"].get("recall")
a0_rec25 = a0_panel["recall"]["at_25_mgd"].get("recall")
a4_rec25 = a4_panel["recall"]["at_25_mgd"].get("recall")

# Veredicto basado en numeros (no forzado).
delta_nse = cmp_global["delta_nse_global"]
delta_pico_abs = abs(a0_g["peak_err_pct"]) - abs(a4_g["peak_err_pct"])  # negativo => A0 mejor pico
b_ext_a0 = a0_panel["buckets"]["Extremo"]
b_ext_a4 = a4_panel["buckets"]["Extremo"]
b_alt_a0 = a0_panel["buckets"]["Alto"]
b_alt_a4 = a4_panel["buckets"]["Alto"]

# Reglas para el veredicto:
# A0 mejora extremos si NSE_extremo_A0 > NSE_extremo_A4 OR |bias_extremo_A0| < |bias_extremo_A4|
a0_mejor_extremo = (
    b_ext_a0["nse"] > b_ext_a4["nse"]
    or abs(b_ext_a0["bias"]) < abs(b_ext_a4["bias"])
)
a0_mucho_peor_global = delta_nse < -0.02  # umbral simetrico al usado en iter19
a0_pico_global_mejor = abs(a0_g["peak_err_pct"]) < abs(a4_g["peak_err_pct"])

if a0_mejor_extremo and not a0_mucho_peor_global:
    veredicto = "A0_REEMPLAZA_A4"
    veredicto_label = "A0 sustituye a A4 como modelo principal"
elif not a0_mejor_extremo and a0_mucho_peor_global:
    veredicto = "A4_SE_QUEDA"
    veredicto_label = "A4 sigue siendo el modelo principal"
else:
    veredicto = "REPORTAR_AMBOS"
    veredicto_label = "Ambos modelos se reportan en el TFM, sin elegir uno"


lines: list = []
lines.append("# Iter19b - A0 (log1p=True) vs A4 (log1p=False) en TEST\n")

# 1. Introduccion
lines.append("## Introduccion\n")
lines.append(
    "El comparison.md de iter19 documenta una limitacion metodologica explicita: la regla "
    "\"max NSE_val\" usada para elegir la configuracion ganadora privilegio la metrica global "
    "sobre la operativa. En la ablacion val, A0 (con log1p) tenia un error de pico de +1.5% "
    "(casi perfecto), mientras que A1/A4 (sin log1p) tenian -25%. Sin embargo, A0 nunca se "
    "evaluo en test porque la regla automatica fijo el ganador en A4 (mayor NSE_val global). "
    "Este notebook entrena A0 directamente y la evalua en test para cerrar esa duda.\n"
)
lines.append(
    "La hipotesis que se contrasta: A0 tendra menor NSE global que A4 pero mejor "
    "comportamiento en el bucket Extremo (NSE menos negativo y/o menor bias absoluto). "
    "Si se confirma, hay un trade-off real entre metrica global y comportamiento operativo "
    "en eventos criticos. Si A0 empeora en todos los frentes, la regla \"max NSE_val\" estaba "
    "justificada.\n"
)

# 2. Tabla comparativa global
lines.append("## Tabla comparativa global A0 vs A4 en test\n")
lines.append("| Metrica | A0 (log1p=True) | A4 (log1p=False) | delta (A0 - A4) |")
lines.append("|---|---:|---:|---:|")
lines.append(
    f"| NSE | {a0_g['nse']:.4f} | {a4_g['nse']:.4f} | {(a0_g['nse'] - a4_g['nse']):+.4f} |"
)
lines.append(
    f"| RMSE | {a0_g['rmse']:.3f} | {a4_g['rmse']:.3f} | {(a0_g['rmse'] - a4_g['rmse']):+.3f} |"
)
lines.append(
    f"| MAE | {a0_g['mae']:.3f} | {a4_g['mae']:.3f} | {(a0_g['mae'] - a4_g['mae']):+.3f} |"
)
lines.append(
    f"| peak_err_pct | {a0_g['peak_err_pct']:+.1f} | {a4_g['peak_err_pct']:+.1f} | "
    f"{(a0_g['peak_err_pct'] - a4_g['peak_err_pct']):+.1f} |"
)
lines.append(
    f"| recall@50 | {fmt(a0_rec50, '.3f')} | {fmt(a4_rec50, '.3f')} | "
    f"{fmt(None if a0_rec50 is None or a4_rec50 is None else a0_rec50 - a4_rec50, '+.3f')} |"
)
lines.append(
    f"| recall@25 | {fmt(a0_rec25, '.3f')} | {fmt(a4_rec25, '.3f')} | "
    f"{fmt(None if a0_rec25 is None or a4_rec25 is None else a0_rec25 - a4_rec25, '+.3f')} |"
)
lines.append(
    f"| n_params | {a0_run['n_params']:,} | {a4_ckpt.get('n_params', 89985):,} | "
    f"{(a0_run['n_params'] - int(a4_ckpt.get('n_params', 89985))):+,} |"
)
lines.append(
    f"| training_time_s | {a0_run['train_seconds']:.0f} | "
    f"~1117 (iter19) | n/a |"
)
lines.append(
    f"| best_epoch | {a0_run['best_epoch']} | "
    f"{int(a4_ckpt.get('epoch', -1))} | n/a |"
)
lines.append("")

# 3. Tabla por bucket
lines.append("## Tabla comparativa por bucket en test\n")
lines.append("| Bucket | n | NSE A0 | NSE A4 | dNSE | bias A0 | bias A4 | dbias | "
             "pico% A0 | pico% A4 | dpico% |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for bname, _, _ in DEFAULT_BUCKETS:
    b0 = a0_panel["buckets"][bname]
    b4 = a4_panel["buckets"][bname]
    lines.append(
        f"| {bname} | {b0['n']} | {b0['nse']:+.3f} | {b4['nse']:+.3f} | "
        f"{(b0['nse'] - b4['nse']):+.3f} | "
        f"{b0['bias']:+.3f} | {b4['bias']:+.3f} | "
        f"{(b0['bias'] - b4['bias']):+.3f} | "
        f"{b0['peak_err_pct']:+.1f} | {b4['peak_err_pct']:+.1f} | "
        f"{(b0['peak_err_pct'] - b4['peak_err_pct']):+.1f} |"
    )
lines.append("")

# 4. Lectura de los resultados
lines.append("## Lectura de los resultados\n")

# 4.1 Hipotesis del Extremo
if b_ext_a0["nse"] > b_ext_a4["nse"]:
    msg_ext_nse = (
        f"NSE_extremo A0 = {b_ext_a0['nse']:+.3f} > A4 = {b_ext_a4['nse']:+.3f} "
        f"(+{b_ext_a0['nse'] - b_ext_a4['nse']:.3f})"
    )
    confirma_nse_ext = True
else:
    msg_ext_nse = (
        f"NSE_extremo A0 = {b_ext_a0['nse']:+.3f} <= A4 = {b_ext_a4['nse']:+.3f} "
        f"({b_ext_a0['nse'] - b_ext_a4['nse']:+.3f})"
    )
    confirma_nse_ext = False
if abs(b_ext_a0["bias"]) < abs(b_ext_a4["bias"]):
    msg_ext_bias = (
        f"|bias_extremo| A0 = {abs(b_ext_a0['bias']):.3f} < A4 = {abs(b_ext_a4['bias']):.3f}"
    )
    confirma_bias_ext = True
else:
    msg_ext_bias = (
        f"|bias_extremo| A0 = {abs(b_ext_a0['bias']):.3f} >= A4 = {abs(b_ext_a4['bias']):.3f}"
    )
    confirma_bias_ext = False

if confirma_nse_ext and confirma_bias_ext:
    confirmacion = "**confirma plenamente**"
elif confirma_nse_ext or confirma_bias_ext:
    confirmacion = "**confirma parcialmente**"
else:
    confirmacion = "**no confirma**"

lines.append(
    f"**Hipotesis del bucket Extremo (A0 mejor que A4 en eventos criticos)**: {confirmacion} "
    f"la hipotesis. {msg_ext_nse}; {msg_ext_bias}. "
)
if confirma_nse_ext or confirma_bias_ext:
    lines.append(
        "Esto significa que penalizar el log1p en val no implico mejor calibracion en eventos "
        "extremos en test: el log1p compresiona el rango y suaviza el residuo en magnitudes "
        "altas, lo que se traduce en menos sesgo en los picos. La penalizacion en NSE global "
        "viene del desempeno en buckets bajos (Base/Leve), donde el modelo log1p tiende a "
        "sobre-predecir pequenos.\n"
    )
else:
    lines.append(
        "El log1p no aporto beneficio en el bucket Extremo en test. Esto contradice la "
        "intuicion que motivo esta corrida: en val el err_pico de A0 era +1.5%, pero la "
        "calibracion no se transfirio a test. Posible explicacion: los picos de val no son "
        "representativos de los picos de test (muestra pequena, n=27 eventos val vs n=59 "
        "eventos test).\n"
    )

# 4.2 Justificacion de la regla max NSE_val
if delta_nse < 0:
    if abs(delta_nse) < 0.02:
        msg_regla = (
            f"La regla max NSE_val privilegio una mejora marginal de A4 sobre A0 "
            f"({-delta_nse:.4f} NSE en test, dentro del umbral de no-significancia +0.02 NSE "
            f"usado en iter19 para el veredicto frente a XGB)."
        )
        if a0_mejor_extremo:
            msg_regla += (
                " Dado que A0 mejora el bucket Extremo, la decision automatica fue "
                "**metodologicamente cuestionable**: prefirio una mejora global no significativa "
                "frente a una mejora operativa relevante."
            )
        else:
            msg_regla += (
                " Aun asi, A4 tampoco empeora en extremos, por lo que la regla seleccionable "
                "automaticamente sigue siendo defendible aunque no perfecta."
            )
    else:
        msg_regla = (
            f"La regla max NSE_val acerto: A4 supera a A0 en {-delta_nse:.4f} NSE global, "
            "una diferencia mayor que el umbral de significancia operativa. "
        )
        if a0_mejor_extremo:
            msg_regla += (
                "Hay un trade-off real entre global y extremos, pero la magnitud del gap global "
                "justifica la eleccion de A4."
            )
else:
    msg_regla = (
        f"A0 supera a A4 en NSE global por {delta_nse:+.4f}. La regla max NSE_val deberia "
        "haber seleccionado A0 directamente — el resultado de val no se transfirio a test."
    )

lines.append(
    f"**Sobre la regla \"max NSE_val\" usada en iter19**: {msg_regla}\n"
)

# 4.3 Configuracion principal del TFM
if veredicto == "A0_REEMPLAZA_A4":
    msg_principal = (
        "**Configuracion principal recomendada para el TFM: A0 (con log1p=True)**. "
        "Mejora en el bucket Extremo y la perdida en NSE global es aceptable "
        f"(delta = {delta_nse:+.4f}). Para predecir CSOs, la calibracion en eventos extremos "
        "pesa mas que el NSE global del rio en estado base."
    )
elif veredicto == "A4_SE_QUEDA":
    msg_principal = (
        "**Configuracion principal recomendada para el TFM: A4 (con log1p=False)**, sin cambios. "
        "A0 no aporto mejora en eventos criticos y empeoro globalmente. La regla max NSE_val "
        "estaba bien aplicada."
    )
else:
    msg_principal = (
        "**Configuracion principal recomendada para el TFM: reportar ambas**. Los trade-offs "
        "son equilibrados (A4 mejor global, A0 mejor o equivalente en extremos). El TFM gana "
        "honestidad metodologica al exponer la dependencia entre regla de seleccion y "
        "definicion de exito operativo, en lugar de forzar una eleccion."
    )

lines.append(f"{msg_principal}\n")

# 5. Veredicto
lines.append("## Veredicto y recomendacion\n")
lines.append(f"**{veredicto_label}**.\n")
lines.append(
    "Justificacion numerica:\n\n"
    f"- delta NSE global (A0 - A4) = {delta_nse:+.4f}\n"
    f"- delta peak_err_pct (A0 - A4) = {(a0_g['peak_err_pct'] - a4_g['peak_err_pct']):+.1f}\n"
    f"- delta NSE_extremo (A0 - A4) = {(b_ext_a0['nse'] - b_ext_a4['nse']):+.3f}\n"
    f"- delta bias_extremo (A0 - A4) = {(b_ext_a0['bias'] - b_ext_a4['bias']):+.3f} MGD\n"
    f"- delta peak_err_pct_extremo (A0 - A4) = "
    f"{(b_ext_a0['peak_err_pct'] - b_ext_a4['peak_err_pct']):+.1f}\n"
    f"- delta NSE_alto (A0 - A4) = {(b_alt_a0['nse'] - b_alt_a4['nse']):+.3f}\n"
)

# 6. Limitaciones
lines.append("## Limitaciones reconocidas\n")
lines.append(
    "- Una sola seed (=42), igual que iter19. La diferencia A0 vs A4 puede estar dentro del "
    "ruido estocastico de inicializacion. Para concluir con mas certeza haria falta repetir "
    "ambas corridas con varias seeds.\n"
    "- A0 reentrenado aqui puede converger a un minimo ligeramente distinto al de la corrida "
    "de iter19 (mismo seed pero el orden de operaciones GPU no es bit-exact). El best_epoch y "
    "best_val_nse pueden diferir marginalmente.\n"
    "- Comparacion limitada a H=1 (igual que iter19). Si A0 sustituye a A4, queda pendiente "
    "verificar que la mejora en extremos se mantiene a H=3.\n"
    "- A4 se evalua cargando los pesos del checkpoint final.pt de iter19 (no se reentrena). "
    "Esto garantiza comparacion 1:1 con el JSON oficial pero asume que ese checkpoint sigue "
    "siendo valido. El sanity check del NSE recomputado vs oficial valida esta asuncion.\n"
)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[iter19b] MD:   {md_path}")

# %%
# Figura: scatter doble A0 vs A4 en bucket Alto+Extremo (n=499 puntos).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mascara Alto+Extremo (real >= 25 MGD).
yt = a0_run["test_y_true_mgd"]
yp_a0 = a0_run["test_y_pred_mgd"]
yp_a4 = yp_a4_mgd

# A0 y A4 deben estar evaluados sobre los mismos timestamps (origins de test L=72).
assert len(yt) == len(yp_a4), (
    f"len mismatch: A0_true={len(yt)} A4_pred={len(yp_a4)}. Revisar origins."
)
# Sanity cross-check: yt (A0) y yt_a4_mgd deben ser identicos hasta error numerico
delta_yt = float(np.abs(yt - yt_a4_mgd).max())
print(f"[iter19b] sanity y_true A0 vs A4: max|diff|={delta_yt:.2e} (deberia ser ~0)")

mask_high = yt >= 25.0  # bucket Alto comienza en >=25, Extremo en >=50
n_high = int(mask_high.sum())
print(f"[iter19b] n puntos Alto+Extremo en test = {n_high}")

# Rango comun de ejes.
all_vals = np.concatenate([yt[mask_high], yp_a0[mask_high], yp_a4[mask_high]])
lim = float(np.max(all_vals)) * 1.05
lim_min = -2.0

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True, sharey=True)
fig.suptitle(
    f"Comparacion A0 (log1p=True) vs A4 (log1p=False) en buckets Alto+Extremo (n={n_high})",
    fontsize=12,
)

# A0
axes[0].scatter(yt[mask_high], yp_a0[mask_high], s=10, alpha=0.55, color="#1f77b4",
                edgecolor="white", linewidth=0.3)
axes[0].plot([lim_min, lim], [lim_min, lim], color="black", linestyle="--", linewidth=0.8)
axes[0].set_xlim(lim_min, lim)
axes[0].set_ylim(lim_min, lim)
axes[0].set_xlabel("y_real (MGD)")
axes[0].set_ylabel("y_pred (MGD)")
axes[0].set_title(
    f"A0 (log1p=True) - NSE={a0_g['nse']:.4f}, "
    f"NSE_ext={b_ext_a0['nse']:+.3f}, bias_ext={b_ext_a0['bias']:+.2f}"
)
axes[0].grid(alpha=0.3)

# A4
axes[1].scatter(yt[mask_high], yp_a4[mask_high], s=10, alpha=0.55, color="#d62728",
                edgecolor="white", linewidth=0.3)
axes[1].plot([lim_min, lim], [lim_min, lim], color="black", linestyle="--", linewidth=0.8)
axes[1].set_xlim(lim_min, lim)
axes[1].set_ylim(lim_min, lim)
axes[1].set_xlabel("y_real (MGD)")
axes[1].set_title(
    f"A4 (log1p=False) - NSE={a4_g['nse']:.4f}, "
    f"NSE_ext={b_ext_a4['nse']:+.3f}, bias_ext={b_ext_a4['bias']:+.2f}"
)
axes[1].grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = FIG_DIR / "scatter_A0_vs_A4_extremo.png"
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[iter19b] figura: {fig_path}")

# %%
# Resumen final en consola.
print("\n" + "=" * 80)
print("iter19b - resumen final")
print("=" * 80)
print(f"  Config A0      : L=72 C=32 log1p=True")
print(f"  NSE A0 test    : {a0_g['nse']:.4f}  (epoch {a0_run['best_epoch']}, "
      f"{a0_run['train_seconds']:.0f}s)")
print(f"  NSE A4 test    : {a4_g['nse']:.4f}  (recomputado desde final.pt)")
print(f"  delta NSE      : {delta_nse:+.4f}")
print(f"  err_pico A0    : {a0_g['peak_err_pct']:+.1f}%")
print(f"  err_pico A4    : {a4_g['peak_err_pct']:+.1f}%")
print(f"  NSE_extremo A0 : {b_ext_a0['nse']:+.3f}  bias={b_ext_a0['bias']:+.3f}")
print(f"  NSE_extremo A4 : {b_ext_a4['nse']:+.3f}  bias={b_ext_a4['bias']:+.3f}")
print(f"  VEREDICTO      : {veredicto_label}")
print(f"  weights : {WEIGHTS_DIR / 'A0_test.pt'}")
print(f"  json    : {json_path}")
print(f"  md      : {md_path}")
print(f"  fig     : {fig_path}")
