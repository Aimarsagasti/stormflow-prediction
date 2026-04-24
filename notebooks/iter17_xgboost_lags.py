# -*- coding: utf-8 -*-
"""iter17_xgboost_lags.py

Notebook de iter17 (rama `iter17-xgboost-lags`).

Objetivo: construir el modelo principal del sistema nuevo segun
`outputs/diagnostic/DIAGNOSTIC_REPORT.md` §6/§7: XGBoost regresivo con
12 lags del target y 10 features exogenas reducidas (S5), para H=1 y H=3.

Ejecucion local sin GPU. No depende del pipeline de normalizacion del
TCN (trabaja en MGD reales).

Estructura de celdas estilo Colab (`# %%`).
"""

# %% [markdown]
# # Iter17 - XGBoost + lags del target (fase 1: scaffolding)
#
# Este commit deja el andamiaje:
# - Verificacion de cache parquet (regenera si falta).
# - Carga + split alineado con S2.
# - Entrenamiento del modelo principal (12 lags + 10 features) a H=1 y H=3.
# - Metricas inline (NSE / RMSE / MAE / error pico) como sanity check.
#
# Las iteraciones posteriores (commits 2 y 3) anaden el panel multi-bucket
# y las ablations / figuras / artefactos.

# %%
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("C:/Dev/TFM")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARQUET_PATH = ROOT / "outputs" / "cache" / "df_with_features.parquet"
GENERATE_STATS_SCRIPT = ROOT / "scripts" / "generate_dataset_stats.py"
OUT_DIR = ROOT / "outputs" / "diagnostic"
FIG_DIR = ROOT / "outputs" / "figures" / "iter17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Fix de reproducibilidad: si el parquet cacheado no existe en una maquina
# nueva, regenerarlo con el script oficial antes de continuar.
if not PARQUET_PATH.exists():
    print(f"[iter17] Cache no encontrado en {PARQUET_PATH}")
    print(f"[iter17] Regenerando via {GENERATE_STATS_SCRIPT}...")
    import subprocess
    r = subprocess.run(
        [sys.executable, str(GENERATE_STATS_SCRIPT)],
        cwd=str(ROOT),
        check=True,
    )
    print(f"[iter17] generate_dataset_stats.py return_code={r.returncode}")
    if not PARQUET_PATH.exists():
        raise RuntimeError(
            f"El cache sigue sin existir tras regenerarlo: {PARQUET_PATH}. "
            "Revisa scripts/generate_dataset_stats.py."
        )
else:
    print(f"[iter17] Cache OK: {PARQUET_PATH}")

# %%
from src.models.xgboost_baseline import (
    DEFAULT_XGB_PARAMS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    FEATURES_10,
    SEQ_LENGTH,
    TARGET_COL,
    get_split_indices,
    train_xgboost_h,
)

print(f"[iter17] SEQ_LENGTH={SEQ_LENGTH}  FEATURES_10={len(FEATURES_10)}")
print(f"[iter17] XGB params iniciales: {DEFAULT_XGB_PARAMS}")
print(f"[iter17] early_stopping_rounds={DEFAULT_EARLY_STOPPING_ROUNDS}")

# %%
t0 = time.time()
df = pd.read_parquet(PARQUET_PATH)
print(f"[iter17] Parquet cargado en {time.time() - t0:.1f}s. shape={df.shape}")
print(f"[iter17] timestamp rango: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

# %%
# Split alineado con S2 para cada horizonte. Sanity check.
for h in [1, 3]:
    idx = get_split_indices(df, horizon=h)
    print(
        f"[iter17] H={h}: n_train={len(idx['train']):,}  "
        f"n_val={len(idx['val']):,}  n_test={len(idx['test']):,}"
    )

# %%
# Entrenamiento del modelo principal: 12 lags + 10 features, H=1 y H=3.
# Usamos los hiperparametros del DIAGNOSTIC_REPORT §7.2 sin modificar.
results_scaffold: dict = {}
for horizon in [1, 3]:
    print(f"\n[iter17] === Entrenando XGB lag12+feat10, H={horizon} ===")
    out = train_xgboost_h(
        df,
        horizon=horizon,
        lags=12,
        features=FEATURES_10,
        include_lags=True,
        include_features=True,
        xgb_params=DEFAULT_XGB_PARAMS,
        early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    results_scaffold[f"H{horizon}_lag12_feat10"] = out
    yt, yp = out["y_true_test"], out["y_pred_test"]
    # Metricas inline minimas como sanity check; el panel completo entra en 7.3/7.4.
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    nse = 1.0 - float(np.sum((yt - yp) ** 2)) / denom
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    peak_real = float(np.max(yt))
    peak_pred = float(np.max(yp))
    peak_err_pct = (peak_pred - peak_real) / max(peak_real, 1e-9) * 100.0
    print(
        f"[iter17] H={horizon}: NSE={nse:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}  "
        f"pico_real={peak_real:.1f}  pico_pred={peak_pred:.1f}  err_pico={peak_err_pct:+.1f}%  "
        f"fit={out['fit_seconds']:.1f}s  best_iter={out['best_iteration']}"
    )

print("\n[iter17] Scaffolding OK. Pendiente en 7.3/7.4: panel multi-bucket y ablations.")
