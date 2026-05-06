"""Multi-bucket metrics panel to evaluate stormflow models.

Functionally replaces `src/evaluation/metrics.py::evaluate_model` with a more
complete battery (bias/peak by bucket, peak_lag_minutes, recall@U, and
quantile coverage). It works on arrays in real MGD; denormalization stays
outside the panel because newer models (iter17+) work directly in MGD.

Severity buckets (same as `evaluate_local.py`):
    Base     <0.5
    Leve     [0.5, 5)
    Moderado [5, 20)
    Alto     [20, 50)
    Extremo  >=50

`evaluate_local.py` keeps two thresholds (Alto [20, 50) and Extremo [50, inf))
that differ from the historical variant in `src/evaluation/metrics.py`
(`grande [12.8, 51)`, `extremo [51, inf)`). The panel uses the thresholds from
`evaluate_local.py` because those are the ones used by diagnostics S2-S5.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_BUCKETS: List[Tuple[str, float, float]] = [
    ("Base",     -np.inf, 0.5),
    ("Leve",     0.5,     5.0),
    ("Moderado", 5.0,     20.0),
    ("Alto",     20.0,    50.0),
    ("Extremo",  50.0,    np.inf),
]

DEFAULT_RECALL_THRESHOLDS: List[float] = [25.0, 50.0]

# Data are at 5 min. `event_gap_minutes=240` = 4h between samples with
# y_true > event_threshold to separate one physical event from the next.
DEFAULT_EVENT_GAP_MINUTES = 240.0
DEFAULT_EVENT_THRESHOLD_MGD = 0.5


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def _nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _peak_err_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Peak error = (max(y_pred) - max(y_true)) / max(y_true) * 100."""
    if y_true.size == 0:
        return float("nan")
    pt = float(np.max(y_true))
    if pt == 0:
        return float("nan")
    pp = float(np.max(y_pred))
    return (pp - pt) / pt * 100.0


def _bucket_mask(y_true: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (y_true >= lo) & (y_true < hi)


def _bucket_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    buckets: Sequence[Tuple[str, float, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in buckets:
        mask = _bucket_mask(y_true, lo, hi)
        n = int(mask.sum())
        if n == 0:
            out[name] = {
                "n": 0,
                "nse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "peak_err_pct": float("nan"),
            }
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        out[name] = {
            "n": n,
            "nse": _nse(yt, yp),
            "rmse": _rmse(yt, yp),
            "mae": _mae(yt, yp),
            "bias": float(np.mean(yp - yt)),
            "peak_err_pct": _peak_err_pct(yt, yp),
        }
    return out


# ---------------------------------------------------------------------------
# Physical event definition and peak_lag_minutes
# ---------------------------------------------------------------------------

def _extract_events(
    y_true: np.ndarray,
    timestamps: pd.DatetimeIndex,
    event_threshold: float,
    gap_minutes: float,
) -> List[Dict[str, int]]:
    """Segment test into physical events.

    Definition (official panel docstring): an event is a continuous interval
    of samples with `y_true > event_threshold`, separated from the next event
    by more than `gap_minutes` without samples above threshold.
    Implementation: samples above threshold are detected and any temporal gap
    > gap_minutes splits the event.

    If the repo had a different convention (for example MSD `is_event`), that
    should be used; here the panel has no access to `is_event`, so threshold +
    gap is the autonomous definition.

    Returns a list of events with start and end (inclusive indices).
    """
    if timestamps is None:
        # Without timestamps, only mask breaks can be used to split events.
        mask = y_true > event_threshold
        if not mask.any():
            return []
        events: List[Dict[str, int]] = []
        in_event = False
        start = 0
        for i, m in enumerate(mask):
            if m and not in_event:
                in_event = True
                start = i
            elif not m and in_event:
                in_event = False
                events.append({"start": start, "end": i - 1})
        if in_event:
            events.append({"start": start, "end": len(mask) - 1})
        return events

    mask = y_true > event_threshold
    if not mask.any():
        return []

    ts_ns = timestamps.to_numpy().astype("datetime64[ns]").astype("int64")
    gap_ns = int(gap_minutes * 60 * 1_000_000_000)

    events = []
    start = None
    prev_idx: Optional[int] = None
    for i in range(len(mask)):
        if not mask[i]:
            continue
        if start is None:
            start = i
            prev_idx = i
            continue
        # i is an active sample and prev_idx is the last previous active sample.
        if ts_ns[i] - ts_ns[prev_idx] > gap_ns:  # type: ignore[arg-type]
            # Close current event.
            events.append({"start": int(start), "end": int(prev_idx)})  # type: ignore[arg-type]
            start = i
        prev_idx = i
    if start is not None and prev_idx is not None:
        events.append({"start": int(start), "end": int(prev_idx)})
    return events


def _peak_lag_minutes_per_event(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    events: List[Dict[str, int]],
) -> List[Dict[str, float]]:
    """For each event, time between real peak and predicted peak (pred - real)."""
    if timestamps is None:
        return []
    ts_ns = timestamps.to_numpy().astype("datetime64[ns]").astype("int64")
    out: List[Dict[str, float]] = []
    for ev in events:
        s, e = ev["start"], ev["end"]
        yt = y_true[s:e + 1]
        yp = y_pred[s:e + 1]
        if yt.size == 0:
            continue
        i_true_peak = int(np.argmax(yt))
        i_pred_peak = int(np.argmax(yp))
        lag_ns = int(ts_ns[s + i_pred_peak] - ts_ns[s + i_true_peak])
        lag_min = lag_ns / (60 * 1_000_000_000)
        out.append({
            "start_index": int(s),
            "end_index": int(e),
            "n_samples": int(e - s + 1),
            "peak_real_mgd": float(yt[i_true_peak]),
            "peak_pred_mgd": float(yp[i_pred_peak]),
            "peak_real_ts": str(timestamps[s + i_true_peak]),
            "peak_pred_ts": str(timestamps[s + i_pred_peak]),
            "peak_lag_minutes": float(lag_min),
        })
    return out


def _summarize_peak_lag(event_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not event_rows:
        return {
            "n_events": 0,
            "mean_lag_minutes": float("nan"),
            "median_lag_minutes": float("nan"),
            "median_abs_lag_minutes": float("nan"),
            "p90_abs_lag_minutes": float("nan"),
        }
    lags = np.array([e["peak_lag_minutes"] for e in event_rows], dtype=float)
    return {
        "n_events": int(len(lags)),
        "mean_lag_minutes": float(np.mean(lags)),
        "median_lag_minutes": float(np.median(lags)),
        "median_abs_lag_minutes": float(np.median(np.abs(lags))),
        "p90_abs_lag_minutes": float(np.percentile(np.abs(lags), 90)),
    }


# ---------------------------------------------------------------------------
# Recall by threshold
# ---------------------------------------------------------------------------

def _recall_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    events: List[Dict[str, int]],
    threshold: float,
) -> Dict[str, float]:
    """recall@U = fraction of events with max(y_true) >= U correctly alerted
    (max(y_pred) >= U in the same event)."""
    pos = 0
    tp = 0
    fn_peak_values: List[float] = []
    for ev in events:
        s, e = ev["start"], ev["end"]
        yt = y_true[s:e + 1]
        yp = y_pred[s:e + 1]
        peak_real = float(np.max(yt))
        peak_pred = float(np.max(yp))
        if peak_real >= threshold:
            pos += 1
            if peak_pred >= threshold:
                tp += 1
            else:
                fn_peak_values.append(peak_real)
    recall = float(tp / pos) if pos > 0 else float("nan")
    return {
        "threshold_mgd": float(threshold),
        "n_events_with_y_true_ge_thr": int(pos),
        "n_events_alerted": int(tp),
        "recall": recall,
        "fn_peak_real_mean": float(np.mean(fn_peak_values)) if fn_peak_values else float("nan"),
        "fn_peak_real_max": float(np.max(fn_peak_values)) if fn_peak_values else float("nan"),
    }


# ---------------------------------------------------------------------------
# Quantile coverage
# ---------------------------------------------------------------------------

def _quantile_coverage(
    y_true: np.ndarray,
    quantiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """`quantiles` = dict `{'0.90': (y_lo, y_hi), '0.95': (y_lo, y_hi)}`.

    Returns empirical coverage (% of samples inside the interval).
    """
    out: Dict[str, Dict[str, float]] = {}
    for label, (lo, hi) in quantiles.items():
        lo = np.asarray(lo, dtype=float).reshape(-1)
        hi = np.asarray(hi, dtype=float).reshape(-1)
        if lo.size != y_true.size or hi.size != y_true.size:
            raise ValueError(f"quantiles[{label}] length mismatch with y_true")
        inside = (y_true >= lo) & (y_true <= hi)
        out[label] = {
            "coverage_empirical": float(np.mean(inside)),
            "avg_interval_width": float(np.mean(hi - lo)),
        }
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_full_panel(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
    timestamps: Optional[Union[pd.DatetimeIndex, pd.Series, np.ndarray, Sequence]] = None,
    buckets: Optional[Sequence[Tuple[str, float, float]]] = None,
    recall_thresholds: Sequence[float] = DEFAULT_RECALL_THRESHOLDS,
    event_threshold: float = DEFAULT_EVENT_THRESHOLD_MGD,
    event_gap_minutes: float = DEFAULT_EVENT_GAP_MINUTES,
    quantiles: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    clip_nonnegative: bool = True,
) -> Dict:
    """Full metrics panel.

    Parameters:
      y_true, y_pred: 1D arrays in MGD (same length).
      timestamps: optional, used to compute peak_lag_minutes and segment events
          by gap. If None, events are segmented only by mask and lags stay
          as nan/missing.
      buckets: list of tuples (name, lo, hi). By default DEFAULT_BUCKETS
          (those from `evaluate_local.py`).
      recall_thresholds: thresholds U for `recall@U`. Default [25, 50].
      event_threshold: threshold (MGD) to consider a sample active.
      event_gap_minutes: minimum temporal gap to separate events. Default 4h
          (enough to separate typical storms in MC-CL-005).
      quantiles: optional dict `{'0.90': (lo, hi), '0.95': (lo, hi)}`.
      clip_nonnegative: if True, forces predictions and target to be >=0 (same
          as `evaluate_local.py`). Useful for models that do not guarantee
          positivity.

    Returns JSON-serializable dict. Field `quantile_coverage` is absent if
    `quantiles` is None.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    if clip_nonnegative:
        y_true = np.clip(y_true, 0.0, None)
        y_pred = np.clip(y_pred, 0.0, None)

    ts: Optional[pd.DatetimeIndex] = None
    if timestamps is not None:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if ts.size != y_true.size:
            raise ValueError("timestamps must have the same length as y_true")

    if buckets is None:
        buckets = DEFAULT_BUCKETS

    # Global
    global_block = {
        "n": int(y_true.size),
        "nse": _nse(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "peak_real_mgd": float(np.max(y_true)) if y_true.size else float("nan"),
        "peak_pred_mgd": float(np.max(y_pred)) if y_pred.size else float("nan"),
        "peak_err_pct": _peak_err_pct(y_true, y_pred),
    }

    # By bucket
    bucket_block = _bucket_metrics(y_true, y_pred, buckets)

    # Events + peak_lag
    events = _extract_events(y_true, ts, event_threshold, event_gap_minutes)
    event_rows = _peak_lag_minutes_per_event(y_true, y_pred, ts, events) if ts is not None else []
    peak_lag_summary = _summarize_peak_lag(event_rows)

    # recall@U
    recall_block = {
        f"at_{int(u)}_mgd": _recall_at_threshold(y_true, y_pred, events, u)
        for u in recall_thresholds
    }

    panel: Dict = {
        "global": global_block,
        "buckets": bucket_block,
        "n_events": int(len(events)),
        "event_definition": {
            "threshold_mgd": float(event_threshold),
            "gap_minutes": float(event_gap_minutes),
        },
        "peak_lag": peak_lag_summary,
        "peak_lag_per_event": event_rows,
        "recall": recall_block,
    }

    if quantiles is not None:
        panel["quantile_coverage"] = _quantile_coverage(y_true, quantiles)
    else:
        panel["quantile_coverage"] = None

    return panel
