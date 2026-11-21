"""Look-ahead inputs for a TRAINING-ONLY teacher (never in a submission: they read future rows).

For the i0 columns (book prices / volumes, trades: 0..51) and each horizon L in FUT_LAGS: mean(x[t+1 .. t+L]) - x[t]
(rows past the sequence end are left out of the mean; the last row gets 0). The teacher's predictions on valid / train
become soft labels for the deployable students (scripts/finetune_valid.py --soft).
"""
from __future__ import annotations

import numpy as np

FUT_COLS = np.arange(0, 52)
FUT_LAGS = (2, 5, 12, 25, 50)
N_FUT = len(FUT_COLS) * len(FUT_LAGS)


def set_lags(lags):
    """Change the horizons (train.py --fut-lags; set before DataLoader workers fork)."""
    global FUT_LAGS, N_FUT
    FUT_LAGS = tuple(int(v) for v in lags); N_FUT = len(FUT_COLS) * len(FUT_LAGS)


def compute_sequence(X: np.ndarray) -> np.ndarray:
    """X: (T, >=52) rows of one sequence -> (T, N_FUT) float32."""
    x = X[:, FUT_COLS].astype(np.float64)
    T = len(x)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])  # cs[i] = sum x[:i]
    t = np.arange(T)
    out = []
    for L in FUT_LAGS:
        hi = np.minimum(t + 1 + L, T)
        n = np.maximum(hi - (t + 1), 1)[:, None]
        m = (cs[hi] - cs[t + 1]) / n
        d = m - x
        d[t + 1 >= T] = 0.0
        out.append(d)
    return np.concatenate(out, 1).astype(np.float32)
