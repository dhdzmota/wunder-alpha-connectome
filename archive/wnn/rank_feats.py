"""Causal per-sequence rank features: for every raw column, the mid-rank of the current value among all values of
that column seen so far in the sequence (expanding window), from a fixed-bin histogram.

Inputs are gauss-transformed (bounded ~±2.33) with sentinel -5.1993, so NB equal bins on [LO, HI] plus clamping
cover them. Output per column: 2 * (count_below + 0.5 * count_same_bin) / n - 1, in [-1, 1].
"""
from __future__ import annotations

import numba as nb
import numpy as np

NB = 64
LO, HI = -2.5, 2.5


@nb.njit(cache=True)
def update(x, first, hist, cnt, out):
    """One row. x: (n,) float64; hist: (n, NB) float64 counts; cnt: (1,) rows seen. out: (n,) float32."""
    n = x.shape[0]
    if first:
        hist[:, :] = 0.0
        cnt[0] = 0.0
    cnt[0] += 1.0
    scale = NB / (HI - LO)
    for i in range(n):
        b = int((x[i] - LO) * scale)
        b = min(max(b, 0), NB - 1)
        hist[i, b] += 1.0
        below = 0.0
        for k in range(b):
            below += hist[i, k]
        out[i] = np.float32(2.0 * (below + 0.5 * hist[i, b]) / cnt[0] - 1.0)


@nb.njit(cache=True)
def _sequence(X, out):
    n = X.shape[1]
    hist, cnt = np.zeros((n, NB)), np.zeros(1)
    x = np.zeros(n)
    for t in range(X.shape[0]):
        for i in range(n):
            x[i] = X[t, i]
        update(x, t == 0, hist, cnt, out[t])


def compute_sequence(X: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape, np.float32)
    _sequence(np.ascontiguousarray(X, dtype=np.float64), out)
    return out
