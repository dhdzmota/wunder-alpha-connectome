"""Causal event-clock features: per raw column, how long ago it last changed and by how much.

Most rows repeat the previous value of most columns (prices change on ~1/3 of rows), so the raw diff is
zero almost everywhere and the nets must remember events through their state. Here, per column j:
    age_j  = rows since x_j last changed (rows since the sequence start before its first change)
    last_j = x_j - x_j(previous row) at that change (0 before the first change)
Output per row: [log1p(min(age, AGE_CAP)) (n), last (n)]. The same `update` kernel runs in training and serving.
"""
from __future__ import annotations

import numba as nb
import numpy as np

AGE_CAP = 5000.0


@nb.njit(cache=True)
def update(x, first, prev, age, last, out):
    """One row. x: (n,) float64; prev / age / last: (n,) float64 state; out: (2 n,) float32."""
    n = x.shape[0]
    for i in range(n):
        if first:
            age[i] = 0.0
            last[i] = 0.0
        else:
            d = x[i] - prev[i]
            if d != 0.0:
                age[i] = 0.0
                last[i] = d
            else:
                age[i] += 1.0
        prev[i] = x[i]
        out[i] = np.float32(np.log1p(min(age[i], AGE_CAP)))
        out[n + i] = np.float32(last[i])


@nb.njit(cache=True)
def _sequence(X, out):
    n = X.shape[1]
    prev, age, last = np.zeros(n), np.zeros(n), np.zeros(n)
    x = np.zeros(n)
    for t in range(X.shape[0]):
        for i in range(n):
            x[i] = X[t, i]
        update(x, t == 0, prev, age, last, out[t])


def compute_sequence(X: np.ndarray) -> np.ndarray:
    """X: (T, n) raw rows of one sequence -> (T, 2 n) float32 features."""
    out = np.zeros((X.shape[0], 2 * X.shape[1]), np.float32)
    _sequence(np.ascontiguousarray(X, dtype=np.float64), out)
    return out


@nb.njit(cache=True)
def update_rel(x, first, prev, age, ia, out):
    """Relative event clock, one row: per column log1p(age) - log1p(typical gap), where the typical gap ia is an
    EMA (over change events, alpha 0.05) of the age at which the column changed. out: (n,) float32."""
    n = x.shape[0]
    for i in range(n):
        if first:
            age[i] = 0.0
            ia[i] = 1.0
        elif x[i] - prev[i] != 0.0:
            ia[i] += 0.05 * (age[i] + 1.0 - ia[i])
            age[i] = 0.0
        else:
            age[i] += 1.0
        prev[i] = x[i]
        out[i] = np.float32(np.log1p(min(age[i], AGE_CAP)) - np.log1p(ia[i]))


@nb.njit(cache=True)
def _sequence_rel(X, out):
    n = X.shape[1]
    prev, age, ia = np.zeros(n), np.zeros(n), np.zeros(n)
    x = np.zeros(n)
    for t in range(X.shape[0]):
        for i in range(n):
            x[i] = X[t, i]
        update_rel(x, t == 0, prev, age, ia, out[t])


def compute_sequence_rel(X: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape, np.float32)
    _sequence_rel(np.ascontiguousarray(X, dtype=np.float64), out)
    return out
