"""Causal per-sequence normalisation features (idea A: make inputs relative to the sequence's own regime).

For every raw column and every span S (alpha = 2 / (S + 1), warm-started as a running mean: a_t = max(alpha, 1/(t+1))):
    d = x - m;  m += a d;  v = (1 - a) (v + a d^2);  z = clip((x - m) / sqrt(v + EPS), -CLIP, CLIP)
Output per row: [z for span 0 (112), z for span 1 (112), ...]. Float64 state, float32 output.
The same `update` kernel runs in training (compute_sequence) and in the submission runtime.
"""
from __future__ import annotations

import numba as nb
import numpy as np

EPS = 1e-4
CLIP = 5.0


@nb.njit(cache=True)
def update(x, t, alphas, m, v, out):
    """One row. x: (n,) float64; t: row index in the sequence; m, v: (S, n) state; out: (S * n,) float32."""
    n = x.shape[0]
    for s in range(alphas.shape[0]):
        a = max(alphas[s], 1.0 / (t + 1.0))
        for i in range(n):
            d = x[i] - m[s, i]
            m[s, i] += a * d
            v[s, i] = (1.0 - a) * (v[s, i] + a * d * d)
            z = (x[i] - m[s, i]) / np.sqrt(v[s, i] + EPS)
            out[s * n + i] = np.float32(min(max(z, -CLIP), CLIP))


@nb.njit(cache=True)
def _sequence(X, alphas, out):
    n = X.shape[1]
    m = np.zeros((alphas.shape[0], n))
    v = np.zeros((alphas.shape[0], n))
    x = np.zeros(n)
    for t in range(X.shape[0]):
        for i in range(n):
            x[i] = X[t, i]
        update(x, t, alphas, m, v, out[t])


def alphas(spans) -> np.ndarray:
    return np.array([2.0 / (s + 1.0) for s in spans], np.float64)


def compute_sequence(X: np.ndarray, spans) -> np.ndarray:
    """X: (T, n) raw rows of one sequence -> (T, len(spans) * n) float32 features."""
    out = np.zeros((X.shape[0], len(spans) * X.shape[1]), np.float32)
    _sequence(np.ascontiguousarray(X, dtype=np.float64), alphas(spans), out)
    return out
