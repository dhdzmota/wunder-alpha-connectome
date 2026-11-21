"""Input representation screening (EXPERIMENTS §17).

"na": the additional columns a0..a7 carry a sentinel (<= -5, ~7 % of values, 33 % in a7). impute() replaces it with 0
      (the centre of the gauss-transformed inputs) before the raw / diff / EMA inputs are built, and missing() returns
      the 8 per-column indicators appended as extra features.
"pb": piecewise-linear encoding of the 44 price-like columns (i0/i1 p0..p21; ~40 distinct values per sequence, one value
      on 57 % of rows): NBIN bins per column with edges at quantiles of the column's distinct train values, each bin
      clip((x - lo) / (hi - lo), 0, 1). Edges are fixed (train) and stored in artifacts/repr/pb_edges.npy.
"""
from __future__ import annotations

import numpy as np

from wnn.data import ARTIFACTS, SequenceReader

A_COLS = np.arange(104, 112)
PRICE_COLS = np.concatenate([np.arange(0, 22), np.arange(52, 74)])
NBIN = 8
SENTINEL = -5.0
EDGES_PATH = ARTIFACTS / "repr" / "pb_edges.npy"
_EDGES = None


def impute(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X: (T, 112) raw rows -> (X with a0..a7 sentinels set to 0, (T, 8) float32 missing indicators)."""
    miss = X[:, A_COLS] <= SENTINEL
    X = X.copy()
    X[:, A_COLS] = np.where(miss, 0.0, X[:, A_COLS])
    return X, miss.astype(np.float32)


def build_edges(n: int = 300, seed: int = 0) -> np.ndarray:
    rd = SequenceReader("train")
    pick = np.random.default_rng(seed).choice(rd.n, n, replace=False)
    V = np.concatenate([rd.read(int(g)).X[:, PRICE_COLS] for g in pick])
    E = np.stack([np.quantile(np.unique(V[:, j]), np.linspace(0, 1, NBIN + 1)) for j in range(len(PRICE_COLS))])
    EDGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EDGES_PATH, E.astype(np.float32))
    return E


def edges() -> np.ndarray:
    global _EDGES
    if _EDGES is None:
        _EDGES = np.load(EDGES_PATH)
    return _EDGES


def ple(X: np.ndarray) -> np.ndarray:
    """X: (T, 112) raw rows -> (T, 44 * NBIN) float32 piecewise-linear bins, column-major (col j: j*NBIN .. j*NBIN+NBIN-1)."""
    E = edges()
    lo, hi = E[:, :-1], E[:, 1:]
    Z = (X[:, PRICE_COLS, None] - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(Z, 0.0, 1.0).reshape(len(X), -1).astype(np.float32)


if __name__ == "__main__":
    E = build_edges()
    print("edges", E.shape, "bin widths min/median", float(np.diff(E, axis=1).min()), float(np.median(np.diff(E, axis=1))))
