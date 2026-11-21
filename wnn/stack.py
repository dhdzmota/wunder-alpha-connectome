"""Row-level LightGBM stack applied after a routed fused model (scripts/train_stack.py).

Inputs per row, float32 as in training: raw row x (n), x - previous row (n), the fused model's blended output (2).
Output: (1 - alpha) * p + alpha * trees(inputs), per target. stack_r<k>.npz: st_<TREE_KEYS>, st_trange, alpha.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .trees import TREE_KEYS, _trees


@nb.njit(cache=True)
def apply(x, prev, p, F, feat, thr, left, right, leaf, node_off, leaf_off, trange, alpha, res):
    """x: current raw row; prev: previous raw row (float32, updated here); p: fused output (float32[2]) -> res."""
    n = x.shape[0]
    for i in range(n):
        xi = np.float32(x[i])
        F[i] = xi
        F[n + i] = xi - prev[i]
        prev[i] = xi
    F[2 * n] = p[0]
    F[2 * n + 1] = p[1]
    s0 = _trees(F, feat, thr, left, right, leaf, node_off, leaf_off, trange[0], trange[1])
    s1 = _trees(F, feat, thr, left, right, leaf, node_off, leaf_off, trange[1], trange[2])
    res[0] = np.float32((1.0 - alpha) * p[0] + alpha * s0)
    res[1] = np.float32((1.0 - alpha) * p[1] + alpha * s1)


class Stack:
    def __init__(self, z, n_raw=112):
        self.tr = tuple(np.ascontiguousarray(z[f"st_{k}"]) for k in TREE_KEYS + ("trange",))
        self.alpha = float(z["alpha"])
        self.F = np.zeros(2 * n_raw + 2, np.float32)
        self.prev = np.zeros(n_raw, np.float32)
        self.res = np.zeros(2, np.float32)

    def set_prev(self, x):
        self.prev[:] = x

    def step(self, x, p):
        apply(x, self.prev, p, self.F, *self.tr, self.alpha, self.res)
        return self.res
