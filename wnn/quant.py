"""Int16 / int8 row-scaled weights and the mat-vec kernel every runtime model uses.

Batch-1 inference streams every weight once per row from L3 cache, so fewer bytes per weight is directly
faster: int16 weights run at 0.82x the float32 latency with no change in the metric.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _qmatvec(W, x, bs, out):
    """out = s * (W x) + b with bs = [b (J), s (J)]: W may be int16 / int8 (row-scaled) or float32 (s = 1).
    Batch-1 inference streams every weight once per row from L3, so fewer bytes per weight = faster."""
    n, J = W.shape[1], W.shape[0]
    J4 = J - J % 4
    for j in range(0, J4, 4):
        s0 = np.float32(0.0)
        s1 = np.float32(0.0)
        s2 = np.float32(0.0)
        s3 = np.float32(0.0)
        for i in range(n):
            xi = x[i]
            s0 += np.float32(W[j, i]) * xi
            s1 += np.float32(W[j + 1, i]) * xi
            s2 += np.float32(W[j + 2, i]) * xi
            s3 += np.float32(W[j + 3, i]) * xi
        out[j] = s0 * bs[J + j] + bs[j]
        out[j + 1] = s1 * bs[J + j + 1] + bs[j + 1]
        out[j + 2] = s2 * bs[J + j + 2] + bs[j + 2]
        out[j + 3] = s3 * bs[J + j + 3] + bs[j + 3]
    for j in range(J4, J):
        s = np.float32(0.0)
        for i in range(n):
            s += np.float32(W[j, i]) * x[i]
        out[j] = s * bs[J + j] + bs[j]


def quantize(w: dict, pairs, bits: int = 32) -> dict:
    """Copy of w where each (W, b) in pairs becomes (W as int<bits> with per-row scale, [b, scale]) for _qmatvec.
    bits = 32 keeps float32 weights (scale 1). Works on stacked arrays (..., J, n) / (..., J)."""
    w = dict(w)
    for kw, kb in pairs:
        W = np.asarray(w[kw], np.float32)
        b = np.asarray(w[kb], np.float32)
        if bits == 32:
            Wq, sc = W, np.ones(b.shape, np.float32)
        else:
            lim = 2 ** (bits - 1) - 1
            sc = np.maximum(np.abs(W).max(-1), 1e-12).astype(np.float32) / np.float32(lim)
            Wq = np.clip(np.round(W / sc[..., None]), -lim, lim).astype({16: np.int16, 8: np.int8}[bits])
        w[kw], w[kb] = np.ascontiguousarray(Wq), np.ascontiguousarray(np.concatenate([b, sc], -1))
    return w
