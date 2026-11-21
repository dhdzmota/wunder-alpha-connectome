"""Single-row stateful GRU inference in numba (numpy + numba only; no scipy BLAS).

Mirrors train_gru.GRUNet: [x, x - x_prev] -> Linear -> GRU(layers) -> Linear(2).
PyTorch gate order is (r, z, n) with n = tanh(W_in x + b_in + r * (W_hn h + b_hn)).
Weights are float32, row-major, so every mat-vec reads contiguous memory.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _matvec(W, x, b, out):
    """out = W x + b. Four rows per pass share each x load (~1.1-1.5x faster than one row at a time)."""
    n, J = W.shape[1], W.shape[0]
    J4 = J - J % 4
    for j in range(0, J4, 4):
        s0 = np.float32(0.0)
        s1 = np.float32(0.0)
        s2 = np.float32(0.0)
        s3 = np.float32(0.0)
        for i in range(n):
            xi = x[i]
            s0 += W[j, i] * xi
            s1 += W[j + 1, i] * xi
            s2 += W[j + 2, i] * xi
            s3 += W[j + 3, i] * xi
        out[j] = s0 + b[j]
        out[j + 1] = s1 + b[j + 1]
        out[j + 2] = s2 + b[j + 2]
        out[j + 3] = s3 + b[j + 3]
    for j in range(J4, J):
        s = np.float32(0.0)
        for i in range(n):
            s += W[j, i] * x[i]
        out[j] = s + b[j]



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

@nb.njit(cache=True, fastmath=True)
def _gru_cell(x, h, Wih, Whh, bih, bhh, gi, gh):
    H = h.shape[0]
    _matvec(Wih, x, bih, gi)
    _matvec(Whh, h, bhh, gh)
    for k in range(H):
        r = 1.0 / (1.0 + np.exp(-(gi[k] + gh[k])))
        z = 1.0 / (1.0 + np.exp(-(gi[H + k] + gh[H + k])))
        n = np.tanh(gi[2 * H + k] + r * gh[2 * H + k])
        h[k] = (1.0 - z) * n + z * h[k]  # in place: gh already holds W_hh h


@nb.njit(cache=True, fastmath=True)
def step(x, prev, first, inp_buf, a, H, Win, bin_, Wih, Whh, bih, bhh, Wout, bout, gi, gh, out):
    """x: (112,) row. prev: (112,) previous row (updated in place). H: (L, hidden) state.
    Wih/Whh/bih/bhh: stacked per layer. Writes the 2 predictions into out."""
    n = x.shape[0]
    for i in range(n):
        inp_buf[i] = x[i]
        inp_buf[n + i] = 0.0 if first else x[i] - prev[i]
        prev[i] = x[i]
    _matvec(Win, inp_buf, bin_, a)
    for l in range(H.shape[0]):
        _gru_cell(a, H[l], Wih[l], Whh[l], bih[l], bhh[l], gi, gh)
        a[:] = H[l]
    _matvec(Wout, a, bout, out)


def export_gru(model) -> dict:
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    L = model.gru.num_layers
    return {"Win": sd["inp.weight"], "bin": sd["inp.bias"],
            "Wih": np.stack([sd[f"gru.weight_ih_l{l}"] for l in range(L)]),
            "Whh": np.stack([sd[f"gru.weight_hh_l{l}"] for l in range(L)]),
            "bih": np.stack([sd[f"gru.bias_ih_l{l}"] for l in range(L)]),
            "bhh": np.stack([sd[f"gru.bias_hh_l{l}"] for l in range(L)]),
            "Wout": sd["head.weight"], "bout": sd["head.bias"]}


class GRURunner:
    """Holds weights + state for one model; call reset() per sequence and step(x) per row."""

    def __init__(self, w: dict):
        self.w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in w.items()}
        L, hidden = self.w["Whh"].shape[0], self.w["Whh"].shape[2]
        self.H = np.zeros((L, hidden), np.float32)
        self.prev = np.zeros(self.w["Win"].shape[1] // 2, np.float32)
        self.inp = np.zeros(self.w["Win"].shape[1], np.float32)
        self.a = np.zeros(hidden, np.float32)
        self.gi = np.zeros(3 * hidden, np.float32)
        self.gh = np.zeros(3 * hidden, np.float32)
        self.out = np.zeros(2, np.float32)
        self.first = True

    def reset(self):
        self.H[:] = 0.0
        self.first = True

    def step(self, x):
        w = self.w
        step(x, self.prev, self.first, self.inp, self.a, self.H, w["Win"], w["bin"], w["Wih"], w["Whh"],
             w["bih"], w["bhh"], w["Wout"], w["bout"], self.gi, self.gh, self.out)
        self.first = False
        return self.out
