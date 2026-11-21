"""Single-row stateful selective-SSM-stack inference in numba (mirrors wnn.nets.SSMNet).

Per block l:  z = rms(x) * n1;  [u, g, dt, B, C] = Wp z + bp;  dt = softplus(dt)
              h[j, n] = exp(-dt_j * expA[j, n]) h[j, n] + dt_j u_j B_n;  y_j = sum_n h[j, n] C_n + D_j u_j
              x = x + Wo (y * silu(g)) + bo;  x = x + W2 gelu_tanh(W1 (rms(x) * n2) + b1) + b2
Output: head(rms(x) * nf). Input row: [x, x - x_prev] -> Linear (same as the GRU).
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .fastmath import exp_neg_fast, gelu_tanh_f, sigmoid_f, softplus_f
from .quant import _qmatvec, quantize
from .mingru_infer import _rms

KEYS = ("Win", "bin", "n1", "Wp", "bp", "expA", "Dsk", "Wo", "bo", "n2", "W1", "b1", "W2", "b2", "nf", "Wout", "bout")
# (weight, bias) pairs run through _qmatvec; their biases carry [b, row scale] (see quant.quantize)
QPAIRS = (("Win", "bin"), ("Wp", "bp"), ("Wo", "bo"), ("W1", "b1"), ("W2", "b2"), ("Wout", "bout"))


@nb.njit(cache=True, fastmath=True)
def step(x, extra, prev, first, inp, H, Win, bin_, n1, Wp, bp, expA, Dsk, Wo, bo, n2, W1, b1, W2, b2, nf, Wout, bout,
         xs, z, pj, v, m, out):
    """x: (n_raw,) row; extra: ema features (used only if Win takes more than 2 * n_raw inputs);
    prev: previous row (updated); H: (L, d, N) state. Writes 2 predictions to out."""
    n = x.shape[0]
    for i in range(n):
        inp[i] = x[i]
        inp[n + i] = 0.0 if first else x[i] - prev[i]
        prev[i] = x[i]
    for i in range(Win.shape[1] - 2 * n):
        inp[2 * n + i] = extra[i]
    _qmatvec(Win, inp, bin_, xs)
    d, N = H.shape[1], H.shape[2]
    for l in range(H.shape[0]):
        _rms(xs, n1[l], z)
        _qmatvec(Wp[l], z, bp[l], pj)  # [u (d), g (d), dt (d), B (N), C (N)]
        for j in range(d):  # torch softplus (threshold 20), in place
            pj[2 * d + j] = softplus_f(pj[2 * d + j])
        for j in range(d):
            u, dt = pj[j], pj[2 * d + j]
            du = dt * u
            y = Dsk[l, j] * u
            for k in range(N):  # branch-free so LLVM vectorises over the state dim
                h = exp_neg_fast(dt * expA[l, j, k]) * H[l, j, k] + du * pj[3 * d + k]
                H[l, j, k] = h
                y += h * pj[3 * d + N + k]
            v[j] = y
        for j in range(d):
            g = pj[d + j]
            v[j] *= g * sigmoid_f(g)
        _qmatvec(Wo[l], v, bo[l], z)
        for j in range(d):
            xs[j] += z[j]
        _rms(xs, n2[l], z)
        _qmatvec(W1[l], z, b1[l], m)
        for j in range(m.shape[0]):
            m[j] = gelu_tanh_f(m[j])
        _qmatvec(W2[l], m, b2[l], v)
        for j in range(d):
            xs[j] += v[j]
    _rms(xs, nf, z)
    _qmatvec(Wout, z, bout, out)


def export_ssm(model) -> dict:
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    L = len(model.blocks)
    st = lambda f: np.stack([sd[f.format(l)] for l in range(L)])  # noqa: E731
    Win, bin_ = sd["inp.weight"], sd["inp.bias"]
    if "mu" in sd:  # fold (x - mu) / sd into the input layer
        Win = (Win / sd["sd"][None, :]).astype(np.float32)
        bin_ = (bin_ - Win.astype(np.float64) @ sd["mu"].astype(np.float64)).astype(np.float32)
    return {"Win": Win, "bin": bin_,
            "n1": st("blocks.{}.n1.weight"), "Wp": st("blocks.{}.in_proj.weight"), "bp": st("blocks.{}.in_proj.bias"),
            "expA": np.exp(st("blocks.{}.A_log")).astype(np.float32), "Dsk": st("blocks.{}.D_skip"),
            "Wo": st("blocks.{}.out_proj.weight"), "bo": st("blocks.{}.out_proj.bias"), "n2": st("blocks.{}.n2.weight"),
            "W1": st("blocks.{}.mlp.0.weight"), "b1": st("blocks.{}.mlp.0.bias"),
            "W2": st("blocks.{}.mlp.2.weight"), "b2": st("blocks.{}.mlp.2.bias"),
            "nf": sd["norm.weight"], "Wout": sd["head.weight"], "bout": sd["head.bias"]}


def buffers(d, n_proj, n_mlp):
    """Scratch arrays for step: xs, z, pj, v, m, out."""
    return (np.zeros(d, np.float32), np.zeros(d, np.float32), np.zeros(n_proj, np.float32),
            np.zeros(d, np.float32), np.zeros(n_mlp, np.float32), np.zeros(2, np.float32))


class SSMRunner:
    """Standalone runner (tests). spans: the model's ema_feats spans, () for plain models."""

    def __init__(self, w: dict, spans=(), bits=32):
        from . import ema_feats
        w = quantize(w, QPAIRS, bits)
        self.w = tuple(w[k] if w[k].dtype.kind == "i" else np.ascontiguousarray(w[k], dtype=np.float32) for k in KEYS)
        L, d, N = self.w[5].shape
        n_in = self.w[0].shape[1]
        n_raw = n_in // (2 + len(spans))
        self.H = np.zeros((L, d, N), np.float32)
        self.prev = np.zeros(n_raw, np.float32)
        self.inp = np.zeros(n_in, np.float32)
        self.ema = (ema_feats.alphas(spans), np.zeros((len(spans), n_raw)), np.zeros((len(spans), n_raw)),
                    np.zeros(len(spans) * n_raw, np.float32))
        self.tmp = buffers(d, self.w[3].shape[1], self.w[10].shape[1])
        self.first, self.t = True, 0

    def reset(self):
        self.H[:] = 0.0
        self.ema[1][:] = 0.0
        self.ema[2][:] = 0.0
        self.first, self.t = True, 0

    def step(self, x):
        from . import ema_feats
        al, m, v, extra = self.ema
        if al.shape[0]:
            ema_feats.update(np.asarray(x, np.float64), self.t, al, m, v, extra)
        step(x, extra, self.prev, self.first, self.inp, self.H, *self.w, *self.tmp)
        self.first, self.t = False, self.t + 1
        return self.tmp[-1]
