"""Single-row stateful minGRU-stack inference in numba (mirrors wnn.nets.MinGRUNet).

Per block l:  z = rms(x) * n1;  k = Wg z + bg;  a = sigmoid(k);  h = a h + (1 - a)(Wc z + bc)
              x = x + h;  x = x + W2 gelu_tanh(W1 (rms(x) * n2) + b1) + b2
Output: head(rms(x) * nf), or with a tick-class head (cls_v: class values) the point E[|y| y] / E[|y|] of the
softmax over head rows [t * K, (t + 1) * K) per target t (see cls_point). Input row: [x, x - x_prev, extra] -> Linear (same as the GRU); extra = optional
per-row features (e.g. wnn.ema_feats), empty for plain models. Input normalisation (mu, sd) is folded into Win.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from .fastmath import gelu_tanh_f, sigmoid_f
from .quant import _qmatvec, quantize

EPS = 1e-6
KEYS = ("Win", "bin", "n1", "Wg", "bg", "Wc", "bc", "n2", "W1", "b1", "W2", "b2", "nf", "Wout", "bout")
# (weight, bias) pairs run through _qmatvec; their biases carry [b, row scale] (see quant.quantize)
QPAIRS = (("Win", "bin"), ("Wg", "bg"), ("Wc", "bc"), ("W1", "b1"), ("W2", "b2"), ("Wout", "bout"))


@nb.njit(cache=True, fastmath=True)
def _rms(x, w, out):
    s = np.float32(0.0)
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    r = np.float32(1.0) / np.sqrt(s / x.shape[0] + np.float32(EPS))
    for i in range(x.shape[0]):
        out[i] = x[i] * r * w[i]


@nb.njit(cache=True, fastmath=True)
def step(x, extra, prev, first, inp, H, Win, bin_, n1, Wg, bg, Wc, bc, n2, W1, b1, W2, b2, nf, Wout, bout,
         xs, z, k, u, m, out):
    """x: (n_raw,) row; extra: (n_extra,) features (may be empty); prev: previous row (updated);
    H: (L, d) state. inp must hold 2 * n_raw + n_extra values. Writes 2 predictions to out."""
    n = x.shape[0]
    for i in range(n):
        inp[i] = x[i]
        inp[n + i] = 0.0 if first else x[i] - prev[i]
        prev[i] = x[i]
    for i in range(extra.shape[0]):
        inp[2 * n + i] = extra[i]
    _qmatvec(Win, inp, bin_, xs)
    d = xs.shape[0]
    for l in range(H.shape[0]):
        _rms(xs, n1[l], z)
        _qmatvec(Wg[l], z, bg[l], k)
        _qmatvec(Wc[l], z, bc[l], u)
        for j in range(d):
            a = sigmoid_f(k[j])
            H[l, j] = a * H[l, j] + (np.float32(1.0) - a) * u[j]
            xs[j] += H[l, j]
        _rms(xs, n2[l], z)
        _qmatvec(W1[l], z, b1[l], m)
        for j in range(m.shape[0]):
            m[j] = gelu_tanh_f(m[j])
        _qmatvec(W2[l], m, b2[l], u)
        for j in range(d):
            xs[j] += u[j]
    _rms(xs, nf, z)
    _qmatvec(Wout, z, bout, out)


@nb.njit(cache=True, fastmath=True)
def cls_point(logits, v, out):
    """Tick-class head -> metric-optimal point per target (mirrors wnn.nets._Net._out)."""
    K = v.shape[0]
    for t in range(2):
        mx = logits[t * K]
        for c in range(1, K):
            mx = max(mx, logits[t * K + c])
        z = np.float32(0.0)
        a = np.float32(0.0)
        b = np.float32(0.0)
        for c in range(K):
            e = np.float32(math.exp(logits[t * K + c] - mx))
            av = abs(v[c])
            z += e
            a += e * av * v[c]
            b += e * av
        out[t] = (a / z) / max(b / z, np.float32(1e-6))


def export_mingru(model) -> dict:
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    L = len(model.blocks)
    st = lambda f: np.stack([sd[f.format(l)] for l in range(L)])  # noqa: E731
    Win, bin_ = sd["inp.weight"], sd["inp.bias"]
    if "mu" in sd:  # fold (x - mu) / sd into the input layer
        Win = (Win / sd["sd"][None, :]).astype(np.float32)
        bin_ = (bin_ - Win.astype(np.float64) @ sd["mu"].astype(np.float64)).astype(np.float32)
    return {"Win": Win, "bin": bin_,
            "n1": st("blocks.{}.n1.weight"), "Wg": st("blocks.{}.gate.weight"), "bg": st("blocks.{}.gate.bias"),
            "Wc": st("blocks.{}.cand.weight"), "bc": st("blocks.{}.cand.bias"), "n2": st("blocks.{}.n2.weight"),
            "W1": st("blocks.{}.mlp.0.weight"), "b1": st("blocks.{}.mlp.0.bias"),
            "W2": st("blocks.{}.mlp.2.weight"), "b2": st("blocks.{}.mlp.2.bias"),
            "nf": sd["norm.weight"], "Wout": sd["head.weight"], "bout": sd["head.bias"],
            **({"cls_v": sd["cls_v"]} if "cls_v" in sd else {})}


class MinGRURunner:
    """Standalone runner (tests). spans: the model's ema_feats spans, () for plain models."""

    def __init__(self, w: dict, spans=(), bits=32):
        self.cls_v = np.asarray(w.get("cls_v", np.zeros(0)), np.float32)
        w = quantize(w, QPAIRS, bits)
        self.w = tuple(w[k] if w[k].dtype.kind == "i" else np.ascontiguousarray(w[k], dtype=np.float32) for k in KEYS)
        L, d = self.w[3].shape[0], self.w[3].shape[1]
        n_in = self.w[0].shape[1]
        self.H = np.zeros((L, d), np.float32)
        n_raw = n_in // (2 + len(spans))
        self.prev = np.zeros(n_raw, np.float32)
        self.inp = np.zeros(n_in, np.float32)
        from . import ema_feats
        self.ema = (ema_feats.alphas(spans), np.zeros((len(spans), n_raw)), np.zeros((len(spans), n_raw)),
                    np.zeros(len(spans) * n_raw, np.float32))
        self.t = 0
        self.tmp = (np.zeros(d, np.float32), np.zeros(d, np.float32), np.zeros(d, np.float32),
                    np.zeros(d, np.float32), np.zeros(self.w[8].shape[1], np.float32), np.zeros(self.w[13].shape[0], np.float32))
        self.out = np.zeros(2, np.float32)
        self.first = True

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
        if self.cls_v.shape[0]:
            cls_point(self.tmp[-1], self.cls_v, self.out)
            return self.out
        return self.tmp[-1]
