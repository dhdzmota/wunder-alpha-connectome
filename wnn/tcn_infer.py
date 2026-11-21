"""Single-row streaming inference of the dilated gated TCN in numba (mirrors wnn.nets.TCNNet).

Per block l (dilation 2^l):  z = rms(x) * n1;  [f, g] = Wc [z_{t-2^l}, z_t] + bc;  x = x + Wo (tanh(f) * sigmoid(g)) + bo
Output: head(rms(x) * nf), or with a tick-class head (cls_v) its logits -> mingru_infer.cls_point. Input row: [x, x - x_prev, extra] -> Linear, as for minGRU / SSM.
State: one ring of the last 2^l values of z per block, all rings in one (2^L - 1, d) array (block l at offset 2^l - 1),
plus the row counter t (ring slot of block l = t mod 2^l).
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .fastmath import sigmoid_f, tanh_f
from .quant import _qmatvec, quantize
from .mingru_infer import _rms, cls_point

KEYS = ("Win", "bin", "n1", "Wc", "bc", "Wo", "bo", "nf", "Wout", "bout")
# (weight, bias) pairs run through _qmatvec; their biases carry [b, row scale] (see quant.quantize)
QPAIRS = (("Win", "bin"), ("Wc", "bc"), ("Wo", "bo"), ("Wout", "bout"))


@nb.njit(cache=True, fastmath=True)
def step(x, extra, prev, first, inp, R, tcount, Win, bin_, n1, Wc, bc, Wo, bo, nf, Wout, bout, xs, z, c2, a, out):
    """x: (n_raw,) row; extra: ema features (used only if Win takes more than 2 * n_raw inputs);
    prev: previous row (updated); R: (2^L - 1, d) rings; tcount: (1,) row counter. Writes 2 predictions to out."""
    n = x.shape[0]
    if first:
        R[:, :] = 0.0
        tcount[0] = 0
    for i in range(n):
        inp[i] = x[i]
        inp[n + i] = 0.0 if first else x[i] - prev[i]
        prev[i] = x[i]
    for i in range(Win.shape[1] - 2 * n):
        inp[2 * n + i] = extra[i]
    _qmatvec(Win, inp, bin_, xs)
    d = xs.shape[0]
    t = tcount[0]
    for l in range(n1.shape[0]):
        dil = 1 << l
        slot = dil - 1 + t % dil
        _rms(xs, n1[l], z)
        for j in range(d):
            c2[j] = R[slot, j]
            c2[d + j] = z[j]
            R[slot, j] = z[j]
        _qmatvec(Wc[l], c2, bc[l], a)
        for j in range(d):
            a[j] = tanh_f(a[j]) * sigmoid_f(a[d + j])
        _qmatvec(Wo[l], a[:d], bo[l], z)
        for j in range(d):
            xs[j] += z[j]
    tcount[0] = t + 1
    _rms(xs, nf, z)
    _qmatvec(Wout, z, bout, out)


def export_tcn(model) -> dict:
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    L = len(model.blocks)
    st = lambda f: np.stack([sd[f.format(l)] for l in range(L)])  # noqa: E731
    Win, bin_ = sd["inp.weight"], sd["inp.bias"]
    if "mu" in sd:  # fold (x - mu) / sd into the input layer
        Win = (Win / sd["sd"][None, :]).astype(np.float32)
        bin_ = (bin_ - Win.astype(np.float64) @ sd["mu"].astype(np.float64)).astype(np.float32)
    return {"Win": Win, "bin": bin_, "n1": st("blocks.{}.n1.weight"),
            "Wc": st("blocks.{}.conv.weight"), "bc": st("blocks.{}.conv.bias"),
            "Wo": st("blocks.{}.out_proj.weight"), "bo": st("blocks.{}.out_proj.bias"),
            "nf": sd["norm.weight"], "Wout": sd["head.weight"], "bout": sd["head.bias"],
            **({"cls_v": sd["cls_v"]} if "cls_v" in sd else {})}


def buffers(d, n_out=2):
    """Scratch arrays for step: xs, z, c2, a, out (n_out = head rows: 2, or 2 * n_classes with a class head)."""
    return (np.zeros(d, np.float32), np.zeros(d, np.float32), np.zeros(2 * d, np.float32),
            np.zeros(2 * d, np.float32), np.zeros(n_out, np.float32))


class TCNRunner:
    """Standalone runner (tests). spans: the model's ema_feats spans, () for plain models."""

    def __init__(self, w: dict, spans=(), bits=32):
        from . import ema_feats
        self.cls_v = np.asarray(w.get("cls_v", np.zeros(0)), np.float32)
        w = quantize(w, QPAIRS, bits)
        self.w = tuple(w[k] if w[k].dtype.kind == "i" else np.ascontiguousarray(w[k], dtype=np.float32) for k in KEYS)
        L, d = self.w[2].shape
        n_in = self.w[0].shape[1]
        n_raw = n_in // (2 + len(spans))
        self.R = np.zeros(((1 << L) - 1, d), np.float32)
        self.tc = np.zeros(1, np.int64)
        self.prev = np.zeros(n_raw, np.float32)
        self.inp = np.zeros(n_in, np.float32)
        self.ema = (ema_feats.alphas(spans), np.zeros((len(spans), n_raw)), np.zeros((len(spans), n_raw)),
                    np.zeros(len(spans) * n_raw, np.float32))
        self.tmp = buffers(d, self.w[8].shape[0])
        self.out = np.zeros(2, np.float32)
        self.first, self.t = True, 0

    def reset(self):
        self.ema[1][:] = 0.0
        self.ema[2][:] = 0.0
        self.first, self.t = True, 0

    def step(self, x):
        from . import ema_feats
        al, m, v, extra = self.ema
        xd = np.asarray(x, np.float64)
        if al.shape[0]:
            ema_feats.update(xd, self.t, al, m, v, extra)
        step(x, extra, self.prev, self.first, self.inp, self.R, self.tc, *self.w, *self.tmp)
        self.first, self.t = False, self.t + 1
        if self.cls_v.shape[0]:
            cls_point(self.tmp[-1], self.cls_v, self.out)
            return self.out
        return self.tmp[-1]
