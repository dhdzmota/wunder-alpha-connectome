"""Fused single-call runtime for one route: minGRU + SSM + TCN members and their blend, one numba call per row.

Separate numba calls per member cost ~0.1 us per array argument plus numpy glue; here every weight and state
array lives in one jitclass (`_holder`), so a Python call unboxes 3 objects instead of ~70.

model.npz layout (written by `pack`):
  mg_<KEY>, ssm_<KEY>, tcn_<KEY>  member weights stacked on a leading axis (n_mingru / n_ssm / n_tcn members;
                                  all members of one kind share a shape; a stack may be empty)
  mg_cls_v, tcn_cls_v             optional tick-class head values (shared by the members of that kind)
  ema_spans                       spans of the causal normalisation features (wnn.ema_feats), computed once per row
  quant_bits                      16: matrices are row-scaled int16 at load time (wnn.quant)
  blend (n_members, 2), bias (2)  output = bias + sum_m blend[m] * member_m, member order minGRUs, SSMs, TCNs
"""
from __future__ import annotations

import numba as nb
import numpy as np

from . import ema_feats
from .mingru_infer import KEYS as MG_KEYS
from .mingru_infer import QPAIRS as MG_QPAIRS
from .mingru_infer import cls_point
from .mingru_infer import step as mg_step
from .quant import quantize
from .ssm_infer import KEYS as SSM_KEYS
from .ssm_infer import QPAIRS as SSM_QPAIRS
from .ssm_infer import step as ssm_step
from .tcn_infer import KEYS as TCN_KEYS
from .tcn_infer import QPAIRS as TCN_QPAIRS
from .tcn_infer import step as tcn_step

N_RAW = 112


@nb.njit(cache=True)
def step(x_in, first, ema_al, ema_m, ema_v, extra, tcount, xd, x32, inp, gout,
         mprev, MH, mWin, mbin, mn1, mWg, mbg, mWc, mbc, mn2, mW1, mb1, mW2, mb2, mnf, mWout, mbout,
         mxs, mz, mk, mu, mm, mlog, mclsv,
         sprev, SH, sWin, sbin, sn1, sWp, sbp, sexpA, sDsk, sWo, sbo, sn2, sW1, sb1, sW2, sb2, snf, sWout, sbout,
         sxs, sz, spj, sv, sm,
         tprev, TR, tcnt, tWin, tbin, tn1, tWc, tbc, tWo, tbo, tnf, tWout, tbout, txs, tz, tc2, ta, tlog, tclsv,
         blend, bias, out):
    """One row. x_in: raw (112,) row (any float dtype); first: first row of a sequence (resets the state).
    Writes the blended (2,) prediction to out."""
    n = x_in.shape[0]
    for i in range(n):
        xd[i] = x_in[i]
        x32[i] = np.float32(x_in[i])
    out[0] = bias[0]
    out[1] = bias[1]
    if ema_al.shape[0]:
        if first:
            ema_m[:] = 0.0
            ema_v[:] = 0.0
            tcount[0] = 0
        ema_feats.update(xd, tcount[0], ema_al, ema_m, ema_v, extra)
        tcount[0] += 1
    c = 0
    for g in range(MH.shape[0]):
        if first:
            MH[g, :, :] = 0.0
        mg_step(x32, extra, mprev[g], first, inp, MH[g], mWin[g], mbin[g], mn1[g], mWg[g], mbg[g], mWc[g], mbc[g],
                mn2[g], mW1[g], mb1[g], mW2[g], mb2[g], mnf[g], mWout[g], mbout[g], mxs, mz, mk, mu, mm, mlog)
        if mclsv.shape[0]:  # tick-class head
            cls_point(mlog, mclsv, gout)
        else:
            gout[0] = mlog[0]
            gout[1] = mlog[1]
        out[0] += blend[c + g, 0] * gout[0]
        out[1] += blend[c + g, 1] * gout[1]
    c += MH.shape[0]
    for g in range(SH.shape[0]):
        if first:
            SH[g, :, :, :] = 0.0
        ssm_step(x32, extra, sprev[g], first, inp, SH[g], sWin[g], sbin[g], sn1[g], sWp[g], sbp[g], sexpA[g], sDsk[g],
                 sWo[g], sbo[g], sn2[g], sW1[g], sb1[g], sW2[g], sb2[g], snf[g], sWout[g], sbout[g],
                 sxs, sz, spj, sv, sm, gout)
        out[0] += blend[c + g, 0] * gout[0]
        out[1] += blend[c + g, 1] * gout[1]
    c += SH.shape[0]
    for g in range(TR.shape[0]):
        tcn_step(x32, extra, tprev[g], first, inp, TR[g], tcnt[g], tWin[g], tbin[g], tn1[g], tWc[g], tbc[g],
                 tWo[g], tbo[g], tnf[g], tWout[g], tbout[g], txs, tz, tc2, ta, tlog)
        if tclsv.shape[0]:  # tick-class head
            cls_point(tlog, tclsv, gout)
        else:
            gout[0] = tlog[0]
            gout[1] = tlog[1]
        out[0] += blend[c + g, 0] * gout[0]
        out[1] += blend[c + g, 1] * gout[1]


def _stack(z, prefix, keys, pairs, bits):
    w = quantize({k: z[prefix + k] for k in keys}, pairs, bits)
    return tuple(w[k] if w[k].dtype.kind == "i" else np.ascontiguousarray(w[k], dtype=np.float32) for k in keys)


def _empty(ndims):
    """Empty member stack with the right ndims: the member loop never runs, the numba signature stays the same."""
    return tuple(np.zeros((0,) + (1,) * (d - 1), np.float32) for d in ndims)


class FusedModel:
    """Owns weights + state; call step(row, first) once per row and read the returned float32[2]."""

    def __init__(self, z):
        files = set(z.files if hasattr(z, "files") else z.keys())
        bits = int(z["quant_bits"]) if "quant_bits" in files else 32
        n_mg, n_ssm, n_tcn = (int(z[k]) if k in files else 0 for k in ("n_mingru", "n_ssm", "n_tcn"))
        self.mw = _stack(z, "mg_", MG_KEYS, MG_QPAIRS, bits) if n_mg else _empty((3, 2, 3, 4, 3, 4, 3, 3, 4, 3, 4, 3, 2, 3, 2))
        self.sw = _stack(z, "ssm_", SSM_KEYS, SSM_QPAIRS, bits) if n_ssm else _empty((3, 2, 3, 4, 3, 4, 3, 4, 3, 3, 4, 3, 4, 3, 2, 3, 2))
        self.tw = _stack(z, "tcn_", TCN_KEYS, TCN_QPAIRS, bits) if n_tcn else _empty((3, 2, 3, 4, 3, 4, 3, 2, 3, 2))
        M, ml, md = self.mw[3].shape[0], self.mw[3].shape[1], self.mw[3].shape[2]
        mh = self.mw[8].shape[2] if n_mg else 1
        Sn, sl, sd_, sN = self.sw[5].shape
        Tn, tl_, td = self.tw[2].shape
        self.blend = np.ascontiguousarray(z["blend"], dtype=np.float64)
        self.bias = np.ascontiguousarray(z["bias"], dtype=np.float64)
        assert self.blend.shape == (M + Sn + Tn, 2), (self.blend.shape, M, Sn, Tn)
        spans = [float(v) for v in z["ema_spans"]] if "ema_spans" in files else []
        n_raw = N_RAW
        self.ema = (ema_feats.alphas(spans), np.zeros((len(spans), n_raw)), np.zeros((len(spans), n_raw)),
                    np.zeros(len(spans) * n_raw, np.float32), np.zeros(1, np.int64))
        self.buf = (np.zeros(n_raw), np.zeros(n_raw, np.float32), np.zeros((2 + len(spans)) * n_raw, np.float32),
                    np.zeros(2, np.float32))
        self.mstate = (np.zeros((M, n_raw), np.float32), np.zeros((M, ml, md), np.float32))
        self.mtmp = tuple(np.zeros(n, np.float32) for n in (md, md, md, md, mh))
        self.mcls = (np.zeros(self.mw[13].shape[1] if n_mg else 2, np.float32),
                     np.ascontiguousarray(z["mg_cls_v"], dtype=np.float32) if "mg_cls_v" in files else np.zeros(0, np.float32))
        self.sstate = (np.zeros((Sn, n_raw), np.float32), np.zeros((Sn, sl, sd_, sN), np.float32))
        self.stmp = tuple(np.zeros(n, np.float32) for n in (sd_, sd_, self.sw[3].shape[2], sd_, self.sw[10].shape[2]))
        self.tstate = (np.zeros((Tn, n_raw), np.float32), np.zeros((Tn, (1 << tl_) - 1, td), np.float32),
                       np.zeros((Tn, 1), np.int64))
        self.ttmp = (np.zeros(td, np.float32), np.zeros(td, np.float32), np.zeros(2 * td, np.float32),
                     np.zeros(2 * td, np.float32))
        self.tcls = (np.zeros(self.tw[8].shape[1] if n_tcn else 2, np.float32),
                     np.ascontiguousarray(z["tcn_cls_v"], dtype=np.float32) if "tcn_cls_v" in files else np.zeros(0, np.float32))
        self.out = np.zeros(2, np.float32)
        self.h = _holder((*self.ema, *self.buf, *self.mstate, *self.mw, *self.mtmp, *self.mcls,
                          *self.sstate, *self.sw, *self.stmp, *self.tstate, *self.tw, *self.ttmp, *self.tcls,
                          self.blend, self.bias, self.out))

    def step(self, x, first):
        return self.h.step(x, first)


_HOLDERS = {}  # numba field types -> compiled jitclass, shared by models with the same array types (startup time)


def _holder(args):
    """jitclass that owns every array of step(); its fields reference the same numpy buffers (no copies)."""
    from numba.experimental import jitclass
    key = tuple(nb.typeof(a) for a in args)
    if key in _HOLDERS:
        return _HOLDERS[key](*args)
    names = [f"a{i}" for i in range(len(args))]
    src = ("class Holder:\n    def __init__(self, " + ", ".join(names) + "):\n"
           + "".join(f"        self.{n} = {n}\n" for n in names)
           + "    def step(self, x, first):\n"
           + "        step(x, first, " + ", ".join(f"self.{n}" for n in names) + ")\n"
           + f"        return self.{names[-1]}\n")
    ns = {"step": step}
    exec(src, ns)
    _HOLDERS[key] = cls = jitclass(list(zip(names, key)))(ns["Holder"])
    return cls(*args)


def pack(mingrus=(), ssms=(), tcns=(), blend=None, bias=None, ema_spans=(), quant_bits=32) -> dict:
    """Arrays for model.npz from per-member export dicts (wnn.*_infer.export_*)."""
    arrays = {}
    for prefix, count, members, keys in (("mg_", "n_mingru", mingrus, MG_KEYS), ("ssm_", "n_ssm", ssms, SSM_KEYS),
                                         ("tcn_", "n_tcn", tcns, TCN_KEYS)):
        arrays[count] = np.array(len(members))
        for k in keys if members else ():
            arrays[prefix + k] = np.stack([g[k] for g in members]).astype(np.float32)
        cls = [g.get("cls_v") for g in members]
        if members and cls[0] is not None and prefix != "ssm_":  # one shared class-value vector per member kind
            assert all(c is not None and np.array_equal(c, cls[0]) for c in cls), f"stacked {prefix} members must share cls_v"
            arrays[prefix + "cls_v"] = np.asarray(cls[0], np.float32)
    if len(ema_spans):
        arrays["ema_spans"] = np.asarray(ema_spans, np.float64)
    if quant_bits != 32:
        arrays["quant_bits"] = np.array(quant_bits)
    arrays["blend"] = np.asarray(blend, np.float64)
    arrays["bias"] = np.zeros(2) if bias is None else np.asarray(bias, np.float64)
    return arrays
