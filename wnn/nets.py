"""PyTorch models, inputs and losses used in training (the submission runs the numba mirrors in wnn/*_infer.py).

Every model maps a chunk of rows (B, T, n_in) and a carried state to per-row outputs, so full 20k-row sequences
are trained statefully with truncated BPTT (`run_sequences`). Input per row: [raw x (112), x - x_prev (112),
extra features] -> Linear -> stack -> head. Members of the final blend:
  MinGRUNet  2 x 128 minGRU blocks (+ MLP), with a tick-class head on the regime specialists
  SSMNet     2 x 128 selective state-space blocks (Mamba-style, no conv), state 16
  TCNNet     10 x 64 gated dilated causal convolutions (receptive field 1024 rows)
  GRUNet     2 x 128 GRU: the training-only look-ahead teacher (its extra inputs read future rows)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_RAW = 112
# per-sequence target tick unit (y = k * unit); regime index = searchsorted(REGIME_EDGES, unit)
REGIME_UNITS = (0.125433, 0.241330, 0.468540, 0.656870)
REGIME_EDGES = (0.2, 0.35, 0.55)


# ---------------------------------------------------------------------------------------------------- inputs

def with_feats(X, feats):
    """Raw rows + extra input columns. feats: tuple of EMA spans (wnn.ema_feats, causal per-sequence normalisation)
    and optionally the entry "fut" (wnn.fut_feats look-ahead columns: teacher only, never at inference)."""
    from wnn import ema_feats
    spans = tuple(f for f in feats if not isinstance(f, str))
    parts = [X]
    if spans:
        parts.append(ema_feats.compute_sequence(X[:, :N_RAW], spans))
    if "fut" in feats:
        from wnn import fut_feats
        parts.append(fut_feats.compute_sequence(X[:, :N_RAW]))
    return np.concatenate(parts, 1)


def n_feats(feats):
    if not feats:
        return 0
    from wnn.fut_feats import N_FUT
    return N_FUT * ("fut" in feats) + N_RAW * len([f for f in feats if not isinstance(f, str)])


def add_diff(xc, prev):
    """xc: (B, T, 112 [+ feats]) chunk; prev: last row of the previous chunk or None (sequence start).
    Output: [raw, raw diff, feats] (diffs of the raw 112 columns only)."""
    r = xc[..., :N_RAW]
    shifted = torch.cat([r[:, :1] if prev is None else prev[:, None, :N_RAW], r[:, :-1]], 1)
    return torch.cat([r, r - shifted, xc[..., N_RAW:]], -1)


# ------------------------------------------------------------------------------------------- scan + blocks

def linear_scan(log_a, b, h0, sub=256):
    """h_t = exp(log_a_t) * h_{t-1} + b_t over dim 1, log_a <= 0. Exact in exact arithmetic;
    log-space within blocks of `sub` steps (signs of b handled by a +/- split) keeps it stable."""
    outs, h = [], h0
    for s in range(0, log_a.shape[1], sub):
        la, bb = log_a[:, s:s + sub], b[:, s:s + sub]
        cum = torch.cumsum(la, 1)
        lp = torch.logcumsumexp(torch.log(bb.clamp_min(1e-30)) - cum, 1)
        ln = torch.logcumsumexp(torch.log((-bb).clamp_min(1e-30)) - cum, 1)
        hs = torch.exp(cum + lp) - torch.exp(cum + ln) + torch.exp(cum) * h[:, None]
        outs.append(hs)
        h = hs[:, -1]
    return torch.cat(outs, 1)


try:  # exact sequential scan on GPU: one program per (batch, channel block), loop over time
    import triton
    import triton.language as tl

    @triton.jit
    def _scan_fwd(A, Bv, H0, Hout, T, D, BLOCK: tl.constexpr):
        b, cb = tl.program_id(0), tl.program_id(1)
        d = cb * BLOCK + tl.arange(0, BLOCK)
        m = d < D
        h = tl.load(H0 + b * D + d, mask=m, other=0.0)
        base = b * T * D
        for t in range(T):
            o = base + t * D + d
            h = tl.load(A + o, mask=m, other=0.0) * h + tl.load(Bv + o, mask=m, other=0.0)
            tl.store(Hout + o, h, mask=m)

    @triton.jit
    def _scan_bwd(A, H, H0, G, dA, dB, dH0, T, D, BLOCK: tl.constexpr):
        b, cb = tl.program_id(0), tl.program_id(1)
        d = cb * BLOCK + tl.arange(0, BLOCK)
        m = d < D
        base = b * T * D
        carry = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(T):
            t = T - 1 - i
            o = base + t * D + d
            g = tl.load(G + o, mask=m, other=0.0) + carry
            tl.store(dB + o, g, mask=m)
            if t > 0:
                hp = tl.load(H + o - D, mask=m, other=0.0)
            else:
                hp = tl.load(H0 + b * D + d, mask=m, other=0.0)
            tl.store(dA + o, g * hp, mask=m)
            carry = tl.load(A + o, mask=m, other=0.0) * g
        tl.store(dH0 + b * D + d, carry, mask=m)

    class _TritonScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b, h0):
            a, b, h0 = a.contiguous().float(), b.contiguous().float(), h0.contiguous().float()
            B, T, D = a.shape
            h = torch.empty_like(a)
            grid = (B, triton.cdiv(D, 64))
            _scan_fwd[grid](a, b, h0, h, T, D, BLOCK=64, num_warps=1)
            ctx.save_for_backward(a, h, h0)
            return h

        @staticmethod
        def backward(ctx, g):
            a, h, h0 = ctx.saved_tensors
            B, T, D = a.shape
            da, db, dh0 = torch.empty_like(a), torch.empty_like(a), torch.empty_like(h0)
            _scan_bwd[(B, triton.cdiv(D, 64))](a, h, h0, g.contiguous().float(), da, db, dh0, T, D, BLOCK=64, num_warps=1)
            return da, db, dh0

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def gated_scan(a, b, h0):
    """h_t = a_t * h_{t-1} + b_t with 0 < a < 1. Triton on GPU, log-space blocks otherwise."""
    if HAVE_TRITON and a.is_cuda:
        return _TritonScan.apply(a, b, h0)
    return linear_scan(torch.log(a), b, h0)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MinGRUBlock(nn.Module):
    """x + minGRU(norm(x)); then x + MLP(norm(x)). minGRU: a = sigmoid(Wa z), h = a h + (1-a) Wu z."""

    def __init__(self, d, mlp_mult=2):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.gate, self.cand = nn.Linear(d, d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(approximate="tanh"), nn.Linear(mlp_mult * d, d))

    def forward(self, x, h0):
        z = self.n1(x)
        k = self.gate(z)
        h = gated_scan(torch.sigmoid(k), torch.sigmoid(-k) * self.cand(z), h0)
        x = x + h
        return x + self.mlp(self.n2(x)), h[:, -1]


class SSMBlock(nn.Module):
    """Selective state-space block (Mamba-style, no conv): per channel an N-dim state with
    input-dependent step size and input/output projections.
      dt = softplus(W_dt z + b), a = exp(-dt * exp(A_log))      (D, N) decay per channel x state
      h_t = a * h_{t-1} + (dt * u) outer B_t                     B_t, C_t = W_B z, W_C z  (N,)
      y = h_t . C_t + D_skip * u;  x += W_out(y * silu(g));  then x + MLP(norm(x))
    The (B, T, D, N) scan reuses the exact sequential gated_scan on D*N channels."""

    def __init__(self, d, n_state=8, mlp_mult=2):
        super().__init__()
        self.d, self.n = d, n_state
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.in_proj = nn.Linear(d, 3 * d + 2 * n_state)  # u, g, dt, B, C
        dt = torch.exp(torch.empty(d).uniform_(math.log(1e-3), math.log(0.3)))  # Mamba dt init
        with torch.no_grad():
            self.in_proj.bias[2 * d:3 * d] = dt + torch.log(-torch.expm1(-dt))  # softplus^-1
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_state + 1, dtype=torch.float32)).repeat(d, 1))
        self.D_skip = nn.Parameter(torch.ones(d))
        self.out_proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(approximate="tanh"), nn.Linear(mlp_mult * d, d))

    def forward(self, x, h0):
        B, L, d = x.shape
        N = self.n
        u, g, dt, Bm, Cm = self.in_proj(self.n1(x)).split([d, d, d, N, N], -1)
        dt = F.softplus(dt)
        a = torch.exp(-dt[..., None] * torch.exp(self.A_log))  # (B, L, d, N)
        b = (dt * u)[..., None] * Bm[:, :, None, :]
        h = gated_scan(a.reshape(B, L, d * N), b.reshape(B, L, d * N), h0.reshape(B, d * N)).reshape(B, L, d, N)
        y = (h * Cm[:, :, None, :]).sum(-1) + self.D_skip * u
        x = x + self.out_proj(y * F.silu(g))
        return x + self.mlp(self.n2(x)), h[:, -1]


class TCNBlock(nn.Module):
    """WaveNet-style gated causal conv (kernel 2, dilation dil) on the normalised input:
      z = norm(x);  [f, g] = W [z_{t-dil}, z_t] + b;  x += W_out(tanh(f) * sigmoid(g))
    The state is the ring of the last dil values of z (zeros before the sequence start)."""

    def __init__(self, d, dil):
        super().__init__()
        self.dil = dil
        self.n1 = RMSNorm(d)
        self.conv = nn.Linear(2 * d, 2 * d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, cache):
        z = self.n1(x)
        zc = torch.cat([cache.to(z.dtype), z], 1)
        f, g = self.conv(torch.cat([zc[:, :z.shape[1]], z], -1)).chunk(2, -1)
        return x + self.out_proj(torch.tanh(f) * torch.sigmoid(g)), zc[:, -self.dil:]


# ---------------------------------------------------------------------------------------------------- models

class _Net(nn.Module):
    """Shared input layer (with optional input normalisation of the extra features) and output head."""

    def _setup(self, n_in, hidden, norm):
        self.inp = nn.Linear(n_in, hidden)
        self.head = nn.Linear(hidden, 2)
        if norm is not None:
            self.register_buffer("mu", torch.as_tensor(norm[0], dtype=torch.float32))
            self.register_buffer("sd", torch.as_tensor(norm[1], dtype=torch.float32))
        else:
            self.mu = None

    def _in(self, x):
        return self.inp(x if self.mu is None else (x - self.mu) / self.sd)

    def set_class_head(self, unit):
        """Distributional head: per target, softmax over tick counts k in [-K, K] (K = ceil(2/unit), the
        end classes hold |k| >= K) with values clip(k*unit, -2, 2). The prediction is the metric-optimal
        point E[|y| y] / E[|y|] under the predicted distribution; logits follow it while training."""
        self.cls_k = math.ceil(2.0 / unit - 1e-6)
        self.cls_unit = unit
        n = 2 * self.cls_k + 1
        self.head = nn.Linear(self.head.in_features, 2 * n)
        k = torch.arange(-self.cls_k, self.cls_k + 1, dtype=torch.float32)
        self.register_buffer("cls_v", (k * unit).clamp(-2, 2))

    def _out(self, z):
        out = self.head(z)
        if getattr(self, "cls_k", 0):
            n = 2 * self.cls_k + 1
            logits = out.unflatten(-1, (2, n)).float()
            pi = logits.softmax(-1)
            av = self.cls_v.abs()
            point = (pi * av * self.cls_v).sum(-1) / (pi * av).sum(-1).clamp_min(1e-6)
            out = torch.cat([point, logits.flatten(-2)], -1) if self.training else point
        return out


class GRUNet(_Net):
    def __init__(self, n_in=224, hidden=128, layers=2, norm=None):
        super().__init__()
        self._setup(n_in, hidden, norm)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)

    def forward(self, x, h=None):
        z, h = self.gru(self._in(x), h)
        return self._out(z), h


class MinGRUNet(_Net):
    def __init__(self, n_in=224, hidden=128, layers=2, norm=None):
        super().__init__()
        self._setup(n_in, hidden, norm)
        self.blocks = nn.ModuleList(MinGRUBlock(hidden) for _ in range(layers))
        self.norm = RMSNorm(hidden)

    def forward(self, x, h=None):
        x = self._in(x)
        if h is None:
            h = x.new_zeros(len(self.blocks), x.shape[0], x.shape[-1])
        hs = []
        for l, blk in enumerate(self.blocks):
            x, hl = blk(x, h[l])
            hs.append(hl)
        return self._out(self.norm(x)), torch.stack(hs)


class SSMNet(_Net):
    def __init__(self, n_in=224, hidden=128, layers=2, norm=None, n_state=8):
        super().__init__()
        self._setup(n_in, hidden, norm)
        self.blocks = nn.ModuleList(SSMBlock(hidden, n_state) for _ in range(layers))
        self.norm = RMSNorm(hidden)
        self.n_state = n_state

    def forward(self, x, h=None):
        x = self._in(x)
        if h is None:
            h = x.new_zeros(len(self.blocks), x.shape[0], x.shape[-1], self.n_state)
        hs = []
        for l, blk in enumerate(self.blocks):
            x, hl = blk(x, h[l])
            hs.append(hl)
        return self._out(self.norm(x)), torch.stack(hs)


class TCNNet(_Net):
    """Stack of dilated gated convs, dilations 1, 2, 4, ... (receptive field 2^layers rows). No recurrence:
    a finite-window view of the sequence, a decorrelated ensemble member next to minGRU / SSM."""

    def __init__(self, n_in=224, hidden=64, layers=10, norm=None):
        super().__init__()
        self._setup(n_in, hidden, norm)
        self.blocks = nn.ModuleList(TCNBlock(hidden, 2 ** l) for l in range(layers))
        self.norm = RMSNorm(hidden)

    def forward(self, x, h=None):
        x = self._in(x)
        if h is None:
            h = tuple(x.new_zeros(x.shape[0], b.dil, x.shape[-1]) for b in self.blocks)
        hs = []
        for blk, c in zip(self.blocks, h):
            x, c = blk(x, c)
            hs.append(c)
        return self._out(self.norm(x)), tuple(hs)


def make_model(arch, hidden, layers, n_in=224, norm=None, n_state=8):
    if arch == "ssm":
        return SSMNet(n_in, hidden, layers, norm=norm, n_state=n_state)
    if arch == "tcn":
        return TCNNet(n_in, hidden, layers, norm=norm)
    return {"gru": GRUNet, "mingru": MinGRUNet}[arch](n_in, hidden, layers, norm=norm)


# ---------------------------------------------------------------------------------------------------- losses

def wcorr_loss(out, y, w):
    """1 - mean over targets of the |y|-weighted Pearson on this chunk (metric surrogate)."""
    w = w.expand_as(y) if w.shape[-1] == 1 else w
    s = w.sum((0, 1)).clamp_min(1e-6)
    po = out - (w * out).sum((0, 1)) / s
    py = y - (w * y).sum((0, 1)) / s
    cov = (w * po * py).sum((0, 1))
    r = cov / ((w * po ** 2).sum((0, 1)) * (w * py ** 2).sum((0, 1))).clamp_min(1e-12).sqrt()
    return 1.0 - r.mean()


def _wloss(out, y, w, corr_w):
    loss = (w * (out - y) ** 2).sum() / w.sum().clamp_min(1e-6)
    return loss + corr_w * wcorr_loss(out, y, w) if corr_w else loss


def _ce_loss(model, out, y_raw, rw):
    """Cross-entropy of the class head (out[..., 2:] logits) on the tick class of the raw target,
    weighted by the row weights rw (B, T, 1) only: zero targets inform the distribution too."""
    K = model.cls_k
    cls = (torch.round(y_raw / model.cls_unit).clamp(-K, K) + K).long()
    logits = out[..., 2:2 + 2 * (2 * K + 1)].unflatten(-1, (2, 2 * K + 1))
    ce = F.cross_entropy(logits.flatten(0, -2), cls.flatten(), reduction="none").view_as(y_raw)
    return (rw * ce).sum() / (rw.sum() * 2).clamp_min(1e-6)


def _detach(h):
    return tuple(_detach(t) for t in h) if isinstance(h, tuple) else h.detach().float()


def run_sequences(model, X, chunk, dev, Y=None, RW=None, opt=None, corr_w=0.0, ce_w=0.0, soft_a=0.8, soft_floor=0.2):
    """Process full sequences chunk by chunk carrying the hidden state (truncated BPTT at chunk borders).
    Inference if opt is None (returns the (B, T, n_out) outputs). Training otherwise, one optimizer step per chunk:
      loss = weighted MSE + corr_w (1 - weighted Pearson) [+ ce_w class-head cross-entropy],
      row weights |clip(y)| * RW; with soft labels (Y columns 2:4 = teacher) the target is (1 - a) y + a teacher
      and the weights (|clip(y)| + floor) * RW, so rows with a zero target still learn the teacher."""
    B, T, _ = X.shape
    h, prev, preds, tot_loss = None, None, [], 0.0
    for a in range(0, T, chunk):
        n = min(chunk, T - a)
        xc = X[:, a:a + chunk].to(dev, non_blocking=True)
        inp = add_diff(xc, prev)
        prev = xc[:, n - 1]
        if opt is None:
            with torch.no_grad():
                out, h = model(inp, h)
            preds.append(out.float().cpu())
            continue
        out, h = model(inp, h)
        out = out.float()
        yc = Y[:, a:a + chunk].to(dev)
        y_raw = yc[..., :2]
        y = y_raw.clamp(-2, 2)
        rw = RW[:, a:a + chunk, None].to(dev)
        if yc.shape[-1] >= 4:
            q = yc[..., 2:4].clamp(-2, 2)
            loss = _wloss(out[..., :2], (1 - soft_a) * y + soft_a * q, (y.abs() + soft_floor) * rw, corr_w)
        else:
            loss = _wloss(out[..., :2], y, y.abs() * rw, corr_w)
        if ce_w:
            loss = loss + ce_w * _ce_loss(model, out, y_raw, rw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        h = _detach(h)
        tot_loss += loss.detach()
    return float(tot_loss) / math.ceil(T / chunk) if opt is not None else torch.cat(preds, 1)
