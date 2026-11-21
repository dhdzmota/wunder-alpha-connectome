"""Stateful GRU trained on full sequences with truncated BPTT.

- inputs: raw 112 features + first differences (computed on the fly, causal)
- loss: |clip(y)|-weighted MSE on clipped targets over required rows (the
  weighted Pearson metric rewards exactly these rows)
- train folds != ES_FOLD; select epoch on ES_FOLD (all required rows);
  report on the official valid set (mask + all rows)
- out-of-fold predictions for ES_FOLD and valid are saved for ensembling
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from wnn.data import ARTIFACTS, WARMUP, SequenceReader, load_splits
from utils import GlobalAccumulator

SEED = 20261007
ES_FOLD = 4
# per-sequence target tick unit (y = k * unit); regime index = searchsorted(REGIME_EDGES, unit)
AUX_EVAL = False  # True: aux heads also in eval (hindsight heads read at inference, EXPERIMENTS §39)
REGIME_UNITS = (0.125433, 0.241330, 0.468540, 0.656870)
REGIME_EDGES = (0.2, 0.35, 0.55)
AMP_DTYPE = None  # set from --amp; mixed precision is used for training forward passes only


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


PMASK_DIR = "pmask"  # artifacts subdir with the dense P(is_scored) arrays (--pmask-dir)


def valid_halves(n):
    """Fixed 50/50 split of valid sequences (half 0, half 1) shared by fine-tune and add-valid runs."""
    perm = np.random.default_rng(SEED + 7).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


class SeqDS(Dataset):
    """Yields (X, Y, third). third = scored mask, or with row_weights=True the per-row loss
    weight need * P(is_scored)^gamma (gamma=0 -> plain required-rows mask)."""

    def __init__(self, name, groups, row_weights=False, gamma=0.0, feats=False, mask_mix=None, aug=0.0, seq_w=None, soft=None,
                 soft_dist=None):
        self.name, self.groups, self.reader, self.mask_mix, self.aug = name, list(groups), None, mask_mix, aug
        self.soft_dist = soft_dist  # optional path of (n_seq * 20000, 2, n_cls) teacher tick-class probabilities -> appended to Y
        self.soft = soft  # optional path of (n_seq * 20000, 2) teacher predictions -> Y gets 4 columns [y, teacher]
        self.seq_w = seq_w  # optional per-sequence loss weight, indexed by group
        self.row_weights, self.gamma, self.pmask, self.feats = row_weights, gamma, None, feats

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        if self.reader is None:
            self.reader = SequenceReader(self.name)
        g = self.groups[i]
        s = self.reader.read(g)
        X = regime_aug(s.X, self.aug, np.random.default_rng([torch.initial_seed() % 2 ** 32, g])) if self.aug else s.X
        X = with_feats(X, self.feats) if self.feats else X
        if not self.row_weights:
            return torch.from_numpy(X), torch.from_numpy(s.Y), torch.from_numpy(s.scored)
        rw = (np.arange(s.X.shape[0]) >= WARMUP).astype(np.float32)
        if self.mask_mix is not None:  # valid rows: true is_scored known -> need * (1 + K * scored)
            rw *= 1.0 + self.mask_mix * s.scored
        elif self.gamma:
            if self.pmask is None:
                self.pmask = np.load(ARTIFACTS / PMASK_DIR / f"{self.name}_pmask.npy", mmap_mode="r")
            rw *= np.asarray(self.pmask[g], np.float32) ** self.gamma
        if self.seq_w is not None:
            rw *= self.seq_w[g]
        Y = s.Y
        if self.soft is not None:
            if not isinstance(self.soft, np.ndarray):
                self.soft = np.load(self.soft, mmap_mode="r").reshape(-1, 20000, 2)
            Y = np.concatenate([Y, np.asarray(self.soft[g], np.float32)], 1)
        if self.soft_dist is not None:
            if not isinstance(self.soft_dist, np.ndarray):
                d = np.load(self.soft_dist, mmap_mode="r")
                self.soft_dist = d.reshape(-1, 20000, d.shape[1] * d.shape[2])
            Y = np.concatenate([Y, np.asarray(self.soft_dist[g], np.float32)], 1)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(rw)


N_RAW = 112
# column groups of the raw row (instrument x price / volume / trade price / trade volume, additional a0..a7)
AUG_GROUPS = ((0, 22), (22, 44), (44, 48), (48, 52), (52, 74), (74, 96), (96, 100), (100, 104), (104, 112))
AUG_P = 0.8


def regime_aug(X, s, rng):
    """Per-sequence random regime shift (train sequences only): with prob AUG_P, every column group gets
    x <- a x + b with log a, b ~ N(0, s), plus a per-column jitter of half that size; sentinels (< -5) untouched.
    Makes the nets less dependent on the absolute level / scale of a period (train vs valid/test shift)."""
    if rng.random() >= AUG_P:
        return X
    a = np.ones(X.shape[1], np.float32); b = np.zeros(X.shape[1], np.float32)
    for lo, hi in AUG_GROUPS:
        a[lo:hi] = np.exp(rng.normal(0, s) + rng.normal(0, s / 2, hi - lo))
        b[lo:hi] = rng.normal(0, s) + rng.normal(0, s / 2, hi - lo)
    return np.where(X > -5, X * a + b, X).astype(np.float32)
NO_RAW = False  # --no-raw: zero the raw-level columns of the input (screening of shift-invariant inputs)
MOVE_W = 0.0  # --move-w: training-only binary "move" head (|k0 - k1| > 1) on the class-head state, BCE weight
SOFT_CONF = 0.0  # soft labels: extra row weight exp(-SOFT_CONF (clip y - teacher)^2) (down-weights labels the teacher calls noise)
SOFT_A = 0.8  # soft labels (Y with 4 columns: y, teacher): point-loss target (1 - a) clip(y) + a teacher
SOFT_FLOOR = 0.2  # ... with row weights (|clip(y)| + floor) * rw, so zero-target rows still learn the teacher
JUMP_P = 1.0  # --jump-p: point-loss row weights raised to this power (> 1: rows before multi-tick moves dominate)
KD_W = 0.0  # class head with a teacher distribution (Y columns 4:): CE target (1 - KD_W) one-hot tick + KD_W teacher probabilities
ROW_ENC = 0  # --row-enc D: per-row MLP encoder (n_in -> D, residual D -> D) before the recurrent stack (0 = one linear layer)


def with_feats(X, feats=True):
    """Raw row + extra input features. feats=True: the 276 engineered streaming features of the LightGBM
    model (wnn.features); feats=(spans...): causal per-sequence normalised columns (wnn.ema_feats), and the
    entry "ev" in the tuple appends the event-clock columns (wnn.event_feats) after them."""
    if feats is True:
        from wnn import features as fe
        return np.concatenate([X, fe.compute_sequence(X)[:, N_RAW:].astype(np.float32)], 1)
    from wnn import ema_feats, event_feats, rank_feats, repr_feats
    spans = tuple(f for f in feats if not isinstance(f, str))
    if "na" in feats:  # a0..a7 sentinels -> 0 before every derived input; indicators appended below
        X, miss = repr_feats.impute(X)
    parts = [X]
    if spans:
        parts.append(ema_feats.compute_sequence(X[:, :N_RAW], spans))
    if "ev" in feats:
        parts.append(event_feats.compute_sequence(X[:, :N_RAW]))
    if "evr" in feats:
        parts.append(event_feats.compute_sequence_rel(X[:, :N_RAW]))
    if "rk" in feats:
        parts.append(rank_feats.compute_sequence(X[:, :N_RAW]))
    if "na" in feats:
        parts.append(miss)
    if "pb" in feats:
        parts.append(repr_feats.ple(X[:, :N_RAW]))
    if "fut" in feats:  # look-ahead inputs: training-only teacher (wnn.fut_feats)
        from wnn import fut_feats
        parts.append(fut_feats.compute_sequence(X[:, :N_RAW]))
    return np.concatenate(parts, 1)


def n_feats(feats):
    if not feats:
        return 0
    if feats is True:
        return len(fe_names())
    from wnn.repr_feats import A_COLS, NBIN, PRICE_COLS
    from wnn.fut_feats import N_FUT
    return (N_FUT * ("fut" in feats) + N_RAW * (len([f for f in feats if not isinstance(f, str)]) + 2 * ("ev" in feats) + ("evr" in feats) + ("rk" in feats))
            + len(A_COLS) * ("na" in feats) + NBIN * len(PRICE_COLS) * ("pb" in feats))


class _Net(nn.Module):
    """Shared input normalisation (engineered features only) and training-only auxiliary head."""

    def _setup(self, n_in, hidden, n_aux, norm, attn=None):
        self.inp = RowEncoder(n_in, ROW_ENC, hidden) if ROW_ENC else nn.Linear(n_in, hidden)
        self.head = nn.Linear(hidden, 2)
        self.aux = nn.Linear(hidden, n_aux) if n_aux else None
        self.attn = LocalAttn(hidden, heads=attn[1], window=attn[0]) if attn else None
        if norm is not None:
            self.register_buffer("mu", torch.as_tensor(norm[0], dtype=torch.float32))
            self.register_buffer("sd", torch.as_tensor(norm[1], dtype=torch.float32))
        else:
            self.mu = None

    def _in(self, x):
        return self.inp(x if self.mu is None else (x - self.mu) / self.sd)

    def _split(self, h):
        """Recurrent state, attention cache (state is a pair when the attention block is on)."""
        if self.attn is None or h is None:
            return h, None
        return h

    def _attend(self, z, cache, hr):
        if self.attn is None:
            return z, hr
        z, cache = self.attn(z, cache)
        return z, (hr, cache)

    def set_class_head(self, unit, joint=False):
        """Distributional head: per target, softmax over tick counts k in [-K, K] (K = ceil(2/unit), the
        end classes hold |k| >= K) with values clip(k*unit, -2, 2). The prediction is the metric-optimal
        point E[|y| y] / E[|y|] under the predicted distribution; logits follow it while training.
        joint: one softmax over the (2K+1)^2 pairs (k0, k1); each target's point uses its marginal."""
        self.cls_k = math.ceil(2.0 / unit - 1e-6)
        self.cls_unit = unit
        self.cls_joint = joint
        n = 2 * self.cls_k + 1
        self.head = nn.Linear(self.head.in_features, n * n if joint else 2 * n)
        k = torch.arange(-self.cls_k, self.cls_k + 1, dtype=torch.float32)
        self.register_buffer("cls_v", (k * unit).clamp(-2, 2))
        self.move = nn.Linear(self.head.in_features, 1) if MOVE_W else None

    def _out(self, z):
        out = self.head(z)
        if getattr(self, "cls_k", 0):
            n = 2 * self.cls_k + 1
            if getattr(self, "cls_joint", False):
                logits = out.float()
                pj = logits.softmax(-1).unflatten(-1, (n, n))
                pi = torch.stack([pj.sum(-1), pj.sum(-2)], -2)
            else:
                logits = out.unflatten(-1, (2, n)).float()
                pi = logits.softmax(-1)
            av = self.cls_v.abs()
            point = (pi * av * self.cls_v).sum(-1) / (pi * av).sum(-1).clamp_min(1e-6)
            flat = logits if getattr(self, "cls_joint", False) else logits.flatten(-2)
            out = torch.cat([point, flat], -1) if self.training else point
            if self.training and getattr(self, "move", None) is not None:  # last column: move logit
                out = torch.cat([out, self.move(z).float()], -1)
        if self.aux is not None and (self.training or AUX_EVAL):
            out = torch.cat([out, self.aux(z)], -1)
        return out


class GRUNet(_Net):
    def __init__(self, n_in=224, hidden=128, layers=2, dropout=0.0, n_aux=0, norm=None, attn=None, slow_k=0):
        super().__init__()
        self._setup(n_in, hidden, n_aux, norm, attn)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.slow_k = slow_k
        if slow_k:  # slow branch: GRU over block means of z, updated once every slow_k rows (cost ~1/slow_k)
            assert attn is None
            self.slow = nn.GRU(hidden, hidden, batch_first=True)
            self.slow_out = nn.Linear(hidden, hidden)

    def forward(self, x, h=None):
        hs = None
        if self.slow_k and h is not None:
            h, hs = h
        h, cache = self._split(h)
        z, h = self.gru(self._in(x), h)
        z, h = self._attend(z, cache, h)
        if self.slow_k:  # rows of block j see the slow state after block j-1 (strictly causal)
            B, T, H = z.shape; k = self.slow_k
            assert T % k == 0, (T, k)
            hs = z.new_zeros(1, B, H) if hs is None else hs
            so, hs_new = self.slow(z.reshape(B, T // k, k, H).mean(2), hs)
            prev = torch.cat([hs.transpose(0, 1), so[:, :-1]], 1)
            z = z + self.slow_out(prev).repeat_interleave(k, 1)
            h = (h, hs_new)
        return self._out(z), h


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


class RowEncoder(nn.Module):
    """Cross-column interactions within a row: e = GELU(W0 x); e = e + W2 GELU(W1 norm(e)); out = W3 e."""

    def __init__(self, n_in, d, hidden):
        super().__init__()
        self.w0, self.n1 = nn.Linear(n_in, d), RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.out = nn.Linear(d, hidden)

    def forward(self, x):
        e = F.gelu(self.w0(x), approximate="tanh")
        return self.out(e + self.mlp(self.n1(e)))


class LocalAttn(nn.Module):
    """x + causal sliding-window attention(norm(x)); then x + MLP(norm(x)).

    Each step attends to itself and the previous window-1 steps. Position enters only through
    a learnable per-head linear distance penalty (ALiBi-style), so streaming inference needs
    just a ring buffer of the last `window` keys/values. Computed blockwise: queries of block j
    (size W) see the keys of blocks j-1 and j, masked to distance 0..W-1. The cache carries the
    normalised inputs of the last W steps across chunks (None at sequence start)."""

    def __init__(self, d, heads=4, window=128, mlp_mult=2):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.w = d, heads, window
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        slopes = torch.logspace(math.log10(0.25), math.log10(0.002), heads)  # short to long memory
        self.slope_raw = nn.Parameter(torch.log(torch.expm1(slopes)))  # softplus^-1
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(approximate="tanh"), nn.Linear(mlp_mult * d, d))

    def forward(self, x, cache=None):
        B, L, d = x.shape
        W, H, dh = self.w, self.h, self.d // self.h
        z = self.n1(x)
        zc = torch.cat([z.new_zeros(B, W, d) if cache is None else cache.to(z.dtype), z], 1)
        new_cache = zc[:, -W:]
        nb = -(-L // W)
        zc = F.pad(zc, (0, 0, 0, nb * W - L))  # (B, W + nb*W, d)
        q, k, v = self.qkv(zc).split(d, -1)
        q = q[:, W:].reshape(B, nb, W, H, dh).permute(0, 1, 3, 2, 4)  # (B, nb, H, W, dh)
        k = k.unfold(1, 2 * W, W).reshape(B, nb, H, dh, 2 * W)  # keys of blocks j-1, j
        v = v.unfold(1, 2 * W, W).reshape(B, nb, H, dh, 2 * W)
        s = (q @ k) / math.sqrt(dh)  # (B, nb, H, W, 2W)
        i = torch.arange(W, device=x.device)[:, None]
        m = torch.arange(2 * W, device=x.device)[None, :]
        dist = (i + W - m).float()  # query i of a block vs key m of its 2W window
        ok = (dist >= 0) & (dist < W)
        bias = -F.softplus(self.slope_raw)[:, None, None] * dist  # (H, W, 2W)
        bias = bias.masked_fill(~ok, float("-inf"))
        s = s + bias
        if cache is None:  # no history before the sequence start
            s[:, 0, :, :, :W] = float("-inf")
        att = torch.softmax(s, -1)
        o = (att @ v.transpose(-1, -2))  # (B, nb, H, W, dh)
        o = o.permute(0, 1, 3, 2, 4).reshape(B, nb * W, d)[:, :L]
        x = x + self.proj(o)
        return x + self.mlp(self.n2(x)), new_cache


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


class DeltaLayer(nn.Module):
    """Per-sequence fast-weight memory (DeltaNet delta rule), a residual layer on the hidden rows.
      k, q = L2-normalised W_k x, W_q x; v = W_v x; beta = sigmoid(W_b x)       per head (H heads, d = hidden / H)
      S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T;  o_t = S_t q_t;  x += W_o o   (W_o zero-init)
    Chunkwise-parallel form (Yang et al. 2024) over blocks of C rows; the state S (B, H, d, d) carries across chunks."""

    def __init__(self, hidden, heads=4, C=100):
        super().__init__()
        self.H, self.d, self.C = heads, hidden // heads, C
        self.norm = RMSNorm(hidden)
        self.qkvb = nn.Linear(hidden, 3 * hidden + heads)
        self.out = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, S=None):
        B, T, D = x.shape; H, d, C = self.H, self.d, self.C
        q, k, v, b = self.qkvb(self.norm(x)).float().split([D, D, D, H], -1)
        f = lambda z: z.view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        q, k, v = F.normalize(f(q), dim=-1), F.normalize(f(k), dim=-1), f(v)
        b = torch.sigmoid(b).transpose(1, 2)  # (B, H, T)
        pad = (-T) % C
        if pad:  # padded rows: k = v = beta = 0 leave the state unchanged
            q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v)); b = F.pad(b, (0, pad))
        n = (T + pad) // C
        q, k, v, b = q.view(B, H, n, C, d), k.view(B, H, n, C, d), v.view(B, H, n, C, d), b.view(B, H, n, C, 1)
        A = torch.eye(C, device=x.device) + (b * k @ k.transpose(-1, -2)).tril(-1)
        W = torch.linalg.solve_triangular(A, b * k, upper=False, unitriangular=True)   # T diag(b) K
        Ub = torch.linalg.solve_triangular(A, b * v, upper=False, unitriangular=True)  # T diag(b) V
        QK = (q @ k.transpose(-1, -2)).tril()
        S = x.new_zeros(B, H, d, d, dtype=torch.float32) if S is None else S.float()
        outs = []
        for i in range(n):
            U = Ub[:, :, i] - W[:, :, i] @ S.transpose(-1, -2)
            outs.append(q[:, :, i] @ S.transpose(-1, -2) + QK[:, :, i] @ U)
            S = S + U.transpose(-1, -2) @ k[:, :, i]
        o = torch.cat(outs, 2)[:, :, :T].transpose(1, 2).reshape(B, T, D)
        return x + self.out(o.to(x.dtype)), S


class MinGRUNet(_Net):
    def __init__(self, n_in=224, hidden=128, layers=2, n_aux=0, norm=None, attn=None, slow_k=0, delta=0):
        super().__init__()
        self._setup(n_in, hidden, n_aux, norm, attn)
        self.blocks = nn.ModuleList(MinGRUBlock(hidden) for _ in range(layers))
        self.norm = RMSNorm(hidden)
        self.slow_k = slow_k
        if slow_k:  # slow branch as in GRUNet: GRU over block means, updated once every slow_k rows
            assert attn is None
            self.slow = nn.GRU(hidden, hidden, batch_first=True)
            self.slow_out = nn.Linear(hidden, hidden)
        self.delta = DeltaLayer(hidden, delta) if delta else None

    def forward(self, x, h=None):
        slow = dS = None
        if self.slow_k and h is not None:
            h, slow = h
        if self.delta is not None and h is not None:
            h, dS = h
        h, cache = self._split(h)
        x = self._in(x)
        if h is None:
            h = x.new_zeros(len(self.blocks), x.shape[0], x.shape[-1])
        hs = []
        for l, blk in enumerate(self.blocks):
            x, hl = blk(x, h[l])
            hs.append(hl)
        x, h = self._attend(x, cache, torch.stack(hs))
        if self.slow_k:  # rows of block j see the slow state after block j-1 (strictly causal)
            B, T, H = x.shape; k = self.slow_k
            assert T % k == 0, (T, k)
            s0 = x.new_zeros(1, B, H) if slow is None else slow
            so, s_new = self.slow(x.reshape(B, T // k, k, H).mean(2), s0)
            prev = torch.cat([s0.transpose(0, 1), so[:, :-1]], 1)
            x = x + self.slow_out(prev).repeat_interleave(k, 1)
            h = (h, s_new)
        if self.delta is not None:
            x, dS = self.delta(x, dS)
            h = (h, dS)
        return self._out(self.norm(x)), h


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


class SSMNet(_Net):
    def __init__(self, n_in=224, hidden=128, layers=2, n_aux=0, norm=None, attn=None, n_state=8):
        super().__init__()
        self._setup(n_in, hidden, n_aux, norm, attn)
        self.blocks = nn.ModuleList(SSMBlock(hidden, n_state) for _ in range(layers))
        self.norm = RMSNorm(hidden)
        self.n_state = n_state

    def forward(self, x, h=None):
        h, cache = self._split(h)
        x = self._in(x)
        if h is None:
            h = x.new_zeros(len(self.blocks), x.shape[0], x.shape[-1], self.n_state)
        hs = []
        for l, blk in enumerate(self.blocks):
            x, hl = blk(x, h[l])
            hs.append(hl)
        x, h = self._attend(x, cache, torch.stack(hs))
        return self._out(self.norm(x)), h


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


class TCNNet(_Net):
    """Stack of dilated gated convs, dilations 1, 2, 4, ... (receptive field 2^layers rows). No recurrence:
    a finite-window view of the sequence, meant as a decorrelated ensemble member next to minGRU / SSM."""

    def __init__(self, n_in=224, hidden=64, layers=10, n_aux=0, norm=None, attn=None):
        super().__init__()
        self._setup(n_in, hidden, n_aux, norm, None)
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


class LANet(_Net):
    """TRAINING-ONLY look-ahead teacher: causal GRU + gated dilated convolutions read backwards over the next rows
    (receptive field 2 ** len(LA_DIL) rows ahead, zero past the sequence end), added to the GRU output through a
    zero-initialised projection. run_sequences hands it each chunk extended by `lookahead` future rows."""
    LA_DIL = (1, 2, 4, 8, 16, 32)

    def __init__(self, n_in=224, hidden=128, layers=2, n_aux=0, norm=None, width=64):
        super().__init__()
        self._setup(n_in, hidden, n_aux, norm)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.lookahead = 2 ** len(self.LA_DIL)
        self.f_in = nn.Linear(n_in, width)
        self.f_conv = nn.ModuleList(nn.Conv1d(width, 2 * width, 2, dilation=d) for d in self.LA_DIL)
        self.f_res = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in self.LA_DIL)
        self.fuse = nn.Linear(width, hidden)
        nn.init.zeros_(self.fuse.weight); nn.init.zeros_(self.fuse.bias)

    def forward(self, x, h=None, n=None):
        n = x.shape[1] if n is None else n
        z, h = self.gru(self._in(x[:, :n]), h)
        xn = x if self.mu is None else (x - self.mu) / self.sd
        f = self.f_in(xn).flip(1).transpose(1, 2)  # reversed time: a causal conv here reads the future
        for conv, res, d in zip(self.f_conv, self.f_res, self.LA_DIL):
            a_, b_ = conv(F.pad(f, (d, 0))).chunk(2, 1)
            f = f + res(torch.tanh(a_) * torch.sigmoid(b_))
        f = f.transpose(1, 2).flip(1)[:, :n]
        return self._out(z + self.fuse(f)), h


def make_model(arch, hidden, layers, n_in=224, n_aux=0, norm=None, attn=None, n_state=8, slow_k=0, delta=0):
    if arch == "la":
        return LANet(n_in, hidden, layers, n_aux=n_aux, norm=norm)
    if arch == "ssm":
        return SSMNet(n_in, hidden, layers, n_aux=n_aux, norm=norm, attn=attn, n_state=n_state)
    if arch == "tcn":
        return TCNNet(n_in, hidden, layers, n_aux=n_aux, norm=norm)
    if delta:
        assert arch == "mingru" and not slow_k and attn is None, "--delta is implemented for the plain minGRU only"
        return MinGRUNet(n_in, hidden, layers, n_aux=n_aux, norm=norm, delta=delta)
    if slow_k:
        assert arch in ("gru", "mingru"), "--slow-k is implemented for the GRU and minGRU only"
        return {"gru": GRUNet, "mingru": MinGRUNet}[arch](n_in, hidden, layers, n_aux=n_aux, norm=norm, slow_k=slow_k)
    return {"gru": GRUNet, "mingru": MinGRUNet}[arch](n_in, hidden, layers, n_aux=n_aux, norm=norm, attn=attn)


def _detach(h):
    return tuple(_detach(t) for t in h) if isinstance(h, tuple) else h.detach().float()


def add_diff(xc, prev):
    """xc: (B, T, 112 [+ feats]) chunk; prev: last row of previous chunk or None (sequence start).
    Output: [raw, raw diff, feats] (diffs of the raw 112 columns only)."""
    r = xc[..., :N_RAW]
    shifted = torch.cat([r[:, :1] if prev is None else prev[:, None, :N_RAW], r[:, :-1]], 1)
    return torch.cat([torch.zeros_like(r) if NO_RAW else r, r - shifted, xc[..., N_RAW:]], -1)


def feat_norm(groups, n=48, seed=SEED, feats=True):
    """Mean / std of the model input (raw + diff + feats) on n train sequences; raw + diff left as is."""
    rd = SequenceReader("train")
    pick = np.random.default_rng(seed).choice(groups, n, replace=False)
    F = np.concatenate([with_feats(rd.read(g).X, feats)[WARMUP:, N_RAW:] for g in pick])
    mu = np.concatenate([np.zeros(2 * N_RAW), F.mean(0)]).astype(np.float32)
    sd = np.concatenate([np.ones(2 * N_RAW), np.maximum(F.std(0), 1e-3)]).astype(np.float32)
    return mu, sd


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


def _ce_loss(model, out, y_raw, rw, tdist=None):
    """Cross-entropy of the distributional head (out[..., 2:] logits) on the tick class of the raw target,
    weighted by the row weights rw (B, T, 1) only: zero targets inform the distribution too.
    tdist (B, T, 2 * n_cls): teacher class probabilities -> target (1 - KD_W) one-hot + KD_W teacher (per-target head)."""
    K = model.cls_k
    cls = (torch.round(y_raw / model.cls_unit).clamp(-K, K) + K).long()
    if tdist is not None and KD_W:
        assert not getattr(model, "cls_joint", False), "teacher distributions: per-target head only"
        n = 2 * K + 1
        logp = out[..., 2:2 + 2 * n].unflatten(-1, (2, n)).float().log_softmax(-1)
        tgt = (1 - KD_W) * F.one_hot(cls, n).float() + KD_W * tdist.unflatten(-1, (2, n))
        return (rw * -(tgt * logp).sum(-1)).sum() / (rw.sum() * 2).clamp_min(1e-6)
    if getattr(model, "cls_joint", False):
        n = 2 * K + 1
        logits = out[..., 2:2 + n * n]
        ce = F.cross_entropy(logits.flatten(0, -2), (cls[..., 0] * n + cls[..., 1]).flatten(), reduction="none")
        return (rw[..., 0] * ce.view_as(rw[..., 0])).sum() / (rw.sum() * 2).clamp_min(1e-6)  # same scale as the per-target mean
    logits = out[..., 2:2 + 2 * (2 * K + 1)].unflatten(-1, (2, 2 * K + 1))
    ce = F.cross_entropy(logits.flatten(0, -2), cls.flatten(), reduction="none").view_as(y_raw)
    return (rw * ce).sum() / (rw.sum() * 2).clamp_min(1e-6)


def _move_loss(model, out, y_raw, rw):
    """BCE of the move logit (last column) on |k0 - k1| > 1 (not a one-tick (0,-1)/(-1,0) pair), weighted by rw."""
    k = torch.round(y_raw / model.cls_unit)
    tgt = ((k[..., 0] - k[..., 1]).abs() > 1).float()
    bce = F.binary_cross_entropy_with_logits(out[..., -1], tgt, reduction="none")
    return (rw[..., 0] * bce).sum() / rw.sum().clamp_min(1e-6)


def run_sequences(model, X, chunk, dev, Y=None, RW=None, opt=None, scaler_clip=1.0, corr_w=0.0,
                  aux_shifts=(), aux_w=0.0, ce_w=0.0):
    """Process full sequences chunk by chunk carrying hidden state. Train if opt is given
    (RW: (B, T) per-row loss weights, multiplied by |y|; ce_w: weight of the class-head cross-entropy)."""
    B, T, _ = X.shape
    h, prev, preds, tot_loss = None, None, [], 0.0
    la = getattr(model, "lookahead", 0)  # look-ahead teacher: chunk + la future rows in, chunk rows out
    for a in range(0, T, chunk):
        n = min(chunk, T - a)
        xc = X[:, a:a + chunk + la].to(dev, non_blocking=True)
        inp = add_diff(xc, prev)
        prev = xc[:, n - 1]
        fwd = (lambda i, hh: model(i, hh, n)) if la else model
        if opt is not None:
            with torch.autocast("cuda", dtype=AMP_DTYPE or torch.float32, enabled=AMP_DTYPE is not None):
                out, h = fwd(inp, h)
            out = out.float()
            yc = Y[:, a:a + chunk].to(dev)
            y_raw = yc[..., :2]
            y = y_raw.clamp(-2, 2)
            rw = RW[:, a:a + chunk, None].to(dev)
            if yc.shape[-1] >= 4:  # soft labels from a teacher in columns 2:4 (then the --kd-dist class probabilities, if any)
                q = yc[..., 2:4].clamp(-2, 2)
                w = (y.abs() + SOFT_FLOOR) ** JUMP_P * rw * (torch.exp(-SOFT_CONF * (y - q) ** 2) if SOFT_CONF else 1.0)
                loss = _wloss(out[..., :2], (1 - SOFT_A) * y + SOFT_A * q, w, corr_w)
            else:
                loss = _wloss(out[..., :2], y, y.abs() ** JUMP_P * rw, corr_w)
            if ce_w:
                loss = loss + ce_w * _ce_loss(model, out, y_raw, rw, yc[..., 4:] if yc.shape[-1] > 4 else None)
                if MOVE_W and getattr(model, "move", None) is not None:
                    loss = loss + MOVE_W * _move_loss(model, out, y_raw, rw)
            if aux_shifts:  # targets h steps ahead (h < 0: h steps back), same weighting (rows off the ends get weight 0)
                L = y.shape[1]
                for j, hs in enumerate(aux_shifts):
                    ya = torch.zeros_like(y)
                    lo = max(0, -(a + hs))
                    seg = Y[:, a + hs + lo:a + hs + L].to(dev).clamp(-2, 2)
                    ya[:, lo:lo + seg.shape[1]] = seg
                    loss = loss + aux_w / len(aux_shifts) * _wloss(out[..., 2 + 2 * j:4 + 2 * j], ya, ya.abs() * rw, corr_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), scaler_clip)
            opt.step()
            h = _detach(h)
            tot_loss += loss.detach()
        else:
            with torch.no_grad():
                out, h = fwd(inp, h)
            preds.append(out.float().cpu())
    return float(tot_loss) / math.ceil(T / chunk) if opt is not None else torch.cat(preds, 1)


def run_windows(model, X, Y, RW, dev, opt, L, burnin, win_bs, corr_w=0.0, rng=None):
    """Windowed training: cut each sequence into non-overlapping windows of L rows (random offset per
    sequence), shuffle the windows of the whole batch and train on mini-batches of win_bs windows,
    each from a zero state. The first `burnin` rows of a window only build state (no loss)."""
    B, T, _ = X.shape
    wins = []
    for b in range(B):
        off = int(rng.integers(1, L + 1))
        wins += [(b, s0) for s0 in range(off, T - L + 1, L)]
    order = rng.permutation(len(wins))
    keep = torch.zeros(L, 1); keep[burnin:] = 1.0
    tot, n = 0.0, 0
    for i in range(0, len(order), win_bs):
        sel = [wins[j] for j in order[i:i + win_bs]]
        xc = torch.stack([X[b, s0 - 1:s0 + L] for b, s0 in sel]).to(dev, non_blocking=True)
        inp = add_diff(xc[:, 1:], xc[:, 0])
        y = torch.stack([Y[b, s0:s0 + L] for b, s0 in sel]).to(dev).clamp(-2, 2)
        rw = (torch.stack([RW[b, s0:s0 + L] for b, s0 in sel])[..., None] * keep).to(dev)
        out, _ = model(inp, None)
        loss = _wloss(out.float()[..., :2], y, y.abs() * rw, corr_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.detach(); n += 1
    return float(tot) / max(n, 1)


def window_steps(n_seqs, bs, L, win_bs, T=20000):
    """Optimizer steps per epoch in windowed mode (upper bound: offsets can drop one window per sequence)."""
    return (n_seqs // bs) * math.ceil(bs * ((T - 1) // L) / win_bs)


class WeightedWP:
    """Global |y|*w weighted Pearson per target from running sums (w: extra row weights)."""

    def __init__(self):
        self.s = np.zeros((6, 2))

    def add(self, Y, P, w):
        y, p = np.clip(Y, -2, 2).astype(np.float64), np.clip(P, -2, 2).astype(np.float64)
        ww = np.abs(y) * w[:, None]
        self.s += np.stack([ww.sum(0), (ww * y).sum(0), (ww * p).sum(0), (ww * y * y).sum(0),
                            (ww * p * p).sum(0), (ww * y * p).sum(0)])

    def result(self):
        W, Sy, Sp, Syy, Spp, Syp = self.s
        cov, vy, vp = Syp / W - Sy * Sp / W ** 2, Syy / W - (Sy / W) ** 2, Spp / W - (Sp / W) ** 2
        r = cov / np.sqrt(vy * vp)
        return {"t0": float(r[0]), "t1": float(r[1]), "weighted_pearson": float(r.mean())}


def predict(model, name, groups, dev, bs, chunk, workers, keep=False, pmask_es=False, feats=False):
    """Streams sequences; metrics via GlobalAccumulator (bounded memory, same result
    as the global formula). Returns (mask metrics, all-rows metrics, preds or None)."""
    """With pmask_es, the first result is the P(is_scored)-weighted WP (estimate of the official
    mask metric on train folds, which have no is_scored column) instead of the scored-mask WP."""
    dl = DataLoader(SeqDS(name, groups, row_weights=pmask_es, gamma=1.0, feats=feats), batch_size=bs, num_workers=workers,
                    pin_memory=False, prefetch_factor=1)
    model.eval()
    acc_mask, acc_all = (WeightedWP() if pmask_es else GlobalAccumulator()), GlobalAccumulator()
    need = np.arange(20000) >= WARMUP
    Ps = []
    for X, Y, M in dl:
        P = run_sequences(model, X, chunk, dev).numpy()
        for b in range(len(P)):
            acc_mask.add(Y[b].numpy(), P[b], M[b].numpy())
            acc_all.add(Y[b].numpy(), P[b], need)
        if keep:
            Ps.append(P.reshape(-1, 2).astype(np.float32))
    return acc_mask.result(), acc_all.result(), (np.concatenate(Ps) if keep else None)


def fe_names():
    from wnn import features as fe
    return fe.FEATURE_NAMES[N_RAW:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gru", choices=["gru", "mingru", "ssm", "tcn", "la"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train-seqs", type=int, default=0, help="debug: limit train sequences")
    ap.add_argument("--corr-w", type=float, default=0.0, help="weight of the 1 - weighted-Pearson loss term")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--select", default="es_all", choices=["es_all", "es_pm"],
                    help="epoch selection on ES fold: all required rows, or P(is_scored)-weighted")
    ap.add_argument("--pmask-gamma", type=float, default=0.0, help="row weight P(is_scored)^gamma")
    ap.add_argument("--amp", default="none", choices=["none", "bf16"], help="mixed precision for training")
    ap.add_argument("--train-frac", type=float, default=1.0, help="fraction of training sequences (fast screening)")
    ap.add_argument("--es-seqs", type=int, default=0, help="ES-fold sequences evaluated per epoch (0 = all)")
    ap.add_argument("--valid-every", type=int, default=1,
                    help="evaluate valid every k epochs (0 = only once, for the selected checkpoint)")
    ap.add_argument("--aux-shifts", default="", help="comma list of h: auxiliary heads predict targets h steps ahead (negative: h steps back)")
    ap.add_argument("--aux-w", type=float, default=0.5, help="total weight of the auxiliary losses")
    ap.add_argument("--feats", action="store_true", help="append the 276 engineered LightGBM features to the input")
    ap.add_argument("--ema-spans", default="", help="comma list of spans: append causal per-sequence normalised columns")
    ap.add_argument("--event-feats", action="store_true", help="append event-clock columns (age, last change) per raw column")
    ap.add_argument("--extra-feats", default="", help="comma list: evr (relative event clock), rk (per-seq ranks), na (a-col sentinels), pb (price bins)")
    ap.add_argument("--row-enc", type=int, default=0, help="per-row MLP encoder width before the recurrent stack (0 = off)")
    ap.add_argument("--no-raw", action="store_true", help="zero the raw-level input columns (keep diffs + extra feats)")
    ap.add_argument("--aug", type=float, default=0.0, help="regime augmentation strength s on train sequences (0 = off)")
    ap.add_argument("--slow-k", type=int, default=0, help="GRU / minGRU: slow branch updated every K rows (0 = off)")
    ap.add_argument("--delta", type=int, default=0, help="minGRU: per-sequence delta-rule memory layer with this many heads (0 = off)")
    ap.add_argument("--wema", type=float, default=0.0, help="EMA decay of the weights per optimizer step (0 = off)")
    ap.add_argument("--window", type=int, default=0, help="windowed training: L rows per window from a zero state (0 = stateful full sequences)")
    ap.add_argument("--burnin", type=int, default=200, help="windowed: first rows of each window build state only (no loss)")
    ap.add_argument("--win-bs", type=int, default=64, help="windowed: windows per optimizer step")
    ap.add_argument("--seq-weights", type=float, default=0.0,
                    help="K: train seq weight 1 + K * rank(P(valid))^gamma (adversarial, artifacts/adv), mean-normalised")
    ap.add_argument("--seq-weights-gamma", type=float, default=2.0)
    ap.add_argument("--attn-window", type=int, default=0, help="local attention block after the recurrent stack (0 = off)")
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--ssm-state", type=int, default=8, help="state size N per channel for --arch ssm")
    ap.add_argument("--pmask-dir", default="pmask", help="P(is_scored) arrays: pmask (v1) or pmask2")
    ap.add_argument("--resume", action="store_true", help="continue tag from last.pt (or best.pt + --resume-epoch)")
    ap.add_argument("--resume-epoch", type=int, default=-1, help="with best.pt only: last completed epoch it holds")
    ap.add_argument("--init-from", default="", help="warm start: load weights from artifacts/nn/<tag>/best.pt")
    ap.add_argument("--add-valid", default="none", choices=["none", "h0", "h1", "all"],
                    help="also train on valid half 0/1 (scored on the other half) or all of valid")
    ap.add_argument("--valid-mix", type=float, default=4.0, help="valid row weights need * (1 + K * is_scored)")
    ap.add_argument("--valid-rep", type=int, default=1, help="valid sequences repeated per epoch (upweighting)")
    ap.add_argument("--regimes", default="", help="comma list of tick regimes 0..3 (units .125/.241/.469/.657) to keep in "
                                                  "train/ES/valid (artifacts/preds/units_*.npy); empty = all")
    ap.add_argument("--class-head", action="store_true", help="distributional tick-class head (needs a single regime)")
    ap.add_argument("--ce-w", type=float, default=1.0, help="with --class-head: weight of the cross-entropy loss")
    ap.add_argument("--joint-head", action="store_true", help="with --class-head: one softmax over (k0, k1) pairs")
    ap.add_argument("--move-w", type=float, default=0.0, help="with --class-head: weight of the training-only move-head BCE")
    ap.add_argument("--soft-train", default="", help="teacher tag: artifacts/teacher/<tag>_train.npy soft labels for train rows")
    ap.add_argument("--kd-dist", default="", help="class-head teacher tag: artifacts/teacher/<tag>_dist_train.npy tick-class "
                         "probabilities for train rows (needs --kd-w; point soft labels still come from --soft-train)")
    ap.add_argument("--kd-w", type=float, default=0.0, help="with --kd-dist: class-head CE target (1 - w) one-hot + w teacher probabilities")
    ap.add_argument("--soft-a", type=float, default=0.8, help="point-loss target (1 - a) y + a teacher")
    ap.add_argument("--soft-floor", type=float, default=0.2, help="row weight |y| + floor with soft labels")
    ap.add_argument("--jump-p", type=float, default=1.0, help="point-loss row weights raised to this power (jump specialist)")
    ap.add_argument("--soft-conf", type=float, default=0.0, help="soft labels: row weight x exp(-c (y - teacher)^2)")
    ap.add_argument("--fut-lags", default="", help="look-ahead teacher horizons for --extra-feats fut (default wnn.fut_feats)")
    ap.add_argument("--tag", default="gru_v1")
    a = ap.parse_args()
    global AMP_DTYPE, PMASK_DIR, NO_RAW, ROW_ENC, MOVE_W, SOFT_A, SOFT_FLOOR, SOFT_CONF, KD_W, JUMP_P
    SOFT_A, SOFT_FLOOR, SOFT_CONF, KD_W, JUMP_P = a.soft_a, a.soft_floor, a.soft_conf, a.kd_w, a.jump_p
    if a.fut_lags:
        from wnn import fut_feats
        fut_feats.set_lags(a.fut_lags.split(","))
    MOVE_W = a.move_w
    PMASK_DIR = a.pmask_dir
    NO_RAW = a.no_raw
    ROW_ENC = a.row_enc
    AMP_DTYPE = {"none": None, "bf16": torch.bfloat16}[a.amp]
    seed_everything(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ARTIFACTS / "nn" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fold = np.asarray(load_splits()["train_fold"])
    tr_groups = np.flatnonzero(fold != ES_FOLD)
    es_groups = np.flatnonzero(fold == ES_FOLD)
    if a.max_train_seqs:
        tr_groups, es_groups = tr_groups[:a.max_train_seqs], es_groups[:max(64, a.max_train_seqs // 4)]
    if a.train_frac < 1.0:
        tr_groups = np.sort(np.random.default_rng(a.seed + 1).choice(tr_groups, int(len(tr_groups) * a.train_frac), replace=False))
    if a.es_seqs:
        es_groups = np.sort(np.random.default_rng(SEED + 2).choice(es_groups, min(a.es_seqs, len(es_groups)), replace=False))
    va_groups = np.arange(SequenceReader("valid").n)
    add_groups = np.array([], int)
    if a.add_valid != "none":  # train on train + valid half (or all), score the other half
        h0, h1 = valid_halves(len(va_groups))
        add_groups, va_groups = {"h0": (h0, h1), "h1": (h1, h0), "all": (va_groups, va_groups)}[a.add_valid]
        print(f"adding {len(add_groups)} valid seqs x{a.valid_rep} (mix{a.valid_mix:g}); scoring {len(va_groups)}"
              + (" (in-sample!)" if a.add_valid == "all" else ""), flush=True)
    regimes = [int(r) for r in a.regimes.split(",") if r]
    if regimes:  # tick-regime specialist: keep only sequences whose target unit is in the chosen regimes
        reg_of = lambda name: np.searchsorted(REGIME_EDGES, np.load(ARTIFACTS / "preds" / f"units_{name}.npy")[:, 0])
        rt, rv = reg_of("train"), reg_of("valid")
        tr_groups, es_groups = tr_groups[np.isin(rt[tr_groups], regimes)], es_groups[np.isin(rt[es_groups], regimes)]
        va_groups, add_groups = va_groups[np.isin(rv[va_groups], regimes)], add_groups[np.isin(rv[add_groups], regimes)]
        print(f"regimes {regimes}", flush=True)
    if a.class_head and len(regimes) != 1:
        sys.exit("--class-head needs exactly one --regimes value (the tick unit sets the classes)")
    if a.max_train_seqs:  # debug runs: small valid too
        va_groups, add_groups = va_groups[:16], add_groups[:4]
    print(f"train seqs {len(tr_groups)}, ES seqs {len(es_groups)}, valid seqs {len(va_groups)}", flush=True)

    aux_shifts = tuple(int(v) for v in a.aux_shifts.split(",") if v)
    feats = True if a.feats else tuple(int(v) for v in a.ema_spans.split(",")) if a.ema_spans else ()
    if a.event_feats:
        feats = feats + ("ev",)
    feats = feats + tuple(f for f in a.extra_feats.split(",") if f)
    feats = feats or False
    if feats and feats is not True:
        (out_dir / "ema.json").write_text(json.dumps({"spans": [f for f in feats if not isinstance(f, str)],
                                                     "extra": [f for f in feats if isinstance(f, str)]}))
    n_in = 224 + n_feats(feats)
    norm = feat_norm(tr_groups, feats=feats) if feats else None
    attn = (a.attn_window, a.attn_heads) if a.attn_window else None
    model = make_model(a.arch, a.hidden, a.layers, n_in, 2 * len(aux_shifts), norm, attn, a.ssm_state, a.slow_k, a.delta)
    if a.class_head:
        assert not aux_shifts and not a.window, "--class-head is implemented for stateful training without aux heads"
        model.set_class_head(REGIME_UNITS[regimes[0]], joint=a.joint_head)
    ce_w = a.ce_w if a.class_head else 0.0
    model = model.to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    if a.init_from:  # weights only: fresh optimizer and schedule (use a lower --lr and fewer --epochs)
        sd = torch.load(ARTIFACTS / "nn" / a.init_from / "best.pt", map_location=dev)
        own = model.state_dict()  # tensors of another shape (e.g. the head of a class-head specialist) keep their fresh init
        skip = sorted(k for k in own if k not in sd or sd[k].shape != own[k].shape)
        model.load_state_dict({k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}, strict=False)
        print(f"init from {a.init_from}" + (f" (fresh: {skip})" if skip else ""), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = ((len(tr_groups) + a.valid_rep * len(add_groups)) // a.bs) * math.ceil(20000 / a.chunk)
    if a.window:
        steps_per_epoch = window_steps(len(tr_groups) + a.valid_rep * len(add_groups), a.bs, a.window, a.win_bs)
        print(f"windowed training: L {a.window}, burn-in {a.burnin}, {a.win_bs} windows/step, {steps_per_epoch} steps/epoch", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.05, anneal_strategy="cos")
    g = torch.Generator(); g.manual_seed(a.seed)
    best, history, start = -1.0, [], 0
    if a.resume:
        # last.pt holds model + optimizer + scheduler (exact resume); best.pt only the weights, so
        # AdamW moments restart and the scheduler is fast-forwarded to the end of --resume-epoch.
        hist_path = out_dir / "history_partial.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text())
        if (out_dir / "last.pt").exists():
            ck = torch.load(out_dir / "last.pt", map_location=dev)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start = ck["epoch"] + 1
        else:
            model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev), strict=False)
            start = a.resume_epoch + 1
            for _ in range(start * steps_per_epoch):
                sched.step()
        done = [r for r in history if r["epoch"] < start]
        if done:
            i = int(np.argmax([r[a.select] for r in done]))
            best, best_ep, best_has_pred = done[i][a.select], done[i]["epoch"], False
        print(f"resumed at epoch {start}, best {a.select} {best:.4f}, lr {sched.get_last_lr()[0]:.2e}", flush=True)
    # Workers are forked once, while the main process is still small, and reused every epoch.
    seq_w = None
    if a.seq_weights:  # importance-style weights towards valid-like train sequences (rank-based: raw p/(1-p) has ESS 0.7%)
        p = np.load(ARTIFACTS / "adv" / "train_pvalid.npy")
        r = np.empty(len(p)); r[np.argsort(p, kind="stable")] = np.arange(len(p)) / (len(p) - 1)
        seq_w = (1.0 + a.seq_weights * r ** a.seq_weights_gamma).astype(np.float32)
        seq_w /= seq_w[tr_groups].mean()
        w = seq_w[tr_groups]
        print(f"seq weights K={a.seq_weights:g} gamma={a.seq_weights_gamma:g}: min {w.min():.2f} max {w.max():.2f}"
              f" ESS {w.sum() ** 2 / (w ** 2).sum() / len(w):.2f}", flush=True)
    soft = str(ARTIFACTS / "teacher" / f"{a.soft_train}_train.npy") if a.soft_train else None
    assert not (soft and (len(add_groups) or a.window or aux_shifts)), "--soft-train: plain stateful train-only runs"
    dist = str(ARTIFACTS / "teacher" / f"{a.kd_dist}_dist_train.npy") if a.kd_dist and a.kd_w else None
    assert not dist or (a.class_head and a.soft_train), "--kd-w needs --class-head and --soft-train"
    ds = SeqDS("train", tr_groups, row_weights=True, gamma=a.pmask_gamma, feats=feats, aug=a.aug, seq_w=seq_w, soft=soft,
               soft_dist=dist)
    if len(add_groups):
        ds = ConcatDataset([ds] + [SeqDS("valid", add_groups, row_weights=True, feats=feats, mask_mix=a.valid_mix)] * a.valid_rep)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    pin_memory=False, drop_last=True, generator=g, prefetch_factor=1, persistent_workers=True)

    params = list(model.parameters())
    wema = [p.detach().clone() for p in params] if a.wema else None  # EMA of weights, evaluated/saved instead of the live ones

    class _Opt:  # one optimizer step per chunk; the scheduler advances with each
        def zero_grad(self, set_to_none=True): opt.zero_grad(set_to_none=set_to_none)

        def step(self):
            opt.step(); sched.step()
            if wema is not None:
                with torch.no_grad():
                    torch._foreach_mul_(wema, a.wema)
                    torch._foreach_add_(wema, [p.detach() for p in params], alpha=1 - a.wema)

    def swap_wema():
        if wema is not None:
            with torch.no_grad():
                for p, w in zip(params, wema):
                    tmp = p.detach().clone(); p.copy_(w); w.copy_(tmp)

    win_rng = np.random.default_rng(a.seed + 3 + 1000 * start)
    for ep in range(start, a.epochs):
        model.train()
        t, losses = time.time(), []
        for i, (X, Y, RW) in enumerate(dl):
            if a.window:
                losses.append(run_windows(model, X, Y, RW, dev, _Opt(), a.window, a.burnin, a.win_bs, a.corr_w, win_rng))
            else:
                losses.append(run_sequences(model, X, a.chunk, dev, Y, RW, _Opt(), corr_w=a.corr_w,
                                            aux_shifts=aux_shifts, aux_w=a.aux_w, ce_w=ce_w))
            if i % 50 == 0:
                print(f"ep{ep} batch {i}/{len(dl)} loss {np.mean(losses[-50:]):.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t:.0f}s", flush=True)
        t_train = time.time() - t
        swap_wema()  # evaluate (and save) the averaged weights
        es_pm, es, _ = predict(model, "train", es_groups, dev, a.bs, 5000, 2, pmask_es=True, feats=feats)
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "es_all": es["weighted_pearson"],
               "es_pm": es_pm["weighted_pearson"], "train_secs": t_train}
        Pv = None
        if a.valid_every and ((ep + 1) % a.valid_every == 0 or ep == a.epochs - 1):
            vm, va_, Pv = predict(model, "valid", va_groups, dev, a.bs, 5000, 2, keep=True, feats=feats)
            rec.update(valid_mask=vm["weighted_pearson"], valid_all=va_["weighted_pearson"],
                       valid_t0=vm["t0"], valid_t1=vm["t1"])
        rec["secs"] = time.time() - t
        history.append(rec)
        print("EPOCH", json.dumps(rec), flush=True)
        if rec[a.select] > best:
            best, best_ep, best_has_pred = rec[a.select], ep, Pv is not None
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("aux.")}, out_dir / "best.pt")
            if aux_shifts:  # keep the aux head too (analysis of the auxiliary predictions)
                torch.save(model.state_dict(), out_dir / "best_with_aux.pt")
            if Pv is not None:
                np.save(out_dir / "pred_valid.npy", Pv)
        del Pv
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep},
                   out_dir / "last.pt")
        (out_dir / "history_partial.json").write_text(json.dumps(history))
        swap_wema()  # back to the live weights (last.pt holds the averaged ones when --wema)
    final = {"best_epoch": best_ep, a.select: best}
    if not best_has_pred:  # valid scored once, for the selected checkpoint only
        model.load_state_dict(torch.load(out_dir / "best.pt"), strict=False)  # aux head is not saved
        vm, va_, Pv = predict(model, "valid", va_groups, dev, a.bs, 5000, 2, keep=True, feats=feats)
        np.save(out_dir / "pred_valid.npy", Pv)
        final.update(valid_mask=vm["weighted_pearson"], valid_all=va_["weighted_pearson"], valid_t0=vm["t0"], valid_t1=vm["t1"])
    else:
        final.update({k: history[best_ep][k] for k in ("valid_mask", "valid_all", "valid_t0", "valid_t1")})
    print("FINAL", json.dumps(final), flush=True)
    (out_dir / "history.json").write_text(json.dumps({"args": vars(a), "history": history, "final": final}, indent=1))
    print("DONE best", a.select, best, flush=True)


if __name__ == "__main__":
    main()
