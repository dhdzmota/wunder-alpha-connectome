"""Branch-free float32 transcendentals that LLVM can vectorise (no SVML on our numba builds).

tanh_f: Eigen's rational approximation (generic_fast_tanh_float), max abs error ~1e-7 on float32.
The others are exact identities on top of it, so every activation costs a few FMAs + one division.
"""
from __future__ import annotations

import numba as nb
import numpy as np

_CLAMP = np.float32(7.90531110763549805)
_A1, _A3, _A5 = np.float32(4.89352455891786e-03), np.float32(6.37261928875436e-04), np.float32(1.48572235717979e-05)
_A7, _A9 = np.float32(5.12229709037114e-08), np.float32(-8.60467152213735e-11)
_A11, _A13 = np.float32(2.00018790482477e-13), np.float32(-2.76076847742355e-16)
_B0, _B2 = np.float32(4.89352518554385e-03), np.float32(2.26843463243900e-03)
_B4, _B6 = np.float32(1.18534705686654e-04), np.float32(1.19825839466702e-06)
_HALF = np.float32(0.5)
_ONE = np.float32(1.0)


@nb.njit(inline="always", fastmath=True)
def tanh_f(x):
    x = min(max(x, -_CLAMP), _CLAMP)
    x2 = x * x
    p = x2 * _A13 + _A11
    p = x2 * p + _A9
    p = x2 * p + _A7
    p = x2 * p + _A5
    p = x2 * p + _A3
    p = x2 * p + _A1
    q = x2 * _B6 + _B4
    q = x2 * q + _B2
    q = x2 * q + _B0
    return x * p / q


@nb.njit(inline="always", fastmath=True)
def sigmoid_f(x):
    return _HALF + _HALF * tanh_f(_HALF * x)


@nb.njit(inline="always", fastmath=True)
def exp_neg_f(q):
    """exp(-q) for q >= 0 via (1 - tanh(q/2)) / (1 + tanh(q/2)); abs error ~1e-7 (tiny decays floor at ~1e-7)."""
    t = tanh_f(_HALF * q)
    return (_ONE - t) / (_ONE + t)


@nb.njit(inline="always", fastmath=True)
def gelu_tanh_f(x):
    c = np.float32(0.7978845608028654)
    return _HALF * x * (_ONE + tanh_f(c * (x + np.float32(0.044715) * x * x * x)))


@nb.njit(inline="always", fastmath=True)
def softplus_f(t):
    """torch softplus (threshold 20): max(t, 0) + log1p(exp(-|t|)), log1p(e) = 2 atanh(e / (2 + e)) series (s <= 1/3)."""
    e = exp_neg_f(abs(t))
    s = e / (np.float32(2.0) + e)
    s2 = s * s
    p = s2 * np.float32(1.0 / 15.0) + np.float32(1.0 / 13.0)
    p = s2 * p + np.float32(1.0 / 11.0)
    p = s2 * p + np.float32(1.0 / 9.0)
    p = s2 * p + np.float32(1.0 / 7.0)
    p = s2 * p + np.float32(1.0 / 5.0)
    p = s2 * p + np.float32(1.0 / 3.0)
    p = s2 * p + _ONE
    r = max(t, np.float32(0.0)) + np.float32(2.0) * s * p
    return t if t > np.float32(20.0) else r


_TAB2 = np.array([2.0 ** -i for i in range(128)], np.float32)  # 2^-n, n = 0..127
_L2E = np.float32(1.4426950408889634)
_P0, _P1, _P2 = np.float32(1.535336188319500e-4), np.float32(1.339887440266574e-3), np.float32(9.618437357674640e-3)
_P3, _P4, _P5 = np.float32(5.550332471162809e-2), np.float32(2.402264791363012e-1), np.float32(6.931472028550421e-1)


@nb.njit(inline="always", fastmath=True)
def exp_neg_fast(q):
    """exp(-q) for q >= 0 with no division: 2^-n from a table times cephes exp2f's polynomial on [-0.5, 0.5]
    (abs error ~8e-8, floors at 2^-126). ~5x cheaper than exp_neg_f in the SSM scan (EXPERIMENTS §25)."""
    t = min(q * _L2E, np.float32(126.0))
    n = np.int32(t + _HALF)
    f = np.float32(n) - t
    p = ((((_P0 * f + _P1) * f + _P2) * f + _P3) * f + _P4) * f + _P5
    return _TAB2[n] * (p * f + _ONE)
