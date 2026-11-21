"""Causal streaming features, one row at a time.

`update(state, x, out)` is the single source of truth: training calls it in a
compiled loop over a sequence, and the submission calls it once per row, so
train/serve features are identical by construction. Only the current and past
rows are used.

Book indices are not depth levels (see data overview), so book summaries use
index-agnostic aggregates (means and sums over each side).
"""
from __future__ import annotations

import numba as nb
import numpy as np

SENTINEL = -5.0  # raw sentinel is -5.1993; anything below this is treated as "missing"
HALF_LIVES = np.array([5.0, 25.0, 100.0, 400.0, 1600.0])
ALPHAS = 1.0 - 0.5 ** (1.0 / HALF_LIVES)
N_EMA = len(HALF_LIVES)

# Column blocks inside the 112-feature row.
I0_PB, I0_PA, I0_VB, I0_VA, I0_DP, I0_DV = (0, 11), (11, 22), (22, 33), (33, 44), (44, 48), (48, 52)
OFF1 = 52
N_RAW = 112

# Summary signals tracked with EMAs (per instrument):
#   mid      = (mean bid p + mean ask p)/2
#   dmid     = mid - previous mid
#   spr      = mean ask p - mean bid p
#   imb      = sum bid v - sum ask v
#   tdp      = mean trade price feature
#   tdv      = mean trade volume feature
#   pchg     = 1 if any price index changed this row
#   tchg     = 1 if trade features changed this row
SIG_PER_INST = ["mid", "dmid", "spr", "imb", "tdp", "tdv", "pchg", "tchg"]
N_SIG = 2 * len(SIG_PER_INST) + 8  # + a0..a7
SIG_NAMES = [f"i{i}_{s}" for i in range(2) for s in SIG_PER_INST] + [f"a{j}" for j in range(8)]

# State layout: prev raw row | prev signal values | EMAs (N_SIG*N_EMA) | EM variance of dmid | row count
S_PREV = 0
S_PSIG = S_PREV + N_RAW
S_EMA = S_PSIG + N_SIG
S_VAR = S_EMA + N_SIG * N_EMA
S_CNT = S_VAR + 2 * N_EMA
STATE_SIZE = S_CNT + 1

FEATURE_NAMES = (
    [f"raw{j}" for j in range(N_RAW)]
    + [f"{s}" for s in SIG_NAMES]
    + [f"{s}_ema{int(h)}" for s in SIG_NAMES for h in HALF_LIVES]
    + [f"{s}_dev{int(h)}" for s in SIG_NAMES for h in HALF_LIVES]
    + [f"i{i}_vol{int(h)}" for i in range(2) for h in HALF_LIVES]
    + ["n_missing", "i0_i1_mid_diff"]
)
N_FEATURES = len(FEATURE_NAMES)


@nb.njit(cache=True, fastmath=False)
def _mean(x, a, b):
    s = 0.0
    for j in range(a, b):
        s += x[j]
    return s / (b - a)


@nb.njit(cache=True, fastmath=False)
def _changed(x, prev, a, b):
    for j in range(a, b):
        if x[j] != prev[j]:
            return 1.0
    return 0.0


@nb.njit(cache=True, fastmath=False)
def reset(state):
    state[:] = 0.0


@nb.njit(cache=True, fastmath=False)
def update(state, x, out):
    """Consume one raw row `x` (112,), write features into `out` (N_FEATURES,)."""
    first = state[S_CNT] == 0.0
    prev = state[S_PREV:S_PREV + N_RAW]
    sig = np.empty(N_SIG)
    for i in range(2):
        o = OFF1 * i
        mid = 0.5 * (_mean(x, o + 0, o + 11) + _mean(x, o + 11, o + 22))
        k = i * 8
        sig[k + 0] = mid
        sig[k + 1] = 0.0 if first else mid - state[S_PSIG + k]
        sig[k + 2] = _mean(x, o + 11, o + 22) - _mean(x, o + 0, o + 11)
        sig[k + 3] = (_mean(x, o + 22, o + 33) - _mean(x, o + 33, o + 44)) * 11.0
        sig[k + 4] = _mean(x, o + 44, o + 48)
        sig[k + 5] = _mean(x, o + 48, o + 52)
        sig[k + 6] = 0.0 if first else _changed(x, prev, o + 0, o + 22)
        sig[k + 7] = 0.0 if first else max(_changed(x, prev, o + 44, o + 48), _changed(x, prev, o + 48, o + 52))
    for j in range(8):
        sig[16 + j] = x[104 + j]

    n = 0
    for j in range(N_RAW):
        out[n] = x[j]; n += 1
    for s in range(N_SIG):
        out[n] = sig[s]; n += 1
    # EMAs: initialised to the first value so early rows are not biased toward 0.
    for s in range(N_SIG):
        for e in range(N_EMA):
            idx = S_EMA + s * N_EMA + e
            if first:
                state[idx] = sig[s]
            else:
                state[idx] += ALPHAS[e] * (sig[s] - state[idx])
            out[n] = state[idx]; n += 1
    for s in range(N_SIG):
        for e in range(N_EMA):
            out[n] = sig[s] - state[S_EMA + s * N_EMA + e]; n += 1
    # Exponentially weighted volatility of mid changes (realised-vol proxy).
    for i in range(2):
        d = sig[i * 8 + 1]
        for e in range(N_EMA):
            idx = S_VAR + i * N_EMA + e
            state[idx] += ALPHAS[e] * (d * d - state[idx])
            out[n] = np.sqrt(state[idx]); n += 1
    miss = 0.0
    for j in range(N_RAW):
        if x[j] < SENTINEL:
            miss += 1.0
    out[n] = miss; n += 1
    out[n] = sig[0] - sig[8]; n += 1

    state[S_PREV:S_PREV + N_RAW] = x
    state[S_PSIG:S_PSIG + N_SIG] = sig
    state[S_CNT] += 1.0


@nb.njit(cache=True)
def compute_sequence(X):
    """Run `update` over a full sequence -> (T, N_FEATURES) float32."""
    T = X.shape[0]
    F = np.empty((T, N_FEATURES), np.float32)
    state = np.zeros(STATE_SIZE)
    out = np.empty(N_FEATURES)
    x = np.empty(N_RAW)
    for t in range(T):
        for j in range(N_RAW):
            x[j] = X[t, j]
        update(state, x, out)
        for j in range(N_FEATURES):
            F[t, j] = out[j]
    return F
