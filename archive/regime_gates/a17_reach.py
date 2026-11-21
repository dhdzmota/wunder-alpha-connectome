"""Probe (EXPERIMENTS §51): does a look-ahead teacher with LONG reach R recover the delayed-label oracle? At row t its output
for row t-R is causal. Same stack as §39 (a12_delayed): per regime, cross-fit on the FT halves, linear on scored rows
(weights |y|): y ~ p(t) + [d, EMA_100(d), EMA_500(d)](t-L), d = source - p; source = teacher or true y (oracle at the same L).
Usage: a17_reach.py <teacher tag>=<reach> ..."""
import sys, numpy as np
from scipy.signal import lfilter
_src = dict(a.split('=') for a in sys.argv[1:])
sys.argv = ['x']
exec(open('./pipeline/regime_gates/proj_soft.py').read())
P = base_g
del G, S, T, X
h = np.zeros(len(det), int); h[H[1]] = 1
Yc = Y.clip(-2, 2)


def lagged(src, L):
    d = src - P; out = np.zeros_like(d); out[:, L:] = d[:, :-L]
    return np.concatenate([out] + [lfilter([a], [1, a - 1], out, axis=1) for a in (0.01, 0.002)], 2).astype(np.float32)


def fit_eval(F):
    out = P.copy()
    for r in range(4):
        for fo in (0, 1):
            tr = np.flatnonzero((det == r) & (h != fo)); te = np.flatnonzero((det == r) & (h == fo))
            for k in range(2):
                mm = m[tr]; y = Yc[tr][..., k][mm]; w = np.abs(y)
                Xt = np.c_[P[tr][..., k][mm], F[tr][mm], np.ones(mm.sum())]
                b = np.linalg.lstsq(Xt * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
                out[te, :, k] = (np.c_[P[te][..., k].reshape(-1), F[te].reshape(-1, F.shape[-1]), np.ones(len(te) * 20000)] @ b).reshape(len(te), 20000)
    return out


def report(name, L, src):
    Z = fit_eval(lagged(src, L))
    d = [wp(Z, H[f]) - wp(P, H[f]) for f in (0, 1)]
    rr = [wp(Z, np.flatnonzero(det == r)) - wp(P, np.flatnonzero(det == r)) for r in range(4)]
    print(f'REACH {name:20s} L{L:4d} {fmt(Z)}  h0 {d[0]:+.4f} h1 {d[1]:+.4f}  r0-r3 ' + ' '.join(f'{x:+.4f}' for x in rr), flush=True)


print('base', fmt(P), flush=True)
for tag, R in _src.items():
    R = int(R); Q = np.load(A / f'nn/{tag}/pred_valid.npy').reshape(-1, 20000, 2)
    print(f'teacher {tag}: scored WP {wp(Q, np.arange(len(det))):.4f}', flush=True)
    report(tag, R, Q); del Q
    report('oracle', R, Yc)
print('DONE')
