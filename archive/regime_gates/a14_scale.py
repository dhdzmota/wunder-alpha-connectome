"""How much global WP does a per-regime affine re-calibration recover when the model comes from another period?
Train-only models scored on valid (period shift, like valid->test) vs the FT cross-fit blend (no shift). In-sample upper bound."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
A = Path('/home/user/data/wnn/artifacts')
Y = np.clip(np.load(A / 'preds/valid_Y.npy').reshape(-1, 20000, 2), -2, 2)
m = (np.load(A / 'preds/valid_mask.npy') & np.load(A / 'preds/valid_need.npy')).astype(bool).reshape(-1, 20000)
det = np.load(A / 'regime/detected_valid.npy')
H = [np.load(A / f'nn_ft/ft_mingru_ema_c3_mix4_f{f}/eval_groups.npy') for f in (0, 1)]

def stats(P, seqs):
    """per regime, per target: weighted sums on scored rows -> (4, 2, 6)"""
    S = np.zeros((4, 2, 6))
    for r in range(4):
        s = seqs[det[seqs] == r]
        mm = m[s]; y = Y[s][mm]; p = np.clip(P[s][mm], -2, 2); w = np.abs(y)
        S[r] = np.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0), (w * y * p).sum(0), (w * p * p).sum(0), (w * y * y).sum(0)], 1)
    return S

def wp(S, a=None, b=None):
    a = np.ones((4, 2)) if a is None else a; b = np.zeros((4, 2)) if b is None else b
    W, Wy, Wp, Wyp, Wpp, Wyy = [S[..., i] for i in range(6)]
    # q = a p + b
    Wq = a * Wp + b * W; Wyq = a * Wyp + b * Wy; Wqq = a * a * Wpp + 2 * a * b * Wp + b * b * W
    W, Wy, Wq, Wyq, Wqq, Wyy = [v.sum(0) for v in (W, Wy, Wq, Wyq, Wqq, Wyy)]
    cov = Wyq / W - Wy / W * Wq / W
    return float(np.mean(cov / np.sqrt((Wyy / W - (Wy / W) ** 2) * (Wqq / W - (Wq / W) ** 2))))

def fit(S, offsets):
    f = lambda v: -wp(S, v[:8].reshape(4, 2), v[8:].reshape(4, 2) if offsets else None)
    v = minimize(f, np.r_[np.ones(8), np.zeros(8 if offsets else 0)], method='L-BFGS-B').x
    return v[:8].reshape(4, 2), (v[8:].reshape(4, 2) if offsets else None)

al = np.arange(len(det))
for tag in sys.argv[1:]:
    P = np.load(A / 'nn' / tag / 'pred_valid.npy').reshape(-1, 20000, 2)
    S = stats(P, al)
    a, _ = fit(S, False); a2, b2 = fit(S, True)
    sl = S[..., 3] / S[..., 4]
    print(f'{tag}: WP {wp(S):.4f} | in-sample scale {wp(S, a):.4f} ({wp(S, a) - wp(S):+.4f}) | affine {wp(S, a2, b2):.4f} ({wp(S, a2, b2) - wp(S):+.4f})')
    print('   scales (rel. to r0)', np.round(a / a[0], 2).T.tolist())
    # cross-fit between halves: does a scale fitted on other sequences of the SAME period transfer?
    Sh = [stats(P, H[f]) for f in (0, 1)]
    d = [wp(Sh[1 - f], fit(Sh[f], False)[0]) - wp(Sh[1 - f]) for f in (0, 1)]
    d2 = [wp(Sh[1 - f], *fit(Sh[f], True)) - wp(Sh[1 - f]) for f in (0, 1)]
    print(f'   cross-fit halves: scale {d[0]:+.4f}/{d[1]:+.4f}  affine {d2[0]:+.4f}/{d2[1]:+.4f}')
