"""Gate for soft-label FT (EXPERIMENTS §23): members fine-tuned on valid halves with target (1-a) y + a teacher
(--soft, suffix _sa*) vs the plain FT, alone and swapped into the sub18-like OOF blend (spec .5 S .25 T .25 per route)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from utils import weighted_pearson
A = Path('/home/user/data/wnn/artifacts')
Y = np.load(A / 'preds/valid_Y.npy').reshape(-1, 20000, 2)
m = (np.load(A / 'preds/valid_mask.npy') & np.load(A / 'preds/valid_need.npy')).astype(bool).reshape(-1, 20000)
det = np.load(A / 'regime/detected_valid.npy')
G, S, T = [np.load(A / f'nn/{t}/pred_valid.npy').reshape(-1, 20000, 2)
           for t in ('ft_mingru_ema_c3_mix4_full', 'ft_ssm_mix4_full', 'ft_tcn64_ema_c3_mix4_full')]
H = [np.load(A / f'nn_ft/ft_mingru_ema_c3_mix4_f{f}/eval_groups.npy') for f in (0, 1)]


def crossfit(tag, into, ep=2):
    """into with the eval rows of nn_ft/<tag>_f{0,1} replaced (None if the runs are missing)."""
    P = into.copy()
    for f in (0, 1):
        d = A / 'nn_ft' / f'{tag}_f{f}'
        if not (d / f'pred_ep{ep}.npy').exists():
            return None
        P[np.load(d / 'eval_groups.npy')] = np.load(d / f'pred_ep{ep}.npy').reshape(-1, 20000, 2)
    return P


def wp(Q, s):
    mm = m[s].ravel(); y = Y[s].reshape(-1, 2)[mm]; q = Q[s].reshape(-1, 2)[mm]
    return np.mean([weighted_pearson(y[:, k], q[:, k]) for k in range(2)])


X = np.full_like(G, np.nan)
for r in range(4):
    X = crossfit(f'ft_mingru_ema_cls{r}_mix4', X)


def blend(X, T):
    Q = .30 * G + .35 * S + .35 * T
    for r in range(4):
        s = (det == r) & ~np.isnan(X[:, 0, 0])
        Q[s] = .5 * X[s] + .25 * (S[s] + T[s])
    return Q


al = np.arange(len(G))
fmt = lambda Q: f'{wp(Q, al):.4f} | h0 {wp(Q, H[0]):.4f} h1 {wp(Q, H[1]):.4f}'
base = blend(X, T)
print(f'blend sub18-like        {fmt(base)}')
print(f'TCN plain FT            {fmt(T)}')
for name in ('tcn64_ema_c3_mix4_sa8', 'tcn64_ema_c3_mix4_sa5', 'tcn64_ema_c3_mix4_c1k', 'tcn64_ema_c3_st8_mix4_sa8', 'tcn64_ema_c3_st16_mix4_sa8', 'tcn64_ema_c3_c8_mix4_sa8'):
    sfx = name.replace('tcn64_ema_c3', '')
    Ts = crossfit(f'ft_{name}', T)
    if Ts is not None:
        print(f'TCN {sfx:19s} {fmt(Ts)}   blend: {fmt(blend(X, Ts))}')
r3 = det == 3
print(f'spec r3 plain (r3 rows) {wp(X, np.flatnonzero(r3 & ~np.isnan(X[:, 0, 0]))):.4f}')
X3 = crossfit('ft_mingru_ema_cls3_mix4_sa8', X)
if X3 is not None:
    print(f'spec r3 soft  (r3 rows) {wp(X3, np.flatnonzero(r3 & ~np.isnan(X3[:, 0, 0]))):.4f}   blend: {fmt(blend(X3, T))}')
    Ts = crossfit('ft_tcn64_ema_c3_mix4_sa8', T)
    if Ts is not None:
        print(f'both soft (a .8)        blend: {fmt(blend(X3, Ts))}')
# Phase 2 (§24): r3 specialist retrained from scratch on la50 soft labels, with the sub23b TCN (st16).
T16 = crossfit('ft_tcn64_ema_c3_st16_mix4_sa8', T)
X3n = crossfit('ft_mingru_ema_cls3_st_mix4_sa8', X)
if X3 is not None and X3n is not None and T16 is not None:
    sel = np.flatnonzero(r3 & ~np.isnan(X3n[:, 0, 0]))
    print(f'P2 spec r3 soft FT (r3 rows) {wp(X3, sel):.4f}   blend+st16: {fmt(blend(X3, T16))}')
    print(f'P2 spec r3 scratch (r3 rows) {wp(X3n, sel):.4f}   blend+st16: {fmt(blend(X3n, T16))}')
# Phase 2 r2: r2 specialist from scratch on la50 (on top of the r3 scratch spec + st16 TCN, i.e. sub23d).
X2n = crossfit('ft_mingru_ema_cls2_st_mix4_sa8', X3n) if X3n is not None else None
X2o = X3n  # r2 rows: plain FT halves (no soft-FT halves exist for cls2); blend = sub23d-like
if X2n is not None and X2o is not None:
    sel = np.flatnonzero((det == 2) & ~np.isnan(X2n[:, 0, 0]))
    print(f'P2 spec r2 plain FT (r2 rows) {wp(X2o, sel):.4f}   blend+st16: {fmt(blend(X2o, T16))}')
    print(f'P2 spec r2 scratch (r2 rows) {wp(X2n, sel):.4f}   blend+st16: {fmt(blend(X2n, T16))}')
# Phase 2 r3, 16 epochs (vs the 8-epoch scratch spec of sub23d).
X3p = crossfit('ft_mingru_ema_cls3_st16_mix4_sa8', X)
if X3p is not None and X3n is not None:
    sel = np.flatnonzero(r3 & ~np.isnan(X3p[:, 0, 0]))
    print(f'P2 spec r3 scratch16 (r3 rows) {wp(X3p, sel):.4f} vs 8ep {wp(X3n, sel):.4f}   blend+st16: {fmt(blend(X3p, T16))}')
# Phase 2 SSM: ssm16_c3 retrained from scratch on la50 (soft FT halves) vs plain FT halves, both cross-fit, on sub23d-like.
Sp = crossfit('ft_ssm16_c3_mix4', S)
Sn = crossfit('ft_ssm16_c3_st_mix4_sa8', S)
if Sn is not None and Sp is not None and X3n is not None:
    def blend_s(X, T, S):
        Q = .30 * G + .35 * S + .35 * T
        for r in range(4):
            s = (det == r) & ~np.isnan(X[:, 0, 0])
            Q[s] = .5 * X[s] + .25 * (S[s] + T[s])
        return Q
    print(f'P2 SSM plain FT halves   {fmt(Sp)}   blend+st16: {fmt(blend_s(X3n, T16, Sp))}')
    print(f'P2 SSM scratch soft      {fmt(Sn)}   blend+st16: {fmt(blend_s(X3n, T16, Sn))}')
# Phase 2 r0/r1: specialists from scratch on la50 on top of sub23g-like (r3 scratch, SSM scratch, TCN st16).
# Note: the runtime r0/r1 routes use the sub21 row stack on top of spec .6 + TCN .4; this gate uses the plain blend.
if Sn is not None and X3n is not None:
    base_g = blend_s(X3n, T16, Sn)
    print(f'P2 base sub23g-like        {fmt(base_g)}')
    for r in (0, 1):
        Xr = crossfit(f'ft_mingru_ema_cls{r}_st_mix4_sa8', X3n)
        if Xr is not None:
            sel = np.flatnonzero((det == r) & ~np.isnan(Xr[:, 0, 0]))
            Q = blend_s(Xr, T16, Sn)
            d0, d1 = wp(Q, H[0]) - wp(base_g, H[0]), wp(Q, H[1]) - wp(base_g, H[1])
            print(f'P2 spec r{r} scratch (r{r} rows) {wp(Xr, sel):.4f} vs {wp(X3n, sel):.4f}   blend: {fmt(Q)}')
            print(f'DECIDE r{r} {int(min(d0, d1) > 0 and d0 + d1 >= 0.0004)}')
# Label recipe on r3: scratch spec with mixed teacher labels (.5 la50 + .5 la64c) vs the la50 one (sub23g-like blend).
if Sn is not None and X3n is not None:
    X3m = crossfit('ft_mingru_ema_cls3_mx_mix4_sa8', X)
    if X3m is not None:
        sel = np.flatnonzero(r3 & ~np.isnan(X3m[:, 0, 0]))
        print(f'P3 spec r3 mix-label (r3 rows) {wp(X3m, sel):.4f} vs la50 {wp(X3n, sel):.4f}   blend: {fmt(blend_s(X3m, T16, Sn))} (base {fmt(blend_s(X3n, T16, Sn))})')
# Label recipe on r3: train-time soft weight a (from-scratch spec; FT kept at a .8) vs a .8 (sub23g-like blend).
if Sn is not None and X3n is not None:
    for tg in ('a6', 'a10', 'cf'):
        Xa = crossfit(f'ft_mingru_ema_cls3_{tg}_mix4_sa8', X)
        if Xa is not None:
            sel = np.flatnonzero(r3 & ~np.isnan(Xa[:, 0, 0]))
            print(f'P3 spec r3 {tg:4s} (r3 rows) {wp(Xa, sel):.4f} vs a.8 {wp(X3n, sel):.4f}   blend: {fmt(blend_s(Xa, T16, Sn))} (base {fmt(blend_s(X3n, T16, Sn))})')
# FT learning rate 1e-4 (jy447 log: +0.0016-0.0020 on one fold) vs 3e-4, 3 ep (lr4) and 6 ep (lr4e6), r3 spec + TCN st16.
if Sn is not None and X3n is not None:
    for sfx, ep in (('lr4', 2), ('lr4e6', 5)):
        X3l = crossfit(f'ft_mingru_ema_cls3_st_mix4_sa8{sfx}', X, ep)
        Tl = crossfit(f'ft_tcn64_ema_c3_st16_mix4_sa8{sfx}', T, ep)
        if X3l is not None and Tl is not None:
            sel = np.flatnonzero(r3 & ~np.isnan(X3l[:, 0, 0]))
            print(f'P3 FT {sfx:5s} r3 {wp(X3l, sel):.4f} vs {wp(X3n, sel):.4f} | TCN {fmt(Tl)} vs {fmt(T16)}')
            print(f'P3 FT {sfx:5s} blend r3+TCN {fmt(blend_s(X3l, Tl, Sn))}  r3 only {fmt(blend_s(X3l, T16, Sn))}  TCN only {fmt(blend_s(X3n, Tl, Sn))}  (base {fmt(blend_s(X3n, T16, Sn))})')
