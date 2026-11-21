"""De-anonymisation probe, r3 (EXPERIMENTS §43): the 22 i0 price columns take ~17 joint values in coarse-tick sequences and each
value fixes the half-tick move of the mid on that row (dictionary from labels: dmid = -median(d(t0 - t1)) / 2). Decoded level
L = cumsum(dmid). Does L relative to its own past (L - EMA_n(L), range position, clocks) add to the blend on r3 scored rows?
Cross-fit on the FT halves, linear and LightGBM stacks on [p0, p1] (+ level features)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from wnn.data import SequenceReader
_argv = sys.argv; sys.argv = sys.argv[:1]
exec(open('./pipeline/regime_gates/proj_soft.py').read().split("al = np.arange")[0])
Sn = crossfit('ft_ssm16_c3_st_mix4_sa8', S); T16 = crossfit('ft_tcn64_ema_c3_st16_mix4_sa8', T); X3n = crossfit('ft_mingru_ema_cls3_st_mix4_sa8', X)
Q = .30 * G + .35 * Sn + .35 * T16
for r in range(4):
    s = (det == r) & ~np.isnan(X3n[:, 0, 0]); Q[s] = .5 * X3n[s] + .25 * (Sn[s] + T16[s])
del G, S, T, X, Sn, T16
u = 0.6568677
r3 = np.flatnonzero(det == 3)
R = SequenceReader('valid')
OUT = A / 'decode'
SPANS = (3, 10, 30, 100, 300, 1000, 3000)


def ema(x, n):
    a = 2.0 / (n + 1); o = np.empty_like(x); acc = x[0]
    for i in range(len(x)):
        acc += a * (x[i] - acc); o[i] = acc
    return o


try:
    import numba
    ema = numba.njit(cache=False)(ema)
except Exception:
    pass

if not (OUT / 'r3_feats.npy').exists():
    # pass 1: dictionary state -> modal d(t0 - t1) from the first 150 r3 sequences (labels), keyed by the rounded price row
    from collections import defaultdict, Counter
    dic = defaultdict(Counter); keys = {}
    for g in r3:
        s = R.read(int(g)); k = [r.tobytes() for r in np.round(s.X[:, :22], 3)]; keys[int(g)] = k
        if len(dic) < 10**9 and g in r3[:150]:
            K = np.rint(s.Y / u).astype(int); dD = np.r_[0, np.diff(K[:, 0] - K[:, 1])]
            for t in range(1, 20000): dic[k[t]][int(dD[t])] += 1
    dm = {k: -c.most_common(1)[0][0] / 2 for k, c in dic.items()}
    print('states', len(dm), 'moves', Counter(dm.values()))
    F = np.zeros((len(r3), 20000, 4 + 2 * len(SPANS)), np.float32); unk = 0
    for i, g in enumerate(r3):
        d = np.array([dm.get(k, np.nan) for k in keys[int(g)]]); unk += np.isnan(d).sum(); d = np.nan_to_num(d)
        L = np.cumsum(d); t = np.arange(20000)
        last = np.maximum.accumulate(np.where(d != 0, t, -1))
        lastdir = np.where(last >= 0, np.sign(d[np.maximum(last, 0)]), 0)
        cols = [d, lastdir, np.log1p(t - last), np.log1p(ema(np.abs(d), 300) * 300)]
        cols += [L - ema(L, n) for n in SPANS] + [ema(d, n) * n for n in SPANS]
        F[i] = np.stack(cols, 1)
    print('unknown-state rows', unk)
    np.save(OUT / 'r3_feats.npy', F)
F = np.load(OUT / 'r3_feats.npy')
mm = m[r3]; y = np.clip(Y[r3], -2, 2); p = Q[r3]
half = np.isin(r3, H[0])
from utils import weighted_pearson
wpr = lambda yy, qq: np.mean([weighted_pearson(yy[:, k], qq[:, k]) for k in range(2)])
print(f'r3 base: all {wpr(y[mm], p[mm]):.4f}  h0 {wpr(y[half][mm[half]], p[half][mm[half]]):.4f}  h1 {wpr(y[~half][mm[~half]], p[~half][mm[~half]]):.4f}')
names = ['dmid', 'lastdir', 'log_since', 'log_act'] + [f'L-ema{n}' for n in SPANS] + [f'flow{n}' for n in SPANS]
res = y[mm] - p[mm]; w = np.abs(y[mm])
print('weighted corr of the blend residual (t0, t1) with each decoded feature, r3 scored rows:')
for j, nm in enumerate(names):
    f = F[..., j][mm]
    print(f'  {nm:10s} t0 {weighted_pearson(res[:, 0], f):+.3f} t1 {weighted_pearson(res[:, 1], f):+.3f} | raw y: t0 {weighted_pearson(y[mm][:, 0], f):+.3f} t1 {weighted_pearson(y[mm][:, 1], f):+.3f}')
import lightgbm as lgb
for label, cols in (('p only', []), ('p + decoded', list(range(F.shape[-1])))):
    out = np.zeros_like(p); outl = np.zeros_like(p)
    for fit_h in (True, False):
        a, b = half == fit_h, half != fit_h
        Xa = np.concatenate([p[a][mm[a]], F[a][mm[a]][:, cols]], 1); Xb = np.concatenate([p[b][mm[b]], F[b][mm[b]][:, cols]], 1)
        ya = y[a][mm[a]]; pb = np.zeros((len(Xb), 2)); pl = np.zeros((len(Xb), 2))
        for k in range(2):
            wa = np.abs(ya[:, k]) + 1e-3
            mdl = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=.05, min_child_samples=500, subsample=.5, subsample_freq=1,
                                    colsample_bytree=.8, n_jobs=4, verbose=-1).fit(Xa, ya[:, k], sample_weight=wa)
            pb[:, k] = mdl.predict(Xb)
            A1 = np.c_[Xa, np.ones(len(Xa))] * np.sqrt(wa)[:, None]
            pl[:, k] = np.c_[Xb, np.ones(len(Xb))] @ np.linalg.lstsq(A1, ya[:, k] * np.sqrt(wa), rcond=None)[0]
        tmp = out[b]; tmp[mm[b]] = pb; out[b] = tmp; tmp = outl[b]; tmp[mm[b]] = pl; outl[b] = tmp
    for nm, o in (('lgbm', out), ('linear', outl)):
        for al_ in (1.0, 0.5):
            q = (1 - al_) * p + al_ * o
            print(f'{label:12s} {nm:6s} a {al_}: r3 {wpr(y[mm], q[mm]):.4f}  h0 {wpr(y[half][mm[half]], q[half][mm[half]]):.4f}  h1 {wpr(y[~half][mm[~half]], q[~half][mm[~half]]):.4f}')
