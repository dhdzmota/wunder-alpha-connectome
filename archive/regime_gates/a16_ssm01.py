"""Gate (EXPERIMENTS §44): SSM back in the r0 / r1 routes under the row stack (spec .5 + SSM .25 + TCN .25, stack refit)
vs the sub24f routes (spec .6 + TCN .4, stack). §22 had +0.0002/+0.0002 for r0 and was dropped only for latency (before fastexp).
Lean in memory (training may be running): only scored rows are held. Cross-fit on the FT halves, stack alpha .35."""
import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
sys.path.insert(0, '.')
from utils import weighted_pearson
A = Path('/home/user/data/wnn/artifacts')
m = (np.load(A / 'preds/valid_mask.npy') & np.load(A / 'preds/valid_need.npy')).astype(bool).ravel()
idx = np.flatnonzero(m); seq = idx // 20000; del m
det = np.load(A / 'regime/detected_valid.npy'); rg = det[seq]
y = np.clip(np.load(A / 'preds/valid_Y.npy', mmap_mode='r').reshape(-1, 2)[idx], -2, 2)
H1 = np.load(A / 'nn_ft/ft_mingru_ema_c3_mix4_f1/eval_groups.npy'); h = np.zeros(len(det), int); h[H1] = 1; hs = h[seq]


def rows(tag, only=None):
    """scored rows of the cross-fit predictions nn_ft/<tag>_f{0,1}/pred_ep2 (NaN where the runs do not cover a sequence)."""
    out = np.full((len(idx), 2), np.nan, np.float32)
    for f in (0, 1):
        d = A / 'nn_ft' / f'{tag}_f{f}'; g = np.load(d / 'eval_groups.npy'); P = np.load(d / 'pred_ep2.npy', mmap_mode='r').reshape(-1, 20000, 2)
        pos = np.full(len(det), -1); pos[g] = np.arange(len(g)); sel = pos[seq] >= 0
        out[sel] = P.reshape(-1, 2)[pos[seq[sel]] * 20000 + idx[sel] % 20000]
    return out


S, T = rows('ft_ssm16_c3_st_mix4_sa8'), rows('ft_tcn64_ema_c3_st16_mix4_sa8')
X = np.full_like(S, np.nan)
for r, tag in enumerate(('ft_mingru_ema_cls0_st_mix4_sa8', 'ft_mingru_ema_cls1_st_mix4_sa8', 'ft_mingru_ema_cls2_st_mix4_sa8', 'ft_mingru_ema_cls3_st_mix4_sa8')):
    v = rows(tag); X[rg == r] = v[rg == r]
assert not np.isnan(S).any() and not np.isnan(T).any()
bad = np.isnan(X[:, 0]); print('rows of mis-detected sequences (general members only):', int(bad.sum())); X[bad] = .5 * (S[bad] + T[bad])
q3 = .5 * X + .25 * (S + T); q2 = .6 * X + .4 * T
g = lambda yy, qq: np.mean([weighted_pearson(yy[:, k], qq[:, k]) for k in range(2)])
rep = lambda name, M: print(f'SSM01 {name:34s} {g(y, M):.5f} | h0 {g(y[hs == 0], M[hs == 0]):.5f} h1 {g(y[hs == 1], M[hs == 1]):.5f}', flush=True)
P = dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4, seed=1, deterministic=True)
SX, SD = (np.load(A / 'stack_rows' / f'stackB_{c}.npy', mmap_mode='r') for c in 'XD')
w = np.abs(y) + 1e-3
Z = {}
for k in (0, 1):
    mk = np.flatnonzero(rg == k); F = np.empty((len(mk), 226), np.float32); F[:, :112] = SX[mk]; F[:, 112:224] = SD[mk]; hk = hs[mk]
    for name, q in (('nossm', q2), ('ssm', q3)):
        F[:, 224:] = q[mk]; z = np.zeros((len(mk), 2))
        for t in range(2):
            for f in (0, 1):
                b = lgb.train(P, lgb.Dataset(F[hk != f], y[mk][hk != f, t], weight=w[mk][hk != f, t]), 200)
                z[hk == f, t] = b.predict(F[hk == f])
        Z[name, k] = .65 * q[mk] + .35 * z
        print('done', k, name, flush=True)
    del F


def compose(v0, v1, stack=True):
    M = q3.copy()
    for k, v in ((0, v0), (1, v1)):
        mk = np.flatnonzero(rg == k); M[mk] = Z[v, k] if stack else (q2 if v == 'nossm' else q3)[mk]
    return M


base = compose('nossm', 'nossm')
rep('no stack, no SSM in r0/r1', compose('nossm', 'nossm', False)); rep('no stack, SSM (robust final)', compose('ssm', 'ssm', False))
rep('stack, no SSM (= sub24f)', base)
for v0, v1 in (('ssm', 'nossm'), ('nossm', 'ssm'), ('ssm', 'ssm')):
    M = compose(v0, v1); d = [g(y[hs == f], M[hs == f]) - g(y[hs == f], base[hs == f]) for f in (0, 1)]
    rep(f'stack, SSM in r0={v0 == "ssm"} r1={v1 == "ssm"}', M)
    print(f'DECIDE_SSM01 r0={v0} r1={v1} {int(min(d) > 0 and sum(d) >= 0.0004)} h0 {d[0]:+.5f} h1 {d[1]:+.5f}')
