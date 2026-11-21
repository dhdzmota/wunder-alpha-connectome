"""Gate for swapping members of the routed blend (EXPERIMENTS §45): proj_prop.py [--incremental] key=<FT-halves tag> ...
--incremental: also score the LAST swap on top of the blend that already holds the earlier ones (INCR lines).
keys: ssm, tcn, r0, r1, r2, r3 (members not named keep the sub24f / sub25 ones). No-stack blend spec .5 + SSM .25 + TCN .25 on
every route (the r0/r1 stack is additive on top, §44), scored rows only (lean in memory), cross-fit FT halves."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from utils import weighted_pearson
A = Path('/home/user/data/wnn/artifacts')
m = (np.load(A / 'preds/valid_mask.npy') & np.load(A / 'preds/valid_need.npy')).astype(bool).ravel()
idx = np.flatnonzero(m); seq = idx // 20000; del m
det = np.load(A / 'regime/detected_valid.npy'); rg = det[seq]
y = np.clip(np.load(A / 'preds/valid_Y.npy', mmap_mode='r').reshape(-1, 2)[idx], -2, 2)
h = np.zeros(len(det), int); h[np.load(A / 'nn_ft/ft_mingru_ema_c3_mix4_f1/eval_groups.npy')] = 1; hs = h[seq]
BASE = dict(ssm='ft_ssm16_c3_st_mix4_sa8', tcn='ft_tcn64_ema_c3_st16_mix4_sa8', r0='ft_mingru_ema_cls0_st_mix4_sa8',
            r1='ft_mingru_ema_cls1_st_mix4_sa8', r2='ft_mingru_ema_cls2_st_mix4_sa8', r3='ft_mingru_ema_cls3_st_mix4_sa8')
_cache = {}


def rows(tag):
    if tag not in _cache:
        out = np.full((len(idx), 2), np.nan, np.float32)
        for f in (0, 1):
            d = A / 'nn_ft' / f'{tag}_f{f}'; g = np.load(d / 'eval_groups.npy'); P = np.load(d / 'pred_ep2.npy', mmap_mode='r')
            pos = np.full(len(det), -1); pos[g] = np.arange(len(g)); sel = pos[seq] >= 0
            out[sel] = P.reshape(-1, 2)[pos[seq[sel]] * 20000 + idx[sel] % 20000]
        _cache[tag] = out
    return _cache[tag]


def blend(cfg):
    S, T = rows(cfg['ssm']), rows(cfg['tcn']); X = np.full_like(S, np.nan)
    for r in range(4):
        v = rows(cfg[f'r{r}']); X[rg == r] = v[rg == r]
    bad = np.isnan(X[:, 0]); X[bad] = .5 * (S[bad] + T[bad])
    return .5 * X + .25 * (S + T)


g = lambda yy, qq: np.mean([weighted_pearson(yy[:, k], qq[:, k]) for k in range(2)])
fmt = lambda M: f'{g(y, M):.5f} | h0 {g(y[hs == 0], M[hs == 0]):.5f} h1 {g(y[hs == 1], M[hs == 1]):.5f}'
incr = '--incremental' in sys.argv
swaps = dict(a.split('=') for a in sys.argv[1:] if '=' in a)
Qb = blend(BASE); print(f'PROP base {fmt(Qb)}')
for keys in [[k] for k in swaps] + ([list(swaps)] if len(swaps) > 1 else []):
    Q = blend({**BASE, **{k: swaps[k] for k in keys}})
    d = [g(y[hs == f], Q[hs == f]) - g(y[hs == f], Qb[hs == f]) for f in (0, 1)]
    one = ''
    if len(keys) == 1:
        k = keys[0]; mk = (np.ones(len(rg), bool) if k in ('ssm', 'tcn') else rg == int(k[1])) & ~np.isnan(rows(swaps[k])[:, 0]) & ~np.isnan(rows(BASE[k])[:, 0])
        one = f' | member alone {g(y[mk], rows(swaps[k])[mk]):.4f} vs {g(y[mk], rows(BASE[k])[mk]):.4f}'
    print(f'PROP {"+".join(keys)}: {fmt(Q)} ({d[0]:+.5f}/{d[1]:+.5f}){one}')
    print(f'DECIDE_PROP {"+".join(keys)} {int(min(d) > 0 and sum(d) >= 0.0004)}')
if incr:
    ks = list(swaps); Qp = blend({**BASE, **{k: swaps[k] for k in ks[:-1]}}); Q = blend({**BASE, **swaps})
    d = [g(y[hs == f], Q[hs == f]) - g(y[hs == f], Qp[hs == f]) for f in (0, 1)]
    print(f'INCR {ks[-1]} on top of {"+".join(ks[:-1]) or "base"}: {fmt(Qp)} -> {fmt(Q)} ({d[0]:+.5f}/{d[1]:+.5f})')
    print(f'INCR_DECIDE {int(min(d) > 0 and sum(d) >= 0.0004)}')
