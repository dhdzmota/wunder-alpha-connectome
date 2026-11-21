"""Probe (EXPERIMENTS §53): in r3 is S = t0 + t1 (= -spread, in ticks) exactly reconstructible from the price states by
integration, and does it add to the blend? Oracle with the true S: r3 0.4363 -> 0.5211 (§16). The price state of a row encodes
the CHANGE of the book, so the spread level is the running sum of the decoded changes (dictionary state -> modal dS from labels
of other sequences) with a causal anchor (running mode of the level = spread 1). Stack on scored rows, cross-fit FT halves:
y ~ p (.) one-hot(S_hat) vs the same with the true S vs p only."""
import sys
import numpy as np
sys.path.insert(0, '.')
from wnn.data import SequenceReader
from collections import Counter, defaultdict
sys.argv = sys.argv[:1]
exec(open('./pipeline/regime_gates/proj_soft.py').read().split("al = np.arange")[0])
Sn = crossfit('ft_ssm16_c3_st_mix4_sa8', S); T16 = crossfit('ft_tcn64_ema_c3_st16_mix4_sa8', T); Xn = crossfit('ft_mingru_ema_cls3_st_mix4_sa8', X)
rs = np.flatnonzero((det == 3) & ~np.isnan(Xn[:, 0, 0]))
p = (.5 * Xn[rs] + .25 * (Sn[rs] + T16[rs])).astype(np.float32)
mm = m[rs]; y = np.clip(Y[rs], -2, 2).astype(np.float32); half = np.isin(rs, H[0]); Yr = Y[rs].copy()
del G, S, T, X, Sn, T16, Xn, Y
R = SequenceReader('valid'); u = 0.6568677
from utils import weighted_pearson
wpr = lambda yy, qq: np.mean([weighted_pearson(yy[:, k], qq[:, k]) for k in range(2)])
keys = []; Ks = []
for g in rs:
    x = np.round(R.read(int(g)).X[:, :22], 3); _, first, inv = np.unique(x, axis=0, return_index=True, return_inverse=True)
    keys.append(np.array([x[f].tobytes() for f in first], dtype=object)[inv])
for i in range(len(rs)):
    K = np.rint(Yr[i] / u).astype(int); Ks.append(K[:, 0] + K[:, 1])
# dictionaries per half (fit on the other half): state -> modal dS
Shat = [None] * len(rs); acc = []
for fo in (True, False):
    dic = defaultdict(Counter)
    for i in np.flatnonzero(half != fo)[:150]:
        dS = np.r_[0, np.diff(Ks[i])]
        for k_, d_ in zip(keys[i][100:], dS[100:]): dic[k_][int(d_)] += 1
    dm = {k_: c.most_common(1)[0][0] for k_, c in dic.items()}
    pur = sum(c.most_common(1)[0][1] for c in dic.values()) / sum(sum(c.values()) for c in dic.values())
    print(f'dictionary: {len(dm)} states, dS modal purity {pur:.4f}, moves {Counter(dm.values())}', flush=True)
    for i in np.flatnonzero(half == fo):
        d = np.array([dm.get(k_, 0) for k_ in keys[i]]); c = np.cumsum(d)
        # causal anchor: the most frequent level so far is spread 1 (S = -1)
        lv, li = np.unique(c, return_inverse=True); cnt = np.zeros(len(lv), int); mode = np.empty(20000, int)
        best = 0
        for t in range(20000):
            cnt[li[t]] += 1
            if cnt[li[t]] > cnt[best]: best = li[t]
            mode[t] = lv[best]
        Shat[i] = (c - mode - 1).astype(np.int8)
        acc.append(((Shat[i] == Ks[i])[mm[i]]).mean())
print(f'S_hat == S on scored rows: mean {np.mean(acc):.4f}, median {np.median(acc):.4f}, p10 {np.percentile(acc, 10):.4f}', flush=True)


def design(ix, src):
    P = np.concatenate([p[i][mm[i]] for i in ix]); s = np.concatenate([np.clip(src[i][mm[i]], -3, 0) for i in ix])
    cols = [P]
    if src is not None:
        for v in (-3, -2, 0):
            ind = (s == v).astype(np.float32)[:, None]; cols += [ind, ind * P]
    return np.concatenate(cols, 1)


for nm, src in (('p only', None), ('p x S_hat', Shat), ('p x true S', Ks)):
    for fo in (True, False):
        tr = np.flatnonzero(half != fo); te = np.flatnonzero(half == fo)
        if src is None:
            Xt = np.concatenate([p[i][mm[i]] for i in tr]); Xe = np.concatenate([p[i][mm[i]] for i in te])
        else:
            Xt = design(tr, src); Xe = design(te, src)
        yt = np.concatenate([y[i][mm[i]] for i in tr]); ye = np.concatenate([y[i][mm[i]] for i in te]); q = np.zeros_like(ye)
        Xt = np.c_[Xt, np.ones(len(Xt))]; Xe2 = np.c_[Xe, np.ones(len(Xe))]
        for k in range(2):
            w = np.sqrt(np.abs(yt[:, k])); b = np.linalg.lstsq(Xt * w[:, None], yt[:, k] * w, rcond=None)[0]; q[:, k] = Xe2 @ b
        print(f'{nm:12s} half {int(not fo)}: {wpr(ye, q):.4f} (raw blend {wpr(ye, Xe[:, :2]):.4f})', flush=True)
print('DONE')
