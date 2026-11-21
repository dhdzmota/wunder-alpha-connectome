"""Probe (EXPERIMENTS §52): does the PATH of the second instrument carry direction for i0 that the models miss?
Permutation importance (§37): the r3 specialist ignores i1 (-0.002) while r2 leans on it (-0.10). The 22 price columns of an
instrument take few joint values per regime (state = book change of the row, §43), so the net drift of i1 over a window is a
linear function of the counts of each state in that window. Features at scored rows: counts of the top-K joint price states
of i1 (control: of i0) over trailing windows. Ridge stack y ~ [p0, p1, counts], weights |y|, cross-fit on the FT halves.
Usage: a18_i1path.py <regime> [K]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from wnn.data import SequenceReader
from collections import Counter
RG = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
sys.argv = sys.argv[:1]
exec(open('./pipeline/regime_gates/proj_soft.py').read().split("al = np.arange")[0])
Sn = crossfit('ft_ssm16_c3_st_mix4_sa8', S); T16 = crossfit('ft_tcn64_ema_c3_st16_mix4_sa8', T)
Xn = crossfit(f'ft_mingru_ema_cls{RG}_st_mix4_sa8', X)
if Xn is None: Xn = crossfit(f'ft_mingru_ema_cls{RG}_mix4_sa8', X)
rs = np.flatnonzero((det == RG) & ~np.isnan(Xn[:, 0, 0]))
p = (.5 * Xn[rs] + .25 * (Sn[rs] + T16[rs])).astype(np.float32)
mm = m[rs]; y = np.clip(Y[rs], -2, 2).astype(np.float32); half = np.isin(rs, H[0])
del G, S, T, X, Sn, T16, Xn, Y
R = SequenceReader('valid')
WIN = (20, 100, 500, 2500)
from utils import weighted_pearson
wpr = lambda yy, qq: np.mean([weighted_pearson(yy[:, k], qq[:, k]) for k in range(2)])


def feats(c0):
    keys = []; cnt = Counter()
    for i, g in enumerate(rs):
        x = np.round(R.read(int(g)).X[:, c0:c0 + 22], 3)
        _, inv = np.unique(x, axis=0, return_inverse=True)
        u, first = np.unique(inv, return_index=True)
        k = np.array([x[f].tobytes() for f in first], dtype=object)[inv]
        keys.append(k)
        if i < 120: cnt.update(k.tolist())
    top = {s: j for j, (s, _) in enumerate(cnt.most_common(K))}
    tot = sum(cnt.values()); print(f'  cols {c0}: {len(cnt)} states in 120 seqs; top-{K} cover {sum(c for _, c in cnt.most_common(K)) / tot:.4f}; '
                                    f'shares {" ".join(f"{c / tot:.3f}" for _, c in cnt.most_common(8))}', flush=True)
    F = []
    for i in range(len(rs)):
        sid = np.array([top.get(s, K) for s in keys[i]]); oh = np.zeros((20001, K + 1), np.float32); oh[np.arange(1, 20001), sid] = 1
        cs = np.cumsum(oh, 0); t = np.flatnonzero(mm[i]) + 1
        F.append(np.concatenate([(cs[t] - cs[np.maximum(t - w, 0)]) / w for w in WIN], 1))
    return F


def stack(F, name):
    out = []
    for fo in (0, 1):
        tr = np.flatnonzero(half != (fo == 0)); te = np.flatnonzero(half == (fo == 0))
        def mat(ix, withF):
            P = np.concatenate([p[i][mm[i]] for i in ix]); cols = [P]
            if withF: cols.append(np.concatenate([F[i] for i in ix]))
            return np.concatenate(cols, 1), np.concatenate([y[i][mm[i]] for i in ix])
        res = {}
        for withF in (False, True):
            Xt, yt = mat(tr, withF); Xe, ye = mat(te, withF)
            mu = Xt.mean(0); sd = Xt.std(0) + 1e-6; Xt = (Xt - mu) / sd; Xe = (Xe - mu) / sd
            pr = np.zeros_like(ye)
            for k in range(2):
                w = np.abs(yt[:, k]); A_ = Xt * np.sqrt(w)[:, None]
                for lam in ((0.0,) if not withF else (1.0, 100.0)):
                    b = np.linalg.solve(A_.T @ A_ + lam * w.sum() * 1e-4 * np.eye(Xt.shape[1]), A_.T @ (np.sqrt(w) * (yt[:, k] - np.average(yt[:, k], weights=w))))
                    q = Xe @ b
                    res.setdefault((withF, lam), np.zeros_like(ye))[:, k] = q
            for key, q in res.items():
                if key[0] == withF: out.append((fo, key, wpr(ye, q), wpr(ye, Xe[:, :2] * sd[:2] + mu[:2])))
    for fo, key, v, raw in out:
        print(f'  {name} half {fo} {"p+counts lam %g" % key[1] if key[0] else "p only      "}: {v:.4f} (raw blend {raw:.4f})', flush=True)


print(f'regime {RG}: {len(rs)} seqs, scored rows {int(mm.sum())}', flush=True)
for c0, nm in ((52, 'i1'), (0, 'i0')):
    F = feats(c0); stack(F, nm); del F
print('DONE')
