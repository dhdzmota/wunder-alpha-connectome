import sys, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from wnn.data import SequenceReader
A = '/home/user/data/wnn/artifacts/'
det = np.load(A + 'regime/detected_valid.npy'); R = SequenceReader('valid')
mask = np.load(A + 'preds/valid_mask.npy').reshape(len(det), -1)
u = 0.6568677
gs = np.flatnonzero(det == 3)[:80]
DEL = (-2, -1, 0, 1, 2, 3, 5, 10, 20, 50, 100)
tab = {d: defaultdict(Counter) for d in DEL}; tabm = {d: defaultdict(Counter) for d in DEL}
Sall = Counter(); Ssc = Counter(); pairs = Counter()
runs = []
for g in gs[:60]:
    s = R.read(int(g)); x = np.round(s.X[:, :22], 3); _, sid = np.unique(x, axis=0, return_inverse=True)
    key = np.array([x[np.flatnonzero(sid == j)[0]].tobytes() for j in range(sid.max() + 1)], dtype=object)[sid]
    K = np.rint(s.Y / u).astype(int); S = K[:, 0] + K[:, 1]; mk = mask[g].astype(bool)
    Sall.update(S[99:].tolist()); Ssc.update(S[mk].tolist()); pairs.update(map(tuple, K[mk].tolist()))
    ch = np.flatnonzero(np.diff(S) != 0); runs += np.diff(ch).tolist()
    for d in DEL:
        t = np.arange(max(99, -d), 20000 - max(d, 0))
        for kk, ss, m_ in zip(key[t + d], S[t], mk[t]):
            tab[d][kk][ss] += 1
            if m_: tabm[d][kk][ss] += 1
print('S (ticks) all rows:', sorted(Sall.items())); print('S scored rows:', sorted(Ssc.items()))
print('top (k0,k1) on scored rows:', pairs.most_common(10))
print('S run length between changes: median %d p10 %d p90 %d' % tuple(np.percentile(runs, [50, 10, 90])))
for d in DEL:
    for nm, T in (('all', tab[d]), ('scored', tabm[d])):
        tot = sum(sum(c.values()) for c in T.values()); hit = sum(c.most_common(1)[0][1] for c in T.values())
        print(f'delta {d:+4d} {nm:6s}: P(S = modal S | price state at t+delta) = {hit / tot:.4f}   ({len(T)} states)')
