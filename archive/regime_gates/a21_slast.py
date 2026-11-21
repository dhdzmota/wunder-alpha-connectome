import sys, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from wnn.data import SequenceReader
A = '/home/user/data/wnn/artifacts/'
det = np.load(A + 'regime/detected_valid.npy'); R = SequenceReader('valid')
mask = np.load(A + 'preds/valid_mask.npy').reshape(len(det), -1).astype(bool)
u = 0.6568677; gs = np.flatnonzero(det == 3)[:240]
D = []
for g in gs:
    s = R.read(int(g)); x = np.round(s.X[:, :22], 3); _, first, inv = np.unique(x, axis=0, return_index=True, return_inverse=True)
    key = np.array([x[f].tobytes() for f in first], dtype=object)[inv]
    K = np.rint(s.Y / u).astype(int); D.append((key, K, mask[g]))
allk = Counter(); [allk.update(d[0].tolist()) for d in D[:60]]
ids = {k: j for j, (k, _) in enumerate(allk.most_common())}; print('state shares', [round(c / sum(allk.values()), 4) for _, c in allk.most_common(12)])
def ctx(key, n):
    sid = np.array([ids.get(k, 99) for k in key]); t = np.arange(20000)
    outs = []; last = np.maximum.accumulate(np.where(sid != 0, t, -1)); cur = last
    for j in range(n):
        v = np.where(cur >= 0, sid[np.maximum(cur, 0)], -1); outs.append(v)
        prev = np.where(cur > 0, last[np.maximum(cur - 1, 0)], -1); cur = prev
    return outs, t - last
for n in (1, 2, 3, 4, 6):
    for agecap in (0, 1):
        tab = defaultdict(Counter); ev = []
        for i, (key, K, mk) in enumerate(D):
            o, age = ctx(key, n); S = np.clip(K[:, 0] + K[:, 1], -3, 0)
            unclipped = (np.abs(K) < 3).all(1)
            sel = mk & unclipped; sel[:100] = False
            a = np.minimum(np.log2(1 + age).astype(int), 6) if agecap else np.zeros(20000, int)
            cx = list(zip(*[v[sel] for v in o], a[sel])); ss = S[sel]
            if i < 120:
                for c, s_ in zip(cx, ss): tab[c][int(s_)] += 1
            else: ev.append((cx, ss))
        mod = {c: v.most_common(1)[0][0] for c, v in tab.items()}
        hit = tot = 0; base = Counter()
        for cx, ss in ev:
            pr = np.array([mod.get(c, -1) for c in cx]); hit += (pr == ss).sum(); tot += len(ss); base.update(ss.tolist())
        print(f'last {n} non-modal states{" + age bucket" if agecap else ""}: acc S on scored unclipped rows {hit / tot:.4f} (base rate {base.most_common(1)[0][1] / tot:.4f}; contexts {len(tab)})', flush=True)
