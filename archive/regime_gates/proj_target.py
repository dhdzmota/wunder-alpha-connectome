"""Target-structure probe (EXPERIMENTS §27, idea 2): tick decomposition of (t0, t1) per regime, predictability of the
sum / difference by the sub23g-like blend, and the share of the metric carried by each tick pattern."""
exec(open('./pipeline/regime_gates/proj_soft.py').read())
from collections import Counter
P = base_g
units = [.125433, .241, .469, .657]
mm = m.ravel(); y = Y.reshape(-1, 2)[mm]; p = P.reshape(-1, 2)[mm]; rr = np.repeat(det, 20000)[mm]
print('rows', len(y), 'WP t0', round(weighted_pearson(y[:, 0], p[:, 0]), 4), 't1', round(weighted_pearson(y[:, 1], p[:, 1]), 4))
s, d = y[:, 0] + y[:, 1], y[:, 0] - y[:, 1]; ps, pd = p[:, 0] + p[:, 1], p[:, 0] - p[:, 1]
print('sum: WP', round(weighted_pearson(s, ps), 4), 'std', round(float(s.std()), 3), '| diff: WP', round(weighted_pearson(d, pd), 4), 'std', round(float(d.std()), 3))
for r in range(4):
    k = rr == r; u = units[r]
    kk = np.round(y[k] / u).astype(int)
    c = Counter(map(tuple, kk)); tot = sum(c.values())
    wt = np.abs(y[k]).sum(1); wsum = wt.sum()
    top = c.most_common(8)
    share = {pat: round(float(wt[(kk[:, 0] == pat[0]) & (kk[:, 1] == pat[1])].sum() / wsum), 3) for pat, _ in top}
    print(f'r{r} rows {k.sum()} resid-off-grid {float(np.abs(y[k] / u - kk).max()):.3f} | top (k0,k1): rows%/metric-weight%',
          [(pat, round(n / tot, 3), share[pat]) for pat, n in top])
    print(f'   k0+k1 dist', sorted(Counter(kk.sum(1)).items(), key=lambda x: -x[1])[:6],
          ' WP sum', round(weighted_pearson(s[k], ps[k]), 4), 'diff', round(weighted_pearson(d[k], pd[k]), 4),
          ' t0', round(weighted_pearson(y[k, 0], p[k, 0]), 4), 't1', round(weighted_pearson(y[k, 1], p[k, 1]), 4))
