"""Gate for per-regime output scaling (EXPERIMENTS §25): the metric is one weighted Pearson over all rows, so the relative
scale (and offset) of each route's predictions matters. On the sub24-like OOF blend (proj_stack2, refit stack a .35),
fit a per-regime, per-target scale (and optionally offset) on one valid half by coordinate search, score on the other.
Usage: proj_scale.py <r0 spec halves tag> <r1 spec halves tag>"""
exec(open('./pipeline/regime_gates/proj_stack2.py').read())
M = nost.copy()
for k in (0, 1):
    mk, Z = Zs['new', k]; M[mk] = .65 * qn[mk] + .35 * Z
regs = np.unique(rg)
print('regimes', regs, 'row share', [round(float(np.mean(rg == r)), 3) for r in regs])
wpk = lambda yy, qq: weighted_pearson(yy, qq)
grid_s = np.exp(np.linspace(np.log(.6), np.log(1.6), 41))


def fit(sel, offsets):
    """per target: scales (and offsets) per regime, last regime fixed at scale 1, by 3 rounds of coordinate search."""
    P = {}
    for t in range(2):
        yy, qq, rr = y[sel, t], M[sel, t], rg[sel]
        sc = {r: 1.0 for r in regs}; of = {r: 0.0 for r in regs}
        sd = float(np.std(qq))
        ap = lambda: qq * np.vectorize(sc.get)(rr) + np.vectorize(of.get)(rr)
        for _ in range(3):
            for r in regs:
                if r != regs[-1]:
                    best = max(grid_s, key=lambda v: (sc.__setitem__(r, v), wpk(yy, ap()))[1]); sc[r] = best
                if offsets:
                    best = max(np.linspace(-.3, .3, 25) * sd, key=lambda v: (of.__setitem__(r, v), wpk(yy, ap()))[1]); of[r] = best
        P[t] = (sc, of)
    return P


def apply(P, sel):
    out = M[sel].copy()
    for t in range(2):
        sc, of = P[t]
        out[:, t] = M[sel, t] * np.vectorize(sc.get)(rg[sel]) + np.vectorize(of.get)(rg[sel])
    return out


for offsets in (False, True):
    d = []
    for f in (0, 1):
        tr, te = np.flatnonzero(hs == f), np.flatnonzero(hs != f)
        P = fit(tr, offsets)
        b, a = g(y[te], M[te]), g(y[te], apply(P, te))
        d.append(a - b)
        print(f'SCALE offsets={offsets} fit h{f} -> eval h{1 - f}: {b:.4f} -> {a:.4f} ({a - b:+.5f})',
              {t: ({int(r): round(float(v), 3) for r, v in P[t][0].items()},
                   {int(r): round(float(v), 4) for r, v in P[t][1].items()}) for t in range(2)}, flush=True)
    print(f'DECIDE_SCALE offsets={offsets} {int(min(d) > 0 and sum(d) >= 0.0004)} sum {sum(d):+.5f}')
