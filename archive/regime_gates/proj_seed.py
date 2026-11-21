"""Gate for a second-seed r3 specialist (EXPERIMENTS §25): spec r3 = mean of the seed-1 (sub24) and seed-2 scratch specs,
both soft-FT cross-fit on the valid halves, in the sub23g-like blend (r3 rows only change; the r0/r1 stacks are untouched).
Usage: proj_seed.py <seed-2 halves tag>"""
exec(open('./pipeline/regime_gates/proj_soft.py').read())
X3s = crossfit(sys.argv[1], X)
sel = np.flatnonzero(r3 & ~np.isnan(X3s[:, 0, 0]))
Xa = X3n.copy(); Xa[sel] = .5 * (X3n[sel] + X3s[sel])
for nm, Xi in (('seed1', X3n), ('seed2', X3s), ('mean', Xa)):
    print(f'SEED r3 {nm:6s} (r3 rows) {wp(Xi, sel):.4f}   blend: {fmt(blend_s(Xi, T16, Sn))}')
Q = blend_s(Xa, T16, Sn)
d = [wp(Q, H[f]) - wp(base_g, H[f]) for f in (0, 1)]
print(f'DECIDE_SEED {int(min(d) > 0 and sum(d) >= 0.0004)} h0 {d[0]:+.5f} h1 {d[1]:+.5f}')
