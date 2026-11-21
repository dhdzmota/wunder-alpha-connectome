"""Gate for label-side variants of the r3 specialist (EXPERIMENTS §41): each <halves tag> (soft FT halves of an r3 spec from
scratch) in the sub23g-like blend, (a) replacing the current r3 spec, (b) averaged with it (pair of r3 specialists: +~33 us/row
on r3 sequences only, fits the sub24f latency headroom). Usage: proj_lab.py <tag> [<tag> ...]"""
import sys
_tags = sys.argv[1:]
exec(open('./pipeline/regime_gates/proj_soft.py').read())
Qb = blend_s(X3n, T16, Sn)
print(f'LAB base sub23g-like {fmt(Qb)}')
X3s = crossfit('ft_mingru_ema_cls3_st_s2_mix4_sa8', X)
for tg in ['ft_mingru_ema_cls3_st_s2_mix4_sa8'] + _tags:
    Xk = X3s if tg.endswith('st_s2_mix4_sa8') else crossfit(tg, X)
    if Xk is None:
        print(f'LAB {tg}: missing halves'); continue
    sel = np.flatnonzero(r3 & ~np.isnan(Xk[:, 0, 0]))
    Xp = X3n.copy(); Xp[sel] = .5 * (X3n[sel] + Xk[sel])
    Qr, Qp = blend_s(Xk, T16, Sn), blend_s(Xp, T16, Sn)
    dr = [wp(Qr, H[f]) - wp(Qb, H[f]) for f in (0, 1)]
    dp = [wp(Qp, H[f]) - wp(Qb, H[f]) for f in (0, 1)]
    c = np.corrcoef(Xk[sel].reshape(-1), X3n[sel].reshape(-1))[0, 1]
    print(f'LAB {tg}: r3 rows {wp(Xk, sel):.4f} vs {wp(X3n, sel):.4f} (corr {c:.3f}) | replace {fmt(Qr)} '
          f'({dr[0]:+.5f}/{dr[1]:+.5f}) | pair {fmt(Qp)} ({dp[0]:+.5f}/{dp[1]:+.5f})')
    print(f'DECIDE_LAB {tg} replace {int(min(dr) > 0 and sum(dr) >= 0.0004)} pair {int(min(dp) > 0 and sum(dp) >= 0.0004)}')
