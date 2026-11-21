"""Gate for a wider r3 specialist (EXPERIMENTS §55): a22_wide.py <FT-halves tag>. Scored rows, cross-fit FT halves, no-stack
blend. (a) replace the r3 specialist, SSM kept (latency: + the extra width on r3 sequences); (b) r3 route without the SSM,
w * wide spec + (1 - w) * TCN (latency ~ neutral for 2x192)."""
import sys
_tag = sys.argv[1]; sys.argv = sys.argv[:1]
exec(open('./pipeline/regime_gates/proj_prop.py').read().split('incr = ')[0])
Qb = blend(BASE); print(f'WIDE base {fmt(Qb)}')
d = lambda Q: [g(y[hs == f], Q[hs == f]) - g(y[hs == f], Qb[hs == f]) for f in (0, 1)]
Xw, Xo, T = rows(_tag), rows(BASE['r3']), rows(BASE['tcn'])
mk = (rg == 3) & ~np.isnan(Xw[:, 0]) & ~np.isnan(Xo[:, 0])
print(f'WIDE member alone r3 rows {g(y[mk], Xw[mk]):.4f} vs {g(y[mk], Xo[mk]):.4f} (corr {np.corrcoef(Xw[mk].ravel(), Xo[mk].ravel())[0, 1]:.3f})')
Q = blend({**BASE, 'r3': _tag}); dd = d(Q)
print(f'WIDE replace, SSM kept: {fmt(Q)} ({dd[0]:+.5f}/{dd[1]:+.5f})  DECIDE {int(min(dd) > 0 and sum(dd) >= 0.0008)}')
for src, nm in ((Xo, 'current spec'), (Xw, 'wide spec')):
    for w in (.5, .6, .7):
        Q = Qb.copy(); Q[mk] = w * src[mk] + (1 - w) * T[mk]; dd = d(Q)
        print(f'WIDE r3 no SSM, {nm} w {w}: {fmt(Q)} ({dd[0]:+.5f}/{dd[1]:+.5f})' + (f'  DECIDE {int(min(dd) > 0 and sum(dd) >= 0.0008)}' if src is Xw else ''))
