"""Gate for distribution distillation (EXPERIMENTS §28): r3 specialist from scratch (sub23d recipe) whose class-head CE
target mixes the one-hot tick with the tick-class probabilities of a look-ahead class-head teacher (--kd-dist, --kd-w),
soft FT halves as before, vs the current r3 specialist in the sub23g-like blend. The second-seed specialist of the same
recipe is printed as the seed-noise reference. Usage: proj_kd.py <halves tag>"""
exec(open('./pipeline/regime_gates/proj_soft.py').read())
X3k = crossfit(sys.argv[1], X)
sel = np.flatnonzero(r3 & ~np.isnan(X3k[:, 0, 0]))
Qb, Qk = blend_s(X3n, T16, Sn), blend_s(X3k, T16, Sn)
print(f'KD r3 spec (r3 rows) base {wp(X3n, sel):.4f} kd {wp(X3k, sel):.4f}   blend: {fmt(Qb)} -> {fmt(Qk)}')
try:
    X3s = crossfit('ft_mingru_ema_cls3_st_s2_mix4_sa8', X)
    print(f'KD seed-2 reference (r3 rows) {wp(X3s, sel):.4f}   blend: {fmt(blend_s(X3s, T16, Sn))}')
except Exception as e:
    print('KD no seed-2 reference:', e)
d = [wp(Qk, H[f]) - wp(Qb, H[f]) for f in (0, 1)]
print(f'DECIDE_KD {int(min(d) > 0 and sum(d) >= 0.0004)} h0 {d[0]:+.5f} h1 {d[1]:+.5f}')
