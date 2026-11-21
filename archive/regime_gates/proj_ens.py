"""Gate for ensemble self-distillation (EXPERIMENTS §27): TCN from scratch, 8 ep, on labels .5 sub24f-route ensemble +
.5 la50 (scripts/ens_label.py), soft FT on the valid halves with la50 as before, vs the la50-only TCN of the same recipe
(st8) — alone and in the sub23g-like blend. Usage: proj_ens.py <halves tag>"""
exec(open('./pipeline/regime_gates/proj_soft.py').read())
T8 = crossfit('ft_tcn64_ema_c3_st8_mix4_sa8', T)
Te = crossfit(sys.argv[1], T)
for nm, Ti in (('la50 st8', T8), ('la50 st16', T16), ('ens+la50', Te)):
    print(f'ENS TCN {nm:9s} {fmt(Ti)}   blend: {fmt(blend_s(X3n, Ti, Sn))}')
d = [wp(blend_s(X3n, Te, Sn), H[f]) - wp(blend_s(X3n, T8, Sn), H[f]) for f in (0, 1)]
print(f'DECIDE_ENS {int(min(d) > 0 and sum(d) >= 0.0004)} vs st8 h0 {d[0]:+.5f} h1 {d[1]:+.5f}')
if len(sys.argv) > 2:  # r3 specialist on the ensemble labels, on top of the new TCN
    X3e = crossfit(sys.argv[2], X)
    sel = np.flatnonzero(r3 & ~np.isnan(X3e[:, 0, 0]))
    Qb, Qe = blend_s(X3n, Te, Sn), blend_s(X3e, Te, Sn)
    print(f'ENS r3 spec la50 (r3 rows) {wp(X3n, sel):.4f} ens {wp(X3e, sel):.4f}   blend: {fmt(Qb)} -> {fmt(Qe)}')
    d = [wp(Qe, H[f]) - wp(Qb, H[f]) for f in (0, 1)]
    print(f'DECIDE_ENS_R3 {int(min(d) > 0 and sum(d) >= 0.0004)} h0 {d[0]:+.5f} h1 {d[1]:+.5f}')
