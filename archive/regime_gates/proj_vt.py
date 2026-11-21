"""Gate for the valid-adapted teacher (EXPERIMENTS §38): soft FT halves of the r3 specialist and the TCN st16 with labels from
the quarter-cross-fit valid teacher (suffix _sa8vt), swapped into the sub23g-like blend (base_g) one at a time and together."""
exec(open('./pipeline/regime_gates/proj_soft.py').read())
sel = np.flatnonzero(r3 & ~np.isnan(X3n[:, 0, 0]))
X3e = crossfit('ft_mingru_ema_cls3_st_mix4_sa8vt', X)
Te = crossfit('ft_tcn64_ema_c3_st16_mix4_sa8vt', T)
print(f'VT r3 {wp(X3e, sel):.4f} vs {wp(X3n, sel):.4f} | TCN {fmt(Te)} vs {fmt(T16)}' if Te is not None else
      f'VT r3 {wp(X3e, sel):.4f} vs {wp(X3n, sel):.4f} | TCN missing')
for nm, Q in (('both', None if Te is None else blend_s(X3e, Te, Sn)), ('r3', blend_s(X3e, T16, Sn)),
              ('TCN', None if Te is None else blend_s(X3n, Te, Sn))):
    if Q is None:
        continue
    d = [wp(Q, H[f]) - wp(base_g, H[f]) for f in (0, 1)]
    print(f'VT {nm:4s} {fmt(Q)}  DECIDE_VT {int(min(d) > 0 and sum(d) >= 0.0004)} h0 {d[0]:+.5f} h1 {d[1]:+.5f}')
