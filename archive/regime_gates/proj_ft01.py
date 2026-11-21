"""Gate for the jy447 FT recipe (EXPERIMENTS §32): soft FT halves with unscored weight 0.1 (mix9), 6 epochs, lr 2e-4 for the r3
specialist and the TCN st16, swapped into the sub23g-like blend (base_g) one at a time and together, read at epochs 3 and 6."""
exec(open('./pipeline/regime_gates/proj_soft.py').read())
sel = np.flatnonzero(r3 & ~np.isnan(X3n[:, 0, 0]))
for ep in (2, 3, 4, 5):
    X3e = crossfit('ft_mingru_ema_cls3_st_mix9_sa8e6', X, ep)
    Te = crossfit('ft_tcn64_ema_c3_st16_mix9_sa8e6', T, ep)
    if X3e is None or Te is None:
        print(f'FT01 ep{ep} missing'); continue
    print(f'FT01 ep{ep} r3 {wp(X3e, sel):.4f} vs {wp(X3n, sel):.4f} | TCN {fmt(Te)} vs {fmt(T16)}')
    for nm, Q in (('both', blend_s(X3e, Te, Sn)), ('r3', blend_s(X3e, T16, Sn)), ('TCN', blend_s(X3n, Te, Sn))):
        d = [wp(Q, H[f]) - wp(base_g, H[f]) for f in (0, 1)]
        print(f'FT01 ep{ep} {nm:4s} {fmt(Q)}  DECIDE {int(min(d) > 0 and sum(d) >= 0.0004)} h0 {d[0]:+.5f} h1 {d[1]:+.5f}')
