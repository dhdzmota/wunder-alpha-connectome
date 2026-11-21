"""Response re-expressed as sum / difference (EXPERIMENTS §27): p0 = (l_s * s_hat + l_d * d_hat) / 2, p1 = (l_s * s_hat - l_d * d_hat) / 2
with s_hat = p0 + p1, d_hat = p0 - p1 and per-regime factors (l_s, l_d) fitted on one valid half, scored on the other
(sub23g-like blend). l_s < 1 shrinks the noisy sum toward 0 relative to the difference."""
exec(open('./pipeline/regime_gates/proj_soft.py').read())
P = base_g
grid = np.round(np.arange(0, 2.01, .1), 2)
def remix(Q, ls):
    out = Q.copy()
    for r, l in ls.items():
        k = det == r
        s = Q[k, :, 0] + Q[k, :, 1]; d = Q[k, :, 0] - Q[k, :, 1]
        out[k, :, 0] = (l * s + d) / 2; out[k, :, 1] = (l * s - d) / 2
    return out
for f in (0, 1):
    tr, te = H[f], H[1 - f]
    ls = {r: 1.0 for r in range(4)}
    for _ in range(2):
        for r in range(4):
            ls[r] = max(grid, key=lambda v: wp(remix(P, {**ls, r: v}), tr))
    print(f'SUMDIFF fit h{f}: l_s {ls}  eval h{1 - f}: {wp(P, te):.4f} -> {wp(remix(P, ls), te):.4f}', flush=True)
