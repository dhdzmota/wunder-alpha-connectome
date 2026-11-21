"""Gate for retraining the r0/r1 row stacks (EXPERIMENTS §24) on the new route blend (soft-label members).
Runtime of a package that keeps the sub21 stacks = trees fitted on the OLD blend (old specs + full-FT TCN) applied to the
NEW blend; candidate = trees refit on the NEW blend. Both cross-fit on the FT halves, scored on the sub23g/h-like global blend.
Usage: proj_stack2.py <r0 spec halves tag> <r1 spec halves tag>   (e.g. ft_mingru_ema_cls0_st_mix4_sa8)
Rows: artifacts/stack_rows/stackB_X / stackB_D (scored, required valid rows; scripts/stack_rows.py)."""
import sys
import numpy as np
import lightgbm as lgb
SP = '/home/user/data/wnn/artifacts/stack_rows/'
exec(open('./pipeline/regime_gates/proj_soft.py').read())
spec_new = sys.argv[1:3]
T16 = crossfit('ft_tcn64_ema_c3_st16_mix4_sa8', T)
Xn = X3n.copy()
for r in (0, 1):
    Xn = crossfit(spec_new[r], Xn)
Qn = blend_s(Xn, T16, Sn)                      # new global blend (r0/r1 rows replaced below)
qn01, qo01 = Qn.copy(), Qn.copy()
for r in (0, 1):
    s = det == r
    qn01[s] = .6 * Xn[s] + .4 * T16[s]          # new r0/r1 route blend (no SSM)
    qo01[s] = .6 * X[s] + .4 * T[s]             # old one, the sub21 stack's training input
idx = np.flatnonzero(m.ravel()); seq = idx // 20000
fl = lambda A_: A_.reshape(-1, 2)[idx]
y, qn, qo, qg = fl(Y), fl(qn01), fl(qo01), fl(Qn)
F = np.c_[np.load(SP + 'stackB_X.npy'), np.load(SP + 'stackB_D.npy'), np.zeros((len(idx), 2), np.float32)]
rg = det[seq]; yc = np.clip(y, -2, 2); w = np.abs(yc) + 1e-3
h = np.zeros(len(det), int); h[H[1]] = 1; hs = h[seq]
g = lambda yy, qq: np.mean([weighted_pearson(yy[:, k], qq[:, k]) for k in range(2)])
P = dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4, seed=1)


def stack_oof(k, fit_q, apply_q, trees=200):
    mk = rg == k; Ff = F[mk].copy(); Fa = F[mk].copy(); Ff[:, -2:] = fit_q[mk]; Fa[:, -2:] = apply_q[mk]
    Z = np.zeros((mk.sum(), 2)); hk = hs[mk]
    for t in range(2):
        for f in (0, 1):
            b = lgb.train(P, lgb.Dataset(Ff[hk != f], yc[mk][hk != f, t], weight=w[mk][hk != f, t]), trees)
            Z[hk == f, t] = b.predict(Fa[hk == f])
    return mk, Z


def report(name, M):
    print(f'STACK {name:28s} {g(y, M):.4f} | h0 {g(y[hs == 0], M[hs == 0]):.4f} h1 {g(y[hs == 1], M[hs == 1]):.4f}', flush=True)


nost = qg.copy()
for k in (0, 1):
    mk = rg == k; nost[mk] = qn[mk]
report('new blend, no stack', nost)
Zs = {}
for k in (0, 1):
    Zs['old', k] = stack_oof(k, qo, qn)
    Zs['new', k] = stack_oof(k, qn, qn)
for kind in ('old', 'new'):
    for a in ((.35,) if kind == 'old' else (.25, .35, .45)):
        M = nost.copy()
        for k in (0, 1):
            mk, Z = Zs[kind, k]; M[mk] = (1 - a) * qn[mk] + a * Z
        report(f'{kind} stack a{a}', M)
# decision: best refit alpha must beat the kept (old) stack on both halves, by >= +0.0004 in sum
Mo = nost.copy()
for k in (0, 1):
    mk, Z = Zs['old', k]; Mo[mk] = .65 * qn[mk] + .35 * Z
best = None
for a in (.25, .35, .45):
    M = nost.copy()
    for k in (0, 1):
        mk, Z = Zs['new', k]; M[mk] = (1 - a) * qn[mk] + a * Z
    d = [g(y[hs == f], M[hs == f]) - g(y[hs == f], Mo[hs == f]) for f in (0, 1)]
    if min(d) > 0 and sum(d) >= 0.0004 and (best is None or sum(d) > best[1]):
        best = (a, sum(d))
print(f'DECIDE_STACK {best[0] if best else 0}')
