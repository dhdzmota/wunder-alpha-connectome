"""Offline check of per-sequence online calibration with delayed teacher pseudo-labels (EXPERIMENTS §26).
At step t the look-ahead teacher's output for row t-LAG only needs rows <= t (teacher_la50 reads up to 50 rows ahead),
so it is a causal, delayed estimate of the realized target (valid corr ~0.80). Per sequence, an EMA of the teacher-vs-blend
gap (shift) or of their covariance (scale) corrects the blend: oracle per-seq shift 0.7119 / scale 0.7079 vs 0.6871 (§16).
Fit on one valid half, score on the other. Base = sub23g-like OOF blend (proj_soft base_g)."""
from scipy.signal import lfilter
exec(open('./pipeline/regime_gates/proj_soft.py').read())
P = base_g.astype(np.float32)
del G, X, X3, X3n, Sp, T16
LAG = int(sys.argv[1]) if len(sys.argv) > 1 else 50
TT = np.load(A / f'nn/{sys.argv[2] if len(sys.argv) > 2 else "teacher_la50_gru"}/pred_valid.npy').reshape(-1, 20000, 2).astype(np.float32)
print(f'base {fmt(P)}   teacher {fmt(TT)}', flush=True)


def ema(z, span):
    a = 2 / (span + 1)
    num = lfilter([a], [1, a - 1], z, axis=1)
    den = lfilter([a], [1, a - 1], np.ones_like(z[:1]), axis=1)
    return (num / den).astype(np.float32)


def lag(z):
    out = np.zeros_like(z); out[:, LAG:] = z[:, :-LAG]
    return out


for span in (500, 2000, 8000):
    d = lag(ema(TT - P, span))                           # delayed mean gap (shift)
    cv = lag(ema(TT * P, span)); vp = lag(ema(P * P, span))
    k = cv / np.maximum(vp, 1e-6)                        # delayed regression slope teacher ~ blend (scale)
    for b in (.25, .5, 1.0):
        print(f'TTA shift span {span:5d} b {b:.2f}  {fmt(P + b * d)}', flush=True)
        print(f'TTA scale span {span:5d} b {b:.2f}  {fmt(P * np.clip(1 + b * (k - 1), .25, 4))}', flush=True)
    del d, cv, vp, k
