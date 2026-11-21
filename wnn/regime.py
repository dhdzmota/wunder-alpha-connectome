"""Target tick regime of a sequence from its warm-up rows (multinomial logistic regression, numpy only).

The target of a sequence is k * unit with one of four units (REGIME_UNITS in wnn/nets.py); the regime
is read once per sequence, from rows 0..98, before the first required prediction. detector.npz: mu, sd, W, b
(fit in artifacts/regime, valid accuracy 0.9989).
"""
import numpy as np


def warmup_stats(X):
    """X: (99, n_raw) warm-up rows -> [mean, std, change rate, mean |change|] per column."""
    X = np.asarray(X, np.float64)
    d = np.diff(X, axis=0)
    return np.concatenate([X.mean(0), X.std(0), (d != 0).mean(0), np.abs(d).mean(0)])


def detect(X, det):
    """Regime index 0..3 (units .125 / .241 / .469 / .657)."""
    f = (warmup_stats(X) - det["mu"]) / det["sd"]
    return int(np.argmax(det["W"] @ f + det["b"]))
