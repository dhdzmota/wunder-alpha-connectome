"""Dense P(is_scored) for every row of train (and valid), for mask-aware NN loss weights.

train has no is_scored column; the mask is predictable from features (AUC ~0.89).
Mask model: compact LightGBM (wnn/features.py row features) on the 128 uniform-random rows per valid
sequence (is_scored = the random draws plus extra rows; the random ones are an unbiased sample), split by
sequence (half of valid sequences fit, half report AUC). Saved normalised to mean 1 over
required rows, float16, shape (n_sequences, 20000), row-group order.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numba as nb
import numpy as np
from sklearn.metrics import roc_auc_score

from wnn import features as fe
from wnn.data import ARTIFACTS, WARMUP, SequenceReader
from wnn.trees import export_booster, predict_row

OUT = ARTIFACTS / "pmask"
SEED = 20261007
KEYS = ("feat", "thr", "left", "right", "leaf", "node_off", "leaf_off")
_ctx = {}


@nb.njit(cache=True)
def _seq_logit(F, m):
    out = np.empty(F.shape[0], np.float32)
    for t in range(F.shape[0]):
        out[t] = predict_row(F[t], m[0], m[1], m[2], m[3], m[4], m[5], m[6])
    return out


def _work(args):
    name, g = args
    if "r" not in _ctx:
        z = np.load(OUT / "mask_model.npz")
        _ctx["m"] = tuple(z[k] for k in KEYS)
        _ctx["r"] = SequenceReader(name)
    F = fe.compute_sequence(_ctx["r"].read(g).X)
    return g, 1.0 / (1.0 + np.exp(-_seq_logit(F, _ctx["m"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--datasets", nargs="+", default=["valid", "train"])
    ap.add_argument("--reuse-model", action="store_true", help="skip fitting; use saved mask_model.npz")
    ap.add_argument("--fit-only", default="", help="fit the mask model into pmask/<this file> and stop (checks)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.fit_only:
        return fit_mask_model(a, a.fit_only)
    if not a.reuse_model:
        fit_mask_model(a)
    for name in a.datasets:
        build(name, a.workers)
    s = SequenceReader("valid").read(0)
    print("sanity valid seq0 AUC", roc_auc_score(s.scored[WARMUP:], np.load(OUT / "valid_pmask.npy")[0, WARMUP:]))


def random_rows(g):
    """Rows of valid sequence g drawn uniformly at random (the seeded draw of the original row sample)."""
    return np.sort(np.random.default_rng([SEED, g, True]).choice(np.arange(WARMUP, 20000), 128, replace=False))


def fit_mask_model(a, fname="mask_model.npz"):
    rd = SequenceReader("valid")
    Xs, ys, gs = [], [], []
    for g in range(rd.n):
        s, r = rd.read(g), random_rows(g)
        Xs.append(fe.compute_sequence(s.X)[r].astype(np.float32))
        ys.append(s.scored[r].astype(float))
        gs.append(np.full(len(r), g))
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    fit = np.random.default_rng(SEED).permutation(np.unique(g))[: len(np.unique(g)) // 2]
    is_fit = np.isin(g, fit)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED,
                  deterministic=True, num_threads=4, verbose=-1)
    m_half = lgb.train(params, lgb.Dataset(X[is_fit], y[is_fit]), a.rounds)
    print(f"mask AUC on held-out valid sequences: {roc_auc_score(y[~is_fit], m_half.predict(X[~is_fit])):.4f}", flush=True)
    model = lgb.train(params, lgb.Dataset(X, y), a.rounds)  # final: all valid sequences
    np.savez(OUT / fname, **export_booster(model))


def build(name, workers):
    n = SequenceReader(name).n
    P = np.lib.format.open_memmap(OUT / f"{name}_pmask.npy", "w+", np.float16, (n, 20000))
    t = time.time()
    with Pool(workers) as pool:
        for i, (gi, p) in enumerate(pool.imap_unordered(_work, [(name, gi) for gi in range(n)], chunksize=4)):
            P[gi] = p
            if i % 1000 == 0:
                print(f"{name} {i}/{n} {time.time() - t:.0f}s", flush=True)
    mean = float(np.mean([P[i, WARMUP:].astype(np.float64).mean() for i in range(n)]))
    for i in range(n):
        P[i] = (P[i].astype(np.float32) / mean).astype(np.float16)
    P.flush()
    print(f"{name}: mean p {mean:.4f}; saved normalised to mean 1", flush=True)


if __name__ == "__main__":
    main()
