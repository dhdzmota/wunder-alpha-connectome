"""End-to-end check of a packaged submission, exactly as the platform calls it.

Each worker imports solution.py from the package dir, creates one
PredictionModel, and feeds its sequences row by row through DataPoint
(sequences are processed back-to-back, so state resets are exercised).
Validates outputs (None on warm-up, finite (2,) after), then computes the
official global WP on the valid mask and on all required rows.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wnn.data import WARMUP, SequenceReader, global_wp
from utils import DataPoint


def _run(args):
    sub_dir, groups = args
    sys.path.insert(0, sub_dir)
    from solution import PredictionModel
    model = PredictionModel()
    reader = SequenceReader("valid")
    Ys, Ps, Ms, secs = [], [], [], 0.0
    for g in groups:
        s = reader.read(g)
        P = np.zeros((len(s.X), 2), np.float32)
        t = time.perf_counter()
        for i in range(len(s.X)):
            v = model.predict(DataPoint(s.seq_ix, i, i >= WARMUP, s.X[i]))
            if i < WARMUP:
                assert v is None, "must return None during warm-up"
            else:
                v = np.asarray(v, np.float32)
                assert v.shape == (2,) and np.isfinite(v).all(), "bad prediction"
                P[i] = v
        secs += time.perf_counter() - t
        Ys.append(s.Y); Ps.append(P); Ms.append(s.scored)
    return np.concatenate(Ys), np.concatenate(Ps), np.concatenate(Ms), secs, len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub_dir")
    ap.add_argument("--n-seqs", type=int, default=0, help="0 = all valid sequences")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    n = SequenceReader("valid").n if a.n_seqs == 0 else a.n_seqs
    chunks = [list(range(w, n, a.workers)) for w in range(a.workers)]
    with Pool(a.workers) as pool:
        res = pool.map(_run, [(str(Path(a.sub_dir).resolve()), c) for c in chunks])
    Y = np.concatenate([r[0] for r in res]); P = np.concatenate([r[1] for r in res]); M = np.concatenate([r[2] for r in res])
    secs, cnt = sum(r[3] for r in res), sum(r[4] for r in res)
    need = np.tile(np.arange(20000) >= WARMUP, cnt)
    us = secs / (cnt * 20000) * 1e6
    print(f"sequences={cnt}  latency={us:.1f} us/row (under {a.workers} parallel workers) -> {us * 39.4e6 / 60e6:.1f} min")
    print("official mask:", global_wp(Y, P, M))
    print("all required rows:", global_wp(Y, P, need))


if __name__ == "__main__":
    main()
