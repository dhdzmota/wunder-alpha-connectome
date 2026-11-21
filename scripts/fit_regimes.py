"""Per-sequence tick units of the target and the warm-up regime detector.

Every sequence has its own target grid: y = k * unit with unit in REGIME_UNITS (the same for t0 and t1, grid
residual ~1e-6). unit = the smallest nonzero |y| of the sequence -> preds/units_<name>.npy ((n_seq, 2), per target).
Regime = searchsorted(REGIME_EDGES, unit). The runtime cannot see y, so the regime is predicted from the 99
warm-up rows: multinomial logistic regression on wnn.regime.warmup_stats, fit on train, accuracy on valid.

The shipped artifacts/regime/detector.npz came from an interactive session (valid accuracy 0.9989); this script
rebuilds an equivalent one and reports its agreement with the shipped detector. Outputs go to artifacts/regime/refit/
unless --write, which replaces preds/units_*.npy and regime/detector.npz.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression

from wnn.data import ARTIFACTS, WARMUP, SequenceReader
from wnn.nets import REGIME_EDGES
from wnn.regime import detect, warmup_stats


def scan(name):
    """(units (n, 2), warm-up stats (n, 448)) of every sequence."""
    rd = SequenceReader(name)
    U, S = np.zeros((rd.n, 2)), []
    for g in range(rd.n):
        s = rd.read(g)
        a = np.abs(s.Y.astype(np.float64))
        U[g] = [a[:, k][a[:, k] > 0].min() for k in range(2)]
        S.append(warmup_stats(s.X[:WARMUP]))  # float32 rows, as DataPoint.state at inference
        if g % 2000 == 0:
            print(f"{name} {g}/{rd.n}", flush=True)
    return U, np.array(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--write", action="store_true", help="overwrite preds/units_*.npy and regime/detector.npz")
    a = ap.parse_args()
    out = ARTIFACTS / "regime" / "refit"
    out.mkdir(parents=True, exist_ok=True)
    data = {n: scan(n) for n in ("train", "valid")}
    for n, (U, _) in data.items():
        old = np.load(ARTIFACTS / "preds" / f"units_{n}.npy")
        print(f"units {n}: max |new - saved| {np.abs(U - old).max():.1e}; regime mix "
              f"{np.bincount(np.searchsorted(REGIME_EDGES, U[:, 0]), minlength=4).tolist()}")
        np.save((ARTIFACTS / "preds" if a.write else out) / f"units_{n}.npy", U)

    (Ut, St), (Uv, Sv) = data["train"], data["valid"]
    rt, rv = np.searchsorted(REGIME_EDGES, Ut[:, 0]), np.searchsorted(REGIME_EDGES, Uv[:, 0])
    mu, sd = St.mean(0), np.maximum(St.std(0), 1e-9)
    lr = LogisticRegression(C=a.C, max_iter=5000).fit((St - mu) / sd, rt)
    det = {"mu": mu, "sd": sd, "W": lr.coef_.astype(np.float64), "b": lr.intercept_.astype(np.float64)}
    shipped = dict(np.load(ARTIFACTS / "regime" / "detector.npz"))
    rd = SequenceReader("valid")
    pn = np.array([detect(rd.read(g).X[:WARMUP], det) for g in range(rd.n)])
    po = np.array([detect(rd.read(g).X[:WARMUP], shipped) for g in range(rd.n)])
    print(f"valid accuracy: refit {np.mean(pn == rv):.4f}, shipped {np.mean(po == rv):.4f}; "
          f"refit == shipped on {np.mean(pn == po):.4f} of valid sequences")
    np.savez((ARTIFACTS / "regime" if a.write else out) / "detector.npz", **det)
    np.save(out / "detected_valid.npy", pn)


if __name__ == "__main__":
    main()
