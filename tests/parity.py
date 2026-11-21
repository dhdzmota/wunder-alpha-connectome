"""Prediction parity between two submission packages (e.g. the submitted zip and a rebuilt one).

Runs each package's PredictionModel in its own subprocess on the same valid sequences, back to back
(state resets exercised), exactly as the platform calls it, and compares the outputs row by row.

    python tests/parity.py submissions/sub26a_route4_r3la25_stack.zip artifacts/submissions/sub26a_clean
"""
import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

RUN = r"""
import sys, numpy as np
pkg, out, groups = sys.argv[1], sys.argv[2], [int(g) for g in sys.argv[3].split(",")]
sys.path.insert(0, sys.argv[4])
from utils import DataPoint
from wnn.data import WARMUP, SequenceReader
sys.path.insert(0, pkg)
from solution import PredictionModel
model, reader, P = PredictionModel(), SequenceReader("valid"), []
for g in groups:
    s = reader.read(g)
    for i in range(len(s.X)):
        v = model.predict(DataPoint(s.seq_ix, i, i >= WARMUP, s.X[i]))
        P.append(np.full(2, np.nan, np.float32) if v is None else np.asarray(v, np.float32))
np.save(out, np.array(P))
"""


def default_groups(per_regime=2):
    """Sequence 0 plus the first `per_regime` valid sequences of each tick regime (all four routes)."""
    from wnn.data import ARTIFACTS
    units = np.load(ARTIFACTS / "preds" / "units_valid.npy")[:, 0]
    reg = np.searchsorted((0.2, 0.35, 0.55), units)
    picks = [0] + [int(g) for r in range(4) for g in np.flatnonzero(reg == r)[:per_regime]]
    return sorted(set(picks))


def run(pkg, groups, tmp):
    pkg = Path(pkg)
    if pkg.suffix == ".zip":
        d = Path(tmp) / pkg.stem
        zipfile.ZipFile(pkg).extractall(d)
        pkg = d
    out = Path(tmp) / f"{pkg.name}.npy"
    subprocess.run([sys.executable, "-c", RUN, str(pkg), str(out), ",".join(map(str, groups)), str(REPO)], check=True)
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="package dir or .zip")
    ap.add_argument("candidate", help="package dir or .zip")
    ap.add_argument("--groups", default="", help="comma list of valid sequences (default: 0 + 2 per regime)")
    ap.add_argument("--tol", type=float, default=0.0, help="max allowed |difference|")
    a = ap.parse_args()
    groups = [int(g) for g in a.groups.split(",")] if a.groups else default_groups()
    with tempfile.TemporaryDirectory() as tmp:
        ref, new = run(a.reference, groups, tmp), run(a.candidate, groups, tmp)
    same_nan = np.array_equal(np.isnan(ref), np.isnan(new))
    diff = float(np.nanmax(np.abs(ref - new)))
    print(f"sequences {groups}: {len(ref)} rows, warm-up None pattern equal: {same_nan}, max |diff| {diff:.3e}")
    if not same_nan or diff > a.tol:
        sys.exit("PARITY FAILED")
    print("PARITY OK")


if __name__ == "__main__":
    main()
