"""A/B latency of packaged submissions under the same machine load (interleaved blocks, median)."""
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from utils import DataPoint
from wnn.data import ARTIFACTS, SequenceReader


def load(d):
    sys.path.insert(0, str(d))
    for m in [m for m in sys.modules if m == "wnn_rt" or m.startswith("wnn_rt.")]:
        del sys.modules[m]
    spec = importlib.util.spec_from_file_location(f"sol_{d.name}", d / "solution.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sys.path.remove(str(d))
    return mod.PredictionModel()


names = sys.argv[1:]
models = [load(ARTIFACTS / "submissions" / n) for n in names]
s = SequenceReader("valid").read(1)
pts = [DataPoint(1, i, i >= 99, s.X[i]) for i in range(20000)]
bt = {n: [] for n in names}
for b in range(0, 20000, 500):
    for n, m in zip(names, models):
        t = time.perf_counter()
        for p in pts[b:b + 500]:
            m.predict(p)
        bt[n].append((time.perf_counter() - t) / 500 * 1e6)
ref = np.median(bt[names[0]])
for n in names:
    us = np.median(bt[n])
    print(f"{n:28s} {us:7.1f} us/row  ratio vs {names[0]} {us / ref:.3f}")
print("server minutes ~= ratio * minutes of the reference sub (sub07: 36.5)")
