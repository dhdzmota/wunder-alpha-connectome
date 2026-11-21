"""Tick-regime routed blend: one fused numba model per regime, chosen per sequence from the warm-up rows.

model.npz is the default (all-regime) model; model_r<k>.npz, if present, replaces it on sequences of regime k
(wnn_rt/regime.py, detector.npz). Every model steps through the 99 warm-up rows (their states must see them);
from step 99 on only the chosen one runs. Runtime deps: numpy + numba.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from wnn_rt.fused import FusedModel
from wnn_rt.regime import detect

WARMUP = 99


class PredictionModel:
    def __init__(self):
        self.default = FusedModel(np.load(HERE / "model.npz"))
        self.routes = {int(p.stem.split("_r")[1]): FusedModel(np.load(p)) for p in sorted(HERE.glob("model_r*.npz"))}
        self.models = [self.default] + list(self.routes.values())
        det = np.load(HERE / "detector.npz")
        self.det = {k: det[k] for k in det.files}
        n_raw = self.default.buf[0].shape[0]
        for m in self.models:  # compile now for float32 rows (DataPoint.state); models share one compiled class
            m.step(np.zeros(n_raw, np.float32), True)
        self.warm = np.zeros((WARMUP, n_raw))
        self.seq_ix, self.active, self.t = None, self.models, 0

    def predict(self, data_point):
        first = data_point.seq_ix != self.seq_ix
        if first:
            self.seq_ix, self.active, self.t = data_point.seq_ix, self.models, 0
        x = data_point.state
        if self.t == WARMUP:  # rows 0..98 seen: pick the model of this sequence's regime
            self.active = [self.routes.get(detect(self.warm, self.det), self.default)]
        elif self.t < WARMUP:
            self.warm[self.t] = x
        self.t += 1
        out = None
        for m in self.active:
            out = m.step(x, first)
        if not data_point.need_prediction:
            return None
        return out.copy()
