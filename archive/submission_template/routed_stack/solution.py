"""Tick-regime routed blend + row-level LightGBM stack on some routes (sub21, EXPERIMENTS §22).

Same as submission_template/routed: model.npz is the default model, model_r<k>.npz replaces it on sequences of
regime k (wnn_rt/regime.py, detector.npz); every model steps through the 99 warm-up rows, then only the chosen one.
stack_r<k>.npz, if present, post-processes route k's output from row 99 on (wnn_rt/stack.py); it needs the previous
raw row, kept for every row. Runtime deps: numpy + numba.
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
from wnn_rt.stack import Stack

WARMUP = 99


class PredictionModel:
    def __init__(self):
        self.default = FusedModel(np.load(HERE / "model.npz"))
        self.routes = {int(p.stem.split("_r")[1]): FusedModel(np.load(p)) for p in sorted(HERE.glob("model_r*.npz"))}
        self.stacks = {int(p.stem.split("_r")[1]): Stack(np.load(p)) for p in sorted(HERE.glob("stack_r*.npz"))}
        self.models = [self.default] + list(self.routes.values())
        det = np.load(HERE / "detector.npz")
        self.det = {k: det[k] for k in det.files}
        n_raw = self.default.buf[0].shape[0]
        x0 = np.zeros(n_raw, np.float32)
        for m in self.models:  # compile now for float32 rows (DataPoint.state); models share one compiled class
            m.step(x0, True)
        for s in self.stacks.values():
            s.step(x0, np.zeros(2, np.float32))
        self.warm = np.zeros((WARMUP, n_raw))
        self.seq_ix, self.active, self.stack, self.t = None, self.models, None, 0

    def predict(self, data_point):
        first = data_point.seq_ix != self.seq_ix
        if first:
            self.seq_ix, self.active, self.stack, self.t = data_point.seq_ix, self.models, None, 0
        x = data_point.state
        if self.t == WARMUP:  # rows 0..98 seen: pick the model of this sequence's regime
            r = detect(self.warm, self.det)
            self.active = [self.routes.get(r, self.default)]
            self.stack = self.stacks.get(r)
            if self.stack is not None:
                self.stack.set_prev(self.warm[WARMUP - 1])
        elif self.t < WARMUP:
            self.warm[self.t] = x
        self.t += 1
        out = None
        for m in self.active:
            out = m.step(x, first)
        if self.stack is not None:
            out = self.stack.step(x, out)
        if not data_point.need_prediction:
            return None
        return out.copy()
