"""Submission entry point: tick-regime routing + per-route blend + optional row-level LightGBM stack.

The target of a sequence is k * unit with one of four tick units. The regime is read from the 99 warm-up rows
(wnn_rt/regime.py, detector.npz); until then every route model steps through the rows, from row 99 on only the
chosen one runs. model_r<k>.npz: the fused minGRU + SSM + TCN blend of route k (wnn_rt/fused.py);
stack_r<k>.npz, if present, post-processes that route's output (wnn_rt/stack.py). Runtime deps: numpy + numba.
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

from wnn_rt.fused import N_RAW, FusedModel
from wnn_rt.regime import detect
from wnn_rt.stack import Stack

WARMUP = 99


class PredictionModel:
    def __init__(self):
        self.routes = {int(p.stem.split("_r")[1]): FusedModel(np.load(p)) for p in sorted(HERE.glob("model_r*.npz"))}
        self.stacks = {int(p.stem.split("_r")[1]): Stack(np.load(p)) for p in sorted(HERE.glob("stack_r*.npz"))}
        self.models = list(self.routes.values())
        det = np.load(HERE / "detector.npz")
        self.det = {k: det[k] for k in det.files}
        x0 = np.zeros(N_RAW, np.float32)
        for m in self.models:  # compile now for float32 rows (DataPoint.state); the routes share one compiled class
            m.step(x0, True)
        for s in self.stacks.values():
            s.step(x0, np.zeros(2, np.float32))
        self.warm = np.zeros((WARMUP, N_RAW))
        self.seq_ix, self.active, self.stack, self.t = None, self.models, None, 0

    def predict(self, data_point):
        first = data_point.seq_ix != self.seq_ix
        if first:
            self.seq_ix, self.active, self.stack, self.t = data_point.seq_ix, self.models, None, 0
        x = data_point.state
        if self.t == WARMUP:  # rows 0..98 seen: keep only this sequence's route
            r = detect(self.warm, self.det)
            self.active = [self.routes[r]]
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
