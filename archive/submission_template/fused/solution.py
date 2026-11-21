"""Blend of LightGBM (streaming features + trees) and stacked GRUs, one numba call per row.

model.npz layout: see wnn_rt/fused.py (pack). Runtime deps: numpy + numba.
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


class PredictionModel:
    def __init__(self):
        self.m = FusedModel(np.load(HERE / "model.npz"))
        for dt in (np.float32, np.float64):  # compile both likely input dtypes now
            self.m.step(np.zeros(self.m.buf[0].shape[0], dt), True)
        self.seq_ix = None

    def predict(self, data_point):
        first = data_point.seq_ix != self.seq_ix
        self.seq_ix = data_point.seq_ix
        out = self.m.step(data_point.state, first)
        if not data_point.need_prediction:
            return None
        return out.copy()
