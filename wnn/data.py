"""Data access, sequence-level splits and metric helpers.

All data is read lossless (float32) from parquet copies on the Linux filesystem.
Each parquet row group is exactly one 20k-row sequence.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, WARMUP, TARGET_COLUMNS, weighted_pearson  # noqa: E402

DATA_DIR = Path(os.environ.get("WNN_DATA", Path.home() / "data" / "wnn"))
ARTIFACTS = Path(os.environ.get("WNN_ARTIFACTS", Path.home() / "data" / "wnn" / "artifacts"))
SPLITS_PATH = REPO / "splits.json"
N_FOLDS = 5
SPLIT_SEED = 20261007


@dataclass
class Sequence:
    seq_ix: int
    X: np.ndarray          # (20000, 112) float32
    Y: np.ndarray          # (20000, 2) float32
    scored: np.ndarray     # (20000,) bool; for train = all required rows


def dataset_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


class SequenceReader:
    """Random access to sequences by row-group index. Cheap to create per worker."""

    def __init__(self, name: str):
        self.name = name
        self.pf = pq.ParquetFile(dataset_path(name))
        self.has_mask = "is_scored" in self.pf.schema_arrow.names
        self.n = self.pf.num_row_groups

    def read(self, g: int) -> Sequence:
        tb = self.pf.read_row_group(g, use_threads=False)
        X = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32, copy=False)
        Y = np.column_stack([tb[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32, copy=False)
        need = tb["need_prediction"].to_numpy().astype(bool)
        scored = tb["is_scored"].to_numpy().astype(bool) & need if self.has_mask else need
        return Sequence(int(tb["seq_ix"][0].as_py()), X, Y, scored)

    def seq_ids(self) -> np.ndarray:
        return self.pf.read(columns=["seq_ix"]).column(0).to_numpy()[::SEQUENCE_LENGTH]


def make_splits() -> dict:
    """K folds over train row groups (by sequence, never by row) + official valid holdout."""
    tr = SequenceReader("train")
    rng = np.random.default_rng(SPLIT_SEED)
    fold = rng.permutation(np.arange(tr.n) % N_FOLDS)
    splits = {"seed": SPLIT_SEED, "n_folds": N_FOLDS,
              "train_fold": fold.tolist(),
              "train_seq_ix": tr.seq_ids().tolist(),
              "valid_seq_ix": SequenceReader("valid").seq_ids().tolist()}
    SPLITS_PATH.write_text(json.dumps(splits))
    return splits


def load_splits() -> dict:
    return json.loads(SPLITS_PATH.read_text())


def global_wp(Y: np.ndarray, P: np.ndarray, mask: np.ndarray) -> dict:
    """Official metric: per-target clipped |y|-weighted Pearson over masked rows, averaged."""
    r = [weighted_pearson(Y[mask, k], P[mask, k]) for k in range(2)]
    return {"t0": r[0], "t1": r[1], "wp": float(np.mean(r))}
