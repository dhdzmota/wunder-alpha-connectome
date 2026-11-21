"""Teacher-free soft labels: centred moving average of the clipped target over rows t-K..t+K of the same sequence
(EXPERIMENTS §41). Writes artifacts/teacher/smooth<K>_train.npy ((n_seq * 20000, 2) float16, for train_gru.py --soft-train)
and artifacts/teacher/smooth<K>_valid.npy (float32, for finetune_valid.py --soft <path>).
Optional --mix <teacher tag> --mix-w w: label = (1 - w) smooth + w teacher (train file of the tag, its pred_valid for valid)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq

from wnn.data import ARTIFACTS, TARGET_COLUMNS, dataset_path

T = 20000


def smooth(Y, K):
    """Y: (n_seq, T, 2) clipped targets -> centred mean over the 2K+1 window, truncated at the sequence ends."""
    cs = np.concatenate([np.zeros((Y.shape[0], 1, 2)), np.cumsum(Y, 1, dtype=np.float64)], 1)
    t = np.arange(T)
    lo, hi = np.maximum(t - K, 0), np.minimum(t + K + 1, T)
    return ((cs[:, hi] - cs[:, lo]) / (hi - lo)[None, :, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--mix", default="")
    ap.add_argument("--mix-w", type=float, default=0.5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or f"smooth{a.k}"
    D = ARTIFACTS / "teacher"
    for name in ("train", "valid"):
        pf = pq.ParquetFile(dataset_path(name))
        res = np.empty((pf.num_row_groups * T, 2), np.float16 if name == "train" else np.float32)
        step = 500
        for g0 in range(0, pf.num_row_groups, step):
            gs = list(range(g0, min(g0 + step, pf.num_row_groups)))
            tb = pf.read_row_groups(gs, columns=list(TARGET_COLUMNS))
            Y = np.column_stack([tb[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32).reshape(len(gs), T, 2)
            res[g0 * T:(g0 + len(gs)) * T] = smooth(np.clip(Y, -2, 2), a.k).reshape(-1, 2)
        if a.mix:
            src = D / f"{a.mix}_train.npy" if name == "train" else ARTIFACTS / "nn" / a.mix / "pred_valid.npy"
            q = np.load(src, mmap_mode="r")
            for i in range(0, len(res), 10_000_000):
                res[i:i + 10_000_000] = (1 - a.mix_w) * res[i:i + 10_000_000].astype(np.float32) + a.mix_w * np.clip(q[i:i + 10_000_000].astype(np.float32), -2, 2)
        np.save(D / f"{out}_{name}.npy", res)
        print(name, res.shape, res.dtype, "->", D / f"{out}_{name}.npy", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
