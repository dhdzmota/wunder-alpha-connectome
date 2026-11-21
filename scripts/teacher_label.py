"""Soft labels of a trained look-ahead teacher on every train sequence -> artifacts/teacher/<tag>_train.npy
((n_seq * 20000, 2) float16, row order = sequence order), read by train.py --soft-train.

The teacher (train.py --arch gru --fut-lags ...) sees future-mean columns of the inputs, so its predictions are a
denoised version of the target; students learn (1 - a) y + a teacher. Its valid predictions
(artifacts/nn/<tag>/pred_valid.npy, written by train.py) are the soft labels of finetune_valid.py --soft."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import SeqDS
from wnn.data import ARTIFACTS, SequenceReader
from wnn.nets import make_model, n_feats, run_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="teacher_la50_gru")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    h = json.loads((ARTIFACTS / "nn" / a.tag / "history.json").read_text())["args"]
    if h.get("fut_lags"):
        from wnn import fut_feats
        fut_feats.set_lags(h["fut_lags"].split(","))
    feats = tuple(int(v) for v in h["ema_spans"].split(",")) if h["ema_spans"] else ()
    feats = (feats + (("fut",) if h.get("fut_lags") or "fut" in h.get("extra_feats", "") else ())) or False
    n_in = 224 + n_feats(feats)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(h["arch"], h["hidden"], h["layers"], n_in, norm=(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)))
    model.load_state_dict(torch.load(ARTIFACTS / "nn" / a.tag / "best.pt", map_location=dev))
    model = model.to(dev).eval()
    n = SequenceReader("train").n
    out_dir = ARTIFACTS / "teacher"
    out_dir.mkdir(exist_ok=True)
    P = np.lib.format.open_memmap(out_dir / f"{a.tag}_train.npy", mode="w+", dtype=np.float16, shape=(n * 20000, 2))
    dl = DataLoader(SeqDS("train", np.arange(n), feats=feats), batch_size=a.bs, num_workers=a.workers, pin_memory=False,
                    prefetch_factor=1)
    i = 0
    for X, _, _ in dl:
        o = run_sequences(model, X, 5000, dev)
        P[i * 20000:(i + len(X)) * 20000] = o[..., :2].reshape(-1, 2).numpy().astype(np.float16)
        i += len(X)
        if i % 960 < len(X):
            print(f"{i}/{n} seqs", flush=True)
    P.flush()
    print("DONE", out_dir / f"{a.tag}_train.npy", i, flush=True)


if __name__ == "__main__":
    main()
