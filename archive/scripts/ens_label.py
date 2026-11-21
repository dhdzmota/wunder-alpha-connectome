"""Ensemble soft labels on every train sequence (EXPERIMENTS §27, self-distillation): the sub24f route blend of the
from-scratch member runs (in-sample on train), regime detected from the 99 warm-up rows as at runtime.
r0/r1: spec .6 + TCN .4 (the row stack is left out); r2/r3: spec .5 + SSM .25 + TCN .25.
Writes artifacts/teacher/<out>_train.npy ((n_seq * 20000, 2) float16) and, with --mix, <out>_<mix>_train.npy =
.5 ensemble + .5 that teacher's labels."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_gru import REGIME_UNITS, SeqDS, make_model, n_feats, run_sequences
from wnn.data import ARTIFACTS, SequenceReader
from wnn.regime import detect

SPECS = ("mingru_ema_cls0_st", "mingru_ema_cls1_st", "mingru_ema_cls2", "mingru_ema_cls3_st")
SSM, TCN = "ssm16_c3_st", "tcn64_ema_c3_st16"


def load(tag):
    h = json.loads((ARTIFACTS / "nn" / tag / "history.json").read_text())["args"]
    feats = tuple(int(v) for v in h["ema_spans"].split(",")) if h["ema_spans"] else ()
    feats = feats or False
    n_in = 224 + n_feats(feats)
    m = make_model(h["arch"], h["hidden"], h["layers"], n_in, norm=(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)) if feats else None,
                   n_state=h.get("ssm_state") or 8)
    if h.get("class_head"):
        m.set_class_head(REGIME_UNITS[int(h["regimes"])], joint=bool(h.get("joint_head")))
    m.load_state_dict(torch.load(ARTIFACTS / "nn" / tag / "best.pt", map_location="cuda"))
    return m.cuda().eval(), feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ens24")
    ap.add_argument("--mix", default="teacher_la50_gru")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    reader = SequenceReader("train"); n = reader.n
    det = dict(np.load(ARTIFACTS / "regime" / "detector.npz"))
    reg = np.array([detect(reader.read(g).X[:99], det) for g in range(n)])
    print("train regimes", np.bincount(reg, minlength=4).tolist(), flush=True)
    P = np.zeros((n, 20000, 2), np.float32)
    w_spec = np.where(reg <= 1, .6, .5); w_ssm = np.where(reg <= 1, 0, .25); w_tcn = 1 - w_spec - w_ssm
    jobs = [(SPECS[r], np.flatnonzero(reg == r), w_spec) for r in range(4)] + \
           [(SSM, np.flatnonzero(reg >= 2), w_ssm), (TCN, np.arange(n), w_tcn)]
    for tag, groups, w in jobs:
        model, feats = load(tag)
        dl = DataLoader(SeqDS("train", groups, feats=feats), batch_size=8 if tag == SSM else a.bs, num_workers=a.workers, pin_memory=False, prefetch_factor=1)
        i = 0
        with torch.no_grad():
            for X, _, _ in dl:
                p = run_sequences(model, X, 5000, "cuda").numpy()[..., :2]
                g = groups[i:i + len(p)]
                P[g] += w[g, None, None] * p; i += len(p)
        print(f"{tag}: {i} seqs", flush=True)
        del model; torch.cuda.empty_cache()
    out = ARTIFACTS / "teacher"
    np.save(out / f"{a.out}_train.npy", P.reshape(-1, 2).astype(np.float16))
    if a.mix:
        q = np.load(out / f"{a.mix}_train.npy", mmap_mode="r")
        np.save(out / f"{a.out}_{a.mix.split('_')[1]}_train.npy", (.5 * P.reshape(-1, 2) + .5 * q).astype(np.float16))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
