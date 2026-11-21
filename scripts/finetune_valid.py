"""Fine-tune a trained model on the validation set, where the true is_scored mask is known.

valid is closer to the test period than train and carries the scoring mask, so a short fine-tune on it was the
largest single gain of the competition. Row weights (times |y|, as in training): mixK = need * (1 + K * is_scored)
(all required rows, scored rows upweighted; mix4 in the final models), or mask = need * is_scored.
With --soft, the target is (1 - a) y + a teacher, the teacher being the look-ahead model's valid predictions.

--fold 0 / 1: 2-fold cross-fit over the fixed valid halves (fit on half f, predict the other half), so both
folds give an out-of-fold prediction for every valid row (gates, row-stack training). --fold -1: fit on all
of valid (the model that ships). Outputs: artifacts/nn_ft/<tag>/{last.pt, history.json[, pred_ep*.npy]}.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train import SEED, predict, regime_of, seed_everything, valid_halves
from wnn.data import ARTIFACTS, WARMUP, SequenceReader
from wnn.nets import REGIME_UNITS, make_model, n_feats, run_sequences, with_feats


class ValidFT(Dataset):
    """Valid sequences with the fine-tune row weights (and optional soft-label columns)."""

    def __init__(self, groups, weight, feats=False, soft=None):
        self.groups, self.weight, self.reader, self.feats, self.soft = list(groups), weight, None, feats, soft

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        g = self.groups[i]
        if self.reader is None:
            self.reader = SequenceReader("valid")
        s = self.reader.read(g)
        rw = (np.arange(20000) >= WARMUP).astype(np.float32)
        if self.weight == "mask":
            rw *= s.scored
        else:  # mixK
            rw *= 1.0 + float(self.weight[3:]) * s.scored
        X = with_feats(s.X, self.feats) if self.feats else s.X
        Y = s.Y if self.soft is None else np.concatenate([s.Y, np.asarray(self.soft[g], np.float32)], 1)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(rw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="tag of the trained run (artifacts/nn/<tag>/best.pt)")
    ap.add_argument("--fold", type=int, required=True, choices=[-1, 0, 1], help="-1: fit on all valid (final model)")
    ap.add_argument("--weight", default="mix4", help="mixK (e.g. mix4) | mask")
    ap.add_argument("--soft", default="", help="teacher tag: artifacts/nn/<tag>/pred_valid.npy as soft labels")
    ap.add_argument("--soft-a", type=float, default=0.8)
    ap.add_argument("--soft-floor", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--corr-w", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--suffix", default="", help="appended to the run tag")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="debug / smoke test: sequences per half")
    a = ap.parse_args()
    h = json.loads((ARTIFACTS / "nn" / a.base / "history.json").read_text())["args"]  # architecture of the base run
    seed_everything(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"ft_{a.base}_{a.weight}{a.suffix}_" + (f"f{a.fold}" if a.fold >= 0 else "full")
    out_dir = ARTIFACTS / "nn_ft" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fit, ev = valid_halves(SequenceReader("valid").n)
    if a.fold == 1:
        fit, ev = ev, fit
    elif a.fold == -1:
        fit, ev = np.arange(SequenceReader("valid").n), None
    regimes = [int(r) for r in h["regimes"].split(",") if r]
    if regimes:  # same halves as the all-regime runs, restricted to the specialist's sequences
        keep = np.isin(regime_of("valid"), regimes)
        fit, ev = fit[keep[fit]], None if ev is None else ev[keep[ev]]
    if a.limit:
        fit, ev = fit[:a.limit], None if ev is None else ev[:a.limit]
    if ev is not None:
        np.save(out_dir / "eval_groups.npy", ev)

    feats = tuple(int(v) for v in h["ema_spans"].split(",")) if h["ema_spans"] else ()
    feats = feats or False
    n_in = 224 + n_feats(feats)
    norm = (np.zeros(n_in, np.float32), np.ones(n_in, np.float32)) if feats else None  # real values come from best.pt
    soft = None
    if a.soft:
        sp = ARTIFACTS / "nn" / a.soft / "pred_valid.npy"
        soft = np.load(sp, mmap_mode="r").reshape(-1, 20000, 2)
        print(f"soft labels {sp} a {a.soft_a} floor {a.soft_floor}", flush=True)
    model = make_model(h["arch"], h["hidden"], h["layers"], n_in, norm=norm, n_state=h["ssm_state"])
    if h["class_head"]:
        model.set_class_head(REGIME_UNITS[regimes[0]])
    ce_w = h["ce_w"] if h["class_head"] else 0.0
    model = model.to(dev)
    model.load_state_dict(torch.load(ARTIFACTS / "nn" / a.base / "best.pt", map_location=dev))
    history = []
    if ev is not None:
        vm, va, _ = predict(model, "valid", ev, dev, 32, 5000, 2, feats=feats)
        history.append({"epoch": -1, "valid_mask": vm["weighted_pearson"], "valid_all": va["weighted_pearson"]})
        print("EPOCH", json.dumps(history[-1]), flush=True)

    dl = DataLoader(ValidFT(fit, a.weight, feats, soft=soft), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    pin_memory=False, drop_last=True, prefetch_factor=1, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = a.epochs * len(dl) * math.ceil(20000 / a.chunk)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")

    class _Opt:
        def zero_grad(self, set_to_none=True): opt.zero_grad(set_to_none=set_to_none)
        def step(self): opt.step(); sched.step()

    for ep in range(a.epochs):
        model.train()
        t, losses = time.time(), []
        for X, Y, RW in dl:
            losses.append(run_sequences(model, X, a.chunk, dev, Y, RW, _Opt(), corr_w=a.corr_w, ce_w=ce_w,
                                        soft_a=a.soft_a, soft_floor=a.soft_floor))
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "secs": time.time() - t}
        if ev is not None:
            vm, va, P = predict(model, "valid", ev, dev, 32, 5000, 2, keep=True, feats=feats)
            rec.update(valid_mask=vm["weighted_pearson"], valid_all=va["weighted_pearson"])
            np.save(out_dir / f"pred_ep{ep}.npy", P)  # rows of eval_groups, in order
        history.append(rec)
        print("EPOCH", json.dumps(rec), flush=True)
    torch.save(model.state_dict(), out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps({"args": vars(a), "history": history}, indent=1))
    if a.fold == -1:  # the shipped member: artifacts/nn/<tag>/best.pt (+ ema.json), where package.py reads it
        d = ARTIFACTS / "nn" / tag
        d.mkdir(exist_ok=True)
        torch.save(model.state_dict(), d / "best.pt")
        if (ARTIFACTS / "nn" / a.base / "ema.json").exists():
            (d / "ema.json").write_text((ARTIFACTS / "nn" / a.base / "ema.json").read_text())
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    main()
