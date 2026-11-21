"""Train one sequence model on the train set (stateful, full 20k-row sequences, truncated BPTT).

- inputs: raw 112 features + first differences (+ EMA-normalised columns, + look-ahead columns for the teacher)
- loss: |clip(y)| * P(is_scored)^gamma weighted MSE + corr_w * (1 - weighted Pearson) on required rows;
  with --soft-train the target is (1 - a) y + a teacher (soft labels from the look-ahead teacher)
- --regimes k --class-head: tick-regime specialist with a distributional tick-class head (+ cross-entropy)
- train folds != ES_FOLD; the epoch is selected on ES_FOLD by the P(is_scored)-weighted WP (es_pm);
  the official valid set is scored once for the selected checkpoint
Outputs: artifacts/nn/<tag>/{best.pt, last.pt, pred_valid.npy, history.json, ema.json}
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import GlobalAccumulator
from wnn.data import ARTIFACTS, WARMUP, SequenceReader, load_splits
from wnn.nets import N_RAW, REGIME_EDGES, REGIME_UNITS, make_model, n_feats, run_sequences, with_feats

SEED = 20261007
ES_FOLD = 4


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def valid_halves(n):
    """Fixed 50/50 split of the valid sequences (half 0, half 1) used by the cross-fitted fine-tunes."""
    perm = np.random.default_rng(SEED + 7).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])


def regime_of(name):
    """Tick regime 0..3 of every sequence of a dataset (from its target unit, scripts/fit_regimes.py)."""
    return np.searchsorted(REGIME_EDGES, np.load(ARTIFACTS / "preds" / f"units_{name}.npy")[:, 0])


class SeqDS(Dataset):
    """Yields (X, Y, third) per sequence. third = scored mask, or with row_weights=True the per-row loss
    weight need * P(is_scored)^gamma (gamma=0 -> plain required-rows mask). soft: (n_seq * 20000, 2) teacher
    predictions -> Y gets 4 columns [y, teacher]."""

    def __init__(self, name, groups, row_weights=False, gamma=0.0, feats=False, soft=None):
        self.name, self.groups, self.reader = name, list(groups), None
        self.row_weights, self.gamma, self.pmask, self.feats, self.soft = row_weights, gamma, None, feats, soft

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        if self.reader is None:
            self.reader = SequenceReader(self.name)
        g = self.groups[i]
        s = self.reader.read(g)
        X = with_feats(s.X, self.feats) if self.feats else s.X
        if not self.row_weights:
            return torch.from_numpy(X), torch.from_numpy(s.Y), torch.from_numpy(s.scored)
        rw = (np.arange(s.X.shape[0]) >= WARMUP).astype(np.float32)
        if self.gamma:
            if self.pmask is None:
                self.pmask = np.load(ARTIFACTS / "pmask" / f"{self.name}_pmask.npy", mmap_mode="r")
            rw *= np.asarray(self.pmask[g], np.float32) ** self.gamma
        Y = s.Y
        if self.soft is not None:
            if not isinstance(self.soft, np.ndarray):
                self.soft = np.load(self.soft, mmap_mode="r").reshape(-1, 20000, 2)
            Y = np.concatenate([Y, np.asarray(self.soft[g], np.float32)], 1)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(rw)


def feat_norm(groups, feats, n=48, seed=SEED):
    """Mean / std of the model input (raw + diff + feats) on n train sequences; raw + diff left as is."""
    rd = SequenceReader("train")
    pick = np.random.default_rng(seed).choice(groups, n, replace=False)
    F = np.concatenate([with_feats(rd.read(g).X, feats)[WARMUP:, N_RAW:] for g in pick])
    mu = np.concatenate([np.zeros(2 * N_RAW), F.mean(0)]).astype(np.float32)
    sd = np.concatenate([np.ones(2 * N_RAW), np.maximum(F.std(0), 1e-3)]).astype(np.float32)
    return mu, sd


class WeightedWP:
    """Global |y|*w weighted Pearson per target from running sums (w: extra row weights)."""

    def __init__(self):
        self.s = np.zeros((6, 2))

    def add(self, Y, P, w):
        y, p = np.clip(Y, -2, 2).astype(np.float64), np.clip(P, -2, 2).astype(np.float64)
        ww = np.abs(y) * w[:, None]
        self.s += np.stack([ww.sum(0), (ww * y).sum(0), (ww * p).sum(0), (ww * y * y).sum(0),
                            (ww * p * p).sum(0), (ww * y * p).sum(0)])

    def result(self):
        W, Sy, Sp, Syy, Spp, Syp = self.s
        cov, vy, vp = Syp / W - Sy * Sp / W ** 2, Syy / W - (Sy / W) ** 2, Spp / W - (Sp / W) ** 2
        r = cov / np.sqrt(vy * vp)
        return {"t0": float(r[0]), "t1": float(r[1]), "weighted_pearson": float(r.mean())}


def predict(model, name, groups, dev, bs, chunk, workers, keep=False, pmask_es=False, feats=False):
    """Streams sequences and returns (mask metrics, all-rows metrics, preds or None). With pmask_es the first
    result is the P(is_scored)-weighted WP (estimate of the official mask metric on train folds, which have no
    is_scored column) instead of the scored-mask WP."""
    dl = DataLoader(SeqDS(name, groups, row_weights=pmask_es, gamma=1.0, feats=feats), batch_size=bs, num_workers=workers,
                    pin_memory=False, prefetch_factor=1)
    model.eval()
    acc_mask, acc_all = (WeightedWP() if pmask_es else GlobalAccumulator()), GlobalAccumulator()
    need = np.arange(20000) >= WARMUP
    Ps = []
    for X, Y, M in dl:
        P = run_sequences(model, X, chunk, dev).numpy()
        for b in range(len(P)):
            acc_mask.add(Y[b].numpy(), P[b], M[b].numpy())
            acc_all.add(Y[b].numpy(), P[b], need)
        if keep:
            Ps.append(P.reshape(-1, 2).astype(np.float32))
    return acc_mask.result(), acc_all.result(), (np.concatenate(Ps) if keep else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mingru", choices=["gru", "mingru", "ssm", "tcn"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ssm-state", type=int, default=16, help="state size N per channel for --arch ssm")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000, help="rows per truncated-BPTT chunk (one optimizer step each)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--corr-w", type=float, default=3.0, help="weight of the 1 - weighted-Pearson loss term")
    ap.add_argument("--pmask-gamma", type=float, default=0.5, help="row weight P(is_scored)^gamma (scripts/build_pmask.py)")
    ap.add_argument("--ema-spans", default="", help="comma list of spans: causal per-sequence normalised input columns")
    ap.add_argument("--fut-lags", default="", help="look-ahead teacher: comma list of horizons of the future-mean inputs")
    ap.add_argument("--regimes", default="", help="comma list of tick regimes 0..3 to keep (specialists); empty = all")
    ap.add_argument("--class-head", action="store_true", help="tick-class head (needs a single regime)")
    ap.add_argument("--ce-w", type=float, default=1.0, help="with --class-head: weight of the cross-entropy loss")
    ap.add_argument("--soft-train", default="", help="teacher tag: artifacts/teacher/<tag>_train.npy soft labels")
    ap.add_argument("--soft-a", type=float, default=0.8, help="soft target (1 - a) y + a teacher")
    ap.add_argument("--soft-floor", type=float, default=0.2, help="soft row weight |y| + floor")
    ap.add_argument("--es-seqs", type=int, default=0, help="ES-fold sequences evaluated per epoch (0 = all)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-train-seqs", type=int, default=0, help="debug / smoke test: limit the sequences")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.fut_lags:
        from wnn import fut_feats
        fut_feats.set_lags(a.fut_lags.split(","))
    seed_everything(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ARTIFACTS / "nn" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fold = np.asarray(load_splits()["train_fold"])
    tr_groups = np.flatnonzero(fold != ES_FOLD)
    es_groups = np.flatnonzero(fold == ES_FOLD)
    if a.max_train_seqs:
        tr_groups, es_groups = tr_groups[:a.max_train_seqs], es_groups[:max(64, a.max_train_seqs // 4)]
    if a.es_seqs:
        es_groups = np.sort(np.random.default_rng(SEED + 2).choice(es_groups, min(a.es_seqs, len(es_groups)), replace=False))
    va_groups = np.arange(SequenceReader("valid").n)
    regimes = [int(r) for r in a.regimes.split(",") if r]
    if regimes:  # tick-regime specialist: keep only sequences whose target unit is in the chosen regimes
        rt, rv = regime_of("train"), regime_of("valid")
        tr_groups, es_groups = tr_groups[np.isin(rt[tr_groups], regimes)], es_groups[np.isin(rt[es_groups], regimes)]
        va_groups = va_groups[np.isin(rv[va_groups], regimes)]
        print(f"regimes {regimes}", flush=True)
    if a.class_head and len(regimes) != 1:
        sys.exit("--class-head needs exactly one --regimes value (the tick unit sets the classes)")
    if a.max_train_seqs:
        va_groups = va_groups[:16]
    print(f"train seqs {len(tr_groups)}, ES seqs {len(es_groups)}, valid seqs {len(va_groups)}", flush=True)

    feats = tuple(int(v) for v in a.ema_spans.split(",")) if a.ema_spans else ()
    feats = (feats + (("fut",) if a.fut_lags else ())) or False
    if feats:
        (out_dir / "ema.json").write_text(json.dumps({"spans": [f for f in feats if not isinstance(f, str)],
                                                     "extra": [f for f in feats if isinstance(f, str)]}))
    n_in = 224 + n_feats(feats)
    norm = feat_norm(tr_groups, feats) if feats else None
    model = make_model(a.arch, a.hidden, a.layers, n_in, norm, a.ssm_state)
    if a.class_head:
        model.set_class_head(REGIME_UNITS[regimes[0]])
    ce_w = a.ce_w if a.class_head else 0.0
    model = model.to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = (len(tr_groups) // a.bs) * math.ceil(20000 / a.chunk)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.05, anneal_strategy="cos")
    g = torch.Generator(); g.manual_seed(a.seed)
    soft = str(ARTIFACTS / "teacher" / f"{a.soft_train}_train.npy") if a.soft_train else None
    ds = SeqDS("train", tr_groups, row_weights=True, gamma=a.pmask_gamma, feats=feats, soft=soft)
    # workers are forked once, while the main process is still small, and reused every epoch
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    pin_memory=False, drop_last=True, generator=g, prefetch_factor=1, persistent_workers=True)

    class _Opt:  # one optimizer step per chunk; the scheduler advances with each
        def zero_grad(self, set_to_none=True): opt.zero_grad(set_to_none=set_to_none)
        def step(self): opt.step(); sched.step()

    best, best_ep, history = -1.0, -1, []
    for ep in range(a.epochs):
        model.train()
        t, losses = time.time(), []
        for i, (X, Y, RW) in enumerate(dl):
            losses.append(run_sequences(model, X, a.chunk, dev, Y, RW, _Opt(), corr_w=a.corr_w, ce_w=ce_w,
                                        soft_a=a.soft_a, soft_floor=a.soft_floor))
            if i % 50 == 0:
                print(f"ep{ep} batch {i}/{len(dl)} loss {np.mean(losses[-50:]):.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t:.0f}s", flush=True)
        t_train = time.time() - t
        es_pm, es, _ = predict(model, "train", es_groups, dev, a.bs, 5000, 2, pmask_es=True, feats=feats)
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "es_all": es["weighted_pearson"],
               "es_pm": es_pm["weighted_pearson"], "train_secs": t_train, "secs": time.time() - t}
        history.append(rec)
        print("EPOCH", json.dumps(rec), flush=True)
        if rec["es_pm"] > best:
            best, best_ep = rec["es_pm"], ep
            torch.save(model.state_dict(), out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep},
                   out_dir / "last.pt")
        (out_dir / "history_partial.json").write_text(json.dumps(history))
    model.load_state_dict(torch.load(out_dir / "best.pt"))
    vm, va_, Pv = predict(model, "valid", va_groups, dev, a.bs, 5000, 2, keep=True, feats=feats)
    np.save(out_dir / "pred_valid.npy", Pv)
    final = {"best_epoch": best_ep, "es_pm": best, "valid_mask": vm["weighted_pearson"], "valid_all": va_["weighted_pearson"],
             "valid_t0": vm["t0"], "valid_t1": vm["t1"]}
    print("FINAL", json.dumps(final), flush=True)
    (out_dir / "history.json").write_text(json.dumps({"args": vars(a), "history": history, "final": final}, indent=1))


if __name__ == "__main__":
    main()
