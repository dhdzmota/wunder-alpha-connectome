"""Row-level LightGBM stack for one tick regime's route, on top of the route's NN blend.

Stack inputs per row: raw row x (112), x - previous row (112, zero on row 0), blend p (2); trained on the scored,
required rows of the valid sequences detected as regime r, weights |clip(y)|; final prediction (1 - a) p + a stack.
The blend p is out-of-fold: specialist .5 + SSM .25 + TCN .25 from the cross-fit fine-tunes (finetune_valid.py
--fold 0 / 1), so the stack never sees predictions of a model fitted on the same rows. (The shipped route blends
SSM / TCN .25 / .27 and .25 / .23 per target; the stack was fitted on the rounded .25 / .25.)
Writes artifacts/<out>/stack_r<k>.npz (exported trees of both targets, trange, alpha), the raw boosters, and 5000
training rows with LightGBM's predictions for the numba check in package.py.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wnn.data import ARTIFACTS, WARMUP, SequenceReader
from wnn.regime import detect
from wnn.trees import export_booster, merge_trees

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=8, seed=1, deterministic=True)


def halves(tag, n):
    """(n, 20000, 2) out-of-fold valid predictions of nn_ft/<tag>_f{0,1} (last epoch); NaN on sequences of neither."""
    P = np.full((n, 20000, 2), np.nan, np.float32)
    for f in (0, 1):
        d = ARTIFACTS / f"nn_ft/{tag}_f{f}"
        P[np.load(d / "eval_groups.npy")] = np.load(d / "pred_ep2.npy").reshape(-1, 20000, 2)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="0,1")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--spec", default="ft_mingru_ema_cls{r}_st_mix4_sa8", help="specialist FT-halves tag ({r} = regime)")
    ap.add_argument("--ssm", default="ft_ssm16_c3_st_mix4_sa8", help="SSM FT-halves tag")
    ap.add_argument("--tcn", default="ft_tcn64_ema_c3_st16_mix4_sa8", help="TCN FT-halves tag")
    ap.add_argument("--out", default="stack_v3", help="artifacts subfolder for the outputs")
    a = ap.parse_args()
    rd = SequenceReader("valid")
    det_model = dict(np.load(ARTIFACTS / "regime/detector.npz"))
    det = np.array([detect(rd.read(g).X[:WARMUP], det_model) for g in range(rd.n)])
    S, T = halves(a.ssm, rd.n), halves(a.tcn, rd.n)
    (ARTIFACTS / a.out).mkdir(exist_ok=True)
    for r in (int(v) for v in a.regimes.split(",")):
        P = .5 * halves(a.spec.format(r=r), rd.n) + .25 * S + .25 * T
        seqs = np.flatnonzero(det == r)
        F, y = [], []
        for g in seqs:
            s = rd.read(g)
            X = s.X.astype(np.float32)
            D = np.vstack([np.zeros((1, X.shape[1]), np.float32), X[1:] - X[:-1]])
            mk = s.scored.astype(bool) & (np.arange(20000) >= WARMUP)
            assert not np.isnan(P[g][mk]).any(), g
            F.append(np.c_[X[mk], D[mk], P[g][mk].astype(np.float32)])
            y.append(s.Y[mk])
        F, y = np.concatenate(F), np.concatenate(y)
        yc = np.clip(y, -2, 2)
        exports, boosters = [], []
        for k in range(2):
            b = lgb.train(PARAMS, lgb.Dataset(F, yc[:, k], weight=np.abs(yc[:, k]) + 1e-3), a.trees)
            b.save_model(str(ARTIFACTS / a.out / f"stack_r{r}_t{k}.txt"))
            exports.append(export_booster(b))
            boosters.append(b)
        arr = {f"st_{k_}": v for k_, v in merge_trees(*exports).items()}
        arr["alpha"] = np.array(a.alpha)
        np.savez(ARTIFACTS / a.out / f"stack_r{r}.npz", **arr)
        np.save(ARTIFACTS / a.out / f"check_r{r}_F.npy", F[:5000])
        np.save(ARTIFACTS / a.out / f"check_r{r}_lgb.npy", np.c_[boosters[0].predict(F[:5000]), boosters[1].predict(F[:5000])])
        print(f"regime {r}: {len(seqs)} seqs, {len(F)} scored rows, trees {arr['st_trange'].tolist()}", flush=True)


if __name__ == "__main__":
    main()
