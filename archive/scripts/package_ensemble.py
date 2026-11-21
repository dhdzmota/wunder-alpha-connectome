"""Fit blend weights on dense valid predictions, then package LightGBM + GRU(s).

Blend (per target k): pred = sum_c w[c,k] * pred_c, with components first
rescaled to unit weighted-std so the single mixing weight alpha is meaningful.
alpha is chosen on a coarse grid (one parameter per target -> low overfit risk)
by the mean of official-mask WP and all-rows WP on valid.

Checks before zipping: GRU numba == torch on real rows, LightGBM numba == npz
trees (already verified at export), latency of the full predict() path, size.
"""
import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from train_gru import GRUNet, MinGRUNet, SSMNet, TCNNet, add_diff, n_feats, with_feats
from utils import DataPoint
from wnn.data import ARTIFACTS, REPO, SequenceReader, global_wp
from wnn import fused
from wnn.gru_infer import GRURunner, export_gru
from wnn.mingru_infer import MinGRURunner, export_mingru
from wnn.ssm_infer import SSMRunner, export_ssm
from wnn.tcn_infer import TCNRunner, export_tcn

P_DIR = ARTIFACTS / "preds"


def wstd(p, y):
    w = np.abs(np.clip(y, -2, 2))
    m = np.average(p, weights=w)
    return np.sqrt(np.average((p - m) ** 2, weights=w))


def simplex_grid(n, steps):
    """All weight vectors of length n with entries in multiples of 1/steps summing to 1."""
    if n == 1:
        return np.ones((1, 1))
    rows = [np.concatenate([[i / steps], r * (steps - i) / steps])
            for i in range(steps + 1) for r in simplex_grid(n - 1, steps - i if steps - i else 1)]
    return np.unique(np.round(np.array(rows), 6), axis=0)


def moment_wp(y, P, W):
    """Weighted Pearson of y vs P @ w for every row w of W, from second moments (y clipped, preds not)."""
    y = np.clip(y.astype(np.float64), -2, 2)
    w = np.abs(y); w /= w.sum()
    P = P.astype(np.float64)
    pm = w @ P
    Pc = P - pm
    yc = y - w @ y
    C = (Pc * w[:, None]).T @ Pc
    c = (Pc * w[:, None]).T @ yc
    var = np.einsum("ij,jk,ik->i", W, C, W)
    return (W @ c) / np.sqrt(np.maximum(var, 1e-20) * (w @ yc ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm-npz", default="", help="exported LightGBM model.npz ('' = none)")
    ap.add_argument("--lgbm-pred", default="sub01_valid.npy")
    ap.add_argument("--gru", nargs="*", default=[], help="tag[:hidden:layers] of GRU runs")
    ap.add_argument("--mingru", nargs="*", default=[], help="tag[:hidden:layers] of minGRU runs")
    ap.add_argument("--ssm", nargs="*", default=[], help="tag[:hidden:layers:state] of SSM runs")
    ap.add_argument("--tcn", nargs="*", default=[], help="tag[:hidden:layers] of TCN runs")
    ap.add_argument("--quant-bits", type=int, default=16, choices=[32, 16, 8],
                    help="minGRU / SSM matrices stored as int16 / int8 with per-row scales (32 = float)")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    Y = np.load(P_DIR / "valid_Y.npy"); M = np.load(P_DIR / "valid_mask.npy"); need = np.load(P_DIR / "valid_need.npy")
    comps, preds = [], []
    if a.lgbm_npz:
        comps.append("lgbm"); preds.append(np.load(P_DIR / a.lgbm_pred))
    gru_specs = []
    for spec in a.gru:
        tag, hidden, layers = (spec.split(":") + ["128", "2"])[:3]
        gru_specs.append((tag, int(hidden), int(layers)))
        comps.append(tag); preds.append(np.load(ARTIFACTS / "nn" / tag / "pred_valid.npy"))
    mg_specs = []
    for spec in a.mingru:
        tag, hidden, layers = (spec.split(":") + ["128", "2"])[:3]
        mg_specs.append((tag, int(hidden), int(layers)))
        comps.append(tag); preds.append(np.load(ARTIFACTS / "nn" / tag / "pred_valid.npy"))

    ssm_specs = []
    for spec in a.ssm:
        tag, hidden, layers, state = (spec.split(":") + ["128", "2", "16"])[:4]
        ssm_specs.append((tag, int(hidden), int(layers), int(state)))
        comps.append(tag); preds.append(np.load(ARTIFACTS / "nn" / tag / "pred_valid.npy"))

    tcn_specs = []
    for spec in a.tcn:
        tag, hidden, layers = (spec.split(":") + ["64", "10"])[:3]
        tcn_specs.append((tag, int(hidden), int(layers)))
        comps.append(tag); preds.append(np.load(ARTIFACTS / "nn" / tag / "pred_valid.npy"))

    # per-component scale normalisation, per target
    scale = np.array([[1.0 / wstd(p[need, k], Y[need, k]) for k in range(2)] for p in preds])
    for c, p in zip(comps, preds):
        r = global_wp(Y, p, M)["wp"], global_wp(Y, p, need)["wp"]
        print(f"{c:12s} mask {r[0]:.4f} all {r[1]:.4f}")

    if not a.lgbm_npz:  # NNs share the target scale; raw convex blend keeps the [-2, 2] clipping honest
        scale = np.ones_like(scale)
    blend = np.zeros((len(preds), 2))
    if len(preds) == 1:
        blend[0] = 1.0  # raw output: rescaling would change the metric's [-2, 2] clipping
    else:
        from utils import weighted_pearson
        grid = simplex_grid(len(preds), 20)  # weights in steps of 0.05, summing to 1
        for k in range(2):
            # rank the grid with weighted moments (ignores the +-2 prediction clip), then score the top few exactly
            approx = [moment_wp(Y[R, k], np.stack([scale[c, k] * p[R, k] for c, p in enumerate(preds)], 1), grid)
                      for R in (M, need)]
            top = np.argsort(-(approx[0] + approx[1]))[:5]
            best = None
            for i in top:
                p = sum(grid[i, c] * scale[c, k] * preds[c][:, k] for c in range(len(preds)))
                rm = weighted_pearson(Y[M, k], p[M]); ra = weighted_pearson(Y[need, k], p[need])
                if best is None or (rm + ra) / 2 > best[0]:
                    best = ((rm + ra) / 2, grid[i], rm, ra)
            _, wts, rm, ra = best
            print(f"t{k}: weights {dict(zip(comps, wts.round(2)))} mask {rm:.4f} all {ra:.4f}")
            blend[:, k] = wts * scale[:, k]
    P = sum(blend[c] * preds[c] for c in range(len(preds)))
    rm, ra = global_wp(Y, P, M), global_wp(Y, P, need)
    print(f"BLEND valid: mask {rm['wp']:.4f} (t0 {rm['t0']:.4f} t1 {rm['t1']:.4f}) | all {ra['wp']:.4f}")

    # ---- package ----
    out = ARTIFACTS / "submissions" / a.name
    if out.exists():
        shutil.rmtree(out)
    (out / "wnn_rt").mkdir(parents=True)
    shutil.copy(REPO / "submission_template/fused/solution.py", out / "solution.py")
    for f in ("features.py", "fastmath.py", "ema_feats.py", "event_feats.py", "rank_feats.py", "trees.py", "gru_infer.py", "mingru_infer.py", "ssm_infer.py", "tcn_infer.py", "fused.py"):
        shutil.copy(REPO / "wnn" / f, out / "wnn_rt" / f)
    (out / "wnn_rt/__init__.py").write_text("")
    s = SequenceReader("valid").read(1)
    def spans_of(tag):  # ema_feats spans of a run (ema.json written by train_gru, copied when staging FT runs)
        f = ARTIFACTS / "nn" / tag / "ema.json"
        if not f.exists():
            return ()
        j = json.loads(f.read_text())
        return tuple(j["spans"]) + tuple(j.get("extra", []))

    tag_spans = {t: spans_of(t) for t in [x[0] for x in mg_specs] + [x[0] for x in ssm_specs] + [x[0] for x in tcn_specs]}
    used = {tuple(f for f in v if not isinstance(f, str)) for v in tag_spans.values()} - {()}
    assert len(used) <= 1, f"models with ema features must share one set of spans: {used}"
    ema_spans = used.pop() if used else ()  # computed once per row in the runtime; plain models ignore it
    xused = {tuple(f for f in v if isinstance(f, str)) for v in tag_spans.values()} - {()}
    assert len(xused) <= 1 and all(f in ("evr", "rk") for x in xused for f in x), xused
    extra_feats = xused.pop() if xused else ()
    assert all(not any(isinstance(f, str) for f in v) for t, v in tag_spans.items() if t not in [x[0] for x in tcn_specs]), \
        "extra feats are supported for TCNs only (runners)"
    exports, mg_exports, ssm_exports, tcn_exports, refs = [], [], [], [], {}
    specs = [(t, h, l, GRUNet, export_gru, GRURunner, exports, {}) for t, h, l in gru_specs]
    specs += [(t, h, l, MinGRUNet, export_mingru, MinGRURunner, mg_exports, {}) for t, h, l in mg_specs]
    specs += [(t, h, l, SSMNet, export_ssm, SSMRunner, ssm_exports, {"n_state": n}) for t, h, l, n in ssm_specs]
    specs += [(t, h, l, TCNNet, export_tcn, TCNRunner, tcn_exports, {}) for t, h, l in tcn_specs]
    for tag, hidden, layers, Net, export, Runner, dest, kw in specs:
        sp = tag_spans.get(tag, ())
        n_in = 224 + n_feats(sp) if sp else 224
        if sp:  # real mu / sd come from the checkpoint
            kw = dict(kw, norm=(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)))
        net = Net(n_in, hidden, layers, **kw)
        net.load_state_dict(torch.load(ARTIFACTS / "nn" / tag / "best.pt", map_location="cpu"))
        net.eval()
        w = export(net)
        dest.append(w)
        dev = "cuda" if Net is SSMNet and torch.cuda.is_available() else "cpu"  # SSM CPU log-space scan overflows
        with torch.no_grad():
            xin = with_feats(s.X, sp) if sp else s.X
            refs[tag] = net.to(dev)(add_diff(torch.from_numpy(xin[None]).to(dev), None))[0][0].cpu().numpy()
        net.cpu()
        ref = refs[tag][:3000]
        r = Runner(w, sp) if sp else Runner(w)
        mine = np.array([r.step(x).copy() for x in s.X[:3000]])
        print(f"{tag}: max|numba-torch| over 3000 rows = {np.abs(mine - ref).max():.2e}")
    arrays = fused.pack(np.load(a.lgbm_npz) if a.lgbm_npz else None, exports, blend, mingrus=mg_exports, ssms=ssm_exports,
                        ema_spans=ema_spans, quant_bits=a.quant_bits, tcns=tcn_exports, extra_feats=extra_feats)
    np.savez_compressed(out / "model.npz", **arrays)
    (out / "blend.json").write_text(json.dumps({"components": comps, "blend": blend.tolist(),
                                                "valid_mask": rm, "valid_all": ra}, indent=1))

    sys.path.insert(0, str(out))
    from solution import PredictionModel  # noqa
    model = PredictionModel()
    pts = [DataPoint(1, i, i >= 99, s.X[i]) for i in range(20000)]
    got = np.array([np.full(2, np.nan, np.float32) if (q := model.predict(p)) is None else q for p in pts])
    # reference: torch output of each packaged net on valid row group 1 (pred_valid may be out-of-fold
    # predictions of other weights, e.g. for models fine-tuned on all of valid)
    exp = sum(blend[c] * (refs[t] if t in refs else preds[c][20000:40000]) for c, t in enumerate(comps))
    print(f"packaged vs offline blend on valid seq 1: max diff {np.nanmax(np.abs(got[99:] - exp[99:])):.2e}")
    bt = []  # block-min timing is robust to other load on the machine
    for b in range(0, 20000, 500):
        t = time.perf_counter()
        for p in pts[b:b + 500]:
            model.predict(p)
        bt.append((time.perf_counter() - t) / 500 * 1e6)
    us = float(np.median(bt))
    print(f"latency median {us:.1f} (min {min(bt):.1f}) us/row local -> ~{us * 39.4e6 / 60e6 / 1.5:.1f} min on server")

    zpath = ARTIFACTS / "submissions" / f"{a.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f.relative_to(out))
    print(f"zip {zpath} {zpath.stat().st_size / 1e6:.2f} MB", zipfile.ZipFile(zpath).namelist())


if __name__ == "__main__":
    main()
