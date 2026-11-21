"""Build the submission zip from a config (configs/final.json): per tick regime k, model_r<k>.npz = the regime
specialist minGRU + the shared SSM + TCN with their blend, optionally stack_r<k>.npz (row-level LightGBM), plus the
regime detector, the numba runtime (wnn/ -> wnn_rt/) and submission/solution.py.

Blend per route: specialist w_spec, SSM / TCN = ssm_tcn_blend rescaled to 1 - w_spec (per target).

Checks: every member numba == torch on 3000 rows of a sequence of its regime; stack trees numba == LightGBM on
stored training rows; solution.predict() == the route's fused model (+ stack) on one sequence per regime; latency.
End-to-end regression against a reference package: tests/parity.py.
Usage: python scripts/package.py configs/final.json   -> artifacts/submissions/<name>{/,.zip}
"""
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from utils import DataPoint
from wnn import fused
from wnn.data import ARTIFACTS, REPO, SequenceReader
from wnn.mingru_infer import MinGRURunner, export_mingru
from wnn.nets import REGIME_EDGES, REGIME_UNITS, MinGRUNet, SSMNet, TCNNet, add_diff, n_feats, with_feats
from wnn.ssm_infer import SSMRunner, export_ssm
from wnn.tcn_infer import TCNRunner, export_tcn
from wnn.trees import TREE_KEYS, _trees

RT_FILES = ("fastmath.py", "quant.py", "ema_feats.py", "mingru_infer.py", "ssm_infer.py", "tcn_infer.py", "fused.py",
            "regime.py", "trees.py", "stack.py")
NETS = {"mingru": (MinGRUNet, export_mingru, MinGRURunner, 128, 2, {}),
        "ssm": (SSMNet, export_ssm, SSMRunner, 128, 2, {"n_state": 16}),
        "tcn": (TCNNet, export_tcn, TCNRunner, 64, 10, {})}


def first_seq_of_regime(r):
    units = np.load(ARTIFACTS / "preds" / "units_valid.npy")[:, 0]
    return int(np.flatnonzero(np.searchsorted(REGIME_EDGES, units) == r)[0])


def load_member(kind, tag, regime, X):
    """Export artifacts/nn/<tag>/best.pt for the numba runtime and check numba == torch on the rows X."""
    Net, export, Runner, hidden, layers, kw = NETS[kind]
    ej = ARTIFACTS / "nn" / tag / "ema.json"
    spans = tuple(json.loads(ej.read_text())["spans"]) if ej.exists() else ()
    n_in = 224 + n_feats(spans)
    sd = torch.load(ARTIFACTS / "nn" / tag / "best.pt", map_location="cpu")
    net = Net(n_in, hidden, layers, norm=(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)) if spans else None, **kw)
    if "cls_v" in sd:  # tick-class head specialist
        net.set_class_head(REGIME_UNITS[regime])
    net.load_state_dict(sd)
    w = export(net.eval())
    dev = "cuda" if kind == "ssm" and torch.cuda.is_available() else "cpu"  # the CPU log-space SSM scan loses precision
    with torch.no_grad():
        xin = torch.from_numpy((with_feats(X, spans) if spans else X)[None]).to(dev)
        ref = net.to(dev)(add_diff(xin, None))[0][0].cpu().numpy()
    run = Runner(w, spans)
    got = np.array([run.step(x).copy() for x in X])
    print(f"  {kind:6s} {tag}: max|numba - torch| over {len(X)} rows = {np.abs(got - ref).max():.2e}")
    return w, spans


def run_predict(model, X, seq_id):
    pts = [DataPoint(seq_id, i, i >= 99, X[i]) for i in range(len(X))]
    return np.array([np.full(2, np.nan, np.float32) if (q := model.predict(p)) is None else q for p in pts])


def main():
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    out = ARTIFACTS / "submissions" / cfg["name"]
    if out.exists():
        shutil.rmtree(out)
    (out / "wnn_rt").mkdir(parents=True)
    shutil.copy(REPO / "submission" / "solution.py", out / "solution.py")
    for f in RT_FILES:
        shutil.copy(REPO / "wnn" / f, out / "wnn_rt" / f)
    (out / "wnn_rt" / "__init__.py").write_text("")
    shutil.copy(ARTIFACTS / cfg["detector"], out / "detector.npz")
    shutil.copy(cfg_path, out / "config.json")

    reader = SequenceReader("valid")
    w_spec = cfg["w_spec"]
    st = np.asarray(cfg["ssm_tcn_blend"], np.float64)
    blend = np.concatenate([np.full((1, 2), w_spec), st / st.sum(0) * (1 - w_spec)])
    spans = tuple(float(v) for v in cfg["ema_spans"])
    for r, route in sorted(cfg["routes"].items()):
        r = int(r)
        print(f"route {r}: blend {blend.round(4).tolist()}")
        X = reader.read(first_seq_of_regime(r)).X[:3000]
        members = {}
        for kind, tag in (("mingru", route["spec"]), ("ssm", cfg["ssm"]), ("tcn", cfg["tcn"])):
            w, sp = load_member(kind, tag, r, X)
            assert not sp or tuple(map(float, sp)) == spans, (tag, sp, spans)  # the EMA features are computed once per row
            members[kind] = w
        arr = fused.pack(mingrus=[members["mingru"]], ssms=[members["ssm"]], tcns=[members["tcn"]], blend=blend,
                         ema_spans=spans, quant_bits=cfg["quant_bits"])
        np.savez_compressed(out / f"model_r{r}.npz", **arr)
        if "stack" in route:
            shutil.copy(ARTIFACTS / route["stack"], out / f"stack_r{r}.npz")
            z = np.load(out / f"stack_r{r}.npz")
            tr, rg = [z[f"st_{k}"] for k in TREE_KEYS], z["st_trange"]
            d = (ARTIFACTS / route["stack"]).parent
            F, ref = np.load(d / f"check_r{r}_F.npy"), np.load(d / f"check_r{r}_lgb.npy")
            got = np.array([[_trees(f, *tr, rg[k], rg[k + 1]) for k in range(2)] for f in F])
            print(f"  stack  r{r}: max|numba - lightgbm| on {len(F)} training rows = {np.abs(got - ref).max():.2e}")

    # solution.predict() == the chosen route's fused model (+ stack), on one sequence per regime
    sys.path.insert(0, str(out))
    from solution import PredictionModel
    from wnn.regime import detect
    from wnn.stack import Stack
    model = PredictionModel()
    for r in sorted(model.routes):
        g = first_seq_of_regime(r)
        X = reader.read(g).X
        got = run_predict(model, X, g)
        rd = detect(X[:99], model.det)
        fm = fused.FusedModel(np.load(out / f"model_r{rd}.npz"))
        exp = np.array([fm.step(x, i == 0).copy() for i, x in enumerate(X)])
        if rd in model.stacks:
            s = Stack(np.load(out / f"stack_r{rd}.npz"))
            s.set_prev(X[98])
            exp[99:] = np.array([s.step(x, p).copy() for x, p in zip(X[99:], exp[99:])])
        print(f"seq {g} (regime {r}, detected {rd}): predict() vs direct max diff {np.nanmax(np.abs(got[99:] - exp[99:])):.2e}")

    bt = []
    for r in sorted(model.routes):
        X = reader.read(first_seq_of_regime(r)).X
        pts = [DataPoint(10_000 + r, i, i >= 99, X[i]) for i in range(20000)]
        for b in range(0, 20000, 500):
            t = time.perf_counter()
            for p in pts[b:b + 500]:
                model.predict(p)
            bt.append((time.perf_counter() - t) / 500 * 1e6)
    us = float(np.median(bt))
    print(f"latency median {us:.1f} us/row locally -> ~{us * 0.45:.1f} min on the scoring server (empirical 0.45 x us/row)")

    zpath = ARTIFACTS / "submissions" / f"{cfg['name']}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f.relative_to(out))
    print(f"zip {zpath} {zpath.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
