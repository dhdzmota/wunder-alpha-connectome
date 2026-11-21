"""Package a tick-regime routed submission: an existing fused package is the default model, and regime k
sequences use model_r<k>.npz = a specialist minGRU (tick-class head) + the default package's SSMs / TCNs.

The routed solution (submission_template/routed) reads the regime from the 99 warm-up rows (wnn/regime.py,
artifacts/regime/detector.npz) and from then on runs only that regime's model.

Checks: specialist numba == torch (class head included); the default model under the current runtime code ==
the default package's own runtime (regression check of wnn/fused.py); routed output == the chosen fused model
on one sequence per route; latency of the full predict() path.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from train_gru import REGIME_EDGES, REGIME_UNITS, MinGRUNet, SSMNet, TCNNet, add_diff, n_feats, with_feats
from utils import DataPoint
from wnn.data import ARTIFACTS, REPO, SequenceReader
from wnn import fused
from wnn.mingru_infer import MinGRURunner, export_mingru
from wnn.ssm_infer import SSMRunner, export_ssm
from wnn.tcn_infer import TCNRunner, export_tcn

RT_FILES = ("features.py", "fastmath.py", "ema_feats.py", "event_feats.py", "rank_feats.py", "trees.py", "gru_infer.py",
            "mingru_infer.py", "ssm_infer.py", "tcn_infer.py", "fused.py", "regime.py")
OLD_RT = """
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from wnn_rt.fused import FusedModel
m = FusedModel(np.load(sys.argv[1] + '/model.npz'))
X = np.load(sys.argv[2])
np.save(sys.argv[3], np.array([m.step(x, i == 0).copy() for i, x in enumerate(X)]))
"""


def first_seq_of_regime(r):
    units = np.load(ARTIFACTS / "preds" / "units_valid.npy")[:, 0]
    return int(np.flatnonzero(np.searchsorted(REGIME_EDGES, units) == r)[0])


def run_fused(m, X):
    return np.array([m.step(x, i == 0).copy() for i, x in enumerate(X)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", required=True, help="packaged submission dir name in artifacts/submissions (default model)")
    ap.add_argument("--route", nargs="+", required=True,
                    help="REGIME=TAG[+TAG2..]:W_SPEC[:SSM_TAG:TCN_TAG]: specialist minGRU run(s) artifacts/nn/<TAG> for that regime "
                         "(several: equal shares of W_SPEC); the default's "
                         "other members are rescaled to 1 - W_SPEC (their relative weights kept, its minGRU dropped). "
                         "SSM_TAG / TCN_TAG (same shapes as the default's: SSM 128x2 state 16, TCN 64x10) replace the "
                         "default's SSM / TCN weights on that route")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    src = ARTIFACTS / "submissions" / a.default
    base = dict(np.load(src / "model.npz"))
    assert int(base["n_mingru"]) == 1 and int(base["n_gru"]) == 0 and "lgbm_feat" not in base, "expects minGRU + SSM/TCN default"
    spans = tuple(float(v) for v in base["ema_spans"]) if "ema_spans" in base else ()
    reader = SequenceReader("valid")

    out = ARTIFACTS / "submissions" / a.name
    if out.exists():
        shutil.rmtree(out)
    (out / "wnn_rt").mkdir(parents=True)
    shutil.copy(REPO / "submission_template/routed/solution.py", out / "solution.py")
    for f in RT_FILES:
        shutil.copy(REPO / "wnn" / f, out / "wnn_rt" / f)
    (out / "wnn_rt/__init__.py").write_text("")
    shutil.copy(src / "model.npz", out / "model.npz")
    shutil.copy(ARTIFACTS / "regime" / "detector.npz", out / "detector.npz")
    for f in ("blend.json",):
        if (src / f).exists():
            shutil.copy(src / f, out / f"default_{f}")

    # regression check: default model under the new runtime == the default package's own runtime
    s1 = reader.read(1)
    tmp = ARTIFACTS / "regime"
    np.save(tmp / "_x.npy", s1.X[:3000])
    subprocess.run([sys.executable, "-c", OLD_RT, str(src), str(tmp / "_x.npy"), str(tmp / "_old.npy")], check=True)
    new = run_fused(fused.FusedModel(np.load(out / "model.npz")), s1.X[:3000])
    print(f"default model, new vs packaged runtime: max diff {np.abs(new - np.load(tmp / '_old.npy')).max():.2e}")

    info = {"default": a.default, "routes": {}}
    for spec in a.route:
        r, rest = spec.split("=")
        tag, w_spec, *st = rest.split(":")
        r, w_spec = int(r), float(w_spec)
        assert len(st) in (0, 2), spec
        wms = []
        s = reader.read(first_seq_of_regime(r))
        for tg in tag.split("+"):  # several specialists (e.g. seeds): equal shares of W_SPEC
            j = json.loads((ARTIFACTS / "nn" / tg / "ema.json").read_text())
            assert tuple(float(v) for v in j["spans"]) == spans and not j.get("extra"), (j, spans)
            n_in = 224 + n_feats(tuple(int(v) for v in j["spans"]))
            net = MinGRUNet(n_in, 128, 2, norm=(np.zeros(n_in, np.float32), np.ones(n_in, np.float32)))
            net.set_class_head(REGIME_UNITS[r])
            net.load_state_dict(torch.load(ARTIFACTS / "nn" / tg / "best.pt", map_location="cpu"))
            net.eval()
            wms.append(export_mingru(net))
            with torch.no_grad():
                ref = net(add_diff(torch.from_numpy(with_feats(s.X, tuple(j["spans"]))[None]), None))[0][0].numpy()
            runner = MinGRURunner(wms[-1], spans)
            mine = np.array([runner.step(x).copy() for x in s.X[:3000]])
            print(f"route {r} {tg}: max|numba-torch| over 3000 rows = {np.abs(mine - ref[:3000]).max():.2e}")

        arr = {k: v for k, v in base.items() if not k.startswith("mg_")}
        for k, v in fused.pack(mingrus=wms, blend=np.zeros((len(wms), 2))).items():
            if k.startswith("mg_"):
                arr[k] = v
        arr["n_mingru"] = np.array(len(wms))
        if st:  # per-route SSM / TCN weights
            assert int(base["n_ssm"]) == 1 and int(base["n_tcn"]) == 1, "per-route S/T expects one SSM and one TCN"
            nets = [(st[0], SSMNet, export_ssm, SSMRunner, 128, 2, {"n_state": 16}),
                    (st[1], TCNNet, export_tcn, TCNRunner, 64, 10, {})]
            ws = []
            for t2, Net, export, Runner, hidden, layers, kw in nets:
                ej = ARTIFACTS / "nn" / t2 / "ema.json"
                sp = tuple(json.loads(ej.read_text())["spans"]) if ej.exists() else ()
                n2 = 224 + n_feats(sp) if sp else 224
                if sp:
                    kw = dict(kw, norm=(np.zeros(n2, np.float32), np.ones(n2, np.float32)))
                n = Net(n2, hidden, layers, **kw)
                sd2 = torch.load(ARTIFACTS / "nn" / t2 / "best.pt", map_location="cpu")
                if "cls_v" in sd2:  # tick-class head specialist
                    n.set_class_head(REGIME_UNITS[r])
                n.load_state_dict(sd2)
                n.eval()
                ws.append(export(n))
                dev = "cuda" if Net is SSMNet and torch.cuda.is_available() else "cpu"  # SSM CPU log-space scan overflows
                with torch.no_grad():
                    xin = with_feats(s.X, sp) if sp else s.X
                    ref2 = n.to(dev)(add_diff(torch.from_numpy(xin[None]).to(dev), None))[0][0].cpu().numpy()
                run2 = Runner(ws[-1], sp) if sp else Runner(ws[-1])
                mine2 = np.array([run2.step(x).copy() for x in s.X[:3000]])
                print(f"route {r} {t2}: max|numba-torch| over 3000 rows = {np.abs(mine2 - ref2[:3000]).max():.2e}")
            q = int(base["quant_bits"]) if "quant_bits" in base else 32
            for k, v in fused.pack(ssms=[ws[0]], tcns=[ws[1]], blend=np.zeros((2, 2)), quant_bits=q).items():
                if k.startswith(("ssm_", "tcn_")):
                    arr[k] = v
            info_st = {"ssm": st[0], "tcn": st[1]}
        others = base["blend"][1:]
        arr["blend"] = np.concatenate([np.full((len(wms), 2), w_spec / len(wms)), others / others.sum(0) * (1 - w_spec)])
        np.savez_compressed(out / f"model_r{r}.npz", **arr)
        info["routes"][r] = {"tag": tag, "blend": arr["blend"].tolist(), **(info_st if st else {})}
        print(f"route {r}: blend {arr['blend'].round(3).tolist()}")
    (out / "routes.json").write_text(json.dumps(info, indent=1))

    sys.path.insert(0, str(out))
    from solution import PredictionModel  # noqa
    from wnn.regime import detect
    model = PredictionModel()
    det = model.det
    for g in [1] + [first_seq_of_regime(int(sp.split("=")[0])) for sp in a.route]:
        s = reader.read(g)
        pts = [DataPoint(g, i, i >= 99, s.X[i]) for i in range(20000)]
        got = np.array([np.full(2, np.nan, np.float32) if (q := model.predict(p)) is None else q for p in pts])
        r = detect(s.X[:99], det)
        exp = run_fused(fused.FusedModel(np.load(out / (f"model_r{r}.npz" if r in model.routes else "model.npz"))), s.X)
        print(f"seq {g}: detected regime {r}, routed vs direct fused max diff {np.nanmax(np.abs(got[99:] - exp[99:])):.2e}")

    bt = []
    for g in (1, first_seq_of_regime(3)):
        s = reader.read(g)
        pts = [DataPoint(10_000 + g, i, i >= 99, s.X[i]) for i in range(20000)]
        for b in range(0, 20000, 500):
            t = time.perf_counter()
            for p in pts[b:b + 500]:
                model.predict(p)
            bt.append((time.perf_counter() - t) / 500 * 1e6)
    us = float(np.median(bt))
    print(f"latency median {us:.1f} (min {min(bt):.1f}) us/row local -> ~{us * 0.45:.1f} min on server (0.45 x us/row)")

    zpath = ARTIFACTS / "submissions" / f"{a.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f.relative_to(out))
    print(f"zip {zpath} {zpath.stat().st_size / 1e6:.2f} MB", zipfile.ZipFile(zpath).namelist())


if __name__ == "__main__":
    main()
