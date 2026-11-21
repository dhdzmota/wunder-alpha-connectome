"""Package a routed submission with row-level stacks (sub21, EXPERIMENTS §22) from an existing routed package.

Copies the routed package --base as is (its wnn_rt runtime included, so its models run unchanged), and for every
regime in --stack (except --keep-ssm): empties the SSM stack of model_r<k>.npz (specialist / TCN reweighted to --w-spec / 1 - --w-spec) and adds
stack_r<k>.npz (artifacts/stack, scripts/train_stack.py) applied on that route. Adds wnn_rt/stack.py and the
routed_stack solution.

Checks: stack numba == LightGBM on stored training rows; routed predict() == offline (fused route model -> numpy
features -> LightGBM booster) on one sequence per stacked regime and one untouched sequence; latency vs --base on the
same sequences.
"""
import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np

from utils import DataPoint
from wnn.data import ARTIFACTS, REPO, SequenceReader
from wnn.regime import detect


def run_predict(model, s, seq_id):
    pts = [DataPoint(seq_id, i, i >= 99, s.X[i]) for i in range(20000)]
    return np.array([np.full(2, np.nan, np.float32) if (q := model.predict(p)) is None else q for p in pts])


def latency(model, reader, gs, tag):
    bt = []
    for g in gs:
        s = reader.read(g)
        pts = [DataPoint(50_000 + g + tag, i, i >= 99, s.X[i]) for i in range(20000)]
        for b in range(0, 20000, 500):
            t = time.perf_counter()
            for p in pts[b:b + 500]:
                model.predict(p)
            bt.append((time.perf_counter() - t) / 500 * 1e6)
    return float(np.median(bt))


def load_solution(path, name):
    import importlib.util
    for k in [k for k in sys.modules if k == "wnn_rt" or k.startswith("wnn_rt.")]:
        del sys.modules[k]
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(name, path / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.remove(str(path))
    return mod.PredictionModel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sub18_route4", help="routed package dir in artifacts/submissions")
    ap.add_argument("--stack-dir", default="stack", help="artifacts subfolder with stack_r<k>.npz / check / booster files")
    ap.add_argument("--stack", default="0,1", help="regimes that get a stack (and lose their SSM)")
    ap.add_argument("--w-spec", type=float, default=0.6)
    ap.add_argument("--keep-ssm", default="", help="stacked regimes whose route keeps its SSM and blend")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    subs = ARTIFACTS / "submissions"
    src, out = subs / a.base, subs / a.name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "submission_template/routed_stack/solution.py", out / "solution.py")
    shutil.copy(REPO / "wnn/stack.py", out / "wnn_rt/stack.py")
    regs = [int(v) for v in a.stack.split(",")]
    info = json.loads((src / "routes.json").read_text())
    info["base_package"] = a.base
    keep = {int(v) for v in a.keep_ssm.split(",") if v}
    for r in regs:
        shutil.copy(ARTIFACTS / a.stack_dir / f"stack_r{r}.npz", out / f"stack_r{r}.npz")
        info["routes"][str(r)]["stack"] = f"stack_r{r}.npz"
        if r in keep:
            continue
        z = dict(np.load(src / f"model_r{r}.npz"))
        assert int(z["n_mingru"]) == 1 and int(z["n_ssm"]) == 1 and int(z["n_tcn"]) == 1, "expects spec + SSM + TCN route"
        # empty SSM stack (leading dim 0) of the same dtypes / ndims: the SSM loop never runs, and the model keeps the
        # numba types of the other routes -> it reuses their compiled jitclass (n_ssm = 0 would add a ~24 s compile)
        z = {k: (v[:0] if k.startswith("ssm_") else v) for k, v in z.items()}
        z["blend"] = np.array([[a.w_spec] * 2, [1 - a.w_spec] * 2])  # order: minGRU, TCN
        np.savez_compressed(out / f"model_r{r}.npz", **z)
        info["routes"][str(r)].update(blend=z["blend"].tolist(), ssm="dropped")
    (out / "routes.json").write_text(json.dumps(info, indent=1))

    # 1) stack trees (numba) vs LightGBM on stored training rows
    sys.path.insert(0, str(REPO))
    from wnn.fused import TREE_KEYS, _trees
    for r in regs:
        z = np.load(out / f"stack_r{r}.npz"); tr = [z[f"st_{k}"] for k in TREE_KEYS]; rg = z["st_trange"]
        F = np.load(ARTIFACTS / a.stack_dir / f"check_r{r}_F.npy"); ref = np.load(ARTIFACTS / a.stack_dir / f"check_r{r}_lgb.npy")
        got = np.array([[_trees(f, *tr, rg[k], rg[k + 1]) for k in range(2)] for f in F])
        print(f"stack r{r}: max|numba - lightgbm| on 5000 training rows = {np.abs(got - ref).max():.2e}")

    # 2) routed predict == offline composition
    reader = SequenceReader("valid")
    model = load_solution(out, "sol_new")
    det_all = np.load(ARTIFACTS / "regime/detected_valid.npy")
    picks = [int(np.flatnonzero(det_all == r)[0]) for r in regs + [min(set(range(4)) - set(regs))]]
    import importlib
    sys.path.insert(0, str(out))
    fused_pkg = importlib.import_module("wnn_rt.fused")
    for g in picks:
        s = reader.read(g)
        got = run_predict(model, s, g)
        r = detect(s.X[:99], model.det)
        mpath = out / (f"model_r{r}.npz" if (out / f"model_r{r}.npz").exists() else "model.npz")
        fm = fused_pkg.FusedModel(np.load(mpath))
        P = np.array([fm.step(x, i == 0).copy() for i, x in enumerate(s.X)])
        exp = P.copy()
        if r in regs:
            X = s.X.astype(np.float32)
            D = np.vstack([np.zeros((1, X.shape[1]), np.float32), X[1:] - X[:-1]])
            F = np.c_[X, D, P]
            al = float(np.load(out / f"stack_r{r}.npz")["alpha"])
            for k in range(2):
                b = lgb.Booster(model_file=str(ARTIFACTS / a.stack_dir / f"stack_r{r}_t{k}.txt"))
                exp[:, k] = (1 - al) * P[:, k] + al * b.predict(F)
        print(f"seq {g}: regime {r}{' (stacked)' if r in regs else ''}: routed vs offline max diff {np.nanmax(np.abs(got[99:] - exp[99:])):.2e}")

    # 3) latency vs base, per regime (one sequence each), weighted by the valid regime mix
    mix = np.bincount(det_all, minlength=4) / len(det_all)
    gs = [int(np.flatnonzero(det_all == r)[0]) for r in range(4)]
    base_model = load_solution(src, "sol_base")
    for rep in range(2):
        ub = np.array([latency(base_model, reader, [g], 0) for g in gs])
        un = np.array([latency(model, reader, [g], 1) for g in gs])
        wb, wn = float(mix @ ub), float(mix @ un)
        print(f"latency rep {rep}: per regime base {ub.round(1).tolist()} new {un.round(1).tolist()} us/row | "
              f"mix-weighted base {wb:.1f} new {wn:.1f} ratio {wn / wb:.3f}")

    zpath = subs / f"{a.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f.relative_to(out))
    print(f"zip {zpath} {zpath.stat().st_size / 1e6:.2f} MB", zipfile.ZipFile(zpath).namelist())


if __name__ == "__main__":
    main()
