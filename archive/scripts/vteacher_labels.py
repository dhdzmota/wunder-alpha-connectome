"""Valid-adapted teacher (EXPERIMENTS §38). quarters: split each FT half into two quarters (artifacts/vteacher/q{0..3}.npy;
h0 = q0+q1, h1 = q2+q3). assemble: soft labels for valid where the rows of quarter q come from the teacher fine-tuned on
the other quarter of the same half (nn_ft/<tag>_q{q}_f0/pred_ep<ep>.npy), so a student FT on half h only sees labels
derived from half h. Output artifacts/vteacher/<out>.npy (n_valid * 20000, 2) float32, usable as finetune_valid --soft."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from train_gru import valid_halves
from wnn.data import ARTIFACTS, SequenceReader

D = ARTIFACTS / "vteacher"
if sys.argv[1] == "quarters":
    for h, half in enumerate(valid_halves(SequenceReader("valid").n)):
        np.save(D / f"q{2 * h}.npy", half[0::2]); np.save(D / f"q{2 * h + 1}.npy", half[1::2])
    print("quarters", [len(np.load(D / f"q{q}.npy")) for q in range(4)])
elif sys.argv[1] == "assemble":
    tag, ep, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    P = np.load(ARTIFACTS / "nn" / "teacher_la50_gru" / "pred_valid.npy").reshape(-1, 20000, 2).astype(np.float32)
    for q in range(4):
        d = ARTIFACTS / "nn_ft" / f"{tag}_q{q}_f0"
        g = np.load(d / "eval_groups.npy"); assert np.array_equal(g, np.load(D / f"q{q}.npy"))
        P[g] = np.load(d / f"pred_ep{ep}.npy").reshape(-1, 20000, 2)
    np.save(D / f"{out}.npy", P.reshape(-1, 2)); print("saved", D / f"{out}.npy", P.shape)
