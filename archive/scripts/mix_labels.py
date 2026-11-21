"""Average of two teachers' soft labels (EXPERIMENTS §41): mix_labels.py <tag a> <tag b> <out>
-> artifacts/teacher/<out>_train.npy (float16, train_gru.py --soft-train) and <out>_valid.npy (finetune_valid.py --soft <path>)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wnn.data import ARTIFACTS

ta, tb, out = sys.argv[1:4]
for name, src in (("train", lambda t: ARTIFACTS / "teacher" / f"{t}_train.npy"), ("valid", lambda t: ARTIFACTS / "nn" / t / "pred_valid.npy")):
    a, b = (np.load(src(t), mmap_mode="r") for t in (ta, tb))
    r = np.empty(a.shape, a.dtype)
    for i in range(0, len(a), 10_000_000):
        r[i:i + 10_000_000] = (.5 * (a[i:i + 10_000_000].astype(np.float32) + b[i:i + 10_000_000].astype(np.float32))).astype(a.dtype)
    np.save(ARTIFACTS / "teacher" / f"{out}_{name}.npy", r)
    print(name, r.shape, r.dtype, flush=True)
print("DONE")
