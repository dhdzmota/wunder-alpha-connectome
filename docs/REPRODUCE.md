# Reproducing the final submission (sub26a)

Public 0.66502, private 0.67220 (rank 137 / 1448), 51m26s on the scoring server.

## Setup
- Python 3.11, `pip install -r requirements.txt`. The submission runtime itself needs only numpy + numba.
- Data: the competition's `train.parquet` / `valid.parquet` (one row group per 20 000-row sequence) under
  `$WNN_DATA` (default `~/data/wnn`). `splits.json` holds the fixed 5-fold split of train sequences
  (fold 4 = early-stopping fold).
- Artifacts (weights, predictions, labels, logs) go to `$WNN_ARTIFACTS`, default `~/data/wnn/artifacts`.
- Hardware used: one laptop GPU (RTX A1000, 4 GB), 16 GB RAM, 20 threads. Member training ≈ 16 h, everything ≈ 1 day.

## Steps
`bash pipeline/run_final.sh` runs all of them in order. Single steps: `bash pipeline/run_final.sh members finetune`.

| step | what | output |
|---|---|---|
| `prep` | tick units per sequence + regime detector (`scripts/fit_regimes.py`); P(is_scored) row weights (`scripts/build_pmask.py`) | `preds/units_*.npy`, `regime/`, `pmask/` |
| `teachers` | look-ahead GRU teachers la50 (horizons 2..50) and la25 (2..25), 4 epochs; soft labels on train | `nn/teacher_*`, `teacher/*_train.npy` |
| `members` | specialists r0–r3 (minGRU 128×2 + tick-class head, 8 ep), SSM 128×2 state 16 (8 ep), TCN 64×10 (16 ep) | `nn/<tag>/best.pt` |
| `finetune` | per member: soft fine-tune on valid halves 0 / 1 (out-of-fold) and on all of valid | `nn_ft/…_f0, _f1, _full`, `nn/ft_…_full/best.pt` |
| `stacks` | LightGBM row stacks for r0 / r1 on the out-of-fold blend | `stack_v3/` |
| `package` | `scripts/package.py configs/final.json`: export, numba-vs-torch / numba-vs-LightGBM checks, latency, zip | `submissions/final_sub26a.zip` |

Labels used per member:

| member | train labels | fine-tune labels |
|---|---|---|
| r0, r1 specialists, SSM, TCN | la50 soft | la50 soft |
| r2 specialist | hard (its soft-label retrain did not pass the gate) | la50 soft |
| r3 specialist | la25 soft | la25 soft |

## Exactness
- The code here was cleaned up after the competition. `tests/parity.py` checks that the package `scripts/package.py`
  builds from the original weights gives **bitwise identical** predictions to the zip that was submitted:
  `python tests/parity.py <submitted sub26a zip> artifacts/submissions/final_sub26a` → `max |diff| 0.000e+00`.
- The cleaned training scripts (`scripts/train.py`, `scripts/finetune_valid.py`) were checked against the
  pre-clean-up code on short CPU runs of every final recipe (identical weights and predictions).
- Retraining from scratch on a GPU is not bitwise reproducible: cuDNN GRU kernels and the order of floating-point
  sums are not deterministic across runs. Expect member scores within the seed noise of ~±0.001.
- Two inputs come from an interactive session, not a script:
  - the shipped `regime/detector.npz`. `fit_regimes.py` recomputes the tick units exactly and refits a detector with
    the same valid accuracy (0.9989), agreeing with the shipped one on 1871 of 1873 valid sequences; the package
    uses the shipped file;
  - the `pmask` mask model. `build_pmask.py --fit-only` refits it on the same rows; the order of LightGBM rows and
    bagging may differ from the original.

## History
`docs/EXPERIMENTS.md` is the lab notebook (§1–§56; early sections in Spanish): every experiment, its gate (both
cross-fit halves must improve) and the public LB results. The research-time queue scripts and gates that produced
these models are in `archive/` and in git history; they refer to the pre-clean-up script names (`train_gru.py`,
`pipeline/queues/`) and are not meant to run as is.

## Figures
`python docs/figures/make_figures.py` redraws every figure in `docs/img/` (light and dark variants). The charts read
the validation data and two prediction files from `$WNN_ARTIFACTS`; `--diagrams` redraws only the diagrams and needs
no data.
