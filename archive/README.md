# Archive

Research-time code kept for reference; nothing here is needed to rebuild the final submission and it does not
run as is (paths and module names are those before the post-competition clean-up).

- `queues/`: the shell queues that ran each experiment (training, fine-tunes, gates, packaging), named in
  `docs/EXPERIMENTS.md`.
- `regime_gates/`: the out-of-fold gate scripts (cross-fit valid halves) that decided what shipped.
- `scripts/`: superseded scripts. `train_gru.py` is the full research trainer (every architecture and option tried);
  `scripts/train.py` + `wnn/nets.py` are its pruned version with the final options only, checked to give identical
  weights. `package_routed.py` / `package_stack.py` built the submitted packages in two layers; `scripts/package.py`
  replaces both.
- `submission_template/`: entry points of earlier package formats.
- `wnn/`: runtime / feature modules used only by dead branches (classic GRU inference, event and rank features).
