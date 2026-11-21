#!/bin/bash
# 2026-10-18 → sub14a/b/c: LB probes of valid fine-tune recipes on all three sub11 members (same G+S+T structure/latency).
#   a: 6 epochs (lr 3e-4)   b: mix16 (3 ep)   c: lr 1e-3 (4 ep)
# Full FT (fold -1) first + package (blend weights = sub11's, via the mix4 OOF preds), then cross-fit OOF for the record.
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts
cd "$(dirname "$0")/../.."
source ~/venvs/wnn/bin/activate
echo "start $(date)"; df -h / | tail -1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# member: base | args | staged sub11 dir
MEM=("mingru_ema_c3|--arch mingru --ema-spans 2000|ft_mingru_ema_c3_mix4_full"
     "ssm16_c3|--arch ssm --ssm-state 16|ft_ssm_mix4_full"
     "tcn64_ema_c3|--arch tcn --hidden 64 --layers 10 --ema-spans 2000|ft_tcn64_ema_c3_mix4_full")
VAR=("a|mix4|_e6|--epochs 6 --lr 3e-4" "b|mix16||--epochs 3" "c|mix4|_lr1e3|--epochs 4 --lr 1e-3")
run_ft() {  # base args weight suffix extra fold
  python scripts/finetune_valid.py --base $1 $2 --weight $3 --suffix "$4" $5 --fold $6 --workers 3 > $L/ft_$1_$3$4_f$6.log 2>&1
  echo "ft $1 $3$4 fold $6 rc=$? $(date)"
}
for v in "${VAR[@]}"; do
  IFS='|' read -r id w suf extra <<< "$v"
  specs=()
  for m in "${MEM[@]}"; do
    IFS='|' read -r base args staged <<< "$m"
    [ -f $A/nn_ft/ft_${base}_${w}${suf}_full/last.pt ] || run_ft $base "$args" $w "$suf" "$extra" -1
    D=$A/nn/ft_${base}_${w}${suf}_full; mkdir -p $D
    cp $A/nn_ft/ft_${base}_${w}${suf}_full/last.pt $D/best.pt
    [ -f $A/nn/$staged/ema.json ] && cp $A/nn/$staged/ema.json $D/
    ln -sf $A/nn/$staged/pred_valid.npy $D/pred_valid.npy   # sub11 OOF -> same blend weights as sub11
  done
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  rm -f wnn/__pycache__/*.nbi wnn/__pycache__/*.nbc
  python scripts/package_ensemble.py --mingru ft_mingru_ema_c3_${w}${suf}_full --ssm ft_ssm16_c3_${w}${suf}_full:128:2:16 \
    --tcn ft_tcn64_ema_c3_${w}${suf}_full:64:10 --name sub14${id}_ft${w}${suf} > $L/package_sub14${id}.log 2>&1
  echo "PACKAGED sub14${id}_ft${w}${suf} rc=$? $(grep -h 'max diff' $L/package_sub14${id}.log) $(date)"
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
done
# cross-fit OOF of each recipe for the record (TCN e6 / lr1e3 folds already exist)
for v in "${VAR[@]}"; do
  IFS='|' read -r id w suf extra <<< "$v"
  for m in "${MEM[@]}"; do
    IFS='|' read -r base args staged <<< "$m"
    for f in 0 1; do
      [ -f $A/nn_ft/ft_${base}_${w}${suf}_f$f/last.pt ] || run_ft $base "$args" $w "$suf" "$extra" $f
    done
  done
done
python - <<'PY'
import sys, numpy as np; sys.path.insert(0, 'scripts')
from ft_crossfit import crossfit
from wnn.data import ARTIFACTS as A, global_wp
Y = np.load(A / "preds/valid_Y.npy"); M = np.load(A / "preds/valid_mask.npy")
wp = lambda p: global_wp(Y, p, M)["wp"]
W = np.array([[.30, .35], [.35, .35], [.35, .30]])
bases = ["mingru_ema_c3", "ssm16_c3", "tcn64_ema_c3"]
for w, ep in (("mix4", 2), ("mix4_e6", 5), ("mix16", 2), ("mix4_lr1e3", 3)):
    P = [crossfit(b, w, ep) for b in bases]
    print(f"{w}: members " + " ".join(f"{wp(p):.4f}" for p in P) + f" | blend {wp(sum(W[i] * P[i] for i in range(3))):.4f}", flush=True)
PY
echo "done $(date)"
