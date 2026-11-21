#!/bin/bash
# Rebuild the final submission (sub26a: public 0.66502, private 0.67220) from the raw data, in order.
# One laptop GPU (RTX A1000, 4 GB) + 16 GB RAM: ~16 h of member training, about a day end to end.
# Every step writes to $WNN_ARTIFACTS (default ~/data/wnn/artifacts); logs in $A/logs. Steps can be run one by one:
#   bash pipeline/run_final.sh [step ...]     steps: prep teachers members finetune stacks package
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
A=${WNN_ARTIFACTS:-~/data/wnn/artifacts}; L=$A/logs; mkdir -p $L
export OMP_NUM_THREADS=4
COMMON="--lr 2e-3 --wd 1e-4 --corr-w 3 --pmask-gamma 0.5 --workers 3"
SPEC="--arch mingru --hidden 128 --layers 2 --ema-spans 2000 --class-head --ce-w 1 --epochs 8 --bs 32 --chunk 2000"
SSM="--arch ssm --hidden 128 --layers 2 --ssm-state 16 --epochs 8 --bs 32 --chunk 1000"
TCN="--arch tcn --hidden 64 --layers 10 --ema-spans 2000 --epochs 16 --bs 24 --chunk 1000 --es-seqs 1000"
FT="--weight mix4 --epochs 3 --lr 3e-4 --bs 16 --corr-w 3 --soft-a 0.8 --suffix _sa8 --workers 3"

run() { echo "$(date +%H:%M) $1"; shift; "$@"; }

prep() {
  # tick units of every sequence + warm-up regime detector (the shipped detector.npz is kept as an artifact;
  # add --write to replace it), then P(is_scored) row weights for train / valid
  run "regimes" $PY scripts/fit_regimes.py > $L/fit_regimes.log 2>&1
  run "pmask" $PY scripts/build_pmask.py > $L/build_pmask.log 2>&1
}

teachers() {
  # look-ahead GRU teachers (future-mean inputs, training only) and their soft labels on train
  for t in "teacher_la50_gru 2,5,12,25,50" "teacher_la25_gru 2,5,12,25"; do
    set -- $t
    run "teacher $1" $PY scripts/train.py --arch gru --hidden 128 --layers 2 --epochs 4 --bs 32 --chunk 2000 --es-seqs 1000 \
      --fut-lags $2 $COMMON --tag $1 > $L/$1.log 2>&1
    run "labels $1" $PY scripts/teacher_label.py --tag $1 > $L/teacher_label_$1.log 2>&1
  done
}

members() {
  # regime specialists on their regime's train sequences (r2's is trained on hard labels: its soft-label retrain
  # did not pass the gate), then the SSM and TCN shared by all routes, on all train sequences
  run "spec r0" $PY scripts/train.py $SPEC $COMMON --regimes 0 --soft-train teacher_la50_gru --tag mingru_ema_cls0_st > $L/mingru_ema_cls0_st.log 2>&1
  run "spec r1" $PY scripts/train.py $SPEC $COMMON --regimes 1 --soft-train teacher_la50_gru --tag mingru_ema_cls1_st > $L/mingru_ema_cls1_st.log 2>&1
  run "spec r2" $PY scripts/train.py $SPEC $COMMON --regimes 2 --tag mingru_ema_cls2 > $L/mingru_ema_cls2.log 2>&1
  run "spec r3" $PY scripts/train.py $SPEC $COMMON --regimes 3 --soft-train teacher_la25_gru --tag mingru_ema_cls3_la25 > $L/mingru_ema_cls3_la25.log 2>&1
  run "ssm" $PY scripts/train.py $SSM $COMMON --soft-train teacher_la50_gru --tag ssm16_c3_st > $L/ssm16_c3_st.log 2>&1
  run "tcn" $PY scripts/train.py $TCN $COMMON --soft-train teacher_la50_gru --tag tcn64_ema_c3_st16 > $L/tcn64_ema_c3_st16.log 2>&1
}

finetune() {
  # soft fine-tune on valid: the two cross-fit halves (out-of-fold predictions for the stacks / gates) and the full fit
  for m in "mingru_ema_cls0_st teacher_la50_gru 2000" "mingru_ema_cls1_st teacher_la50_gru 2000" \
           "mingru_ema_cls2 teacher_la50_gru 2000" "mingru_ema_cls3_la25 teacher_la25_gru 2000" \
           "ssm16_c3_st teacher_la50_gru 1000" "tcn64_ema_c3_st16 teacher_la50_gru 1000"; do
    set -- $m
    for f in 0 1 -1; do
      run "FT $1 fold $f" $PY scripts/finetune_valid.py --base $1 --soft $2 --chunk $3 $FT --fold $f > $L/ft_$1_f$f.log 2>&1
    done
  done
}

stacks() {
  # row-level LightGBM stacks for routes 0 / 1 on the out-of-fold route blend
  for r in 0 1; do
    run "stack r$r" $PY scripts/train_stack.py --regimes $r --alpha 0.35 --out stack_v3 >> $L/train_stack_v3.log 2>&1
  done
}

package() {
  run "package" $PY scripts/package.py configs/final.json > $L/package_final.log 2>&1
  tail -3 $L/package_final.log
}

for s in "${@:-prep teachers members finetune stacks package}"; do
  for t in $s; do $t; done
done
