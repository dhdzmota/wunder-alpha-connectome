#!/bin/bash
# 2026-11-10 (EXPERIMENTS §54): r3 jump specialist. r3 metric weight sits on rows before multi-tick moves; the point-loss
# row weights (|y| + floor) are raised to the power p (--jump-p, train and soft FT) so those rows dominate the fit.
# Same recipe as queue_lab.sh (la50 soft labels, 8 ep from scratch, soft FT halves), gate proj_lab.py (replace / pair).
# Bar for a sub: both halves > 0 and sum >= +0.0008 (la25 at +0.00048 did not carry to public).
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
TR="$SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-a 0.8"
DONE_TAGS=""
for P in $*; do
  V=jp${P/./}; B=mingru_ema_cls3_$V
  $PY scripts/train_gru.py $TR --soft-train teacher_la50_gru --jump-p $P --tag $B > $L/$B.log 2>&1
  echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
  for f in 0 1; do
    $PY scripts/finetune_valid.py --base $B $SP --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --jump-p $P --suffix _sa8 --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
    echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
  done
  DONE_TAGS="$DONE_TAGS ft_${B}_mix4_sa8"
  $PY pipeline/regime_gates/proj_lab.py $DONE_TAGS > $L/proj_jump.log 2>&1
  grep -E "^LAB ft_${B}|DECIDE_LAB ft_${B}|Error" $L/proj_jump.log | sed 's/^/GATE /'
done
echo QUEUE_DONE
