#!/bin/bash
# 2026-11-10 (EXPERIMENTS §51): long-reach look-ahead teachers (future means up to 200 / 400 rows) as CAUSAL delayed estimates
# of y(t-R). Oracle y(t-L) is worth +0.095 at L100, +0.041 at L200, +0.005 at L500 (§39); la64c (reach 113, es_pm 0.90) gave 0.
# Question: does the teacher's accuracy on the slow level grow fast enough with reach to keep part of the oracle at L = R?
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
for spec in "200 2,5,12,25,50,100,200" "400 2,5,12,25,50,100,200,400"; do
  set -- $spec; T=teacher_la$1_gru
  $PY scripts/train_gru.py --arch gru --hidden 128 --layers 2 --epochs 4 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
    --pmask-gamma 0.5 --select es_pm --valid-every 0 --es-seqs 1000 --extra-feats fut --fut-lags $2 --tag $T > $L/$T.log 2>&1
  echo "$(date +%H:%M) $T: $(grep -h 'FINAL\|Error' $L/$T.log | cut -c1-200)"
  $PY pipeline/regime_gates/a17_reach.py $T=$1 > $L/a17_reach_$1.log 2>&1
  grep -E "^base|^teacher|^REACH|Error" $L/a17_reach_$1.log | sed 's/^/GATE /'
done
echo QUEUE_DONE
