#!/bin/bash
# 2026-11-14 (EXPERIMENTS §55): capacity re-test on the r3 specialist. Same recipe as mingru_ema_cls3_st (la50 soft labels,
# 8 ep from scratch) with --hidden $1 (default 192). Kill rule: after epoch 3, es_pm must be >= 0.3797 (reference 0.3827 - 0.003).
# Then soft FT halves and gate a22_wide.py (replace with SSM kept / r3 route without SSM). Bar: both halves > 0, sum >= +0.0008.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
W=${1:-192}; B=mingru_ema_cls3_w$W
SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1 --hidden $W --layers 2"
$PY scripts/train_gru.py $SP --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 --pmask-gamma 0.5 \
  --select es_pm --valid-every 0 --soft-train teacher_la50_gru --soft-a 0.8 --tag $B > $L/$B.log 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do
  sleep 60
  E3=$(grep -h '^EPOCH' $L/$B.log | grep '"epoch": 3,' | sed 's/.*"es_pm": \([0-9.]*\).*/\1/')
  if [ -n "$E3" ] && [ "$(echo "$E3 < 0.3797" | bc -l)" = 1 ]; then
    kill $TP; wait $TP 2>/dev/null
    echo "$(date +%H:%M) KILLED $B at epoch 3: es_pm $E3 < 0.3797"; grep -h '^EPOCH' $L/$B.log | cut -c1-150; echo QUEUE_DONE_KILLED; exit 0
  fi
done
wait $TP
grep -h '^EPOCH' $L/$B.log | sed 's/"train_loss.*"es_pm"/"es_pm"/' | cut -c1-110
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
grep -q '^FINAL' $L/$B.log || { echo QUEUE_FAILED; tail -5 $L/$B.log; exit 1; }
for f in 0 1; do
  $PY scripts/finetune_valid.py --base $B $SP --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8 --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/a22_wide.py ft_${B}_mix4_sa8 2>&1 | grep -E "^WIDE|Error" | sed 's/^/GATE /'
echo QUEUE_DONE
