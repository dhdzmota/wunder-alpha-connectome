#!/bin/bash
# 2026-11-10 (EXPERIMENTS §46): pretrain -> specialise. The regime specialists are trained from scratch on their own sequences
# only (train: r0 1305, r1 1257, r2 3488, r3 4557 of 10607). Here the trunk starts from the general minGRU-EMA trained on all
# regimes (mingru_ema_c3, hard labels) and only the tick-class head is fresh; then the usual recipe on the regime's sequences
# (la50 soft labels, 8 ep, lr 1e-3), soft FT halves, gate proj_prop.py vs the from-scratch specialist.
# Usage: queue_pt.sh "<regimes>" [lr] [suffix]
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
T=teacher_la50_gru; LR=${2:-1e-3}; SFX=${3:-pt}
for r in $1; do
  B=mingru_ema_cls${r}_$SFX; SP="--arch mingru --ema-spans 2000 --regimes $r --class-head --ce-w 1"
  $PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr $LR --workers 3 --corr-w 3 \
    --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train $T --soft-a 0.8 --init-from mingru_ema_c3 --tag $B > $L/$B.log 2>&1
  echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
  for f in 0 1; do
    $PY scripts/finetune_valid.py --base $B $SP --weight mix4 --workers 3 --soft $T --soft-a 0.8 --suffix _sa8 --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
    echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-140)"
  done
  $PY pipeline/regime_gates/proj_prop.py r$r=ft_${B}_mix4_sa8 > $L/proj_prop_r${r}_$SFX.log 2>&1
  grep -E "^PROP r|DECIDE|Error" $L/proj_prop_r${r}_$SFX.log | sed "s/^/GATE $SFX /"
done
echo QUEUE_DONE
