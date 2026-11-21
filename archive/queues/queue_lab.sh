#!/bin/bash
# 2026-11-10 (EXPERIMENTS §41): label-side round on the r3 specialist (sub23d recipe from scratch, 8 ep, soft FT halves),
# gate proj_lab.py (replace / pair with the current r3 spec in the sub23g-like blend). Variants, by prior:
#  la25 look-ahead teacher with less future (horizons 2,5,12,25; capacity gap / label-horizon argument, jy447 used 25)
#  cf1  la50 labels with privileged-noise row weights exp(-(y - teacher)^2) (heteroscedastic LUPI)
#  m25  average of la50 and la25 labels (teacher ensemble)
#  la12 teacher with horizons 2,5,12
# Dropped: s16 (teacher-free centred mean of y over +-16 rows; stopped at ep 2, es_pm below the hard-label run) and its mix sm.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
TR="$SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-a 0.8"
DONE_TAGS=""

r3run() {  # $1 variant name, $2 train label tag (artifacts/teacher/<tag>_train.npy), $3 FT soft (tag or .npy), $4 extra train args
  B=mingru_ema_cls3_$1
  $PY scripts/train_gru.py $TR --soft-train $2 $4 --tag $B > $L/$B.log 2>&1
  echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
  for f in 0 1; do
    $PY scripts/finetune_valid.py --base $B $SP --weight mix4 --workers 3 --soft $3 --soft-a 0.8 --suffix _sa8 --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
    echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
  done
  DONE_TAGS="$DONE_TAGS ft_${B}_mix4_sa8"
  $PY pipeline/regime_gates/proj_lab.py $DONE_TAGS > $L/proj_lab.log 2>&1
  grep -E "^LAB ft_mingru_ema_cls3_$1|DECIDE_LAB ft_mingru_ema_cls3_$1|Error" $L/proj_lab.log | sed 's/^/GATE /'
}

teacher() {  # $1 tag, $2 horizons
  $PY scripts/train_gru.py --arch gru --hidden 128 --layers 2 --epochs 4 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
    --pmask-gamma 0.5 --select es_pm --valid-every 0 --es-seqs 1000 --extra-feats fut --fut-lags $2 --tag $1 > $L/$1.log 2>&1
  echo "$(date +%H:%M) $1: $(grep -h 'FINAL\|Error' $L/$1.log | cut -c1-200)"
  $PY scripts/teacher_label.py --tag $1 > $L/teacher_label_$1.log 2>&1
  echo "$(date +%H:%M) labels $1: $(grep -h 'DONE\|Error' $L/teacher_label_$1.log | cut -c1-200)"
  grep -q DONE $L/teacher_label_$1.log
}

if teacher teacher_la25_gru 2,5,12,25; then
  r3run la25 teacher_la25_gru teacher_la25_gru
fi
r3run cf1 teacher_la50_gru teacher_la50_gru "--soft-conf 1"
if [ -f $A/teacher/teacher_la25_gru_train.npy ]; then
  $PY scripts/mix_labels.py teacher_la50_gru teacher_la25_gru m25 > $L/mix_m25.log 2>&1
  r3run m25 m25 $A/teacher/m25_valid.npy
fi
if teacher teacher_la12_gru 2,5,12; then
  r3run la12 teacher_la12_gru teacher_la12_gru
fi
echo QUEUE_DONE
