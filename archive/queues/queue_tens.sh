#!/bin/bash
# 2026-11-14 (EXPERIMENTS §56): teacher seed average. 2nd seed of the la50 look-ahead teacher (same recipe as teacher_la50_gru),
# labels = mean of the two teachers (train: teacher_label.py; valid: pred_valid), r3 specialist from scratch on them (sub23d
# recipe), soft FT halves, gate proj_lab.py (replace / pair). Bar: both halves > 0 and sum >= +0.0008.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
T=teacher_la50_gru_s2
$PY scripts/train_gru.py --arch gru --extra-feats fut --epochs 4 --bs 32 --workers 3 --corr-w 3 --pmask-gamma 0.5 --select es_pm \
  --valid-every 0 --es-seqs 1000 --seed 2 --tag $T > $L/$T.log 2>&1
echo "$(date +%H:%M) $T: $(grep -h 'FINAL\|Error' $L/$T.log | cut -c1-200)"
grep -q '^FINAL' $L/$T.log || { echo QUEUE_FAILED teacher; tail -5 $L/$T.log; exit 1; }
$PY scripts/teacher_label.py --tag $T > $L/teacher_label_$T.log 2>&1
grep -q DONE $L/teacher_label_$T.log || { echo QUEUE_FAILED labels; tail -5 $L/teacher_label_$T.log; exit 1; }
$PY scripts/mix_labels.py teacher_la50_gru $T t2avg > $L/mix_t2avg.log 2>&1
grep -q DONE $L/mix_t2avg.log || { echo QUEUE_FAILED mix; tail -5 $L/mix_t2avg.log; exit 1; }
echo "$(date +%H:%M) labels t2avg ready"
SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
B=mingru_ema_cls3_t2
$PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 --pmask-gamma 0.5 \
  --select es_pm --valid-every 0 --soft-a 0.8 --soft-train t2avg --tag $B > $L/$B.log 2>&1
grep -h '^EPOCH' $L/$B.log | sed 's/"train_loss.*"es_pm"/"es_pm"/' | cut -c1-110
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
grep -q '^FINAL' $L/$B.log || { echo QUEUE_FAILED r3; tail -5 $L/$B.log; exit 1; }
for f in 0 1; do
  $PY scripts/finetune_valid.py --base $B $SP --weight mix4 --workers 3 --soft $A/teacher/t2avg_valid.npy --soft-a 0.8 --suffix _sa8 --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_lab.py ft_${B}_mix4_sa8 2>&1 | grep -E "^LAB base|^LAB ft_${B}|DECIDE_LAB ft_${B}|Error" | sed 's/^/GATE /'
echo QUEUE_DONE
