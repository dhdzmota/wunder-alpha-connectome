#!/bin/bash
# 2026-10-29 (EXPERIMENTS §28, idea b): distribution distillation for r3. After PID $1: look-ahead GRU teacher with a
# tick-class head on r3 only (la50 recipe + --regimes 3 --class-head, 8 ep), its train labels and class probabilities
# (teacher_label.py), r3 specialist from scratch with la50 point labels + KD of the teacher's distribution (kd-w .5),
# soft FT halves -> gate (proj_kd), full FT only if the gate passes.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
while kill -0 "$1" 2> /dev/null; do sleep 30; done
TT=teacher_la50_cls3
$PY scripts/train_gru.py --arch gru --extra-feats fut --regimes 3 --class-head --ce-w 1 --epochs 8 --bs 32 --workers 3 \
  --corr-w 3 --pmask-gamma 0.5 --select es_pm --valid-every 0 --es-seqs 1000 --tag $TT > $L/$TT.log 2>&1
echo "$(date +%H:%M) $TT: $(grep -h 'FINAL\|Error' $L/$TT.log | cut -c1-200)"
$PY scripts/teacher_label.py --tag $TT > $L/teacher_label_$TT.log 2>&1
echo "$(date +%H:%M) labels: $(grep -h 'DONE\|Error' $L/teacher_label_$TT.log | cut -c1-200)"
grep -q DONE $L/teacher_label_$TT.log || exit 1
B=mingru_ema_cls3_kd5; SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
$PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
  --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train teacher_la50_gru --soft-a 0.8 --kd-dist $TT --kd-w 0.5 \
  --tag $B > $L/$B.log 2>&1
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
FT="--base $B $SP --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
done
$PY pipeline/regime_gates/proj_kd.py ft_${B}_mix4_sa8 > $L/proj_kd.log 2>&1
grep -E "^KD|DECIDE_KD|Error" $L/proj_kd.log | sed 's/^/GATE /'
grep -q "DECIDE_KD 1" $L/proj_kd.log || { echo "gate failed"; echo QUEUE_DONE; exit 0; }
NM=ft_${B}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT --fold -1 > $L/$NM.log 2>&1
D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
echo "$(date +%H:%M) FT $B soft full done"
echo QUEUE_DONE
