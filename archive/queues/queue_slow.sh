#!/bin/bash
# 2026-10-30 (EXPERIMENTS §29): slow branch (GRU over 100-row block means, --slow-k 100) on the current r3 specialist
# (sub23d recipe, la50 soft labels). Earlier screening on the plain GRU tied (§ line 164); this re-tests it on the
# soft-label r3 specialist. Soft FT halves -> gate in the blend (proj_kd.py: replaces the r3 specialist). Full FT if it passes.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
B=mingru_ema_cls3_slow100; SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1 --slow-k 100"
$PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
  --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train teacher_la50_gru --soft-a 0.8 --tag $B > $L/$B.log 2>&1
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
FT="--base $B $SP --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_kd.py ft_${B}_mix4_sa8 > $L/proj_slow.log 2>&1
grep -E "^KD|DECIDE_KD|Error" $L/proj_slow.log | sed 's/^/GATE /'
grep -q "DECIDE_KD 1" $L/proj_slow.log || { echo "gate failed"; echo QUEUE_DONE; exit 0; }
NM=ft_${B}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT --fold -1 > $L/$NM.log 2>&1
D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
echo "$(date +%H:%M) FT $B soft full done"
echo QUEUE_DONE
