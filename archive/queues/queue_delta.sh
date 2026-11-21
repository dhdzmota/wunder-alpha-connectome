#!/bin/bash
# 2026-10-31 (EXPERIMENTS §33): TTT-style per-sequence memory: DeltaNet delta-rule fast-weight layer (--delta 4, 4 heads x 32)
# on the r3 specialist (sub23d recipe, la50 soft labels), from scratch. jy447 reported "delta memory" within noise;
# this is the smoke test. Soft FT halves -> gate in the blend (proj_kd.py: replaces the r3 specialist). Full FT if it passes.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
B=mingru_ema_cls3_delta4; SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1 --delta 4"
$PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
  --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train teacher_la50_gru --soft-a 0.8 --tag $B > $L/$B.log 2>&1
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
FT="--base $B $SP --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_kd.py ft_${B}_mix4_sa8 > $L/proj_delta.log 2>&1
grep -E "^KD|DECIDE_KD|Error" $L/proj_delta.log | sed 's/^/GATE /'
grep -q "DECIDE_KD 1" $L/proj_delta.log || { echo "gate failed"; echo QUEUE_DONE; exit 0; }
NM=ft_${B}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT --fold -1 > $L/$NM.log 2>&1
D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
echo "$(date +%H:%M) FT $B soft full done"
echo QUEUE_DONE
