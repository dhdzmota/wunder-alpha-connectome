#!/bin/bash
# 2026-10-29 (§25): GRU 2x256 lost to the SSM (es_pm 0.5547 vs 0.5656 at ep 4) -> GRU 2x128 cancelled.
# After the PID $1 (2026-10-29 20:20: reordered after queue_ens, the bigger lever): per-regime scaling gate (CPU, needs ~11 GB RAM, so not alongside training), then a second-seed
# r3 specialist (sub23d recipe, seed 2), soft FT halves -> gate (mean of both seeds), full FT for packaging.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
while kill -0 "$1" 2> /dev/null; do sleep 30; done
grep -q DECIDE_SCALE $L/proj_scale.log 2>/dev/null || $PY pipeline/regime_gates/proj_scale.py ft_mingru_ema_cls0_st_mix4_sa8 ft_mingru_ema_cls1_st_mix4_sa8 > $L/proj_scale.log 2>&1
grep -E "STACK new stack a0.35|regimes|SCALE|Error" $L/proj_scale.log | cut -c1-400
T=teacher_la50_gru; B=mingru_ema_cls3_st_s2; SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
$PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
  --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train $T --soft-a 0.8 --seed 2 --tag $B > $L/$B.log 2>&1
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
FT="--base $B $SP --weight mix4 --workers 3 --soft $T --soft-a 0.8 --suffix _sa8 --seed 2"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_seed.py ft_${B}_mix4_sa8 2>&1 | grep -E "SEED|Error" | sed 's/^/GATE /'
NM=ft_${B}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT --fold -1 > $L/$NM.log 2>&1
echo "$(date +%H:%M) FT $B soft full: $(grep -h 'EPOCH\|DONE\|Error' $L/$NM.log | tail -1 | cut -c1-120)"
D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
echo QUEUE_DONE
