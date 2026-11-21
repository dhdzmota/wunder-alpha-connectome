#!/bin/bash
# 2026-10-29 (EXPERIMENTS §27, idea 1): ensemble self-distillation. After queue_seed_r3 (PID $1): sub24f-route ensemble
# labels on train (scripts/ens_label.py, in-sample member runs), mixed .5/.5 with la50; TCN from scratch with the st8 recipe
# on them; soft FT halves with la50 as before; gate vs st8 (proj_ens). Full FT only if the gate passes.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
while kill -0 "$1" 2> /dev/null; do sleep 30; done
$PY scripts/ens_label.py --out ens24 --mix teacher_la50_gru > $L/ens_label.log 2>&1
echo "$(date +%H:%M) ens labels: $(grep -h 'regimes\|DONE\|Error' $L/ens_label.log | tr '\n' ' ' | cut -c1-200)"
grep -q DONE $L/ens_label.log || exit 1
T=ens24_la50; B=tcn64_ema_c3_ens8; TCN="--arch tcn --hidden 64 --layers 10 --ema-spans 2000"
$PY scripts/train_gru.py $TCN --epochs 8 --bs 24 --chunk 1000 --lr 2e-3 --workers 3 --corr-w 3 --pmask-gamma 0.5 --select es_pm \
  --es-seqs 1000 --valid-every 0 --soft-train $T --soft-a 0.8 --tag $B > $L/$B.log 2>&1
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
FT="--base $B $TCN --chunk 1000 --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_ens.py ft_${B}_mix4_sa8 > $L/proj_ens.log 2>&1
grep -E "ENS|Error" $L/proj_ens.log | sed 's/^/GATE /'
grep -q "DECIDE_ENS 1" $L/proj_ens.log || { echo "gate failed"; echo QUEUE_DONE; exit 0; }
NM=ft_${B}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT --fold -1 > $L/$NM.log 2>&1
echo "$(date +%H:%M) FT $B soft full: $(grep -h 'EPOCH\|DONE\|Error' $L/$NM.log | tail -1 | cut -c1-120)"
D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
# r3 specialist from scratch on the same labels (sub23d recipe), gated on top of the new TCN.
B3=mingru_ema_cls3_ens; SP="--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
$PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
  --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train $T --soft-a 0.8 --tag $B3 > $L/$B3.log 2>&1
echo "$(date +%H:%M) $B3: $(grep -h 'FINAL\|Error' $L/$B3.log | cut -c1-200)"
FT3="--base $B3 $SP --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT3 --fold $f > $L/ft_${B3}_mix4_sa8_f$f.log 2>&1
done
$PY pipeline/regime_gates/proj_ens.py ft_${B}_mix4_sa8 ft_${B3}_mix4_sa8 > $L/proj_ens_r3.log 2>&1
grep -E "ENS r3|DECIDE_ENS_R3|Error" $L/proj_ens_r3.log | sed 's/^/GATE /'
grep -q "DECIDE_ENS_R3 1" $L/proj_ens_r3.log || { echo "r3 gate failed"; echo QUEUE_DONE; exit 0; }
NM3=ft_${B3}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT3 --fold -1 > $L/$NM3.log 2>&1
D=$A/nn/$NM3; mkdir -p $D; cp $A/nn_ft/$NM3/last.pt $D/best.pt; [ -f $A/nn/$B3/ema.json ] && cp $A/nn/$B3/ema.json $D/
echo "$(date +%H:%M) FT $B3 soft full done"
echo QUEUE_DONE
