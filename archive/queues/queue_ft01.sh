#!/bin/bash
# 2026-10-30 (EXPERIMENTS §32, jy447 recipe): soft FT with unscored rows at weight 0.1 (mix9: need*(1+9*scored)) and 6 epochs,
# lr 2e-4, vs the current mix4 / 3 ep / 3e-4. Halves of the r3 specialist and the TCN st16 first; gate proj_ft01.py at ep 2 and 5.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
R="--weight mix9 --epochs 6 --lr 2e-4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8e6"
SP="--base mingru_ema_cls3_st --arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
TC="--base tcn64_ema_c3_st16 --arch tcn --hidden 64 --layers 10 --ema-spans 2000 --chunk 1000"
for f in 0 1; do
  $PY scripts/finetune_valid.py $SP $R --fold $f > $L/ft_cls3_st_mix9e6_f$f.log 2>&1
  echo "$(date +%H:%M) r3 fold $f: $(grep -h 'EPOCH\|Error' $L/ft_cls3_st_mix9e6_f$f.log | tail -1 | cut -c1-160)"
done
for f in 0 1; do
  $PY scripts/finetune_valid.py $TC $R --fold $f > $L/ft_tcn_st16_mix9e6_f$f.log 2>&1
  echo "$(date +%H:%M) TCN fold $f: $(grep -h 'EPOCH\|Error' $L/ft_tcn_st16_mix9e6_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_ft01.py > $L/proj_ft01.log 2>&1
grep -E "FT01|DECIDE|Error" $L/proj_ft01.log | sed 's/^/GATE /'
echo QUEUE_DONE
