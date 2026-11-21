#!/bin/bash
# 2026-10-27: sub22 = sub20 structure (specialists + SSM + TCN, FT mix4 on all valid) with soft labels from the look-ahead
# teacher (teacher_la50_gru, train only -> leak-free on valid), then the sub21 stack swap on r0/r1 (EXPERIMENTS §23).
# First: control for the gate's chunk confound (TCN plain FT at --chunk 1000, as the _sa8 run).
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
SA=${SA:-0.8}; SFX=_sa${SA#0.}; T=teacher_la50_gru
while pgrep -f queue_teacher.sh > /dev/null; do sleep 60; done
TCN="--arch tcn --hidden 64 --layers 10 --ema-spans 2000"
for f in 0 1; do
  $PY scripts/finetune_valid.py --base tcn64_ema_c3 $TCN --chunk 1000 --weight mix4 --fold $f --workers 3 --suffix _c1k > $L/ft_tcn64_ema_c3_mix4_c1k_f$f.log 2>&1
  echo "$(date +%H:%M) CONTROL TCN plain chunk1000 fold $f: $(grep -h 'EPOCH\|Error' $L/ft_tcn64_ema_c3_mix4_c1k_f$f.log | tail -1 | cut -c1-140)"
done
full() {  # base, spec, staged name
  $PY scripts/finetune_valid.py --base $1 $2 --weight mix4 --fold -1 --workers 3 --soft $T --soft-a $SA --suffix $SFX > $L/ft_$1_mix4${SFX}_full.log 2>&1
  echo "$(date +%H:%M) FT $1 soft full: $(grep -h 'EPOCH\|DONE\|Error' $L/ft_$1_mix4${SFX}_full.log | tail -1 | cut -c1-120)"
  D=$A/nn/$3; mkdir -p $D; cp $A/nn_ft/ft_$1_mix4${SFX}_full/last.pt $D/best.pt; [ -f $A/nn/$1/ema.json ] && cp $A/nn/$1/ema.json $D/
}
for r in 0 1 2 3; do
  full mingru_ema_cls$r "--arch mingru --ema-spans 2000 --regimes $r --class-head --ce-w 1" ft_mingru_ema_cls${r}_mix4${SFX}_full
done
full tcn64_ema_c3 "$TCN" ft_tcn64_ema_c3_mix4${SFX}_full
full ssm16_c3 "--arch ssm --hidden 128 --layers 2 --ssm-state 16" ft_ssm_mix4${SFX}_full
ST=ft_ssm_mix4${SFX}_full:ft_tcn64_ema_c3_mix4${SFX}_full
N=sub22_route4_soft
$PY scripts/package_routed.py --default sub14b_ftmix16 --name $N \
  --route 0=ft_mingru_ema_cls0_mix4${SFX}_full:0.5:$ST 1=ft_mingru_ema_cls1_mix4${SFX}_full:0.5:$ST \
          2=ft_mingru_ema_cls2_mix4${SFX}_full:0.5:$ST 3=ft_mingru_ema_cls3_mix4${SFX}_full:0.5:$ST > $L/package_$N.log 2>&1
echo "$(date +%H:%M) PACKAGED $N: $(tail -2 $L/package_$N.log | cut -c1-200)"
ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/package_stack.py --base $N --name ${N}_stack > $L/package_${N}_stack.log 2>&1
echo "$(date +%H:%M) PACKAGED ${N}_stack: $(tail -9 $L/package_${N}_stack.log | cut -c1-220)"
echo QUEUE_DONE
