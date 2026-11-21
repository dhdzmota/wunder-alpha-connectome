#!/bin/bash
# 2026-10-29 (§24): r0/r1 scratch specs pass UNDER the row stack (proj_stack2: old stack 0.6922, refit a .35 0.6923 vs
# sub23g-like 0.6916, both halves up). Full soft FT of both specs now (GPU, small); after queue_p3_mix (PID $1) the stack
# refit into stack_v2 (CPU) and sub24 = sub23g + r0/r1 scratch specs + refit stacks; also sub24_oldstack for comparison.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
T=teacher_la50_gru
for r in 0 1; do
  B=mingru_ema_cls${r}_st; NM=ft_${B}_mix4_sa8_full
  $PY scripts/finetune_valid.py --base $B --arch mingru --ema-spans 2000 --regimes $r --class-head --ce-w 1 --weight mix4 --workers 3 \
    --soft $T --soft-a 0.8 --suffix _sa8 --fold -1 > $L/$NM.log 2>&1
  echo "$(date +%H:%M) FT $B soft full: $(grep -h 'EPOCH\|DONE\|Error' $L/$NM.log | tail -1 | cut -c1-120)"
  D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
done
ST=ft_ssm16_c3_st_mix4_sa8_full:ft_tcn64_ema_c3_st16_mix4_sa8_full
N=sub24_route4_r01scratch
$PY scripts/package_routed.py --default sub14b_ftmix16 --name $N \
  --route 0=ft_mingru_ema_cls0_st_mix4_sa8_full:0.5:$ST 1=ft_mingru_ema_cls1_st_mix4_sa8_full:0.5:$ST \
          2=ft_mingru_ema_cls2_mix4_sa8_full:0.5:$ST 3=ft_mingru_ema_cls3_st_mix4_sa8_full:0.5:$ST > $L/package_$N.log 2>&1
echo "$(date +%H:%M) PACKAGED $N: $(tail -2 $L/package_$N.log | cut -c1-200)"
while kill -0 "$1" 2> /dev/null; do sleep 30; done
rm -f $L/train_stack_v2.log
for r in 0 1; do
  ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/train_stack.py --regimes $r --spec ft_mingru_ema_cls{r}_st_mix4_sa8 \
    --tcn ft_tcn64_ema_c3_st16_mix4_sa8 --alpha 0.35 --out stack_v2 >> $L/train_stack_v2.log 2>&1
done
echo "$(date +%H:%M) stacks v2: $(tail -2 $L/train_stack_v2.log | cut -c1-160)"
ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/package_stack.py --base $N --stack-dir stack_v2 --name ${N}_stack2 > $L/package_${N}_stack2.log 2>&1
echo "$(date +%H:%M) PACKAGED ${N}_stack2: $(tail -9 $L/package_${N}_stack2.log | cut -c1-220)"
echo QUEUE_DONE
