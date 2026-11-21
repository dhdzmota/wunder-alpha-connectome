#!/bin/bash
# 2026-10-28 (§24 phase 2): SSM (ssm16_c3 recipe) retrained from scratch on la50 soft labels (a .8), soft FT on valid halves,
# gate vs plain-FT SSM halves on the sub23d-like blend, full FT, package sub23g = sub23d with this SSM.
# Usage: queue_p2_ssm.sh <PID to wait for>
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
while kill -0 "$1" 2> /dev/null; do sleep 30; done
T=teacher_la50_gru; B=ssm16_c3_st; SP="--arch ssm --hidden 128 --layers 2 --ssm-state 16"
$PY scripts/train_gru.py $SP --epochs 8 --bs 32 --chunk 1000 --lr 2e-3 --workers 3 --corr-w 3 \
  --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train $T --soft-a 0.8 --tag $B > $L/$B.log 2>&1
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
FT="--base $B $SP --chunk 1000 --weight mix4 --workers 3 --soft $T --soft-a 0.8 --suffix _sa8"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_soft.py 2>&1 | grep -E "blend sub18|P2" | sed 's/^/GATE /'
NM=ft_${B}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT --fold -1 > $L/$NM.log 2>&1
echo "$(date +%H:%M) FT $B soft full: $(grep -h 'EPOCH\|DONE\|Error' $L/$NM.log | tail -1 | cut -c1-120)"
D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
ST=$NM:ft_tcn64_ema_c3_st16_mix4_sa8_full
N=sub23g_route4_ssmscratch
$PY scripts/package_routed.py --default sub14b_ftmix16 --name $N \
  --route 0=ft_mingru_ema_cls0_mix4_sa8_full:0.5:$ST 1=ft_mingru_ema_cls1_mix4_sa8_full:0.5:$ST \
          2=ft_mingru_ema_cls2_mix4_sa8_full:0.5:$ST 3=ft_mingru_ema_cls3_st_mix4_sa8_full:0.5:$ST > $L/package_$N.log 2>&1
echo "$(date +%H:%M) PACKAGED $N: $(tail -2 $L/package_$N.log | cut -c1-200)"
cmp -s $A/stack/stack_r0.npz $A/stack_sub21/stack_r0.npz && cmp -s $A/stack/stack_r1.npz $A/stack_sub21/stack_r1.npz || { echo "stack files != sub21"; exit 1; }
ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/package_stack.py --base $N --name ${N}_stack > $L/package_${N}_stack.log 2>&1
echo "$(date +%H:%M) PACKAGED ${N}_stack: $(tail -9 $L/package_${N}_stack.log | cut -c1-220)"
echo QUEUE_DONE
