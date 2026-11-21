#!/bin/bash
# 2026-10-29 (§24 phase 2): r1 and r0 specialists from scratch on la50 soft labels (cls recipe, 8 ep), soft FT halves,
# gate on the sub23g-like blend (both halves up, sum >= +0.0004); full FT of the passing ones; package sub23h = sub23g with them.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
T=teacher_la50_gru
for r in 1 0; do
  B=mingru_ema_cls${r}_st; SP="--arch mingru --ema-spans 2000 --regimes $r --class-head --ce-w 1"
  $PY scripts/train_gru.py $SP --hidden 128 --layers 2 --epochs 8 --bs 32 --chunk 2000 --lr 2e-3 --workers 3 --corr-w 3 \
    --pmask-gamma 0.5 --select es_pm --valid-every 0 --soft-train $T --soft-a 0.8 --tag $B > $L/$B.log 2>&1
  echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
  for f in 0 1; do
    $PY scripts/finetune_valid.py --base $B $SP --weight mix4 --workers 3 --soft $T --soft-a 0.8 --suffix _sa8 --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
    echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
  done
done
$PY pipeline/regime_gates/proj_soft.py 2>&1 | grep -E "P2 base|P2 spec r0|P2 spec r1|DECIDE" | sed 's/^/GATE /' | tee $L/gate_p2_r01.txt
R0=ft_mingru_ema_cls0_mix4_sa8_full; R1=ft_mingru_ema_cls1_mix4_sa8_full; NEW=0
for r in 0 1; do
  grep -q "DECIDE r$r 1" $L/gate_p2_r01.txt || continue
  B=mingru_ema_cls${r}_st; NM=ft_${B}_mix4_sa8_full; NEW=1
  $PY scripts/finetune_valid.py --base $B --arch mingru --ema-spans 2000 --regimes $r --class-head --ce-w 1 --weight mix4 --workers 3 \
    --soft $T --soft-a 0.8 --suffix _sa8 --fold -1 > $L/$NM.log 2>&1
  echo "$(date +%H:%M) FT $B soft full: $(grep -h 'EPOCH\|DONE\|Error' $L/$NM.log | tail -1 | cut -c1-120)"
  D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
  [ $r = 0 ] && R0=$NM || R1=$NM
done
[ $NEW = 1 ] || { echo "no r0/r1 spec passed the gate"; echo QUEUE_DONE; exit 0; }
ST=ft_ssm16_c3_st_mix4_sa8_full:ft_tcn64_ema_c3_st16_mix4_sa8_full
N=sub23h_route4_r01scratch
$PY scripts/package_routed.py --default sub14b_ftmix16 --name $N \
  --route 0=$R0:0.5:$ST 1=$R1:0.5:$ST \
          2=ft_mingru_ema_cls2_mix4_sa8_full:0.5:$ST 3=ft_mingru_ema_cls3_st_mix4_sa8_full:0.5:$ST > $L/package_$N.log 2>&1
echo "$(date +%H:%M) PACKAGED $N: $(tail -2 $L/package_$N.log | cut -c1-200)"
cmp -s $A/stack/stack_r0.npz $A/stack_sub21/stack_r0.npz && cmp -s $A/stack/stack_r1.npz $A/stack_sub21/stack_r1.npz || { echo "stack files != sub21"; exit 1; }
ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/package_stack.py --base $N --name ${N}_stack > $L/package_${N}_stack.log 2>&1
echo "$(date +%H:%M) PACKAGED ${N}_stack: $(tail -9 $L/package_${N}_stack.log | cut -c1-220)"
echo QUEUE_DONE
