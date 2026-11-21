#!/bin/bash
# 2026-10-28 (EXPERIMENTS §24): patient distillation (Beyer et al. 2022): same TCN as tcn64_ema_c3_st8 (soft la50, a .8)
# but 16 epochs; soft FT on valid halves -> gate (proj_soft) vs st8; then full FT on all valid and sub23b package
# (sub22 with this TCN) so it can be submitted right away if the gate holds.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
B=tcn64_ema_c3_st16; TCN="--arch tcn --hidden 64 --layers 10 --ema-spans 2000"
$PY scripts/train_gru.py $TCN --epochs 16 --bs 24 --chunk 1000 --lr 2e-3 --workers 3 --corr-w 3 --pmask-gamma 0.5 --select es_pm \
  --es-seqs 1000 --valid-every 0 --soft-train teacher_la50_gru --soft-a 0.8 --tag $B > $L/$B.log 2>&1
echo "$(date +%H:%M) $B: $(grep -h 'FINAL\|Error' $L/$B.log | cut -c1-200)"
FT="--base $B $TCN --chunk 1000 --weight mix4 --workers 3 --soft teacher_la50_gru --soft-a 0.8 --suffix _sa8"
for f in 0 1; do
  $PY scripts/finetune_valid.py $FT --fold $f > $L/ft_${B}_mix4_sa8_f$f.log 2>&1
  echo "$(date +%H:%M) FT $B fold $f: $(grep -h 'EPOCH\|Error' $L/ft_${B}_mix4_sa8_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_soft.py 2>&1 | grep -E "blend sub18|st8|st16" | sed 's/^/GATE /'
NM=ft_${B}_mix4_sa8_full
$PY scripts/finetune_valid.py $FT --fold -1 > $L/$NM.log 2>&1
echo "$(date +%H:%M) FT $B soft full: $(grep -h 'EPOCH\|DONE\|Error' $L/$NM.log | tail -1 | cut -c1-120)"
D=$A/nn/$NM; mkdir -p $D; cp $A/nn_ft/$NM/last.pt $D/best.pt; [ -f $A/nn/$B/ema.json ] && cp $A/nn/$B/ema.json $D/
ST=ft_ssm_mix4_sa8_full:$NM
N=sub23b_route4_softtcn16
$PY scripts/package_routed.py --default sub14b_ftmix16 --name $N \
  --route 0=ft_mingru_ema_cls0_mix4_sa8_full:0.5:$ST 1=ft_mingru_ema_cls1_mix4_sa8_full:0.5:$ST \
          2=ft_mingru_ema_cls2_mix4_sa8_full:0.5:$ST 3=ft_mingru_ema_cls3_mix4_sa8_full:0.5:$ST > $L/package_$N.log 2>&1
echo "$(date +%H:%M) PACKAGED $N: $(tail -2 $L/package_$N.log | cut -c1-200)"
cmp -s $A/stack/stack_r0.npz $A/stack_sub21/stack_r0.npz && cmp -s $A/stack/stack_r1.npz $A/stack_sub21/stack_r1.npz || { echo "stack files != sub21"; exit 1; }
ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/package_stack.py --base $N --name ${N}_stack > $L/package_${N}_stack.log 2>&1
echo "$(date +%H:%M) PACKAGED ${N}_stack: $(tail -9 $L/package_${N}_stack.log | cut -c1-220)"
