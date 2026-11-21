#!/bin/bash
# 2026-11-10 (EXPERIMENTS §45): propagate a label recipe that passed the r3 gate (queue_lab.sh) to the other members.
# Usage: queue_prop.sh <X> <train label tag> <valid soft: teacher tag or .npy> "<stages>"
#   X = variant suffix (r3 base mingru_ema_cls3_<X> and its soft-FT halves already exist from queue_lab.sh)
#   stages, in order of public gain per GPU hour: r3pkg ssm r01 tcn
# Every stage: member from scratch on the labels -> soft FT halves -> gate proj_prop.py (vs the sub25 members) -> if it passes,
# full FT and its own package sub26<stage>_... = previous passing package with the member swapped (+ stack_v3, SSM kept in r0/r1).
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
X=$1; LT=$2; LV=$3; STAGES=$4
R0=ft_mingru_ema_cls0_st_mix4_sa8; R1=ft_mingru_ema_cls1_st_mix4_sa8; R2F=ft_mingru_ema_cls2_mix4_sa8_full
R3=ft_mingru_ema_cls3_st_mix4_sa8; SS=ft_ssm16_c3_st_mix4_sa8; TC=ft_tcn64_ema_c3_st16_mix4_sa8
SWAPS=""   # accumulated passing swaps for the gate (key=halves tag)

train() {  # $1 tag, $2 arch args, $3 extra train args
  $PY scripts/train_gru.py $2 --lr 2e-3 --workers 3 --corr-w 3 --pmask-gamma 0.5 --select es_pm --valid-every 0 \
    --soft-train $LT --soft-a 0.8 $3 --tag $1 > $L/$1.log 2>&1
  echo "$(date +%H:%M) $1: $(grep -h 'FINAL\|Error' $L/$1.log | cut -c1-200)"
}
ft() {  # $1 base tag, $2 arch args, $3 fold (-1 = all valid -> nn/ft_<base>_mix4_sa8_full)
  local sfx=$([ "$3" = -1 ] && echo full || echo f$3)
  $PY scripts/finetune_valid.py --base $1 $2 --weight mix4 --workers 3 --soft $LV --soft-a 0.8 --suffix _sa8 --fold $3 > $L/ft_$1_mix4_sa8_$sfx.log 2>&1
  echo "$(date +%H:%M) FT $1 $sfx: $(grep -h 'EPOCH\|DONE\|Error' $L/ft_$1_mix4_sa8_$sfx.log | tail -1 | cut -c1-140)"
  if [ "$3" = -1 ]; then
    local D=$A/nn/ft_$1_mix4_sa8_full; mkdir -p $D; cp $A/nn_ft/ft_$1_mix4_sa8_full/last.pt $D/best.pt; [ -f $A/nn/$1/ema.json ] && cp $A/nn/$1/ema.json $D/
  fi
}
gate() {  # $1 key, $2 halves tag -> 0 if the new member improves the blend that already holds the earlier passing swaps
  $PY pipeline/regime_gates/proj_prop.py --incremental $SWAPS $1=$2 > $L/proj_prop_$1_$X.log 2>&1
  grep -E "^PROP|^INCR|Error" $L/proj_prop_$1_$X.log | sed "s/^/GATE $1 /"
  grep -q "^INCR_DECIDE 1" $L/proj_prop_$1_$X.log
}
package() {  # $1 name; uses the *_FULL member tags
  local ST=$SSF:$TCF
  $PY scripts/package_routed.py --default sub14b_ftmix16 --name $1 --route 0=$R0F:0.5:$ST 1=$R1F:0.5:$ST 2=$R2F:0.5:$ST 3=$R3F:0.5:$ST > $L/package_$1.log 2>&1
  ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/package_stack.py --base $1 --stack-dir $STACK --stack 0,1 --keep-ssm 0,1 --name $1_stack > $L/package_$1_stack.log 2>&1
  cp $A/submissions/$1_stack.zip submissions/
  echo "$(date +%H:%M) PACKAGED $1_stack: $(grep -E 'max diff|max\|numba|latency|Error' $L/package_$1_stack.log | cut -c1-200 | tr '\n' ';')"
}
R0F=${R0}_full; R1F=${R1}_full; R3F=${R3}_full; SSF=${SS}_full; TCF=${TC}_full; STACK=stack_v3
SPM="--arch mingru --ema-spans 2000 --class-head --ce-w 1"
for st in $STAGES; do
  case $st in
  r3pkg)  # r3 gate already passed in queue_lab.sh: full FT + package
    B=mingru_ema_cls3_$X; SWAPS="$SWAPS r3=ft_${B}_mix4_sa8"
    ft $B "$SPM --regimes 3" -1; R3F=ft_${B}_mix4_sa8_full; package sub26a_route4_r3$X ;;
  r3set)  # r3 already packaged by an earlier call: only carry its state
    B=mingru_ema_cls3_$X; SWAPS="$SWAPS r3=ft_${B}_mix4_sa8"; R3F=ft_${B}_mix4_sa8_full ;;
  ssm)
    B=ssm16_c3_$X; AR="--arch ssm --hidden 128 --layers 2 --ssm-state 16"
    train $B "$AR" "--epochs 8 --bs 32 --chunk 1000"
    ft $B "$AR --chunk 1000" 0; ft $B "$AR --chunk 1000" 1
    if gate ssm ft_${B}_mix4_sa8; then SWAPS="$SWAPS ssm=ft_${B}_mix4_sa8"; ft $B "$AR --chunk 1000" -1; SSF=ft_${B}_mix4_sa8_full; package sub26b_route4_ssm$X; fi ;;
  r01)
    for r in 0 1; do
      B=mingru_ema_cls${r}_$X
      train $B "$SPM --regimes $r --hidden 128 --layers 2" "--epochs 8 --bs 32 --chunk 2000"
      ft $B "$SPM --regimes $r" 0; ft $B "$SPM --regimes $r" 1
      if gate r$r ft_${B}_mix4_sa8; then
        SWAPS="$SWAPS r$r=ft_${B}_mix4_sa8"; ft $B "$SPM --regimes $r" -1
        [ $r = 0 ] && { R0=ft_${B}_mix4_sa8; R0F=${R0}_full; } || { R1=ft_${B}_mix4_sa8; R1F=${R1}_full; }
        NEW01=1
      fi
    done
    if [ -n "$NEW01" ]; then   # stacks refit on the routes that changed
      STACK=stack_$X; rm -f $L/train_stack_$X.log; SSH=$(echo "$SWAPS" | tr ' ' '\n' | grep '^ssm=' | cut -d= -f2); TCH=$(echo "$SWAPS" | tr ' ' '\n' | grep '^tcn=' | cut -d= -f2)
      for r in 0 1; do
        SPEC=$([ $r = 0 ] && echo $R0 || echo $R1)
        ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/train_stack.py --regimes $r --with-ssm $r --spec $SPEC --tcn ${TCH:-$TC} --ssm ${SSH:-$SS} \
          --alpha 0.35 --out $STACK >> $L/train_stack_$X.log 2>&1
      done
      package sub26c_route4_r01$X
    fi ;;
  tcn)
    B=tcn64_ema_c3_$X; AR="--arch tcn --hidden 64 --layers 10 --ema-spans 2000"
    train $B "$AR" "--epochs 16 --bs 24 --chunk 1000"
    ft $B "$AR --chunk 1000" 0; ft $B "$AR --chunk 1000" 1
    if gate tcn ft_${B}_mix4_sa8; then SWAPS="$SWAPS tcn=ft_${B}_mix4_sa8"; ft $B "$AR --chunk 1000" -1; TCF=ft_${B}_mix4_sa8_full; package sub26d_route4_tcn$X; fi ;;
  esac
done
echo QUEUE_DONE
