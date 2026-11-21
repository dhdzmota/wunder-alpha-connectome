#!/bin/bash
# 2026-11-09 (EXPERIMENTS §38): valid-adapted look-ahead teacher. teacher_la50_gru fine-tuned (mix4, 3 ep) on each valid quarter,
# predicting the other quarter of the same FT half -> soft labels for valid (vteacher/vt_la50.npy) -> soft FT halves of the r3
# specialist and the TCN st16 with them (suffix _sa8vt) -> gate proj_vt.py vs base_g (labels from the train-only teacher).
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; A=~/data/wnn/artifacts; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
$PY scripts/vteacher_labels.py quarters
TE="--base teacher_la50_gru --arch gru --hidden 128 --layers 2 --extra-feats fut --weight mix4 --workers 3 --fold 0"
for q in 0 1 2 3; do
  o=$((q ^ 1))  # partner quarter in the same half
  $PY scripts/finetune_valid.py $TE --fit-groups $A/vteacher/q$o.npy --eval-groups $A/vteacher/q$q.npy --suffix _q$q > $L/ft_vt_q$q.log 2>&1
  echo "$(date +%H:%M) teacher q$q: $(grep -h 'EPOCH\|Error' $L/ft_vt_q$q.log | sed -n '1p;$p' | cut -c1-120 | tr '\n' ' ')"
done
$PY scripts/vteacher_labels.py assemble ft_teacher_la50_gru_mix4 2 vt_la50 || exit 1
R="--weight mix4 --workers 3 --soft $A/vteacher/vt_la50.npy --soft-a 0.8 --suffix _sa8vt"
SP="--base mingru_ema_cls3_st --arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1"
TC="--base tcn64_ema_c3_st16 --arch tcn --hidden 64 --layers 10 --ema-spans 2000 --chunk 1000"
for f in 0 1; do
  $PY scripts/finetune_valid.py $SP $R --fold $f > $L/ft_cls3_st_vt_f$f.log 2>&1
  echo "$(date +%H:%M) r3 fold $f: $(grep -h 'EPOCH\|Error' $L/ft_cls3_st_vt_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_vt.py > $L/proj_vt_r3.log 2>&1; grep -E "^VT|Error" $L/proj_vt_r3.log | sed 's/^/GATE r3-only /'
for f in 0 1; do
  $PY scripts/finetune_valid.py $TC $R --fold $f > $L/ft_tcn_st16_vt_f$f.log 2>&1
  echo "$(date +%H:%M) TCN fold $f: $(grep -h 'EPOCH\|Error' $L/ft_tcn_st16_vt_f$f.log | tail -1 | cut -c1-160)"
done
$PY pipeline/regime_gates/proj_vt.py > $L/proj_vt.log 2>&1
grep -E "^VT|Error" $L/proj_vt.log | sed 's/^/GATE /'
echo QUEUE_DONE
