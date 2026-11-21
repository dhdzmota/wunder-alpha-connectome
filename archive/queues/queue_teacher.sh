#!/bin/bash
# 2026-10-27: soft-label distillation, step 1 + gate (EXPERIMENTS §23). Look-ahead teacher (training only: wnn/fut_feats.py,
# mean of the next 2..50 rows of the i0 columns) trained on train only -> its valid predictions are leak-free soft labels.
# Gate: FT of existing members on valid halves with target (1-a) y + a teacher, weights |y| + 0.2, vs the plain FT.
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; PY=~/venvs/wnn/bin/python
export OMP_NUM_THREADS=4
T=teacher_la50_gru
$PY scripts/train_gru.py --arch gru --extra-feats fut --epochs 4 --bs 32 --workers 3 --corr-w 3 --pmask-gamma 0.5 --select es_pm --valid-every 0 --es-seqs 1000 --tag $T > $L/$T.log 2>&1
echo "$(date +%H:%M) $T: $(grep -h 'FINAL\|Error' $L/$T.log | cut -c1-200)"
ft() {  # base, spec, suffix, a
  for f in 0 1; do
    $PY scripts/finetune_valid.py --base $1 $2 --weight mix4 --fold $f --workers 3 --soft $T --soft-a $4 --suffix $3 > $L/ft_$1_mix4$3_f$f.log 2>&1
    echo "$(date +%H:%M) FT $1$3 fold $f: $(grep -h 'EPOCH\|Error' $L/ft_$1_mix4$3_f$f.log | tail -1 | cut -c1-160)"
  done
}
TCN="--arch tcn --hidden 64 --layers 10 --ema-spans 2000 --chunk 1000"
ft tcn64_ema_c3 "$TCN" _sa8 0.8
ft tcn64_ema_c3 "$TCN" _sa5 0.5
ft mingru_ema_cls3 "--arch mingru --ema-spans 2000 --regimes 3 --class-head --ce-w 1" _sa8 0.8
echo QUEUE_DONE
