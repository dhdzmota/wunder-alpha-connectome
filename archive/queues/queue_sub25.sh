#!/bin/bash
# 2026-11-10 (EXPERIMENTS §44): sub25 = sub24f with the SSM back in the r0 / r1 routes under refit row stacks (stack_v3).
cd "$(dirname "$0")/../.."
L=~/data/wnn/artifacts/logs; PY=~/venvs/wnn/bin/python
rm -f $L/train_stack_v3.log
for r in 0 1; do
  ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/train_stack.py --regimes $r --with-ssm $r --spec ft_mingru_ema_cls{r}_st_mix4_sa8 \
    --tcn ft_tcn64_ema_c3_st16_mix4_sa8 --ssm ft_ssm16_c3_st_mix4_sa8 --alpha 0.35 --out stack_v3 >> $L/train_stack_v3.log 2>&1
done
echo "$(date +%H:%M) stacks v3: $(tail -2 $L/train_stack_v3.log | cut -c1-160)"
N=sub25_route4_ssm01_stack
ARROW_DEFAULT_MEMORY_POOL=system $PY scripts/package_stack.py --base sub24f_route4_fastexp --stack-dir stack_v3 --stack 0,1 --keep-ssm 0,1 \
  --name $N > $L/package_$N.log 2>&1
echo "$(date +%H:%M) PACKAGED $N: $(tail -9 $L/package_$N.log | cut -c1-240)"
echo QUEUE_DONE
