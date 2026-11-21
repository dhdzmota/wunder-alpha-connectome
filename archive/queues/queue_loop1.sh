#!/bin/bash
# 2026-11-10 (§47): la25 passed the r3 gate (replace +0.00016/+0.00032). Chain: r3 la25 package (sub26a) -> pretrain->specialise
# gates on r0, r1 (la50 labels, queue_pt.sh) -> la25 propagated to SSM, r0/r1, TCN (each its own package if its gate passes).
cd "$(dirname "$0")/../.."
T=teacher_la25_gru
bash pipeline/queues/queue_prop.sh la25 $T $T "r3pkg" | sed 's/QUEUE_DONE/STAGE_DONE r3pkg/'
bash pipeline/queues/queue_pt.sh "0 1" | sed 's/QUEUE_DONE/STAGE_DONE pt/'
bash pipeline/queues/queue_prop.sh la25 $T $T "r3set ssm r01 tcn"
