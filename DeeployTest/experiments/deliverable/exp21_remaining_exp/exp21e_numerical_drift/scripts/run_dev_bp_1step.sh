#!/usr/bin/env bash
# exp21e: BP 1-step device run with DUMP_WEIGHTS (weights after update 0) + AMSIG; worker x21e_bp_1s.
# Waits for the 165-step AMSIG run (runner PID 74792, same generation dir) to finish first.
H=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift
while docker exec traindeeploy kill -0 74792 >/dev/null 2>&1; do sleep 20; done
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && PYTEST_XDIST_WORKER=x21e_bp_1s python3 -u deeployTrainingRunner_tiled_siracusa.py \
    -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref --n-steps 2 --n-accum 4 --cores 8 \
    --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
    -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON DUMP_ARGMAX=ON ARGMAX_MASK_SIG=ON" > $H/logs/dev_bp_2steps_wdump.log 2>&1
