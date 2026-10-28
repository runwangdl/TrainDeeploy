#!/usr/bin/env bash
# exp21e control: BP 2-step device run on the current tree WITHOUT any exp21e instrumentation (no DUMP_ARGMAX,
# no ARGMAX_MASK_SIG) -- is the current-tree BP optimizer step itself off? worker x21e_bp_ctl. Waits for PID 82651.
H=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift
while docker exec traindeeploy kill -0 82651 >/dev/null 2>&1; do sleep 20; done
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && PYTEST_XDIST_WORKER=x21e_bp_ctl python3 -u deeployTrainingRunner_tiled_siracusa.py     -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref --n-steps 2 --n-accum 4 --cores 8     --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max     -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON" > $H/logs/dev_bp_2steps_control.log 2>&1
