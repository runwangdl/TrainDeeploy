#!/usr/bin/env bash
# exp21e/fix task 4: COMPLETE BP round 1 on Siracusa GVSoC (exp17 settings, 540 steps, n_accum 4) against the C-order
# reference fixture speechnet_train_fullfrozen_b1_fold3_corder, device built STRICT fp32 (generated networks +
# whole PULPOpen kernel library: -fno-fast-math -ffp-contract=off), raw loss bits and final weights dumped.
# Run INSIDE traindeeploy: bash run_sir_bp_round.sh <worker> <n_steps> <log> [extra -D]           -- QW exp21e
WT=/app/ETH/TrainDeeploy/.worktrees/exp21e_bp_debug
TAG=$1; NS=$2; LOG=$3; shift 3
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy;print("Deeploy from", Deeploy.__file__)' > $LOG 2>&1
PYTEST_XDIST_WORKER=$TAG python3 -u deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_corder --n-steps $NS --n-accum 4 --cores 8 \
  --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON DUMP_LOSS_HEX=ON DEEPLOY_STRICT_FP32=ON DEEPLOY_STRICT_FP32_ALL=ON "$@" >> $LOG 2>&1
echo "EXIT $?" >> $LOG
