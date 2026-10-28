#!/usr/bin/env bash
# exp21e/fix task 3: COMPLETE ZO round 1 (1620 steps = 120 epochs, n_accum 4) on Siracusa GVSoC against the
# out-of-place reference fixture SpeechNet_Test/speechnet_{train,update}_ZO_round1_oop. Settings = exp18 phase 3.
# Run INSIDE traindeeploy: bash run_sir_zo_round.sh <worker> <n_steps> <log> [extra -D]   -- QW exp21e
WT=/app/ETH/TrainDeeploy/.worktrees/exp21e_bp_debug
TAG=$1; NS=$2; LOG=$3; shift 3
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy;print("Deeploy from", Deeploy.__file__)' > $LOG 2>&1
PYTEST_XDIST_WORKER=$TAG python3 -u deeployMezoRunner_tiled_siracusa.py \
  -t ../SpeechNet_Test/speechnet_train_ZO_round1_oop --optimizer-dir ../SpeechNet_Test/speechnet_update_ZO_round1_oop \
  --n-steps $NS --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max --cores 8 \
  -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON "$@" >> $LOG 2>&1
echo "EXIT $?" >> $LOG
