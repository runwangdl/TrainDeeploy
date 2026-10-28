#!/usr/bin/env bash
# exp21e/fix: BP device sim on Siracusa GVSoC with the exp17 settings, fixture speechnet_train_fullfrozen_b1_fold3_ref.
# Run INSIDE traindeeploy:  bash run_sir_bp.sh <worktree-in-container> <worker-tag> <n_steps> <logfile> [extra -D flags]
# PYTHONPATH points Deeploy at the given worktree (traindeeploy has the MAIN tree pip-installed editable). -- QW exp21e
set -u
WT=$1; TAG=$2; NS=$3; LOG=$4; shift 4
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy;print("Deeploy from", Deeploy.__file__)' > $LOG 2>&1
(cd $WT && git log -1 --format='commit %H %s' 2>/dev/null) >> $LOG
PYTEST_XDIST_WORKER=$TAG python3 -u deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3_ref --n-steps $NS --n-accum 4 --cores 8 \
  --l1 128000 --l2 1500000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON "$@" >> $LOG 2>&1
echo "EXIT $?" >> $LOG
