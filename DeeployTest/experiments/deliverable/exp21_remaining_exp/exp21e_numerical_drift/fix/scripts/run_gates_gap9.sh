#!/usr/bin/env bash
# exp21e/fix: the SpeechNet GAP9 CI gates (ci-speechnet.yml: pytest -m "gap9_tiled and speechnet_gap9") on a worktree:
# ZO / QZO / QZO+ECO-5-steps with exact trained-weight hashes (ZO_EXPECT_WSUM), BP single step, fused QZO+ECO
# (NE16 b0 / NE16 b0-2) with the unfused run's hash. Run INSIDE deeploy_gap9: bash run_gates_gap9.sh <worktree> <log>
WT=$1; LOG=$2
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy;print("Deeploy from", Deeploy.__file__)' > $LOG 2>&1
PYTEST_XDIST_WORKER=x21e2_gates python3 -m pytest test_platforms.py -v -s -m "gap9_tiled and speechnet_gap9" >> $LOG 2>&1
echo "EXIT $?" >> $LOG
