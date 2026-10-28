#!/usr/bin/env bash
# exp21e/fix: BP device sim on GAP9 GVSoC at Run's BOARD BP settings (silentwear-zo/scripts/pw_sw_build_all.sh, pw_bp),
# fixture SpeechNet_Test/speechnet_train_BP_round1 (== Tests/.../speechnet_train_fullfrozen_b1_fold3_ref, byte-identical npz).
# Run INSIDE deeploy_gap9:  [FIXTURE=<SpeechNet_Test dir>] bash run_gap9_bp.sh <worker-tag> <n_steps> <logfile> [extra -D flags...]
# Uses the exp21e worktree's Deeploy (PYTHONPATH) -- the gap9 venv otherwise imports /app/Deeploy (main tree). -- QW exp21e
# (no set -u: the GAP9 SDK env scripts reference unset variables)
TAG=$1; NS=$2; LOG=$3; shift 3
WT=${WT:-/app/Deeploy/.worktrees/exp21e_bp_debug}
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy;print("Deeploy from", Deeploy.__file__)' > $LOG 2>&1
# testUtils/deeployTrainingRunner.py prepends /usr/bin to PATH (hides the venv's kconfiglib) -> restore PATH (exp9 wrapper)
cat > /tmp/x21e2_bp_wrap.py <<'PYEOF'
import os, runpy, sys
p = os.environ["PATH"]
sys.path.insert(0, os.getcwd())
import testUtils.deeployTrainingRunner  # noqa: F401
os.environ["PATH"] = p
sys.argv = ["deeployTrainingRunner_tiled_gap9.py"] + sys.argv[1:]
runpy.run_path("deeployTrainingRunner_tiled_gap9.py", run_name="__main__")
PYEOF
T=../SpeechNet_Test
PYTEST_XDIST_WORKER=$TAG python3 -u /tmp/x21e2_bp_wrap.py -t $T/${FIXTURE:-speechnet_train_BP_round1} --n-steps $NS --n-accum 4 \
  --num-data-inputs 2 --l1 122000 --l2 1300000 --defaultMemLevel L3 --doublebuffer --promoteToL2 \
  --promoteToL2Strategy=traffic-per-peak --promoteToL2Headroom=131072 --memAllocStrategy MiniMalloc --cores 8 \
  -D BN_FROZEN_STATS=ON CC_STACK_SIZE=4096 "$@" >> $LOG 2>&1
echo "EXIT $?" >> $LOG
