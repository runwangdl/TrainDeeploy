#!/usr/bin/env bash
# exp9 memory comparison (2026-10-23, feat/memory_opt = state 3, FuseFrozenBNTailPass on): ONE update step (n_accum 4)
# simulated on GVSoC with Run's BOARD settings (silentwear-zo/scripts/pw_sw_build_all.sh) and --plotMemAlloc:
#   bp       BP, deeployTrainingRunner_tiled_gap9.py, L3 default level + L2 promotion (--l1 122000 --l2 1300000)
#   zo       ZO fp32 (--l1 114000 --l2 1300000)
#   qzo      QZO unfused, cluster (--l1 114000 --l2 1340000)
#   eco      QZO+ECO (SR) unfused, cluster
#   eco_fus  QZO+ECO (SR) fused dual-pass, cluster only (--zo-fused-dual, no NE16)
#   eco_fus_ne16  QZO+ECO (SR) fused dual-pass, NE16 for blocks 0-2 (the board configuration)
# Run inside deeploy_gap9. Output: memsim/<cfg>/{run.log, *Network.c, deeployStates/ (memory_alloc.html, graphs)}. -- QW
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
O=experiments/dual_pass_fusion/exp9_memory_optim/memsim; T=../SpeechNet_Test; W=${WTAG:-m9s}
ZA="--n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --defaultMemLevel L2 --cores 8 --plotMemAlloc"
ZD="BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON"
Q="-t $T/speechnet_train_QZOECO_full_fused_round1 --optimizer-dir $T/speechnet_update_QZOECO_full_fused_round1"
G=deeployMezoRunner_tiled_GAP9.py
run() {  # cfg runner fixture-name args...   (ONLY="cfg1 cfg2" runs a subset)
  local cfg=$1 runner=$2 fix=$3; shift 3
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $cfg "* ]] && return
  mkdir -p $O/$cfg
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
  PYTEST_XDIST_WORKER=${W}_$cfg python3 $runner "$@" > $O/$cfg/run.log 2>&1
  local d; d=$(ls -d TEST_GAP9$([[ $runner == *NE16* ]] && echo _W_NE16)/$fix 2>/dev/null | head -1)
  cp $d/*Network.c $O/$cfg/ 2>/dev/null; cp -r $d/deeployStates $O/$cfg/ 2>/dev/null
  echo "$cfg: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $O/$cfg/run.log | head -1) | $(grep -ao '\[ECO\].*momentum' $O/$cfg/run.log | head -1) | $(for f in $O/$cfg/*Network.c; do echo -n "$(basename $f): $(grep -o '[a-z_0-9]*malloc(sizeof([a-z0-9_]*) \* [0-9]*)' $f | tr '\n' ' ') "; done)"
}
# BP: testUtils/deeployTrainingRunner.py prepends /usr/bin to PATH, whose python3 lacks the GAP9 SDK packages in the
# deeploy_gap9 container (kconfiglib, xxhash, ...: CMake configure fails). Run it through a wrapper that restores PATH.
cat > /tmp/x11_bp_wrap.py <<'PYEOF'
import os, runpy, sys
p = os.environ["PATH"]
sys.path.insert(0, os.getcwd())
import testUtils.deeployTrainingRunner  # noqa: F401  (applies the PATH override)
os.environ["PATH"] = p
sys.argv = ["deeployTrainingRunner_tiled_gap9.py"] + sys.argv[1:]
runpy.run_path("deeployTrainingRunner_tiled_gap9.py", run_name="__main__")
PYEOF
run bp  /tmp/x11_bp_wrap.py speechnet_train_BP_round1 -t $T/speechnet_train_BP_round1 --n-steps 1 --n-accum 4 \
  --num-data-inputs 2 --l1 122000 --l2 1300000 --defaultMemLevel L3 --doublebuffer --promoteToL2 \
  --promoteToL2Strategy=traffic-per-peak --promoteToL2Headroom=131072 --memAllocStrategy MiniMalloc --cores 8 --plotMemAlloc \
  -D BN_FROZEN_STATS=ON CC_STACK_SIZE=4096
run zo  $G speechnet_train_ZO_round1 -t $T/speechnet_train_ZO_round1 --optimizer-dir $T/speechnet_update_ZO_round1 $ZA --l2 1300000 --lr 3e-6 -D $ZD
run qzo $G speechnet_train_QZOECO_full_fused_round1 $Q $ZA --l2 1340000 --lr 1e-5 -D $ZD
run eco $G speechnet_train_QZOECO_full_fused_round1 $Q $ZA --l2 1340000 --lr 3e-6 -D $ZD ZO_ECO=ON ZO_ECO_SR=ON
run eco_fus $G speechnet_train_QZOECO_full_fused_round1 $Q $ZA --l2 1340000 --lr 3e-6 --zo-fused-dual -D $ZD ZO_ECO=ON ZO_ECO_SR=ON
# eco_fus_ne16: the board configuration (pw_sw_build_all.sh): fused, NE16 for blocks 0-2. Output dir TEST_GAP9_W_NE16.
run eco_fus_ne16 deeployMezoRunner_tiled_GAP9_w_NE16.py speechnet_train_QZOECO_full_fused_round1 $Q $ZA --l2 1340000 --lr 3e-6 \
  --zo-fused-dual --zo-fused-dual-ne16 0,1,2 -D $ZD ZO_ECO=ON ZO_ECO_SR=ON
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
