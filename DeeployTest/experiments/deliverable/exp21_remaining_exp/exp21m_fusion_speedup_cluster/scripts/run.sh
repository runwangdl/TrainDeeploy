#!/usr/bin/env bash
# exp21m: speed-up of the dual-pass fusion ALONE, GAP9 cluster only (no NE16), QZO-ECO, ONE update step (n_accum 4
# training-graph runs + 1 update-graph run), plain GVSoC (BENCH, no --profileTiling). Command lines = exp21c
# scripts/run_network.sh (Run's board settings --l1 114000 --l2 1340000, frozen BN, GAP9_NETWORK_O3, ECO + SR, lr 3e-6).
#   eco_plain   unfused, pulp-nn convs + shipped perturbation nodes                     (reference A, = exp21c eco_plain)
#   ctrl_plain  unfused control: block 0 FlatConv (DEEPLOY_RQCONV_FLAT=1), blocks 1-4 PerturbConv4x2 (perturbation in
#               the kernel, single pass; graph from build_ctrl_fixture.py) -> everything of the fused build but fusion
#   fus_plain   fused dual pass, cluster only (DualConv block 0, DualConv2In blocks 1-4)
# Bit-exactness: all three must give Errors 0/8 and [WSUM s=0] 0xfa1f7dd2 (exp21c / merge-gate value).
# Run inside deeploy_gap9:  bash scripts/run.sh        (ONLY="ctrl_plain" etc. to run a subset)
# gvsoc killed only by PID, only processes whose cwd is the run's own build dir (build_x21m_<cfg>/). -- QW exp21m
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
E=experiments/deliverable/exp21_remaining_exp/exp21m_fusion_speedup_cluster; O=$E/logs; F=$E/fixtures; CG=$E/codegen
ZA="--n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --l2 1340000 --lr 3e-6"
ZA="$ZA --defaultMemLevel L2 --cores 8"
ZD="BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_ECO=ON ZO_ECO_SR=ON"
U="--optimizer-dir $F/speechnet_update_QZOECO_full_fused_round1"
G=deeployMezoRunner_tiled_GAP9.py
cfgs=(${ONLY:-ctrl_plain eco_plain fus_plain})

kill_mine() {  # $1 = cfg: only that run's build dir (build_x21m_<cfg>/), so parallel lanes never kill each other
  for p in $(pgrep -f "[g]vsoc_launcher" 2>/dev/null); do
    if readlink /proc/$p/cwd 2>/dev/null | grep -qE "build_x21m_$1/"; then kill -9 $p 2>/dev/null; fi
  done
}
run() {  # cfg fixture extra-args...
  local cfg=$1 fx=$2; shift 2
  kill_mine $cfg
  PYTEST_XDIST_WORKER=x21m_$cfg timeout 7200 python3 $G -t $F/$fx $U $ZA "$@" > $O/net_$cfg.log 2>&1
  kill_mine $cfg
  local d; d=$(ls -d TEST_GAP9/$F/$fx 2>/dev/null | head -1)
  for f in $d/*Network.c; do cp $f $CG/net_${cfg}_$(basename $f); done
  echo "$cfg: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $O/net_$cfg.log | head -1) | $(grep -a '^BENCH' $O/net_$cfg.log | head -1) | $(grep -a 'WSUM s=0' $O/net_$cfg.log | tail -1)" | tee -a $E/results/summary.txt
}
for c in "${cfgs[@]}"; do
  case $c in
    eco_plain)  run eco_plain  speechnet_train_QZOECO_full_fused_round1 -D $ZD ;;
    ctrl_plain) DEEPLOY_RQCONV_FLAT=1 run ctrl_plain speechnet_train_QZOECO_ctrl_round1 -D $ZD ;;
    fus_plain)  run fus_plain  speechnet_train_QZOECO_full_fused_round1 --zo-fused-dual -D $ZD ;;
  esac
done
