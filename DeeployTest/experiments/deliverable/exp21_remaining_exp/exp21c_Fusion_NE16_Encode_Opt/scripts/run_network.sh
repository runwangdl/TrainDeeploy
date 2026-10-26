#!/usr/bin/env bash
# exp21c task 3: whole-network per-node accounting of the exp9 memsim runs (ONE update step = n_accum 4 training-graph
# runs + 1 update-graph run, Run's board settings --l1 114000 --l2 1340000, frozen BN, GAP9_NETWORK_O3, QZO+ECO SR,
# lr 3e-6), exactly exp9's scripts/mem_single_step.sh command lines minus --plotMemAlloc, with --profileTiling (wall
# clock per node and tile, exp8 metric) and without (plain BENCH, to re-check 13.27 M / 6.57 M on today's tree).
#   eco_prof / eco_plain            unfused (cluster, pulp-nn convs)
#   ne16_prof / ne16_plain          fused dual pass, NE16 for blocks 0-2, cluster DualConv2In for 3-4 (board recipe)
#   fus_prof                        fused dual pass, cluster only (optional, ONLY=fus_prof)
# Fixtures are copies of SpeechNet_Test/speechnet_{train,update}_QZOECO_full_fused_round1 in ../fixtures (private
# generated-code directory). Run inside deeploy_gap9. gvsoc killed only by PID (own build dirs). -- QW exp21c
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
E=experiments/deliverable/exp21_remaining_exp/exp21c_Fusion_NE16_Encode_Opt; O=$E/logs; F=$E/fixtures; CG=$E/codegen
ZA="--n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --defaultMemLevel L2 --cores 8"
ZD="BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON"
Q="-t $F/speechnet_train_QZOECO_full_fused_round1 --optimizer-dir $F/speechnet_update_QZOECO_full_fused_round1"
G=deeployMezoRunner_tiled_GAP9.py; GN=deeployMezoRunner_tiled_GAP9_w_NE16.py
cfgs=(${ONLY:-eco_prof ne16_prof eco_plain ne16_plain})

kill_mine() {  # only gvsoc_launcher processes whose working directory is one of THIS script's build dirs (build_x21c_net_*);
  # gvsoc_launcher's command line carries no build path, its cwd (<build>/gvsoc_workdir) does. Never by name. -- QW exp21c
  for p in $(pgrep -f "[g]vsoc_launcher" 2>/dev/null); do
    if readlink /proc/$p/cwd 2>/dev/null | grep -qE "build_x21c_net_"; then kill -9 $p 2>/dev/null; fi
  done
}
run() {  # cfg runner args...
  local cfg=$1 runner=$2; shift 2
  kill_mine
  PYTEST_XDIST_WORKER=x21c_net_$cfg timeout 7200 python3 $runner "$@" > $O/net_$cfg.log 2>&1
  kill_mine
  local d; d=$(ls -d TEST_GAP9$([[ $runner == *NE16* ]] && echo _W_NE16)/$F/speechnet_train_QZOECO_full_fused_round1 2>/dev/null | head -1)
  for f in $d/*Network.c; do cp $f $CG/net_${cfg}_$(basename $f); done
  echo "$cfg: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $O/net_$cfg.log | head -1) | $(grep -a '^BENCH' $O/net_$cfg.log | head -1)"
}
for c in "${cfgs[@]}"; do
  case $c in
    eco_prof)   run eco_prof   $G  $Q $ZA --l2 1340000 --lr 3e-6 --profileTiling -D $ZD ZO_ECO=ON ZO_ECO_SR=ON ;;
    eco_plain)  run eco_plain  $G  $Q $ZA --l2 1340000 --lr 3e-6 -D $ZD ZO_ECO=ON ZO_ECO_SR=ON ;;
    ne16_prof)  run ne16_prof  $GN $Q $ZA --l2 1340000 --lr 3e-6 --zo-fused-dual --zo-fused-dual-ne16 0,1,2 --profileTiling -D $ZD ZO_ECO=ON ZO_ECO_SR=ON ;;
    ne16_plain) run ne16_plain $GN $Q $ZA --l2 1340000 --lr 3e-6 --zo-fused-dual --zo-fused-dual-ne16 0,1,2 -D $ZD ZO_ECO=ON ZO_ECO_SR=ON ;;
    fus_prof)   run fus_prof   $G  $Q $ZA --l2 1340000 --lr 3e-6 --zo-fused-dual --profileTiling -D $ZD ZO_ECO=ON ZO_ECO_SR=ON ;;
  esac
done
kill_mine
