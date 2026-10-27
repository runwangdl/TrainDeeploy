#!/usr/bin/env bash
# exp21c impl: whole-network fused QZO+ECO on GAP9 (board settings: --l1 114000 --l2 1340000, frozen BN,
# GAP9_NETWORK_O3, ECO + SR, lr 3e-6, --zo-fused-dual --zo-fused-dual-ne16 <blocks>) under the optimisation configs.
# Run inside deeploy_gap9:   bash run_network.sh <lane> "<run> <run> ..."
#   run = <cfg>:<kind>[:<ne16 blocks, default 0,1,2>]
#   cfg  in base|B|BH|BHS  (opt-out flags passed explicitly ON/OFF, see run_blocks.sh)
#   kind plain   one update step (n_accum 4), BENCH + Errors + WSUM
#        prof    the same with --profileTiling (wall clock per node and tile)
#        multi   3 update steps x n_accum 4, DUMP_WEIGHTS after every step (bit-exactness across updates)
#        multi2  2 update steps x n_accum 2, DUMP_WEIGHTS after every step (n_accum != 4)
# Each lane uses its own copy of the train/update fixtures (impl/fixtures/net_<lane>/...) so the generated code of
# concurrent lanes never collides, and its own build dir (worker x21c2_net_<lane>). gvsoc killed only by PID.
# The command line is exp21c's scripts/run_network.sh (= exp9 mem_single_step.sh) plus the flags above. -- QW exp21c
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
WT=/app/Deeploy/.worktrees/exp21c_ne16_encode_opt
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy,sys; assert Deeploy.__file__.startswith("'$WT'"), Deeploy.__file__' || exit 9
E=experiments/deliverable/exp21_remaining_exp/exp21c_Fusion_NE16_Encode_Opt; I=$E/impl; O=$I/logs; CG=$I/codegen
lane=$1; runs=($2)
F=$I/fixtures/net_$lane
mkdir -p $F
for f in speechnet_train_QZOECO_full_fused_round1 speechnet_update_QZOECO_full_fused_round1; do
  [ -d $F/$f ] || cp -r $E/fixtures/$f $F/$f
done
ZD="BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_ECO=ON ZO_ECO_SR=ON"
Q="-t $F/speechnet_train_QZOECO_full_fused_round1 --optimizer-dir $F/speechnet_update_QZOECO_full_fused_round1"
C="--num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --l2 1340000 --lr 3e-6 --defaultMemLevel L2 --cores 8"
GN=deeployMezoRunner_tiled_GAP9_w_NE16.py

kill_mine() {
  for p in $(pgrep -f "[g]vsoc_launcher" 2>/dev/null); do
    if readlink /proc/$p/cwd 2>/dev/null | grep -q "build_x21c2_net_${lane}/"; then kill -9 $p 2>/dev/null; fi
  done
}
flags() {
  local b=ON h=ON s=ON
  case $1 in base) ;; B) b=OFF ;; BH) b=OFF; h=OFF ;; BHS) b=OFF; h=OFF; s=OFF ;; *) echo "bad cfg $1" >&2; exit 2 ;; esac
  echo "NE16DUAL_NOOPT_B=$b NE16DUAL_NOOPT_H=$h NE16DUAL_NOOPT_S=$s"
}
for r in "${runs[@]}"; do
  IFS=: read cfg kind blocks <<< "$r"; blocks=${blocks:-0,1,2}
  tag=${cfg}_${kind}_ne16_${blocks//,/}
  extra=""; steps="--n-steps 1 --n-accum 4"; dump="DUMP_WEIGHTS=OFF"
  case $kind in
    prof)   extra="--profileTiling" ;;
    multi)  steps="--n-steps 3 --n-accum 4"; dump="DUMP_WEIGHTS=ON DUMP_STEP_LO=0 DUMP_STEP_HI=2" ;;
    multi2) steps="--n-steps 2 --n-accum 2"; dump="DUMP_WEIGHTS=ON DUMP_STEP_LO=0 DUMP_STEP_HI=1" ;;
  esac
  log=$O/net_${tag}.log
  kill_mine
  PYTEST_XDIST_WORKER=x21c2_net_$lane timeout 10800 python3 $GN $Q $C $steps --zo-fused-dual --zo-fused-dual-ne16 $blocks \
      $extra -D $ZD $dump NE16DUAL_PROBE_WALL=OFF $(flags $cfg) > $log 2>&1
  kill_mine
  d=$(ls -d TEST_GAP9_W_NE16/$F/speechnet_train_QZOECO_full_fused_round1 2>/dev/null | head -1)
  [ $kind == plain ] && for f in $d/*Network.c; do cp $f $CG/net_${tag}_$(basename $f); done
  echo "$(date +%H:%M) $lane $tag: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) | $(grep -a '^BENCH' $log | head -1) | $(grep -a 'WSUM s=' $log | tail -1)" | tee -a $O/network_summary.txt
done
kill_mine
