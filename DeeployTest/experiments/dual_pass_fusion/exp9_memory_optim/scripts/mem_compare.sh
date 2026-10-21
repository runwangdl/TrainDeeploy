#!/usr/bin/env bash
# exp9 add-on (2026-10-16): L2 memory plan of the four board configurations (Run's pw_sw_build_all.sh), code generation
# only (--skipsim), with --plotMemAlloc. QZO configs twice: DEEPLOY_FUSE_BN_TAIL=0 (= board code fe1ffa2) and 1 (exp9).
# Run inside deeploy_gap9. Output: memcompare/<tag>/{gen.log, *Network.c, deeployStates plots}.
source /app/install/gap9-sdk/.gap9-venv/bin/activate; source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest; O=experiments/dual_pass_fusion/exp9_memory_optim/memcompare; T=../SpeechNet_Test
F="-t $T/speechnet_train_QZOECO_full_fused_round1 --optimizer-dir $T/speechnet_update_QZOECO_full_fused_round1"
ZA="--n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --defaultMemLevel L2 --cores 8 --skipsim --plotMemAlloc"
ZD="-D BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_QUIET=ON"
gen() {  # tag tail runner fixture args...
  local tag=$1 tail=$2 runner=$3 fix=$4; shift 4
  local plat=TEST_GAP9; [[ $runner == *NE16* ]] && plat=TEST_GAP9_W_NE16
  mkdir -p $O/$tag
  DEEPLOY_FUSE_BN_TAIL=$tail PYTEST_XDIST_WORKER=mc_$tag python3 $runner "$@" > $O/$tag/gen.log 2>&1
  cp $plat/$fix/*Network.c $O/$tag/ 2>/dev/null
  cp -r $plat/$fix/deeployStates $O/$tag/ 2>/dev/null
  echo "$tag: exit=$? $(for f in $O/$tag/*Network.c; do echo "$(basename $f) $(grep -o 'pi_l2_malloc(sizeof(int8_t) \* [0-9]*)' $f | tr '\n' ' ')"; done | tr '\n' ' ')"
}
gen bp      1 deeployTrainingRunner_tiled_gap9.py speechnet_train_BP_round1 -t $T/speechnet_train_BP_round1 --n-steps 1 --n-accum 4 \
   --num-data-inputs 2 --l1 122000 --l2 1300000 --defaultMemLevel L3 --doublebuffer --promoteToL2 \
   --promoteToL2Strategy=traffic-per-peak --promoteToL2Headroom=131072 --memAllocStrategy MiniMalloc --cores 8 --skipsim --plotMemAlloc \
   -D BN_FROZEN_STATS=ON CC_STACK_SIZE=4096
gen zo      1 deeployMezoRunner_tiled_GAP9.py speechnet_train_ZO_round1 -t $T/speechnet_train_ZO_round1 --optimizer-dir $T/speechnet_update_ZO_round1 $ZA --l2 1300000 --lr 3e-6 $ZD
gen eco_unf_t0 0 deeployMezoRunner_tiled_GAP9.py speechnet_train_QZOECO_full_fused_round1 $F $ZA --l2 1340000 --lr 3e-6 $ZD ZO_ECO=ON ZO_ECO_SR=ON
gen eco_unf_t1 1 deeployMezoRunner_tiled_GAP9.py speechnet_train_QZOECO_full_fused_round1 $F $ZA --l2 1340000 --lr 3e-6 $ZD ZO_ECO=ON ZO_ECO_SR=ON
gen eco_fus_t0 0 deeployMezoRunner_tiled_GAP9_w_NE16.py speechnet_train_QZOECO_full_fused_round1 $F $ZA --zo-fused-dual --zo-fused-dual-ne16 0,1,2 --l2 1340000 --lr 3e-6 $ZD ZO_ECO=ON ZO_ECO_SR=ON
gen eco_fus_t1 1 deeployMezoRunner_tiled_GAP9_w_NE16.py speechnet_train_QZOECO_full_fused_round1 $F $ZA --zo-fused-dual --zo-fused-dual-ne16 0,1,2 --l2 1340000 --lr 3e-6 $ZD ZO_ECO=ON ZO_ECO_SR=ON
