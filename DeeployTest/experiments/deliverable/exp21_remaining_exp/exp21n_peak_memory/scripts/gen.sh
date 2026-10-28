#!/usr/bin/env bash
# exp21n: peak-memory comparison BP / ZO / QZO-ECO on GAP9 -- static memory plans from Deeploy code generation (+ build,
# no simulation: --skipsim). Same conditions for every recipe: S01 round-1 fixtures of Board_exp (BP lr 3e-5; ZO and
# QZO-ECO = the board fixtures), one update step (n_accum 4), EVERY tensor homed in L2 (--defaultMemLevel L2,
# --l2 1600000), no promotion, no double buffering, MiniMalloc, the board --l1 of each recipe.
# Run inside deeploy_gap9:  bash scripts/gen.sh [bp zo eco]
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
E=experiments/deliverable/exp21_remaining_exp/exp21n_peak_memory; B=experiments/Board_exp
L2=${L2:-1600000}
ZA="--n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --defaultMemLevel L2 --l2 $L2 --cores 8"
for c in ${@:-bp zo eco}; do
  case $c in
    bp)  R=deeployTrainingRunner_tiled_gap9.py; FX=$B/BP/fixtures/speechnet_train_BP_r1
         A="-t $FX --n-steps 1 --n-accum 4 --num-data-inputs 2 --l1 122000 --l2 $L2 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --cores 8"
         D="BN_FROZEN_STATS=ON CC_STACK_SIZE=4096" ;;
    zo)  R=deeployMezoRunner_tiled_GAP9.py; FX=$B/ZOfam/fixtures/speechnet_train_ZO_r1
         A="-t $FX --optimizer-dir $B/ZOfam/fixtures/speechnet_update_ZO_r1 $ZA --l1 114000 --lr 3e-6"
         D="BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON" ;;
    eco) R=deeployMezoRunner_tiled_GAP9_w_NE16.py; FX=$B/ZOfam/fixtures/speechnet_train_QZO_r1
         A="-t $FX --optimizer-dir $B/ZOfam/fixtures/speechnet_update_QZO_r1 $ZA --l1 114000 --lr 3e-6 --zo-fused-dual --zo-fused-dual-ne16 0,1,2"
         D="BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_ECO=ON ZO_ECO_SR=ON" ;;
  esac
  PYTEST_XDIST_WORKER=x21n_$c python3 $R $A --skipsim -D $D > $E/logs/gen_$c.log 2>&1
  P=TEST_GAP9; [[ $R == *NE16* ]] && P=TEST_GAP9_W_NE16
  mkdir -p $E/codegen/$c; cp $P/$FX/*Network.[ch] $E/codegen/$c/ 2>/dev/null
  cp $P/$FX/deeployStates/memory_alloc.html $E/codegen/$c/ 2>/dev/null
  echo "$c: $(grep -aE 'PASSED|FAILED|Error' $E/logs/gen_$c.log | tail -1) -> $(ls $E/codegen/$c | tr '\n' ' ')"
done
