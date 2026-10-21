#!/usr/bin/env bash
# exp9 verification on Siracusa (run inside traindeeploy): exp8 fused setting + the unfused frozen-BN QZO, 4 steps with
# weight dumps, without (TAIL=0) and with the pass; then all Siracusa SpeechNet CI gates.

cd /app/ETH/TrainDeeploy/DeeployTest; O=experiments/dual_pass_fusion/exp9_memory_optim/verify; T=../SpeechNet_Test
C="-t $T/speechnet_train_QZO_single_step --optimizer-dir $T/speechnet_update_QZO_single_step --n-steps 4 --n-accum 1 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON -D DUMP_WEIGHTS=ON -D DUMP_STEP_LO=0 -D DUMP_STEP_HI=3"
run() {  # tag tail args...
  local tag=$1 tail=$2; shift 2
  rm -rf TEST_SIRACUSA/build_v9s_$tag
  DEEPLOY_FUSE_BN_TAIL=$tail PYTEST_XDIST_WORKER=v9s_$tag python3 deeployMezoRunner_tiled_siracusa.py $C "$@" > $O/sir_$tag.log 2>&1
  cp TEST_SIRACUSA/speechnet_train_QZO_single_step/TrainingNetwork.c $O/sir_$tag.TrainingNetwork.c
  echo "sir_$tag: $(grep -aE 'Errors: [0-9]+ out of' $O/sir_$tag.log) fused_calls=$(grep -c 'PULP_FrozenBNPoolReluQuantHWC_i8_i8(' $O/sir_$tag.TrainingNetwork.c) $(grep -o 'pi_l2_malloc(sizeof(int8_t) \* [0-9]*)' $O/sir_$tag.TrainingNetwork.c | head -1) $(grep -aoE 'train_cycles_lo=[0-9]+' $O/sir_$tag.log)"
}
run fused_tail1 1 --zo-fused-dual
run unfused_tail0 0
run unfused_tail1 1
python3 -m pytest test_platforms.py -k "siracusa and (SpeechNet or speechnet)" -v > $O/ci_siracusa_speechnet.log 2>&1; tail -18 $O/ci_siracusa_speechnet.log
