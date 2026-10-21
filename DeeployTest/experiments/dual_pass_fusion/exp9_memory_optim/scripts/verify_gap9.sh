#!/usr/bin/env bash
# exp9 verification on GAP9 (run inside deeploy_gap9): every configuration the new pass can reach, without (TAIL=0)
# and with it; logs + generated TrainingNetwork.c + lowered graph per run in verify/.

source /app/install/gap9-sdk/.gap9-venv/bin/activate; source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest; O=experiments/dual_pass_fusion/exp9_memory_optim/verify; T=../SpeechNet_Test
run() {  # tag tail runner fixture-name args...
  local tag=$1 tail=$2 runner=$3 fix=$4; shift 4
  local plat=TEST_GAP9; [[ $runner == *NE16* ]] && plat=TEST_GAP9_W_NE16
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
  rm -rf $plat/build_v9_$tag
  DEEPLOY_FUSE_BN_TAIL=$tail PYTEST_XDIST_WORKER=v9_$tag python3 $runner "$@" > $O/$tag.log 2>&1
  cp $plat/$fix/TrainingNetwork.c $O/$tag.TrainingNetwork.c
  cp $plat/$fix/deeployStates/middleware_post_lowering.onnx $O/$tag.post_lowering.onnx
  echo "$tag: $(grep -aE 'Errors: [0-9]+ out of|WSUM\] final' $O/$tag.log | tr '\n' ' ') fused_calls=$(grep -c 'PULP_FrozenBNPoolReluQuantHWC_i8_i8(' $O/$tag.TrainingNetwork.c) $(grep -o 'pi_l2_malloc(sizeof(int8_t) \* [0-9]*)' $O/$tag.TrainingNetwork.c | head -1) $(grep -aoE 'train_cycles_lo=[0-9]+' $O/$tag.log)"
}
NE=deeployMezoRunner_tiled_GAP9_w_NE16.py; G=deeployMezoRunner_tiled_GAP9.py
FS="-t $T/speechnet_train_QZOECO_full_fused_single_update_step --optimizer-dir $T/speechnet_update_QZOECO_full_fused_single_update_step --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --lr 3e-3 --l1 114000 --l2 1340000 --defaultMemLevel L2 --cores 8 --zo-fused-dual --zo-fused-dual-ne16 0,1,2 -D BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_ECO=ON ZO_EXPECT_WSUM=0xb11cc64f"
R1="-t $T/speechnet_train_QZOECO_full_fused_round1 --optimizer-dir $T/speechnet_update_QZOECO_full_fused_round1 --n-steps 14 --n-accum 4 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --lr 1e-5 --l1 114000 --l2 1340000 --defaultMemLevel L2 --cores 8 --zo-fused-dual --zo-fused-dual-ne16 0,1,2 -D BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON"
UQ="-t $T/speechnet_train_QZO_single_step --optimizer-dir $T/speechnet_update_QZO_single_step --n-accum 1 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --l2 1500000 --defaultMemLevel L2 --cores 8"
ZO="-t $T/speechnet_train_ZO_single_step --optimizer-dir $T/speechnet_update_ZO_single_step --n-steps 1 --n-accum 1 --num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --lr 3e-6 --l1 114000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D BN_FROZEN_STATS=ON ZO_EXPECT_WSUM=0xc14819e6"
run fused_step_tail0 0 $NE speechnet_train_QZOECO_full_fused_single_update_step $FS
run fused_step_tail1 1 $NE speechnet_train_QZOECO_full_fused_single_update_step $FS
run round1_tail1     1 $NE speechnet_train_QZOECO_full_fused_round1 $R1
run unfused_qzo_tail0 0 $G speechnet_train_QZO_single_step $UQ --n-steps 1 --lr 1e-5 -D BN_FROZEN_STATS=ON ZO_EXPECT_WSUM=0x97e57471
run unfused_qzo_tail1 1 $G speechnet_train_QZO_single_step $UQ --n-steps 1 --lr 1e-5 -D BN_FROZEN_STATS=ON ZO_EXPECT_WSUM=0x97e57471
run unfused_eco5_tail1 1 $G speechnet_train_QZO_single_step $UQ --n-steps 5 --lr 3e-3 -D BN_FROZEN_STATS=ON ZO_ECO=ON ZO_PERSIST=ON ZO_EXPECT_WSUM=0x6706ea8a
run zo_fp32_tail1    1 $G speechnet_train_ZO_single_step $ZO
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
python3 -m pytest test_platforms.py -m "gap9_tiled and speechnet_gap9" -v > $O/ci_gap9_speechnet.log 2>&1; tail -12 $O/ci_gap9_speechnet.log
