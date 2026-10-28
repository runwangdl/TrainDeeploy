#!/usr/bin/env bash
# merge gates 2026-10-27 (feat/memory_opt after merging exp21e + exp21c, 92027907), GAP9 part. Run INSIDE deeploy_gap9:
#   bash run_gap9_gates.sh
# 1) whole-network fused QZO+ECO update step on NE16 (exp21c run_network.sh "plain", board settings), blocks 0-2 and
#    0-3: expect Errors 0/8 and WSUM s=0 0xfa1f7dd2 (= exp21c checkpoint / branch runs)
# 2) SpeechNet GAP9 CI gates (pytest -m "gap9_tiled and speechnet_gap9", exact ZO_EXPECT_WSUM hashes)
# gvsoc killed only by PID, only processes of this script's build dirs.                                    -- QW
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
WT=/app/Deeploy
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy,sys; assert Deeploy.__file__.startswith("'$WT'/"), Deeploy.__file__' || exit 9
G=experiments/deliverable/exp21_remaining_exp/merge_gates_2026-10-27; O=$G/logs
E=experiments/deliverable/exp21_remaining_exp/exp21c_Fusion_NE16_Encode_Opt
F=$G/fixtures; mkdir -p $F
for f in speechnet_train_QZOECO_full_fused_round1 speechnet_update_QZOECO_full_fused_round1; do
  [ -d $F/$f ] || cp -r $E/fixtures/$f $F/$f
done
kill_mine() {
  for p in $(pgrep -f "[g]vsoc_launcher" 2>/dev/null); do
    readlink /proc/$p/cwd 2>/dev/null | grep -q "build_mg1027" && kill -9 $p 2>/dev/null
  done
}
Q="-t $F/speechnet_train_QZOECO_full_fused_round1 --optimizer-dir $F/speechnet_update_QZOECO_full_fused_round1"
C="--num-data-inputs 2 --eps 0.01 --q 1 --seed 42 --l1 114000 --l2 1340000 --lr 3e-6 --defaultMemLevel L2 --cores 8"
for blocks in 0,1,2 0,1,2,3; do
  tag=net_BHS_plain_ne16_${blocks//,/}; log=$O/gap9_$tag.log
  kill_mine
  PYTEST_XDIST_WORKER=build_mg1027_$tag timeout 10800 python3 deeployMezoRunner_tiled_GAP9_w_NE16.py $Q $C \
    --n-steps 1 --n-accum 4 --zo-fused-dual --zo-fused-dual-ne16 $blocks \
    -D BN_FROZEN_STATS=ON GAP9_NETWORK_O3=ON ZO_ECO=ON ZO_ECO_SR=ON DUMP_WEIGHTS=OFF NE16DUAL_PROBE_WALL=OFF > $log 2>&1
  echo "EXIT $?" >> $log
  kill_mine
  echo "$tag: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) | $(grep -a '^BENCH ' $log | head -1) | $(grep -a 'WSUM s=' $log | tail -1)" | tee -a $O/gap9_summary.txt
done
kill_mine
PYTEST_XDIST_WORKER=build_mg1027_ci python3 -m pytest test_platforms.py -v -s -m "gap9_tiled and speechnet_gap9" > $O/gap9_ci_speechnet.log 2>&1
echo "EXIT $?" >> $O/gap9_ci_speechnet.log
kill_mine
echo "CI: $(grep -aE '^(=+ .*(passed|failed).* =+)$' $O/gap9_ci_speechnet.log | tail -1)" | tee -a $O/gap9_summary.txt
echo DONE >> $O/gap9_summary.txt
