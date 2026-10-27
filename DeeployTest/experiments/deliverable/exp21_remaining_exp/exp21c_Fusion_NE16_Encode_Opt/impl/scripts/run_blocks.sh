#!/usr/bin/env bash
# exp21c impl: per-block NE16 fixtures under the optimisation configs, GAP9 GVSoC, exp11 settings
# (--l1 86000 --l2 1500000, GAP9_NETWORK_O3=ON, 8 cores). Run inside deeploy_gap9 (from anywhere):
#   bash run_blocks.sh <lane-fixture> "<run> <run> ..."      run = <cfg>:<variant>, cfg in base|B|BH|BHS,
#                                                            variant in plain|wall|mb
# cfg -> opt-out flags (ALWAYS passed explicitly ON/OFF, so a reused build dir never keeps a stale cache value):
#   base = NOOPT_B,H,S all ON (= the exp21 checkpoint code)   B = only B on   BH = B+H on   BHS = all on
# Fixtures:
#   B0          impl/fixtures/QZO_DualConv_NE16_B0            (copy of Tests/Kernels/Integer/QZO_DualConv_NE16, block 0,
#                                                              stacked [W+;W-] 1x1 job, input_offset 128)
#   off0_B<i>   impl/fixtures/QZO_DualConv2in_NE16P_off0_B<i>  (exp21c fixtures with ne16_input_offset = 0 on the bias
#                                                              and conv node, as in the network; inputs are >= 0)
#   B<i>        ../fixtures/QZO_DualConv2in_NE16P_B<i>         (exp11 fixtures, input_offset 128)
# One build dir per lane (worker x21c2_blk_<lane>); gvsoc killed only by PID (cwd in a build_x21c2_blk_<lane> dir).
# Deeploy is imported from THIS worktree (PYTHONPATH). -- QW exp21c
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
WT=/app/Deeploy/.worktrees/exp21c_ne16_encode_opt
export PYTHONPATH=$WT
cd $WT/DeeployTest
python3 -c 'import Deeploy,sys; assert Deeploy.__file__.startswith("'$WT'"), Deeploy.__file__' || exit 9
E=experiments/deliverable/exp21_remaining_exp/exp21c_Fusion_NE16_Encode_Opt; I=$E/impl; O=$I/logs; CG=$I/codegen
lane=$1; runs=($2)
M="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8"
NE=deeployRunner_tiled_gap9_w_ne16.py

case $lane in
  B0) FIX=$I/fixtures/QZO_DualConv_NE16_B0
      T2="--input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0" ;;
  off0_B*|B*) i=${lane: -1}
      [[ $lane == off0_* ]] && FIX=$I/fixtures/QZO_DualConv2in_NE16P_off0_B$i || FIX=$E/fixtures/QZO_DualConv2in_NE16P_B$i
      A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
      T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0" ;;
esac

kill_mine() {
  for p in $(pgrep -f "[g]vsoc_launcher" 2>/dev/null); do
    if readlink /proc/$p/cwd 2>/dev/null | grep -q "build_x21c2_blk_${lane}/"; then kill -9 $p 2>/dev/null; fi
  done
}
flags() {  # cfg -> -D list
  local b=ON h=ON s=ON
  case $1 in base) ;; B) b=OFF ;; BH) b=OFF; h=OFF ;; BHS) b=OFF; h=OFF; s=OFF ;; *) echo "bad cfg $1" >&2; exit 2 ;; esac
  echo "NE16DUAL_NOOPT_B=$b NE16DUAL_NOOPT_H=$h NE16DUAL_NOOPT_S=$s"
}
for r in "${runs[@]}"; do
  cfg=${r%%:*}; var=${r##*:}
  extra=""; probe=OFF
  [ $var == wall ] && probe=ON
  [ $var == mb ] && extra="--profileMicrobenchmark"
  log=$O/blk_${lane}_${cfg}_${var}.log
  kill_mine
  PYTEST_XDIST_WORKER=x21c2_blk_$lane timeout 3600 python3 $NE -t $FIX $M $T2 $extra \
      -D GAP9_NETWORK_O3=ON NE16DUAL_PROBE_WALL=$probe $(flags $cfg) > $log 2>&1
  kill_mine
  d=$(ls -d TEST_GAP9*/$FIX 2>/dev/null | head -1)
  [ $var == plain ] && [ -n "$d" ] && cp $d/Network.c $CG/blk_${lane}_${cfg}_Network.c
  echo "$(date +%H:%M) $lane $cfg $var: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) $(grep -aE '^Runtime: [0-9]+' $log | head -1)" | tee -a $O/blocks_summary.txt
done
kill_mine
