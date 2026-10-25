#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
#
# exp11: per-node breakdown of blocks 1-4 on GAP9 (state fe1ffa2 = before exp9), the exp6 block-0 table for each block.
# Run inside deeploy_gap9 (TrainDeeploy at /app/Deeploy):  bash run_breakdown.sh [blocks...] -- variants via VARIANTS env
#   A    2 x pulp-nn (cluster)                    QZO_Conv_B<i>              deeployRunner_tiled_gap9.py         (x2)
#   B    2 x NE16 im2col route                    QZO_Conv_NE16_B<i>         deeployRunner_tiled_gap9_w_ne16.py  (x2)
#   Bp   2 x Conv4x2 (all non-fusion opts)        QZO_Conv_B<i> + DEEPLOY_RQCONV_KERNEL=4x2                      (x2)
#   Ccl  fused cluster DualConv2In                QZO_DualConv2in_B<i>       deeployRunner_tiled_gap9.py
#   C    fused NE16, pipelined (shipped form)     QZO_DualConv2in_NE16P_B<i> deeployRunner_tiled_gap9_w_ne16.py
#   Cpr  Ccl with -D DUALCONV2IN_PROBE=ON (per-core phase-1 / phase-2 work cycles), plain only
# Every variant plain (Runtime = total) and --profileMicrobenchmark (per-node wall time: on GAP9 the wrapper runs on the
# cluster master core 8, which dispatches the workers and waits). All with -D GAP9_NETWORK_O3=ON (board setting).
# One build directory per (runner, CMake flags, mode) -- shared across blocks, never across flag sets. -- QW
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
E=experiments/dual_pass_fusion/exp11_b1to4_breakdown; O=$E/logs; CG=$E/codegen; mkdir -p $O $CG
blocks=("$@"); [ ${#blocks[@]} -eq 0 ] && blocks=(1 2 3 4)
variants=(${VARIANTS:-A B Bp Ccl C Cpr})
W=${WTAG:-x11a}
M="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8 -D GAP9_NETWORK_O3=ON"

run() {  # block variant mode runner fixture typeflags kernelenv extra worker
  local i=$1 v=$2 mode=$3 runner=$4 fix=$5 tf=$6 kenv=$7 extra=$8 wk=$9
  local mb=""; [ $mode = mb ] && mb="--profileMicrobenchmark"
  local log=$O/b${i}_${v}_${mode}.log
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
  env DEEPLOY_RQCONV_KERNEL=$kenv PYTEST_XDIST_WORKER=$wk python3 $runner -t Tests/Kernels/Integer/$fix $M $tf $extra $mb > $log 2>&1
  local d; d=$(ls -d TEST_GAP9*/Tests/Kernels/Integer/$fix 2>/dev/null | head -1)
  [ $mode = plain ] && [ -n "$d" ] && cp $d/Network.c $CG/b${i}_${v}_Network.c
  echo "b$i $v $mode: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' $log | head -1) $(grep -aE '^Runtime: [0-9]+' $log | head -1) records=$(grep -ao 'records=[0-9]*' $log | head -1)"
}

for i in "${blocks[@]}"; do
  A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
  T1="--input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0"
  T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"
  for v in "${variants[@]}"; do
    case $v in
      A)   for m in plain mb; do run $i A   $m deeployRunner_tiled_gap9.py        QZO_Conv_B$i              "$T1" pulpnn "" ${W}_g9_$m; done ;;
      Bp)  for m in plain mb; do run $i Bp  $m deeployRunner_tiled_gap9.py        QZO_Conv_B$i              "$T1" 4x2    "" ${W}_g9_$m; done ;;
      Ccl) for m in plain mb; do run $i Ccl $m deeployRunner_tiled_gap9.py        QZO_DualConv2in_B$i       "$T2" pulpnn "" ${W}_g9_$m; done ;;
      B)   for m in plain mb; do run $i B   $m deeployRunner_tiled_gap9_w_ne16.py QZO_Conv_NE16_B$i         "$T1" pulpnn "" ${W}_ne_$m; done ;;
      C)   for m in plain mb; do run $i C   $m deeployRunner_tiled_gap9_w_ne16.py QZO_DualConv2in_NE16P_B$i "$T2" pulpnn "" ${W}_ne_$m; done ;;
      Cpr) run $i Cpr plain deeployRunner_tiled_gap9.py QZO_DualConv2in_B$i "$T2" pulpnn "-D DUALCONV2IN_PROBE=ON" ${W}_g9_probe ;;
    esac
  done
done
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9
