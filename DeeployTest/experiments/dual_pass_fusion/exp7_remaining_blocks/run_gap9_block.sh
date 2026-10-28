#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
#
# exp7 step 7: exp6's four GAP9 measurements for block i (container deeploy_gap9, TrainDeeploy at /app/Deeploy):
#   A    2 x pulp-nn on the GAP9 cluster        QZO_Conv_B<i>             deeployRunner_tiled_gap9.py        (x2 = both passes)
#   B    2 x NE16 im2col route, single pass     QZO_Conv_NE16_B<i>        deeployRunner_tiled_gap9_w_ne16.py (x2)
#   C_cl fused two-input kernel on the cluster  QZO_DualConv2in_B<i>      deeployRunner_tiled_gap9.py
#   C    fused on NE16 (two jobs, shared prep)  QZO_DualConv2in_NE16_B<i> deeployRunner_tiled_gap9_w_ne16.py
#   C_S / C_P  one NE16DualConv2in node: word-wise HWC column, sequential / SDK-pipelined (QZO_DualConv2in_NE16{S,P}_B<i>)
# each plain (valid Runtime) and with --profileMicrobenchmark (master-core = per-node wall time on GAP9, exp6 §5).
#   bash run_gap9_block.sh <i> [variant ...]      default: A B C_cl C
set -uo pipefail
i=${1:?block}; shift
variants=("$@"); [ ${#variants[@]} -eq 0 ] && variants=(A B C_cl C)
letter=$(printf "\\x$(printf %x $((96 + i)))")
H=$(cd "$(dirname "$0")" && pwd)
OUT=$H/exp7${letter}_block${i}/block${i}_fused_NE16
REL=DeeployTest/experiments/dual_pass_fusion/exp7_remaining_blocks/exp7${letter}_block${i}/block${i}_fused_NE16
mkdir -p "$OUT"/{logs,results,codegen}
A="/blocks_${i}_conv_input_quant_1/Clip_output_0"; B="blocks.${i}.conv.bias_rqsadd"
# L1 86000 = Run's kernel-fixture setting (silentwear-zo scripts/run_smoke_local.sh G9K, notes §4.5-4.7): the GAP9 kernel
# harness (deeploytest.c) keeps the cluster stacks in L1 (SLAVESTACKSIZE 3800 + 14 kB master stack), so exp6's 92000 no
# longer fits on this branch (exp6's own block-0 fixture fails at 92000). Same as her block-0 kernel numbers.
P="--l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8"
T1="--input-type-map $B=int32_t $A=int8_t --input-offset-map $B=0 $A=0"
T2="--input-type-map $B=int32_t $A=int8_t ${A}_minus=int8_t --input-offset-map $B=0 $A=0 ${A}_minus=0"

run() {  # $1 variant, $2 runner, $3 test, $4 type flags
  for mode in plain microbench; do
    local extra=""; [ $mode = microbench ] && extra="--profileMicrobenchmark"
    local log=/app/Deeploy/$REL/logs/gap9_$1_$mode.log
    docker exec deeploy_gap9 bash -lc "
      source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
      source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
      export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
      pgrep -f '[g]vsoc_launcher' | xargs -r kill -9
      cd /app/Deeploy/DeeployTest
      rm -rf TEST_GAP9*/build_e7$1 2>/dev/null
      PYTEST_XDIST_WORKER=e7$1 python3 $2 -t Tests/Kernels/Integer/$3 $P $4 $extra > $log 2>&1
      if [ $mode = plain ]; then f=\$(ls -d TEST_GAP9*/Tests/Kernels/Integer/$3 2>/dev/null | head -1); [ -n \"\$f\" ] && cp \$f/Network.c /app/Deeploy/$REL/codegen/Network_$1.c; fi"
    echo "  block $i $1 $mode: $(grep -aE 'Errors: [0-9]+ out of [0-9]+' "$OUT/logs/gap9_$1_$mode.log" | head -1)  $(grep -aE '^Runtime:' "$OUT/logs/gap9_$1_$mode.log" | head -1)"
  done
  python3 "$H/../exp6_NE16_fusion/parse_microbench.py" "$OUT/logs/gap9_$1_microbench.log" "$OUT/results/gap9_$1_per_node" > /dev/null 2>&1 \
    || echo "  (no per-node table for $1)"
}

for v in "${variants[@]}"; do
  case $v in
    A)    run A    deeployRunner_tiled_gap9.py        QZO_Conv_B$i             "$T1" ;;
    B)    run B    deeployRunner_tiled_gap9_w_ne16.py QZO_Conv_NE16_B$i        "$T1" ;;
    C_cl) run C_cl deeployRunner_tiled_gap9.py        QZO_DualConv2in_B$i      "$T2" ;;
    C)    run C    deeployRunner_tiled_gap9_w_ne16.py QZO_DualConv2in_NE16_B$i "$T2" ;;
    C_S)  run C_S  deeployRunner_tiled_gap9_w_ne16.py QZO_DualConv2in_NE16S_B$i "$T2" ;;   # one node, word-wise, sequential
    C_P)  run C_P  deeployRunner_tiled_gap9_w_ne16.py QZO_DualConv2in_NE16P_B$i "$T2" ;;   # one node, word-wise, pipelined
  esac
done
